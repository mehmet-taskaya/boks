"""
Proje 2: Kamerayla El & Vücut Takibi, Hareketle Fare Kontrolü ve Tam Ekran Boks Oyunu
Kütüphaneler: opencv-python, mediapipe, numpy

Özellikler:
1. Webcam'den canlı görüntü alma (ayna görünümü).
2. Eli tespit edip 21 eklem noktasını (landmarks) ve bağlantılarını ekrana çizme.
3. Vücut (Pose) tespiti ile baş, omuz, kol ve gövde iskeletini çizme; sağa/sola/aşağı eğilmeyi algılama.
4. Kaç parmağın açık olduğunu ve yumruk (boks) vuruşlarını gerçek zamanlı hesaplayıp ekrana yazma.
5. BONUS 1 - FARE KONTROLÜ:
   - Başparmak + İşaret parmağını birleştirerek (Pinch) fare imlecini yönetme.
   - Serçe parmağı 1x aç-kapat -> Tek Sol Tık, 2x aç-kapat -> Çift Sol Tık.
   - Orta parmağı kaldırıp indirince Sağ Tık.
   - Yumruk yaparak (0 parmak açık) ekrandaki nesneleri ve masaüstü pencerelerini/sayfalarını kaydırma.
6. BONUS 2 - [G] TUŞU İLE TAM EKRAN BOKS OYUNU MODU:
   - [G] tuşuna basınca tam ekran interaktif Boks Oyunu moduna geçer.
   - Karşıdaki rakibin yumruklarından vücudunuzla SAĞA veya SOLA eğilerek kaçın.
   - Yumruklarınızı gerçek zamanlı algılar ve ekranda boks eldiveni/efekt olarak gösterir.
   - Yumruk atarak karşıdaki rakibin canını bitirip NAKAVT (K.O.) ile yenin!
"""

from __future__ import annotations

import ctypes
import math
import os
import random
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


def disable_windows_background_throttling() -> None:
    """
    Windows 11'de pencere arka plana düştüğünde (masaüstüne tıklandığında)
    işletim sisteminin Python sürecini 'EcoQoS / Verimlilik Modu'na alıp
    FPS'i düşürmesini engeller ve zamanlayıcı çözünürlüğünü 1ms yapar.
    """
    if sys.platform != "win32":
        return
    try:
        kernel32 = ctypes.windll.kernel32
        # ABOVE_NORMAL_PRIORITY_CLASS (0x00008000)
        current_process = kernel32.GetCurrentProcess()
        kernel32.SetPriorityClass(current_process, 0x00008000)

        # PROCESS_POWER_THROTTLING_STATE yapısıyla arka plan güç kısıtlamasını kapat
        class PROCESS_POWER_THROTTLING_STATE(ctypes.Structure):
            _fields_ = [
                ("Version", ctypes.c_ulong),
                ("ControlMask", ctypes.c_ulong),
                ("StateMask", ctypes.c_ulong),
            ]

        throttling = PROCESS_POWER_THROTTLING_STATE()
        throttling.Version = 1
        throttling.ControlMask = 0x1  # PROCESS_POWER_THROTTLING_EXECUTION_SPEED
        throttling.StateMask = 0x0    # Kısıtlamayı devre dışı bırak (0)
        # ProcessPowerThrottling = 4
        kernel32.SetProcessInformation(
            current_process,
            4,
            ctypes.byref(throttling),
            ctypes.sizeof(throttling),
        )
    except Exception:
        pass

    try:
        # Windows uyku/zamanlayıcı hassasiyetini 15.6ms yerine 1ms yap
        ctypes.windll.winmm.timeBeginPeriod(1)
    except Exception:
        pass


def focus_window(window_title: str) -> None:
    """
    Windows üzerinde belirtilen pencereyi en öne getirir, tam ekran yapar
    ve kullanıcı klavye/fare odağını doğrudan oyun penceresine verir.
    """
    if sys.platform != "win32":
        return
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, window_title)
        if hwnd:
            user32.ShowWindow(hwnd, 3)  # SW_MAXIMIZE
            user32.SetForegroundWindow(hwnd)
            user32.BringWindowToTop(hwnd)
    except Exception:
        pass


class ThreadedCamera:
    """
    Kameradan kare okuma işlemini ayrı bir arka plan iş parçacığında (thread)
    yürütür. Böylece MediaPipe çalışırken kamera sürücüsü beklemez ve masaüstünde
    FPS düşüşü yaşanmaz.
    """

    def __init__(self, index: int = 0, width: int = 960, height: int = 540) -> None:
        # Windows'ta DirectShow (CAP_DSHOW) MSMF'ye göre çok daha düşük gecikmelidir
        if sys.platform == "win32":
            self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(index)
        else:
            self.cap = cv2.VideoCapture(index)

        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, 60)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.lock = threading.Lock()
        self.running = False
        self.frame: Optional[np.ndarray] = None
        self.success = False
        self.new_frame_event = threading.Event()

        if self.cap.isOpened():
            ok, first_frame = self.cap.read()
            if ok and first_frame is not None:
                self.success = True
                self.frame = first_frame
                self.new_frame_event.set()
            self.running = True
            self.thread = threading.Thread(target=self._reader_loop, daemon=True)
            self.thread.start()

    def isOpened(self) -> bool:
        return self.cap.isOpened()

    def _reader_loop(self) -> None:
        while self.running:
            ok, frm = self.cap.read()
            if ok and frm is not None:
                with self.lock:
                    self.success = True
                    self.frame = frm
                self.new_frame_event.set()
            else:
                time.sleep(0.005)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        # Yeni kare gelene kadar en fazla 35ms bekle (CPU'yu boşuna %100 yormaz)
        self.new_frame_event.wait(timeout=0.035)
        self.new_frame_event.clear()
        with self.lock:
            if not self.success or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def release(self) -> None:
        self.running = False
        try:
            if hasattr(self, "thread") and self.thread.is_alive():
                self.thread.join(timeout=0.5)
        except Exception:
            pass
        self.cap.release()


# ============================================================================
# WINDOWS NATIVE MOUSE CONTROLLER (Ek paket gerektirmeden sıfır gecikmeli)
# ============================================================================
class WindowsMouseController:
    """Windows user32 API üzerinden hızlı fare kontrolü (SetCursorPos & mouse_event)."""

    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004
    MOUSEEVENTF_RIGHTDOWN = 0x0008
    MOUSEEVENTF_RIGHTUP = 0x0010
    MOUSEEVENTF_WHEEL = 0x0800

    def __init__(self) -> None:
        self.available = sys.platform == "win32"
        self.left_Button_down = False
        if self.available:
            try:
                # DPI ölçeklendirmesinden etkilenmemek için Process DPI Awareness aktif et
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except Exception:
                try:
                    ctypes.windll.user32.SetProcessDPIAware()
                except Exception:
                    pass
            self.user32 = ctypes.windll.user32
            self.screen_w = int(self.user32.GetSystemMetrics(0))
            self.screen_h = int(self.user32.GetSystemMetrics(1))
        else:
            self.user32 = None
            self.screen_w = 1920
            self.screen_h = 1080

    def move(self, x: int, y: int) -> None:
        if not self.available:
            return
        cx = max(0, min(self.screen_w - 1, int(x)))
        cy = max(0, min(self.screen_h - 1, int(y)))
        self.user32.SetCursorPos(cx, cy)

    def left_click(self, pos: Optional[Tuple[int, int]] = None) -> None:
        if not self.available:
            return
        if pos is not None:
            self.move(pos[0], pos[1])
        self.user32.mouse_event(self.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        self.user32.mouse_event(self.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

    def double_click(self, pos: Optional[Tuple[int, int]] = None) -> None:
        if not self.available:
            return
        if pos is not None:
            self.move(pos[0], pos[1])
        self.user32.mouse_event(self.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        self.user32.mouse_event(self.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        time.sleep(0.03)
        self.user32.mouse_event(self.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        self.user32.mouse_event(self.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

    def right_click(self, pos: Optional[Tuple[int, int]] = None) -> None:
        if not self.available:
            return
        if pos is not None:
            self.move(pos[0], pos[1])
        self.user32.mouse_event(self.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
        self.user32.mouse_event(self.MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)

    def start_drag(self) -> None:
        if not self.available or self.left_Button_down:
            return
        self.user32.mouse_event(self.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        self.left_Button_down = True

    def stop_drag(self) -> None:
        if not self.available or not self.left_Button_down:
            return
        self.user32.mouse_event(self.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        self.left_Button_down = False

    def scroll(self, delta: int) -> None:
        if not self.available or delta == 0:
            return
        self.user32.mouse_event(self.MOUSEEVENTF_WHEEL, 0, 0, int(delta), 0)


# ============================================================================
# SANAL NESNELER (Yumruk yaparak kamera ekranında kaydırılabilen kutular)
# ============================================================================
@dataclass
class DraggableObject:
    label: str
    x: int
    y: int
    w: int
    h: int
    color: Tuple[int, int, int]
    is_grabbed: bool = False
    offset_x: int = 0
    offset_y: int = 0

    def contains(self, px: int, py: int, margin: int = 25) -> bool:
        return (
            (self.x - margin) <= px <= (self.x + self.w + margin)
            and (self.y - margin) <= py <= (self.y + self.h + margin)
        )


def create_default_objects(frame_w: int = 960) -> List[DraggableObject]:
    x_pos = max(20, frame_w - 170)
    return [
        DraggableObject("Nesne 1", x_pos, 130, 140, 85, (255, 140, 50)),
        DraggableObject("Nesne 2", x_pos, 235, 140, 85, (80, 200, 120)),
        DraggableObject("Nesne 3", x_pos, 340, 140, 85, (180, 105, 255)),
    ]


# ============================================================================
# MEDIAPIPE EL TAKİP SARMALAYICI (Hem mp.solutions hem mp.tasks uyumlu)
# ============================================================================
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Başparmak
    (0, 5), (5, 6), (6, 7), (7, 8),          # İşaret parmağı
    (5, 9), (9, 10), (10, 11), (11, 12),     # Orta parmak
    (9, 13), (13, 14), (14, 15), (15, 16),   # Yüzük parmağı
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)  # Serçe parmak ve avuç
]


@dataclass
class Point3D:
    x: float
    y: float
    z: float = 0.0


@dataclass
class DetectedHand:
    landmarks: List[Point3D]
    label: str  # "Right" veya "Left"
    score: float = 1.0


class HandDetector:
    """MediaPipe Hands sarmalayıcısı (eski ve yeni MediaPipe sürümleriyle tam uyumlu)."""

    MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    )

    def __init__(
        self,
        max_hands: int = 2,
        detection_conf: float = 0.7,
        tracking_conf: float = 0.7,
    ) -> None:
        self.use_solutions = hasattr(mp, "solutions") and hasattr(mp.solutions, "hands")
        self.hands = None
        self.landmarker = None
        self._timestamp_ms = 0

        if self.use_solutions:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=max_hands,
                model_complexity=1,
                min_detection_confidence=detection_conf,
                min_tracking_confidence=tracking_conf,
            )
        else:
            # MediaPipe 0.10.30+ Tasks API yedek desteği
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task"
            )
            if not os.path.exists(model_path):
                print("MediaPipe HandLandmarker modeli indiriliyor...")
                urllib.request.urlretrieve(self.MODEL_URL, model_path)

            # Windows'ta Türkçe karakterli klasör yollarında ("görev" vb.) MediaPipe C++
            # dosya okuyucusunun hata vermemesi için modeli bellek tamponu olarak yüklüyoruz.
            with open(model_path, "rb") as f:
                model_bytes = f.read()

            base_options = mp_python.BaseOptions(model_asset_buffer=model_bytes)
            options = mp_vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.VIDEO,
                num_hands=max_hands,
                min_hand_detection_confidence=detection_conf,
                min_hand_presence_confidence=detection_conf,
                min_tracking_confidence=tracking_conf,
            )
            self.landmarker = mp_vision.HandLandmarker.create_from_options(options)

    def detect(self, frame_bgr: np.ndarray) -> List[DetectedHand]:
        # Yumruk yapıldığında (parmaklar kapandığında) ve hızlı harekette el takibinin
        # kopmaması için 640x360 çözünürlük kullanıyoruz.
        h, w = frame_bgr.shape[:2]
        if w > 640:
            small_bgr = cv2.resize(frame_bgr, (640, int(h * 640 / w)), interpolation=cv2.INTER_LINEAR)
        else:
            small_bgr = frame_bgr

        rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
        detected: List[DetectedHand] = []

        if self.use_solutions:
            rgb.flags.writeable = False
            results = self.hands.process(rgb)
            rgb.flags.writeable = True

            if results.multi_hand_landmarks:
                handedness_list = results.multi_handedness or []
                for idx, hand_lms in enumerate(results.multi_hand_landmarks):
                    label = "Right"
                    score = 1.0
                    if idx < len(handedness_list):
                        cls = handedness_list[idx].classification[0]
                        label = cls.label
                        score = float(cls.score)
                    pts = [Point3D(lm.x, lm.y, lm.z) for lm in hand_lms.landmark]
                    detected.append(DetectedHand(landmarks=pts, label=label, score=score))
        else:
            rgb_contiguous = np.ascontiguousarray(rgb)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_contiguous)
            now_ms = int(time.perf_counter() * 1000)
            self._timestamp_ms = max(self._timestamp_ms + 1, now_ms)
            result = self.landmarker.detect_for_video(mp_image, self._timestamp_ms)
            if result.hand_landmarks:
                for idx, hand_lms in enumerate(result.hand_landmarks):
                    label = "Right"
                    score = 1.0
                    if idx < len(result.handedness) and result.handedness[idx]:
                        label = result.handedness[idx][0].category_name
                        score = float(result.handedness[idx][0].score)
                    pts = [Point3D(lm.x, lm.y, lm.z) for lm in hand_lms]
                    detected.append(DetectedHand(landmarks=pts, label=label, score=score))

        return detected

    def close(self) -> None:
        if self.hands is not None:
            self.hands.close()
        if self.landmarker is not None:
            self.landmarker.close()


# ============================================================================
# MEDIAPIPE VÜCUT (POSE) TAKİP SARMALAYICI (Sağa/Sola Eğilme & Kol Takibi)
# ============================================================================
POSE_CONNECTIONS = [
    (11, 12),                  # Omuz hattı
    (11, 13), (13, 15),        # Sol kol
    (12, 14), (14, 16),        # Sağ kol
    (11, 23), (12, 24),        # Gövde yanları
    (23, 24),                  # Kalça hattı
    (0, 2), (2, 7),            # Yüz sol
    (0, 5), (5, 8),            # Yüz sağ
]


@dataclass
class PosePoint3D:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0


@dataclass
class DetectedPose:
    landmarks: List[PosePoint3D]
    lean_state: str = "MERKEZ"     # "SOL", "MERKEZ", "SAG", "EGILDI"
    lean_score: float = 0.0        # [-1.0 .. +1.0] (Negatif: Sol, Pozitif: Sağ)
    is_ducking: bool = False
    head_norm: Tuple[float, float] = (0.5, 0.35)
    shoulder_mid_norm: Tuple[float, float] = (0.5, 0.60)


class PoseDetector:
    """MediaPipe Pose sarmalayıcısı (eski mp.solutions ve yeni mp.tasks API uyumlu)."""

    MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    )

    def __init__(
        self,
        detection_conf: float = 0.55,
        tracking_conf: float = 0.55,
    ) -> None:
        self.use_solutions = hasattr(mp, "solutions") and hasattr(mp.solutions, "pose")
        self.pose = None
        self.landmarker = None
        self._timestamp_ms = 0
        self.smooth_lean: float = 0.0
        self.smooth_head_x: float = 0.5
        self.smooth_head_y: float = 0.35

        if self.use_solutions:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=0,
                smooth_landmarks=True,
                min_detection_confidence=detection_conf,
                min_tracking_confidence=tracking_conf,
            )
        else:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "pose_landmarker_lite.task"
            )
            if not os.path.exists(model_path):
                print("MediaPipe PoseLandmarker (Vücut Algılama) modeli indiriliyor...")
                urllib.request.urlretrieve(self.MODEL_URL, model_path)

            with open(model_path, "rb") as f:
                model_bytes = f.read()

            base_options = mp_python.BaseOptions(model_asset_buffer=model_bytes)
            options = mp_vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=detection_conf,
                min_pose_presence_confidence=detection_conf,
                min_tracking_confidence=tracking_conf,
            )
            self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    def detect(self, frame_bgr: np.ndarray) -> Optional[DetectedPose]:
        h, w = frame_bgr.shape[:2]
        if w > 384:
            small_bgr = cv2.resize(
                frame_bgr, (384, int(h * 384 / w)), interpolation=cv2.INTER_LINEAR
            )
        else:
            small_bgr = frame_bgr

        rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
        pts: List[PosePoint3D] = []

        if self.use_solutions:
            rgb.flags.writeable = False
            results = self.pose.process(rgb)
            rgb.flags.writeable = True
            if results.pose_landmarks:
                for lm in results.pose_landmarks.landmark:
                    vis = float(getattr(lm, "visibility", 1.0))
                    pts.append(PosePoint3D(lm.x, lm.y, lm.z, vis))
        else:
            rgb_contiguous = np.ascontiguousarray(rgb)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_contiguous)
            now_ms = int(time.perf_counter() * 1000)
            self._timestamp_ms = max(self._timestamp_ms + 1, now_ms)
            result = self.landmarker.detect_for_video(mp_image, self._timestamp_ms)
            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                for lm in result.pose_landmarks[0]:
                    vis = float(getattr(lm, "visibility", 1.0))
                    pts.append(PosePoint3D(lm.x, lm.y, lm.z, vis))

        if len(pts) < 17:
            return None

        # Baş (0: Burun) ve Omuzlar (11, 12) üzerinden sağa/sola/aşağı eğilme analizi
        nose = pts[0]
        sh1 = pts[11]
        sh2 = pts[12]

        # Ekranda soldaki ve sağdaki omuzu x koordinatına göre sırala
        left_sh, right_sh = (sh1, sh2) if sh1.x < sh2.x else (sh2, sh1)
        sh_mid_x = (left_sh.x + right_sh.x) * 0.5
        sh_mid_y = (left_sh.y + right_sh.y) * 0.5
        sh_width = max(0.08, right_sh.x - left_sh.x)

        self.smooth_head_x = 0.65 * self.smooth_head_x + 0.35 * nose.x
        self.smooth_head_y = 0.65 * self.smooth_head_y + 0.35 * nose.y

        # 1) Kamera merkezine göre baş/gövde yatay kayması
        pos_offset = (self.smooth_head_x * 0.65 + sh_mid_x * 0.35) - 0.50
        # 2) Başın omuz merkezine göre yana yatması (boyun/üst gövde eğimi)
        neck_tilt = (self.smooth_head_x - sh_mid_x) / sh_width
        # 3) Omuzların eğimi (sağa eğilince sağ omuz düşer -> right_sh.y > left_sh.y)
        shoulder_slope = (right_sh.y - left_sh.y) / sh_width

        raw_lean = (pos_offset * 2.3) + (neck_tilt * 0.45) + (shoulder_slope * 0.45)
        raw_lean = max(-1.0, min(1.0, raw_lean))
        self.smooth_lean = 0.60 * self.smooth_lean + 0.40 * raw_lean

        # Aşağı eğilme (Duck) kontrolü
        is_ducking = (self.smooth_head_y > 0.60) or (sh_mid_y > 0.78)

        if self.smooth_lean < -0.21:
            lean_state = "SOL"
        elif self.smooth_lean > 0.21:
            lean_state = "SAG"
        elif is_ducking:
            lean_state = "EGILDI"
        else:
            lean_state = "MERKEZ"

        return DetectedPose(
            landmarks=pts,
            lean_state=lean_state,
            lean_score=self.smooth_lean,
            is_ducking=is_ducking,
            head_norm=(self.smooth_head_x, self.smooth_head_y),
            shoulder_mid_norm=(sh_mid_x, sh_mid_y),
        )

    def close(self) -> None:
        if self.pose is not None:
            self.pose.close()
        if self.landmarker is not None:
            self.landmarker.close()


# ============================================================================
# PARMAK SAYMA VE JEST (HAREKET) ANALİZİ
# ============================================================================
def dist2d(a: Point3D, b: Point3D) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def joint_angle_deg(a: Point3D, b: Point3D, c: Point3D) -> float:
    """b noktasındaki (a -> b -> c) eklem açısını derece cinsinden (0..180) hesaplar."""
    v1x, v1y = a.x - b.x, a.y - b.y
    v2x, v2y = c.x - b.x, c.y - b.y
    n1 = math.hypot(v1x, v1y)
    n2 = math.hypot(v2x, v2y)
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0
    cos_val = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (n1 * n2)))
    return math.degrees(math.acos(cos_val))


def count_open_fingers(hand: DetectedHand) -> List[int]:
    """
    Her parmak için [Başparmak, İşaret, Orta, Yüzük, Serçe] açık (1) / kapalı (0) döndürür.
    Hem sağ hem sol el, hem avuç içi hem el sırtı dönükken doğru çalışır.
    """
    lms = hand.landmarks
    wrist = lms[0]
    fingers: List[int] = [0, 0, 0, 0, 0]

    # --- 1. Başparmak (Landmark 4: Uç, 3: IP, 2: MCP, 5: İşaret MCP, 17: Serçe MCP) ---
    # Avuç genişliği referansı
    palm_width = max(dist2d(lms[5], lms[17]), 1e-4)
    # Başparmak ucu serçe parmak köküne (17) ve orta parmak köküne (9),
    # başparmak orta ekleminden (3 ve 2) belirgin şekilde daha uzaksa açıktır.
    thumb_tip_to_pinky = dist2d(lms[4], lms[17])
    thumb_ip_to_pinky = dist2d(lms[3], lms[17])
    thumb_mcp_to_pinky = dist2d(lms[2], lms[17])
    thumb_tip_to_index_mcp = dist2d(lms[4], lms[5])

    if (
        thumb_tip_to_pinky > thumb_ip_to_pinky * 1.07
        and thumb_tip_to_pinky > thumb_mcp_to_pinky * 1.12
        and thumb_tip_to_index_mcp > palm_width * 0.48
    ):
        fingers[0] = 1

    # --- 2. Diğer 4 Parmak (İşaret: 8, Orta: 12, Yüzük: 16, Serçe: 20) ---
    tip_ids = [8, 12, 16, 20]
    pip_ids = [6, 10, 14, 18]
    mcp_ids = [5, 9, 13, 17]

    # Yüzük ve serçe parmakları tek başına kaldırıldığında anatomik olarak tam 180 derece
    # açılamadığı için eşik değerlerini parmağa özel hassasiyetle belirliyoruz.
    pip_mults = [1.07, 1.07, 1.01, 1.03]
    mcp_mults = [1.22, 1.22, 1.12, 1.15]

    for i in range(4):
        tip = lms[tip_ids[i]]
        pip = lms[pip_ids[i]]
        mcp = lms[mcp_ids[i]]

        tip_wrist_dist = dist2d(tip, wrist)
        pip_wrist_dist = dist2d(pip, wrist)
        mcp_wrist_dist = dist2d(mcp, wrist)

        if (
            tip_wrist_dist > pip_wrist_dist * pip_mults[i]
            and tip_wrist_dist > mcp_wrist_dist * mcp_mults[i]
        ):
            if tip.y < pip.y + 0.03:
                fingers[i + 1] = 1

    return fingers


def is_fist_gesture(hand: DetectedHand, fingers: List[int], is_pinching: bool) -> bool:
    """
    Yumruk (0 parmak açık) durumunu tespit eder.
    El kameraya dik/karşıdan yumruk olarak tutulduğunda (perspektif kısalmasında)
    bile doğru algılaması için hem boyuna hem enine avuç ölçeğini referans alır.
    """
    if is_pinching:
        return False
    lms = hand.landmarks
    wrist = lms[0]
    palm_ref = max(dist2d(wrist, lms[9]), dist2d(lms[5], lms[17]), 1e-4)

    # Parmak uçları (8, 12, 16, 20) avuç merkezine (9) veya bileğe (0) yakınsa yumruktur
    index_curled = (
        dist2d(lms[8], wrist) < palm_ref * 1.65
        or dist2d(lms[8], lms[9]) < palm_ref * 0.95
    )
    middle_curled = (
        dist2d(lms[12], wrist) < palm_ref * 1.65
        or dist2d(lms[12], lms[9]) < palm_ref * 0.95
    )
    ring_curled = (
        dist2d(lms[16], wrist) < palm_ref * 1.65
        or dist2d(lms[16], lms[9]) < palm_ref * 0.98
    )
    if sum(fingers[1:]) == 0:
        return index_curled and middle_curled
    # Kameraya karşıdan yumruk tutulduğunda açıdan dolayı 1 parmak açık görünse bile
    # parmak uçları avuç içine toplanmışsa yumruk kabul et
    if sum(fingers[1:]) == 1 and index_curled and middle_curled and ring_curled:
        return True
    return False


# ============================================================================
# HAREKET VE FARE YÖNETİCİSİ (Pinch İmleç, Serçe 1x/2x Sol Tık, Orta Sağ Tık, Yumruk Kaydırma)
# ============================================================================
@dataclass
class GestureMouseState:
    enabled: bool = True
    fist_mode: str = "DRAG"  # "DRAG" (Sürükle-Bırak) veya "SCROLL" (Sayfa Kaydır)
    show_virtual_objects: bool = True

    # Pinch (Başparmak + İşaret birleştirme) durumu
    is_pinching: bool = False

    # Serçe parmak: 1 kez aç-kapat -> Tek Sol Tık, 2 kez aç-kapat -> Çift Sol Tık
    pinky_is_up: bool = False
    pinky_up_time: float = 0.0
    pinky_cycle_count: int = 0
    pinky_last_close_time: float = 0.0
    double_cycle_window: float = 0.48  # 1. kapatmadan sonra 2. açma için bekleme penceresi (sn)

    # Orta parmak: Sağ Tık durumu
    middle_lifted: bool = False
    middle_up_frames: int = 0
    middle_lift_time: float = 0.0

    last_click_time: float = 0.0
    click_anchor_screen: Optional[Tuple[int, int]] = None

    # Yumruk durumu
    is_fist: bool = False
    prev_fist_pos: Optional[Tuple[float, float]] = None

    # İmleç yumuşatma
    smooth_x: Optional[float] = None
    smooth_y: Optional[float] = None

    # Görsel bildirim (Son yapılan aksiyon)
    action_text: str = "Hazir"
    action_color: Tuple[int, int, int] = (200, 200, 200)
    action_until: float = 0.0

    # Aktif kamera alanı (kenarlara kolay ulaşmak için çerçeve marjı)
    margin_x: float = 0.16
    margin_y: float = 0.16

    def set_feedback(self, text: str, color: Tuple[int, int, int], duration: float = 1.0) -> None:
        self.action_text = text
        self.action_color = color
        self.action_until = time.perf_counter() + duration

    def map_to_screen(self, norm_x: float, norm_y: float, screen_w: int, screen_h: int) -> Tuple[int, int]:
        # Kamera içindeki aktif dikdörtgeni [0, 1] aralığına ölçekle
        nx = (norm_x - self.margin_x) / max(1e-4, (1.0 - 2.0 * self.margin_x))
        ny = (norm_y - self.margin_y) / max(1e-4, (1.0 - 2.0 * self.margin_y))
        nx = max(0.0, min(1.0, nx))
        ny = max(0.0, min(1.0, ny))

        target_x = nx * screen_w
        target_y = ny * screen_h

        # Adaptif üstel yumuşatma (titremeyi önler, hızlı harekette gecikmez)
        if self.smooth_x is None or self.smooth_y is None:
            self.smooth_x = target_x
            self.smooth_y = target_y
        else:
            jump = math.hypot(target_x - self.smooth_x, target_y - self.smooth_y)
            alpha = min(0.65, max(0.18, jump / 180.0))
            self.smooth_x += alpha * (target_x - self.smooth_x)
            self.smooth_y += alpha * (target_y - self.smooth_y)

        return int(self.smooth_x), int(self.smooth_y)


def process_gestures(
    hand: Optional[DetectedHand],
    fingers: List[int],
    state: GestureMouseState,
    mouse: WindowsMouseController,
    objects: List[DraggableObject],
    frame_w: int,
    frame_h: int,
) -> dict:
    """
    El hareketlerini değerlendirir ve fare/nesne aksiyonlarını çalıştırır:
    - Başparmak + İşaret birleştirme (Pinch): Fare imlecini hareket ettirir.
    - Serçe parmağı 1 kez aç-kapat: Tek Sol Tık.
    - Serçe parmağı 2 kez aç-kapat: Çift Sol Tık.
    - Orta parmağı kaldırıp indirme: Sağ Tık.
    - Yumruk (0 parmak açık): Nesneleri/pencereleri sürükler veya sayfayı kaydırır.
    """
    now = time.perf_counter()
    info = {
        "pinch_px": None,
        "thumb_px": None,
        "index_px": None,
        "palm_px": None,
        "pinch_ratio": 1.0,
        "gesture_name": "El Yok",
    }

    # Bekleyen 1x Serçe Aç-Kapat süresi dolduysa (ve 2. kez serçe kaldırılmadıysa) -> TEK SOL TIK!
    if (
        state.pinky_cycle_count == 1
        and not state.pinky_is_up
        and (now - state.pinky_last_close_time) > state.double_cycle_window
    ):
        if state.enabled:
            mouse.left_click(state.click_anchor_screen)
        state.last_click_time = now
        state.pinky_cycle_count = 0
        state.click_anchor_screen = None
        state.set_feedback("TEK SOL TIK! (1x Serce Ac-Kapat)", (0, 255, 120), 1.1)

    if hand is None:
        # El ekrandan çıktıysa basılı tutulan sürüklemeleri ve durumları sıfırla
        state.is_pinching = False
        state.pinky_is_up = False
        state.pinky_cycle_count = 0
        state.middle_lifted = False
        state.middle_up_frames = 0
        state.click_anchor_screen = None
        state.is_fist = False
        state.prev_fist_pos = None
        mouse.stop_drag()
        for obj in objects:
            obj.is_grabbed = False
        return info

    lms = hand.landmarks
    thumb_tip = lms[4]
    index_tip = lms[8]
    palm_center = lms[9]  # Orta parmak MCP (avuç merkezi)

    thumb_px = (int(thumb_tip.x * frame_w), int(thumb_tip.y * frame_h))
    index_px = (int(index_tip.x * frame_w), int(index_tip.y * frame_h))
    pinch_mid_norm = ((thumb_tip.x + index_tip.x) * 0.5, (thumb_tip.y + index_tip.y) * 0.5)
    pinch_px = (int(pinch_mid_norm[0] * frame_w), int(pinch_mid_norm[1] * frame_h))
    palm_px = (int(palm_center.x * frame_w), int(palm_center.y * frame_h))

    info["thumb_px"] = thumb_px
    info["index_px"] = index_px
    info["pinch_px"] = pinch_px
    info["palm_px"] = palm_px

    # Avuç büyüklüğüne göre normalize edilmiş başparmak-işaret mesafesi
    palm_scale = max(dist2d(lms[0], lms[9]), dist2d(lms[5], lms[17]), 0.05)
    pinch_dist = dist2d(thumb_tip, index_tip)
    pinch_ratio = pinch_dist / palm_scale
    info["pinch_ratio"] = pinch_ratio

    # İşaret parmağı avuç içine tamamen gömülü değilse (yumruk değilse) pinch adaylığı
    index_extended_enough = dist2d(lms[8], lms[0]) > dist2d(lms[5], lms[0]) * 0.85
    other_fingers_open = sum(fingers[2:]) > 0

    # Histerezisli Pinch (Başparmak + İşaret Birleştirme) tespiti
    if state.is_pinching:
        now_pinching = (pinch_ratio < 0.45) and (index_extended_enough or other_fingers_open)
    else:
        now_pinching = (pinch_ratio < 0.33) and (index_extended_enough or other_fingers_open)

    # Yumruk (Fist) tespiti
    now_fist = is_fist_gesture(hand, fingers, now_pinching)

    # --- 1. SERÇE (1x AÇ-KAPAT: TEK TIK, 2x AÇ-KAPAT: ÇİFT TIK) & ORTA PARMAK (SAĞ TIK) ---
    # fingers = [Başparmak(0), İşaret(1), Orta(2), Yüzük(3), Serçe(4)]
    middle_open = fingers[2] == 1
    ring_open = fingers[3] == 1
    pinky_open = fingers[4] == 1

    # El tamamen açıksa (parmak sayma gösterimi: 3+ ana parmak açık ve pinch yoksa),
    # tıklama tetiklemelerini devre dışı bırak.
    full_open_palm = (not now_pinching) and (sum(fingers[1:]) >= 3)

    if full_open_palm:
        state.pinky_is_up = False
        state.pinky_cycle_count = 0
        state.middle_lifted = False
        state.middle_up_frames = 0
        state.click_anchor_screen = None
    else:
        # --------------------------------------------------------------------
        # A) SERÇE PARMAK: 1 Kez Aç-Kapat -> Tek Tık | 2 Kez Aç-Kapat -> Çift Tık
        # --------------------------------------------------------------------
        if pinky_open and not middle_open and not ring_open:
            if not state.pinky_is_up:
                state.pinky_is_up = True
                state.pinky_up_time = now
                # İlk açılışta imleç konumunu sabitle
                if state.pinky_cycle_count == 0 and state.smooth_x is not None and state.smooth_y is not None:
                    state.click_anchor_screen = (int(state.smooth_x), int(state.smooth_y))
            if state.pinky_cycle_count == 0:
                state.set_feedback("Serce Acik (1/2) -> Kapatinca Tik", (120, 255, 160), 0.25)
            else:
                state.set_feedback("Serce Acik (2/2) -> Kapatinca CIFT TIK!", (0, 255, 255), 0.3)

        elif state.pinky_is_up and not pinky_open:
            # Serçe açıldıktan sonra kapatıldı -> 1 "Aç-Kapat" döngüsü tamamlandı
            up_dur = now - state.pinky_up_time
            state.pinky_is_up = False

            if 0.02 <= up_dur <= 1.80:
                # Önceki döngünün üzerinden çok vakit geçtiyse sayacı yeniden başlat
                if state.pinky_cycle_count > 0 and (now - state.pinky_last_close_time) > 1.0:
                    state.pinky_cycle_count = 1
                else:
                    state.pinky_cycle_count += 1

                state.pinky_last_close_time = now

                if state.pinky_cycle_count >= 2:
                    # 2 kez Aç-Kapat tamamlandı -> ÇİFT SOL TIK!
                    if state.enabled:
                        mouse.double_click(state.click_anchor_screen)
                    state.last_click_time = now
                    state.pinky_cycle_count = 0
                    state.click_anchor_screen = None
                    state.set_feedback("CIFT SOL TIK! (2x Serce Ac-Kapat)", (0, 255, 255), 1.2)
                else:
                    state.set_feedback("1x Ac-Kapat (Cift tik icin 1 kez daha...)", (255, 230, 80), 0.5)
            else:
                state.pinky_cycle_count = 0
                state.click_anchor_screen = None

        # --------------------------------------------------------------------
        # B) ORTA PARMAK: Kaldırıp İndirince SAĞ TIK
        # --------------------------------------------------------------------
        if middle_open and not ring_open and not pinky_open:
            state.middle_up_frames += 1
            if not state.middle_lifted and state.middle_up_frames >= 2:
                state.middle_lifted = True
                state.middle_lift_time = now
                if state.smooth_x is not None and state.smooth_y is not None:
                    state.click_anchor_screen = (int(state.smooth_x), int(state.smooth_y))
                state.set_feedback("Orta Parmak Havada -> Indirince SAG TIK", (80, 200, 255), 0.3)
        else:
            state.middle_up_frames = 0
            if state.middle_lifted and not middle_open:
                lift_dur = now - state.middle_lift_time
                state.middle_lifted = False
                if 0.03 <= lift_dur <= 2.0 and (now - state.last_click_time) > 0.25:
                    if state.enabled:
                        mouse.right_click(state.click_anchor_screen)
                    state.last_click_time = now
                    state.set_feedback("SAG TIK! (Orta Parmak)", (0, 165, 255), 1.1)
                state.click_anchor_screen = None

    # --- 2. PINCH (BAŞPARMAK + İŞARET PARMAĞI BİRLEŞTİRME) İLE İMLEÇ YÖNETİMİ ---
    if now_pinching:
        # Yumruk sürüklemesi varsa bırak
        if state.is_fist:
            mouse.stop_drag()
            for obj in objects:
                obj.is_grabbed = False
            state.is_fist = False
            state.prev_fist_pos = None

        state.is_pinching = True
        screen_pos = state.map_to_screen(
            pinch_mid_norm[0], pinch_mid_norm[1], mouse.screen_w, mouse.screen_h
        )

        # Tıklama (serçe veya orta parmak) hareketi devam ederken imleci hedefte sabitle
        freeze_for_click = (
            state.middle_lifted
            or state.pinky_is_up
            or state.pinky_cycle_count > 0
        )
        if state.enabled and not freeze_for_click:
            mouse.move(screen_pos[0], screen_pos[1])

        info["gesture_name"] = "Imlec Kontrol (Pinch)"
    else:
        state.is_pinching = False

    # --- 2. YUMRUK (NESNE KAYDIRMA / SÜRÜKLEME) MANTIĞI ---
    if now_fist:
        state.is_fist = True
        info["gesture_name"] = f"Yumruk: Kaydirma ({state.fist_mode})"

        # A) Ekrandaki sanal nesneleri yakala ve kaydır
        grabbed_any_virtual = False
        if state.show_virtual_objects:
            for obj in reversed(objects):
                if obj.is_grabbed:
                    grabbed_any_virtual = True
                    break
            if not grabbed_any_virtual:
                for obj in reversed(objects):
                    if obj.contains(palm_px[0], palm_px[1]):
                        obj.is_grabbed = True
                        obj.offset_x = palm_px[0] - obj.x
                        obj.offset_y = palm_px[1] - obj.y
                        grabbed_any_virtual = True
                        state.set_feedback(f"{obj.label} Kaydiriliyor!", obj.color, 0.5)
                        break

            for obj in objects:
                if obj.is_grabbed:
                    obj.x = max(10, min(frame_w - obj.w - 10, palm_px[0] - obj.offset_x))
                    obj.y = max(70, min(frame_h - obj.h - 10, palm_px[1] - obj.offset_y))

        # B) Windows Fare Sürükleme (DRAG) veya Sayfa Kaydırma (SCROLL)
        if state.enabled:
            screen_pos = state.map_to_screen(
                palm_center.x, palm_center.y, mouse.screen_w, mouse.screen_h
            )
            if state.fist_mode == "DRAG":
                mouse.move(screen_pos[0], screen_pos[1])
                if not grabbed_any_virtual:
                    mouse.start_drag()
                    state.set_feedback("Yumruk: Nesne Surukleniyor", (255, 120, 255), 0.3)
            else:  # SCROLL modu
                mouse.stop_drag()
                if state.prev_fist_pos is not None:
                    dy = palm_center.y - state.prev_fist_pos[1]
                    if abs(dy) > 0.006:
                        scroll_amount = int(-dy * 1800)
                        mouse.scroll(scroll_amount)
                        direction = "Yukari" if scroll_amount > 0 else "Asagi"
                        state.set_feedback(f"Yumruk: Sayfa Kaydir ({direction})", (255, 180, 50), 0.3)
            state.prev_fist_pos = (palm_center.x, palm_center.y)
    else:
        if state.is_fist:
            state.is_fist = False
            state.prev_fist_pos = None
            mouse.stop_drag()
            for obj in objects:
                obj.is_grabbed = False

    if not now_pinching and not now_fist:
        info["gesture_name"] = f"Acik El ({sum(fingers)} Parmak)"

    return info


# ============================================================================
# YUMRUK ALGILAMA, BOKS ELDİVENİ GÖRSELLEŞTİRME VE TAM EKRAN BOKS OYUNU ([G])
# ============================================================================
@dataclass
class HandFistStatus:
    side: str                      # "SOL" veya "SAG" (ekran tarafı)
    px: Tuple[int, int]            # Ekrandaki piksel konumu
    norm_pos: Tuple[float, float]  # Normalize [0..1] konum
    is_fist: bool                  # Yumruk sıkılı mı?
    is_punching: bool              # Şu an yumruk vuruşu ivmesinde mi?
    speed: float                   # Hız büyüklüğü
    scale: float                   # El büyüklüğü (kameraya yakınlık)
    radius_px: int                 # Çizilecek eldiven/yumruk yarıçapı


@dataclass
class PunchImpactEffect:
    x: int
    y: int
    text: str
    color: Tuple[int, int, int]
    created_at: float
    duration: float = 0.55
    is_crit: bool = False


class PlayerLockTracker:
    """
    Oyuna başlayan tek bir oyuncuyu kilitler (Player Lock) ve sahnedeki başka
    kişilerin / arkadan veya yandan uzatılan yabancı ellerin algılanmasını engeller:
    1. MediaPipe Pose ile kameradaki tek ana oyuncunun gövdesi ve sol/sağ bilekleri takip edilir.
    2. Tespit edilen eller arasından SADECE oyuncunun sol ve sağ bileğine/koluna
       en yakın ve gövde sınırları içindeki eller kabul edilir.
    3. Arkadaki, uzaktaki veya başka bir kişiye ait yabancı eller elenir.
    4. Konum sürekliliği (Spatial tracking): Oyuncunun elleri kareler arasında takip edilir,
       aniden başka yerde beliren eller yok sayılır.
    """

    def __init__(self) -> None:
        self.is_locked: bool = False
        self.last_seen_time: float = 0.0
        self.prev_left_pos: Optional[Tuple[float, float]] = None
        self.prev_right_pos: Optional[Tuple[float, float]] = None
        self.locked_sh_center: Optional[Tuple[float, float]] = None
        self.locked_sh_width: float = 0.25

    def filter_hands(
        self,
        raw_hands: List[DetectedHand],
        pose: Optional[DetectedPose],
        now: float,
    ) -> List[DetectedHand]:
        if not raw_hands:
            if self.is_locked and (now - self.last_seen_time) > 2.5:
                self.is_locked = False
                self.prev_left_pos = None
                self.prev_right_pos = None
            return []

        has_pose = pose is not None and len(pose.landmarks) >= 17
        if has_pose:
            self.is_locked = True
            self.last_seen_time = now
            s1 = pose.landmarks[11]
            s2 = pose.landmarks[12]
            left_sh, right_sh = (s1, s2) if s1.x < s2.x else (s2, s1)
            sh_mid_x = (left_sh.x + right_sh.x) * 0.5
            sh_mid_y = (left_sh.y + right_sh.y) * 0.5
            sh_width = max(0.10, right_sh.x - left_sh.x)
            self.locked_sh_center = (sh_mid_x, sh_mid_y)
            self.locked_sh_width = sh_width

            # Sol ve sağ kol/bilek referansları
            p_l_wrist = pose.landmarks[15]
            p_r_wrist = pose.landmarks[16]
            p_l_index = pose.landmarks[19] if len(pose.landmarks) > 19 else p_l_wrist
            p_r_index = pose.landmarks[20] if len(pose.landmarks) > 20 else p_r_wrist
            p_l_elbow = pose.landmarks[13]
            p_r_elbow = pose.landmarks[14]

            # Ekranda soldaki ve sağdaki kol noktalarını belirle
            if s1.x < s2.x:
                sol_wr, sag_wr = p_l_wrist, p_r_wrist
                sol_idx, sag_idx = p_l_index, p_r_index
                sol_elb, sag_elb = p_l_elbow, p_r_elbow
            else:
                sol_wr, sag_wr = p_r_wrist, p_l_wrist
                sol_idx, sag_idx = p_r_index, p_l_index
                sol_elb, sag_elb = p_r_elbow, p_l_elbow

            # Oyuncunun vücut genişliğine göre kolun uzanabileceği alan sınırları
            max_reach_x = sh_width * 2.4
            max_wrist_dist = max(0.40, sh_width * 1.7)

            valid_candidates: List[Tuple[DetectedHand, float, float, float]] = []
            for hand in raw_hands:
                lms = hand.landmarks
                cx = lms[0].x * 0.35 + lms[9].x * 0.65
                cy = lms[0].y * 0.35 + lms[9].y * 0.65
                palm_sz = math.hypot(lms[0].x - lms[9].x, lms[0].y - lms[9].y)

                # Yandaki yabancı kişilerin gövde dışındaki ellerini ele
                if abs(cx - sh_mid_x) > max_reach_x:
                    continue

                # Arkadaki kişilere ait küçük/derindeki elleri ele
                if palm_sz < 0.032:
                    continue

                valid_candidates.append((hand, cx, cy, palm_sz))

            if not valid_candidates:
                return []

            best_left_hand: Optional[DetectedHand] = None
            best_left_dist = float("inf")
            best_left_pos: Optional[Tuple[float, float]] = None

            best_right_hand: Optional[DetectedHand] = None
            best_right_dist = float("inf")
            best_right_pos: Optional[Tuple[float, float]] = None

            for hand, cx, cy, sz in valid_candidates:
                d_sol = min(
                    math.hypot(cx - sol_wr.x, cy - sol_wr.y),
                    math.hypot(cx - sol_idx.x, cy - sol_idx.y),
                    math.hypot(cx - sol_elb.x, cy - sol_elb.y) * 1.15,
                )
                d_sag = min(
                    math.hypot(cx - sag_wr.x, cy - sag_wr.y),
                    math.hypot(cx - sag_idx.x, cy - sag_idx.y),
                    math.hypot(cx - sag_elb.x, cy - sag_elb.y) * 1.15,
                )

                # El oyuncunun ne sol koluna ne sağ koluna yakınsa yabancıdır, yoksay
                if d_sol > max_wrist_dist and d_sag > max_wrist_dist:
                    continue

                if d_sol < d_sag and d_sol < max_wrist_dist:
                    if d_sol < best_left_dist:
                        best_left_dist = d_sol
                        best_left_hand = hand
                        best_left_pos = (cx, cy)
                elif d_sag <= d_sol and d_sag < max_wrist_dist:
                    if d_sag < best_right_dist:
                        best_right_dist = d_sag
                        best_right_hand = hand
                        best_right_pos = (cx, cy)

            selected_hands: List[DetectedHand] = []
            if best_left_hand is not None:
                selected_hands.append(best_left_hand)
                self.prev_left_pos = best_left_pos
            if best_right_hand is not None and best_right_hand is not best_left_hand:
                selected_hands.append(best_right_hand)
                self.prev_right_pos = best_right_pos

            if selected_hands:
                self.is_locked = True
                self.last_seen_time = now

            return selected_hands

        else:
            # Pose o an yoksa ama oyuncu kilitliyse konum sürekliliğiyle yabancı elleri engelle
            if self.is_locked and (self.prev_left_pos or self.prev_right_pos):
                selected_hands = []
                for hand in raw_hands:
                    lms = hand.landmarks
                    cx = lms[0].x * 0.35 + lms[9].x * 0.65
                    cy = lms[0].y * 0.35 + lms[9].y * 0.65
                    palm_sz = math.hypot(lms[0].x - lms[9].x, lms[0].y - lms[9].y)
                    if palm_sz < 0.032:
                        continue

                    d_l = math.hypot(cx - self.prev_left_pos[0], cy - self.prev_left_pos[1]) if self.prev_left_pos else 999.0
                    d_r = math.hypot(cx - self.prev_right_pos[0], cy - self.prev_right_pos[1]) if self.prev_right_pos else 999.0

                    if min(d_l, d_r) < 0.30:
                        selected_hands.append(hand)
                        if d_l < d_r:
                            self.prev_left_pos = (cx, cy)
                        else:
                            self.prev_right_pos = (cx, cy)

                if selected_hands:
                    self.last_seen_time = now
                    return selected_hands[:2]

            # İlk açılışta ve pose henüz oturmamışken en öndeki (en büyük) elleri al ve kilitlen
            sorted_by_size = sorted(
                raw_hands,
                key=lambda h: math.hypot(h.landmarks[0].x - h.landmarks[9].x, h.landmarks[0].y - h.landmarks[9].y),
                reverse=True,
            )
            candidates = sorted_by_size[:2]
            if candidates:
                self.is_locked = True
                self.last_seen_time = now
            return candidates


class PunchTracker:
    """
    Kullanıcının sağ ve sol yumruklarını (HandDetector + PoseDetector birleşimiyle)
    kesintisiz takip eder:
    - Hızlı yumruk sırasında el bulanıklaşsa bile Pose bilek/eklem noktalarıyla takibi sürdürür.
    - Sağa/sola eğilirken omuz ve el birlikte hareket ettiği için yanlış vuruş tetiklemez.
    - El kameraya doğru ileri uzatıldığında (büyüme) VEYA omuza göre ileri/yukarı hızlıca
      fırlatıldığında (kol uzatma) anında yumruk vuruşu tetikler.
    """

    def __init__(self) -> None:
        self.smooth_scale: dict[str, float] = {}
        self.scale_history: dict[str, List[Tuple[float, float]]] = {"SOL": [], "SAG": []}
        self.prev_pos: dict[str, Tuple[float, float]] = {}
        self.prev_rel_pos: dict[str, Tuple[float, float]] = {}
        self.prev_time: dict[str, float] = {}
        self.fist_memory_until: dict[str, float] = {"SOL": 0.0, "SAG": 0.0}
        self.is_armed: dict[str, bool] = {"SOL": True, "SAG": True}
        self.peak_punch_scale: dict[str, float] = {"SOL": 0.0, "SAG": 0.0}

        self.cooldown_until: dict[str, float] = {"SOL": 0.0, "SAG": 0.0}
        self.punch_active_until: dict[str, float] = {"SOL": 0.0, "SAG": 0.0}
        self.trails: dict[str, List[Tuple[int, int, float]]] = {"SOL": [], "SAG": []}
        self.impacts: List[PunchImpactEffect] = []
        self.total_punches: int = 0
        self.last_punch_text: str = ""
        self.last_punch_until: float = 0.0

    @staticmethod
    def _rigid_palm_scale(lms: List[Point3D]) -> float:
        d05 = dist2d(lms[0], lms[5])
        d09 = dist2d(lms[0], lms[9])
        d017 = dist2d(lms[0], lms[17])
        d517 = dist2d(lms[5], lms[17])
        return max(0.035, (d05 + d09 + d017 + d517) * 0.25)

    def _get_shoulder_anchors(
        self, pose: Optional[DetectedPose]
    ) -> dict[str, Tuple[float, float]]:
        """Ekranda sol ('SOL') ve sağ ('SAG') omuzların normalize koordinatlarını döndürür."""
        if pose is None or len(pose.landmarks) < 13:
            return {"SOL": (0.38, 0.58), "SAG": (0.62, 0.58)}
        s1, s2 = pose.landmarks[11], pose.landmarks[12]
        left_sh, right_sh = (s1, s2) if s1.x < s2.x else (s2, s1)
        return {
            "SOL": (left_sh.x, left_sh.y),
            "SAG": (right_sh.x, right_sh.y),
        }

    def update(
        self,
        detected_hands: List[DetectedHand],
        hand_fingers_list: List[List[int]],
        pose: Optional[DetectedPose],
        frame_w: int,
        frame_h: int,
        game_mode: bool = False,
    ) -> Tuple[List[HandFistStatus], List[HandFistStatus]]:
        now = time.perf_counter()
        statuses: List[HandFistStatus] = []
        new_punches: List[HandFistStatus] = []
        seen_sides: set[str] = set()

        shoulders = self._get_shoulder_anchors(pose)

        # 1. HandDetector tarafından bulunan elleri işle
        for idx, hand in enumerate(detected_hands):
            fingers = hand_fingers_list[idx] if idx < len(hand_fingers_list) else [0, 0, 0, 0, 0]
            lms = hand.landmarks
            wrist = lms[0]
            knuckle = lms[9]

            cx = wrist.x * 0.25 + knuckle.x * 0.75
            cy = wrist.y * 0.25 + knuckle.y * 0.75

            # Hangi omuza / ekran tarafına daha yakınsa o taraf olarak eşle
            d_sol = abs(cx - shoulders["SOL"][0])
            d_sag = abs(cx - shoulders["SAG"][0])
            side = "SOL" if d_sol <= d_sag else "SAG"
            if side in seen_sides:
                side = "SAG" if side == "SOL" else "SOL"
            seen_sides.add(side)

            raw_scale = self._rigid_palm_scale(lms)

            # Pinch kontrolü (normal modda pinch yaparken yumruk sayılmasın)
            pinch_ratio = dist2d(lms[4], lms[8]) / max(raw_scale, 0.04)
            is_pinch = (not game_mode) and (pinch_ratio < 0.35) and (sum(fingers[2:]) > 0)

            strict_fist = is_fist_gesture(hand, fingers, is_pinch)
            # Oyun modunda el hızlı hareket ederken en fazla 2 parmak açık görünüyorsa da yumruk kabul et
            game_fist = game_mode and (sum(fingers[1:]) <= 2) and (not is_pinch)
            if strict_fist or game_fist:
                self.fist_memory_until[side] = now + 0.45

            is_fist = strict_fist or game_fist or (now < self.fist_memory_until.get(side, 0.0))

            st, triggered = self._process_hand_kinematics(
                side=side,
                cx=cx,
                cy=cy,
                raw_scale=raw_scale,
                is_fist=is_fist,
                shoulder_pos=shoulders[side],
                frame_w=frame_w,
                frame_h=frame_h,
                game_mode=game_mode,
                now=now,
                has_hand_landmarks=True,
            )
            statuses.append(st)
            if triggered:
                new_punches.append(st)

        # 2. Hızlı yumruk sırasında HandDetector eli kaçırırsa PoseDetector kol/bilek takibiyle devam et!
        if pose is not None and len(pose.landmarks) >= 21:
            # Pose kol çiftleri: (Omuz, Dirsek, Bilek, İşaret Kökü, Serçe Kökü)
            arm_sets = [
                (11, 13, 15, 19, 17),
                (12, 14, 16, 20, 18),
            ]
            for sh_i, el_i, wr_i, idx_i, pnk_i in arm_sets:
                sh_lm = pose.landmarks[sh_i]
                wr_lm = pose.landmarks[wr_i]
                idx_lm = pose.landmarks[idx_i]
                pnk_lm = pose.landmarks[pnk_i]

                if wr_lm.visibility < 0.35:
                    continue

                cx = wr_lm.x * 0.35 + idx_lm.x * 0.40 + pnk_lm.x * 0.25
                cy = wr_lm.y * 0.35 + idx_lm.y * 0.40 + pnk_lm.y * 0.25
                if not (0.02 <= cx <= 0.98 and 0.03 <= cy <= 0.96):
                    continue

                side = "SOL" if sh_lm.x < 0.5 else "SAG"
                if side in seen_sides:
                    continue

                # Sadece oyun modunda VEYA yakın zamanda yumruk hafızası varsa Pose yedeğini devreye sok
                recently_fist = now < self.fist_memory_until.get(side, 0.0)
                hand_raised_in_guard = cy < (sh_lm.y + 0.22)
                if not (game_mode or recently_fist):
                    continue
                if not (recently_fist or hand_raised_in_guard):
                    continue

                seen_sides.add(side)
                pose_knuckle_scale = max(
                    0.05,
                    math.hypot(wr_lm.x - idx_lm.x, wr_lm.y - idx_lm.y) * 1.45,
                    self.smooth_scale.get(side, 0.08),
                )
                st, triggered = self._process_hand_kinematics(
                    side=side,
                    cx=cx,
                    cy=cy,
                    raw_scale=pose_knuckle_scale,
                    is_fist=True,
                    shoulder_pos=(sh_lm.x, sh_lm.y),
                    frame_w=frame_w,
                    frame_h=frame_h,
                    game_mode=game_mode,
                    now=now,
                    has_hand_landmarks=False,
                )
                statuses.append(st)
                if triggered:
                    new_punches.append(st)

        self.impacts = [imp for imp in self.impacts if (now - imp.created_at) < imp.duration]
        return statuses, new_punches

    def _process_hand_kinematics(
        self,
        side: str,
        cx: float,
        cy: float,
        raw_scale: float,
        is_fist: bool,
        shoulder_pos: Tuple[float, float],
        frame_w: int,
        frame_h: int,
        game_mode: bool,
        now: float,
        has_hand_landmarks: bool,
    ) -> Tuple[HandFistStatus, bool]:
        # Omuza göre bağıl el konumu (Sağa/sola eğilirken omuz ve el birlikte kaydığı için değişmez!)
        rel_x = cx - shoulder_pos[0]
        rel_y = cy - shoulder_pos[1]

        last_t = self.prev_time.get(side, 0.0)
        if (now - last_t) > 0.35:
            self.smooth_scale[side] = raw_scale
            self.scale_history[side] = [(now, raw_scale)]
            self.prev_pos[side] = (cx, cy)
            self.prev_rel_pos[side] = (rel_x, rel_y)
            self.is_armed[side] = True
            dt = 0.033
        else:
            dt = max(1e-3, now - last_t)

        prev_s = self.smooth_scale.get(side, raw_scale)
        curr_s = (0.50 * prev_s + 0.50 * raw_scale) if has_hand_landmarks else prev_s
        self.smooth_scale[side] = curr_s

        hist = self.scale_history[side]
        hist.append((now, curr_s))
        self.scale_history[side] = [(t, s) for (t, s) in hist if (now - t) <= 0.25]

        # Omuza göre bağıl hız (Vücut eğilmesinden bağımsız, saf kol uzatma/yumruk hızı)
        prev_rx, prev_ry = self.prev_rel_pos.get(side, (rel_x, rel_y))
        rel_vx = (rel_x - prev_rx) / dt
        rel_vy = (rel_y - prev_ry) / dt
        rel_speed = math.hypot(rel_vx, rel_vy)

        self.prev_pos[side] = (cx, cy)
        self.prev_rel_pos[side] = (rel_x, rel_y)
        self.prev_time[side] = now

        # El yavaşladığında veya hafif geri çekildiğinde bir sonraki yumruk için hemen hazırla
        if not self.is_armed.get(side, True):
            if (
                curr_s < self.peak_punch_scale.get(side, curr_s) * 0.94
                or rel_speed < 0.45
                or (now - self.cooldown_until.get(side, 0.0)) > 0.05
            ):
                self.is_armed[side] = True

        # 1) İleri doğru büyüme (El kameraya yaklaşırken %9+ büyüme)
        past_samples = [s for (t, s) in self.scale_history[side] if (now - t) >= 0.04]
        base_s = min(past_samples) if past_samples else curr_s
        scale_ratio = curr_s / max(0.035, base_s)
        scale_delta = curr_s - base_s

        forward_zoom_punch = (
            has_hand_landmarks
            and scale_ratio >= 1.085
            and scale_delta >= 0.0075
            and (curr_s >= prev_s)
        )

        # 2) Omuza göre hızlı kol fırlatma (Yana eğilmeden etkilenmez; kol yukarı/merkeze/ileri uzanır)
        # El aşağı düşmüyorsa (rel_vy < 0.22) ve omuza göre hızla fırladıysa:
        inward_vx = rel_vx if side == "SOL" else -rel_vx
        arm_thrust_punch = (
            rel_speed >= (0.82 if game_mode else 1.05)
            and rel_vy < 0.20
            and (rel_vy < -0.35 or inward_vx > 0.35 or scale_ratio >= 1.04)
        )

        px = int(cx * frame_w)
        py = int(cy * frame_h)
        radius_px = int(max(26, min(80, curr_s * frame_w * 0.50)))

        self.trails[side].append((px, py, now))
        self.trails[side] = [t for t in self.trails[side] if (now - t[2]) < 0.20]

        triggered = False
        if (
            is_fist
            and self.is_armed.get(side, True)
            and now >= self.cooldown_until.get(side, 0.0)
            and (forward_zoom_punch or arm_thrust_punch)
        ):
            triggered = True
            self.is_armed[side] = False
            self.peak_punch_scale[side] = curr_s
            self.cooldown_until[side] = now + 0.24
            self.punch_active_until[side] = now + 0.24
            self.total_punches += 1
            self.last_punch_text = f"{side} YUMRUK ATILDI!"
            self.last_punch_until = now + 0.85

            if not game_mode:
                self.impacts.append(
                    PunchImpactEffect(
                        x=px,
                        y=py,
                        text=f"{side} YUMRUK!",
                        color=(0, 215, 255),
                        created_at=now,
                        duration=0.5,
                    )
                )

        is_punching_now = now < self.punch_active_until.get(side, 0.0)
        punch_power = max(rel_speed * 1.4, (scale_ratio - 1.0) * 8.0, 1.0)
        st = HandFistStatus(
            side=side,
            px=(px, py),
            norm_pos=(cx, cy),
            is_fist=is_fist,
            is_punching=is_punching_now,
            speed=punch_power,
            scale=curr_s,
            radius_px=radius_px,
        )
        return st, triggered


def draw_pose_skeleton(
    frame: np.ndarray,
    pose: Optional[DetectedPose],
    game_mode: bool = False,
) -> None:
    """Vücut (Pose) iskeletini, baş/omuz hizasını ve sağa/sola eğilme durumunu çizer."""
    if pose is None:
        return

    h, w = frame.shape[:2]
    lms = pose.landmarks

    # Gövde ve kol bağlantılarını çiz
    for idx_a, idx_b in POSE_CONNECTIONS:
        if idx_a < len(lms) and idx_b < len(lms):
            la, lb = lms[idx_a], lms[idx_b]
            if la.visibility > 0.35 and lb.visibility > 0.35:
                pa = (int(la.x * w), int(la.y * h))
                pb = (int(lb.x * w), int(lb.y * h))
                cv2.line(frame, pa, pb, (255, 210, 60), 3, cv2.LINE_AA)

    # Boyun ve omurga merkez çizgisi (Eğilme açısını net gösterir)
    hx = int(pose.head_norm[0] * w)
    hy = int(pose.head_norm[1] * h)
    sx = int(pose.shoulder_mid_norm[0] * w)
    sy = int(pose.shoulder_mid_norm[1] * h)

    lean_col = (0, 255, 120)
    if pose.lean_state == "SOL":
        lean_col = (255, 220, 0)
    elif pose.lean_state == "SAG":
        lean_col = (0, 200, 255)
    elif pose.lean_state == "EGILDI":
        lean_col = (255, 120, 255)

    cv2.line(frame, (sx, sy), (hx, hy), lean_col, 4, cv2.LINE_AA)

    # Omuz ve dirsek eklem noktaları
    for idx in (11, 12, 13, 14, 15, 16, 23, 24):
        if idx < len(lms) and lms[idx].visibility > 0.35:
            px, py = int(lms[idx].x * w), int(lms[idx].y * h)
            cv2.circle(frame, (px, py), 6, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 8, (30, 30, 30), 2, cv2.LINE_AA)

    # Baş takip halkası ve eğilme etiketi
    head_r = 28 if not game_mode else 34
    cv2.circle(frame, (hx, hy), head_r, lean_col, 2, cv2.LINE_AA)
    cv2.circle(frame, (hx, hy), 5, lean_col, -1, cv2.LINE_AA)

    if not game_mode:
        lean_txt = f"Vucut: {pose.lean_state}"
        cv2.putText(
            frame,
            lean_txt,
            (max(10, hx - 55), max(135, hy - head_r - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            lean_col,
            2,
            cv2.LINE_AA,
        )


def draw_fists_and_punches(
    frame: np.ndarray,
    fist_statuses: List[HandFistStatus],
    tracker: PunchTracker,
    game_mode: bool = False,
) -> None:
    """
    Algılanan yumrukları (boks eldiveni / enerji halkası), hareket izlerini (trails)
    ve yumruk vuruş patlamalarını (POW! / YUMRUK!) ekrana çizer.
    """
    now = time.perf_counter()

    # 1. Yumruk hareket izlerini (motion trail) çiz
    for side, trail in tracker.trails.items():
        if len(trail) >= 2:
            for i in range(1, len(trail)):
                age = now - trail[i][2]
                alpha_ratio = max(0.1, 1.0 - age / 0.22)
                thickness = max(2, int(10 * alpha_ratio))
                col = (0, 180, 255) if side == "SOL" else (80, 100, 255)
                cv2.line(
                    frame,
                    (trail[i - 1][0], trail[i - 1][1]),
                    (trail[i][0], trail[i][1]),
                    col,
                    thickness,
                    cv2.LINE_AA,
                )

    # 2. Her yumruk için boks eldiveni / güç halkası çiz
    for st in fist_statuses:
        if not st.is_fist and not game_mode:
            continue

        px, py = st.px
        r = st.radius_px
        if st.is_punching:
            # Yumruk atıldığı anda parlayan büyük boks halkası
            outer_col = (0, 255, 255)
            inner_col = (0, 80, 255)
            r_draw = int(r * 1.28)
        elif st.is_fist:
            outer_col = (50, 220, 255) if st.side == "SOL" else (120, 120, 255)
            inner_col = (20, 60, 200) if st.side == "SOL" else (40, 40, 180)
            r_draw = r
        else:
            # Oyun modunda el açıkken yarı saydam el göstergesi
            cv2.circle(frame, (px, py), r, (180, 180, 180), 2, cv2.LINE_AA)
            cv2.putText(
                frame,
                "YUMRUK YAP!",
                (px - 45, py + r + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )
            continue

        # Boks eldiveni dolgusu (yarı saydam)
        fh, fw = frame.shape[:2]
        x1, y1 = max(0, px - r_draw), max(0, py - r_draw)
        x2, y2 = min(fw, px + r_draw), min(fh, py + r_draw)
        if x2 > x1 and y2 > y1:
            roi = frame[y1:y2, x1:x2]
            overlay = roi.copy()
            cv2.circle(overlay, (px - x1, py - y1), r_draw, inner_col, -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.48, roi, 0.52, 0, roi)

        cv2.circle(frame, (px, py), r_draw, outer_col, 3, cv2.LINE_AA)
        cv2.circle(frame, (px, py), max(6, r_draw // 3), (255, 255, 255), 2, cv2.LINE_AA)

        label = f"{st.side} YUMRUK!" if st.is_punching else f"{st.side} YUMRUK"
        cv2.putText(
            frame,
            label,
            (max(5, px - 52), max(25, py - r_draw - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55 if st.is_punching else 0.46,
            (0, 255, 255) if st.is_punching else (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # 3. Vuruş patlama efektleri (Shockwave & Floating Text)
    for imp in tracker.impacts:
        age = now - imp.created_at
        prog = min(1.0, age / max(1e-3, imp.duration))
        ring_r = int(22 + prog * (95 if imp.is_crit else 68))
        thickness = max(1, int(4 * (1.0 - prog)))
        cv2.circle(frame, (imp.x, imp.y), ring_r, imp.color, thickness, cv2.LINE_AA)
        if imp.is_crit:
            cv2.circle(frame, (imp.x, imp.y), int(ring_r * 0.65), (255, 255, 255), thickness, cv2.LINE_AA)

        ty = int(imp.y - prog * 55)
        cv2.putText(
            frame,
            imp.text,
            (max(10, imp.x - 65), max(35, ty)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78 if imp.is_crit else 0.65,
            (20, 20, 20),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            imp.text,
            (max(10, imp.x - 65), max(35, ty)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78 if imp.is_crit else 0.65,
            imp.color,
            2,
            cv2.LINE_AA,
        )


# ============================================================================
# TAM EKRAN BOKS OYUNU MOTORU ([G] TUŞU)
# ============================================================================
class BoxingGame:
    """
    [G] tuşuyla açılan tam ekran Boks Oyunu:
    - Karşıdaki rakip boksör belirli aralıklarla yumruk hazırlar (WINDUP) ve vurur (STRIKE).
    - Oyuncu vücudunu SAĞA veya SOLA (veya aşağı) eğerek rakibin yumruklarından kaçar.
    - Kaçış sonrası 'KARŞI ATAK (COUNTER)' penceresi açılır ve oyuncunun yumrukları 2x hasar vurur.
    - Oyuncu kameraya/rakibe yumruk atarak rakibin canını 0'a indirip NAKAVT (K.O.) eder!
    """

    def __init__(self) -> None:
        self.active: bool = False
        self.round_num: int = 1
        self.ko_time: float = 0.0
        self.reset_match(keep_round=False)

    def reset_match(self, keep_round: bool = False) -> None:
        if not keep_round:
            self.round_num = 1
        now = time.perf_counter()
        self.ko_time = 0.0
        self.player_max_hp: int = 100
        self.player_hp: int = 100
        # Sabit zorluk: Zamanla veya rauntlar ilerledikçe can artışı iptal edildi
        self.opp_max_hp: int = 1000
        self.opp_hp: int = self.opp_max_hp

        self.state: str = "FIGHT"  # "FIGHT", "VICTORY_KO", "DEFEAT_KO"
        self.opp_action: str = "IDLE"  # "IDLE", "WINDUP", "STRIKE", "HURT"
        # Rakip atak türleri:
        # "DUZ_MERKEZ": Merkeze vurur -> Sola veya Sağa eğilerek kaçılır!
        # "SOL_KROSE": Ekranın soluna ve merkezine vurur -> SAĞA eğilerek kaçılır!
        # "SAG_KROSE": Ekranın sağına ve merkezine vurur -> SOLA eğilerek kaçılır!
        self.attack_type: str = "DUZ_MERKEZ"
        self.attack_start_time: float = now
        # Sabit zorluk: Rakibin vuruş hazırlık süresi kısaltılmaz, her zaman sabit 1.05 sn
        self.windup_duration: float = 1.05
        self.next_attack_time: float = now + 2.0
        self.strike_applied: bool = False

        self.opp_hurt_until: float = 0.0
        self.opp_recoil_x: float = 0.0
        self.opp_recoil_y: float = 0.0

        self.counter_window_until: float = 0.0
        self.combo_count: int = 0
        self.last_player_hit_time: float = 0.0
        self.score: int = 0

        self.banner_text: str = "DOVUS BASLADI! YUMRUK AT & SAGA/SOLA EGIL!"
        self.banner_color: Tuple[int, int, int] = (0, 255, 255)
        self.banner_until: float = now + 2.5

        self.screen_flash_col: Optional[Tuple[int, int, int]] = None
        self.screen_flash_until: float = 0.0
        self.shake_until: float = 0.0

    def set_banner(self, text: str, color: Tuple[int, int, int], duration: float = 1.3) -> None:
        self.banner_text = text
        self.banner_color = color
        self.banner_until = time.perf_counter() + duration

    def _is_dodge_successful(self, attack_type: str, lean_state: str) -> bool:
        if lean_state == "EGILDI":
            return True
        if attack_type == "DUZ_MERKEZ":
            return lean_state in ("SOL", "SAG")
        if attack_type == "SOL_KROSE":
            # Rakip ekranın sol tarafına vuruyor -> Oyuncu SAĞA eğilmeli
            return lean_state == "SAG"
        if attack_type == "SAG_KROSE":
            # Rakip ekranın sağ tarafına vuruyor -> Oyuncu SOLA eğilmeli
            return lean_state == "SOL"
        return False

    def update_and_render(
        self,
        base_frame: np.ndarray,
        detected_hands: List[DetectedHand],
        hand_fingers_list: List[List[int]],
        pose: Optional[DetectedPose],
        fist_statuses: List[HandFistStatus],
        new_punches: List[HandFistStatus],
        tracker: PunchTracker,
        fps: float,
        player_locked: bool = True,
    ) -> np.ndarray:
        """
        Tam ekran boks oyunu sahnesini günceller ve 1280x720 HD çözünürlükte çizer.
        Kamera görüntüsü yerine boks ringi, rakip ve oyuncu maskotu ekranda yer alır;
        sağ alt köşede ise minik bir canlı webcam penceresi (PiP) gösterilir.
        """
        now = time.perf_counter()
        game_w, game_h = 1280, 720

        # 1. Arka plana kamera yerine Boks Arenası / Ring sahnesi çiz
        canvas = np.zeros((game_h, game_w, 3), dtype=np.uint8)
        self._draw_arena_background(canvas, game_w, game_h)

        scale_x = game_w / float(base_frame.shape[1])
        scale_y = game_h / float(base_frame.shape[0])

        # Ölçeklenmiş yumruk konumları
        scaled_fists: List[HandFistStatus] = []
        for st in fist_statuses:
            scaled_fists.append(
                HandFistStatus(
                    side=st.side,
                    px=(int(st.px[0] * scale_x), int(st.px[1] * scale_y)),
                    norm_pos=st.norm_pos,
                    is_fist=st.is_fist,
                    is_punching=st.is_punching,
                    speed=st.speed,
                    scale=st.scale,
                    radius_px=int(st.radius_px * scale_x),
                )
            )

        lean_state = pose.lean_state if pose is not None else "MERKEZ"
        lean_score = pose.lean_score if pose is not None else 0.0

        # Rakip merkez konumu (bobbing & recoil animasyonu)
        bob_x = math.sin(now * 3.2) * 18.0
        bob_y = abs(math.cos(now * 6.4)) * 10.0
        if now >= self.opp_hurt_until:
            self.opp_recoil_x *= 0.80
            self.opp_recoil_y *= 0.80

        opp_cx = int(game_w * 0.50 + bob_x + self.opp_recoil_x)
        opp_cy = int(game_h * 0.39 + bob_y + self.opp_recoil_y)

        # --------------------------------------------------------------------
        # 2. OYUN MANTIĞI GÜNCELLEMESİ (FIGHT DURUMU)
        # --------------------------------------------------------------------
        if self.state == "FIGHT":
            # A) Oyuncunun attığı yeni yumrukları değerlendir
            for punch in new_punches:
                px_g = int(punch.px[0] * scale_x)
                py_g = int(punch.px[1] * scale_y)

                # Rakip vuruş kutusu (ileri doğru atılan gerçek yumruklar)
                in_target_x = abs(px_g - opp_cx) < 520
                in_target_y = 40 <= py_g <= 680
                if in_target_x and in_target_y:
                    # Kombo ve Karşı Atak (Counter-Hit) hesapla
                    if (now - self.last_player_hit_time) < 1.8:
                        self.combo_count += 1
                    else:
                        self.combo_count = 1
                    self.last_player_hit_time = now

                    is_counter = now < self.counter_window_until
                    base_dmg = int(min(38, max(20, 18 + punch.speed * 7.5)))
                    if is_counter:
                        dmg = int(base_dmg * 2.0)
                        self.counter_window_until = 0.0
                        hit_label = f"KARSI ATAK KRITIK! -{dmg}"
                        hit_col = (0, 255, 255)
                        self.set_banner(f"MUHTESEM KARSI ATAK! (-{dmg} HP)", (0, 255, 255), 1.1)
                    elif self.combo_count >= 3:
                        dmg = int(base_dmg * 1.4)
                        hit_label = f"KOMBO x{self.combo_count}! -{dmg}"
                        hit_col = (0, 215, 255)
                    else:
                        dmg = base_dmg
                        hit_label = f"BAM! -{dmg}"
                        hit_col = (50, 255, 120)

                    self.opp_hp = max(0, self.opp_hp - dmg)
                    self.score += dmg * 5
                    self.opp_hurt_until = now + 0.30
                    self.opp_recoil_x = -46.0 if punch.side == "SAG" else 46.0
                    self.opp_recoil_y = -24.0

                    # Eğer rakip vuruş hazırlığındayken (WINDUP) karşı atak yerse atağı bozulur!
                    if is_counter and self.opp_action == "WINDUP":
                        self.opp_action = "HURT"
                        self.next_attack_time = now + 1.35

                    # Vuruş efektini rakibin gövde/baş hizasında patlat
                    impact_x_orig = int((opp_cx + (28 if punch.side == "SAG" else -28)) / scale_x)
                    impact_y_orig = int((opp_cy + 10) / scale_y)
                    tracker.impacts.append(
                        PunchImpactEffect(
                            x=impact_x_orig,
                            y=impact_y_orig,
                            text=hit_label,
                            color=hit_col,
                            created_at=now,
                            duration=0.65,
                            is_crit=(is_counter or self.combo_count >= 3),
                        )
                    )

                    if self.opp_hp <= 0:
                        self.state = "VICTORY_KO"
                        self.opp_action = "KO"
                        self.ko_time = now
                        self.score += 1000
                        self.set_banner(
                            "NAKAVT (K.O.)! RAKIBI YENDIN! [R] Yeni Raund",
                            (0, 255, 120),
                            10.0,
                        )
                        break

            # B) Rakibin Atak Durum Makinesi (IDLE -> WINDUP -> STRIKE)
            if self.state == "FIGHT":
                if self.opp_action == "HURT" and now >= self.opp_hurt_until:
                    self.opp_action = "IDLE"

                if self.opp_action == "IDLE" and now >= self.next_attack_time:
                    self.opp_action = "WINDUP"
                    self.attack_type = random.choice(["DUZ_MERKEZ", "SOL_KROSE", "SAG_KROSE"])
                    self.attack_start_time = now
                    self.strike_applied = False

                elif self.opp_action == "WINDUP":
                    elapsed = now - self.attack_start_time
                    if elapsed >= self.windup_duration:
                        self.opp_action = "STRIKE"
                        self.attack_start_time = now
                        self.strike_applied = False

                elif self.opp_action == "STRIKE":
                    if not self.strike_applied:
                        self.strike_applied = True
                        dodged = self._is_dodge_successful(self.attack_type, lean_state)
                        if dodged:
                            self.counter_window_until = now + 1.9
                            self.score += 100
                            self.screen_flash_col = (0, 255, 160)
                            self.screen_flash_until = now + 0.25
                            self.set_banner(
                                "HARIKA KACIS! SIMDI YUMRUK AT! (2x KARSI ATAK)",
                                (0, 255, 160),
                                1.5,
                            )
                        else:
                            # Sabit hasar: Zamanla veya raunt ilerledikçe artış iptal edildi
                            dmg_taken = 15
                            self.player_hp = max(0, self.player_hp - dmg_taken)
                            self.combo_count = 0
                            self.screen_flash_col = (0, 0, 255)
                            self.screen_flash_until = now + 0.32
                            self.shake_until = now + 0.28
                            self.set_banner(
                                f"YUMRUK YEDIN! (-{dmg_taken} HP) -> SAGA/SOLA EGIL!",
                                (60, 80, 255),
                                1.4,
                            )
                            if self.player_hp <= 0:
                                self.state = "DEFEAT_KO"
                                self.ko_time = now
                                self.set_banner(
                                    "NAKAVT OLDUN! [R] Tusu ile Tekrar Dene",
                                    (0, 80, 255),
                                    10.0,
                                )

                    if (now - self.attack_start_time) >= 0.30 and self.state == "FIGHT":
                        self.opp_action = "IDLE"
                        # Sabit bekleme süresi: Saldırı sıklığı zamanla hızlanmaz, sabit 1.95 sn kalır
                        cooldown = 1.95
                        self.next_attack_time = now + cooldown

        # --------------------------------------------------------------------
        # 3. KARŞIDAKİ RAKİP BOKSÖRÜ VE OYUNCU MASKOTUNU ÇİZ
        # --------------------------------------------------------------------
        self._draw_opponent(canvas, opp_cx, opp_cy, now)

        self._draw_player_mascot(
            canvas,
            lean_state,
            lean_score,
            scaled_fists,
            opp_cx,
            opp_cy,
            now,
        )

        # Vuruş patlama efektlerini (POW! / BAM! / KRİTİK!) çiz
        orig_impacts = tracker.impacts
        scaled_impacts = [
            PunchImpactEffect(
                x=int(imp.x * scale_x),
                y=int(imp.y * scale_y),
                text=imp.text,
                color=imp.color,
                created_at=imp.created_at,
                duration=imp.duration,
                is_crit=imp.is_crit,
            )
            for imp in orig_impacts
        ]
        tracker.impacts = scaled_impacts
        draw_fists_and_punches(canvas, [], tracker, game_mode=True)
        tracker.impacts = orig_impacts

        # --------------------------------------------------------------------
        # 4. EKRAN SARSINTISI (SHAKE) VE HASAR/KAÇIŞ FLAŞI
        # --------------------------------------------------------------------
        if now < self.screen_flash_until and self.screen_flash_col is not None:
            flash_layer = np.full_like(canvas, self.screen_flash_col)
            cv2.addWeighted(flash_layer, 0.25, canvas, 0.75, 0, canvas)

        if now < self.shake_until:
            dx = random.randint(-14, 14)
            dy = random.randint(-10, 10)
            m_mat = np.float32([[1, 0, dx], [0, 1, dy]])
            canvas = cv2.warpAffine(canvas, m_mat, (game_w, game_h))

        # --------------------------------------------------------------------
        # 5. SAĞ ALT KÖŞEDE MİNİK CANLI WEBCAM GÖRÜNTÜSÜ (PiP)
        # --------------------------------------------------------------------
        self._draw_mini_webcam_pip(
            canvas,
            base_frame,
            detected_hands,
            hand_fingers_list,
            pose,
            lean_state,
            player_locked=player_locked,
        )

        # --------------------------------------------------------------------
        # 6. BOKS OYUNU ARAYÜZÜ (CAN BARLARI, EĞİLME RADARI, UYARILAR)
        # --------------------------------------------------------------------
        self._draw_game_hud(canvas, lean_state, lean_score, fps, now)

        return canvas

    def _draw_arena_background(self, canvas: np.ndarray, w: int, h: int) -> None:
        """Tam ekran boks ringi arka planını (spot ışıkları, ipler, köşe direkleri ve zemin) çizer."""
        # Stadyum üst karanlık gökyüzü / tavan
        canvas[0 : int(h * 0.56), :] = (32, 22, 18)

        # Arena spot ışığı huzmeleri
        overlay = canvas.copy()
        beam1 = np.array([[w // 2 - 280, 0], [w // 2 - 80, 0], [w // 2 - 40, int(h * 0.62)], [w // 2 - 480, int(h * 0.62)]], dtype=np.int32)
        beam2 = np.array([[w // 2 + 80, 0], [w // 2 + 280, 0], [w // 2 + 480, int(h * 0.62)], [w // 2 + 40, int(h * 0.62)]], dtype=np.int32)
        cv2.fillPoly(overlay, [beam1, beam2], (75, 65, 55))
        cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)

        # Perspektif Boks Ringi Zemini (Mavi/Gri Mat)
        ring_top_y = int(h * 0.52)
        ring_poly = np.array(
            [
                [140, ring_top_y],
                [w - 140, ring_top_y],
                [w + 80, h],
                [-80, h],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(canvas, [ring_poly], (85, 52, 32))
        cv2.polylines(canvas, [ring_poly], True, (220, 190, 140), 4, cv2.LINE_AA)

        # Ring Köşe Direkleri (Sol Kırmızı Köşe, Sağ Mavi Köşe)
        cv2.rectangle(canvas, (115, 210), (155, ring_top_y + 20), (40, 40, 200), -1, cv2.LINE_AA)
        cv2.rectangle(canvas, (w - 155, 210), (w - 115, ring_top_y + 20), (200, 70, 30), -1, cv2.LINE_AA)

        # Boks Ringi İpleri (3 yatay halat)
        for ry in (255, 315, 375):
            cv2.line(canvas, (145, ry), (w - 145, ry), (210, 210, 235), 5, cv2.LINE_AA)
            cv2.line(canvas, (0, ry + 45), (145, ry), (160, 160, 185), 4, cv2.LINE_AA)
            cv2.line(canvas, (w - 145, ry), (w, ry + 45), (160, 160, 185), 4, cv2.LINE_AA)

    def _draw_player_mascot(
        self,
        canvas: np.ndarray,
        lean_state: str,
        lean_score: float,
        scaled_fists: List[HandFistStatus],
        opp_cx: int,
        opp_cy: int,
        now: float,
    ) -> None:
        """
        Kameradaki vücut eğilmesini (Sağa/Sola/Aşağı) ve yumruk atışlarını
        birebir taklit eden Oyuncu Maskotunu (Boksör Karakterimizi) ekrana çizer.
        """
        h, w = canvas.shape[:2]

        # Oyuncunun eğilme puanına göre maskotun yatay/dikey kayması ve gövde açısı
        duck_offset = 65 if lean_state == "EGILDI" else 0
        mascot_cx = int(w * 0.50 + lean_score * 265.0)
        mascot_base_y = int(h - 92 + duck_offset)
        head_tilt_x = int(lean_score * 55.0)

        # Maskot Omuz ve Baş Koordinatları
        torso_top_y = mascot_base_y - 125
        head_cx = mascot_cx + head_tilt_x
        head_cy = torso_top_y - 48

        l_sh_x = mascot_cx - 92 + head_tilt_x // 2
        l_sh_y = torso_top_y + int(lean_score * 24)
        r_sh_x = mascot_cx + 92 + head_tilt_x // 2
        r_sh_y = torso_top_y - int(lean_score * 24)

        # Kullanıcının gerçek sol/sağ el durumunu bul
        left_fist_st: Optional[HandFistStatus] = None
        right_fist_st: Optional[HandFistStatus] = None
        for st in scaled_fists:
            if st.side == "SOL":
                left_fist_st = st
            elif st.side == "SAG":
                right_fist_st = st

        # Sol Eldiven Hedef Konumu (Yumruk atılıyorsa rakibe fırlar, değilse gard/el konumunda durur)
        if left_fist_st is not None and left_fist_st.is_punching:
            l_gx = opp_cx - 28
            l_gy = opp_cy + 18
            l_gr = 54
            l_punching = True
        elif left_fist_st is not None:
            l_gx = int(l_sh_x - 35 + (left_fist_st.norm_pos[0] - 0.30) * 220)
            l_gy = int(head_cy - 10 + (left_fist_st.norm_pos[1] - 0.55) * 160)
            l_gr = 40
            l_punching = False
        else:
            l_gx = l_sh_x - 28
            l_gy = head_cy - 12 + int(math.sin(now * 5.0) * 6)
            l_gr = 38
            l_punching = False

        # Sağ Eldiven Hedef Konumu
        if right_fist_st is not None and right_fist_st.is_punching:
            r_gx = opp_cx + 28
            r_gy = opp_cy + 18
            r_gr = 54
            r_punching = True
        elif right_fist_st is not None:
            r_gx = int(r_sh_x + 35 + (right_fist_st.norm_pos[0] - 0.70) * 220)
            r_gy = int(head_cy - 10 + (right_fist_st.norm_pos[1] - 0.55) * 160)
            r_gr = 40
            r_punching = False
        else:
            r_gx = r_sh_x + 28
            r_gy = head_cy - 12 + int(math.cos(now * 5.0) * 6)
            r_gr = 38
            r_punching = False

        # 1. Maskot Kolları (Omuzlardan eldivenlere uzanan kaslı kollar)
        for sh_x, sh_y, gx, gy, is_p in (
            (l_sh_x, l_sh_y, l_gx, l_gy, l_punching),
            (r_sh_x, r_sh_y, r_gx, r_gy, r_punching),
        ):
            arm_col = (0, 255, 255) if is_p else (95, 175, 225)
            elbow_x = (sh_x + gx) // 2 + (-25 if sh_x < mascot_cx else 25)
            elbow_y = (sh_y + gy) // 2 + 22
            cv2.line(canvas, (sh_x, sh_y), (elbow_x, elbow_y), arm_col, 20, cv2.LINE_AA)
            cv2.line(canvas, (elbow_x, elbow_y), (gx, gy), arm_col, 18, cv2.LINE_AA)

        # 2. Maskot Gövdesi (Yeşil/Turkuaz Boksör Sırtı & Atlet)
        torso_pts = np.array(
            [
                [l_sh_x, l_sh_y],
                [r_sh_x, r_sh_y],
                [mascot_cx + 68, mascot_base_y],
                [mascot_cx - 68, mascot_base_y],
            ],
            dtype=np.int32,
        )
        mascot_col = (110, 210, 40) if lean_state in ("SOL", "SAG") else (70, 175, 45)
        cv2.fillPoly(canvas, [torso_pts], mascot_col)
        cv2.polylines(canvas, [torso_pts], True, (255, 255, 255), 3, cv2.LINE_AA)

        # Sırt yazısı ("SEN")
        cv2.putText(
            canvas,
            "SEN",
            (mascot_cx - 32 + head_tilt_x // 3, torso_top_y + 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )

        # 3. Maskot Başı, Saçı ve Kafa Bandı (Headband)
        cv2.circle(canvas, (head_cx, head_cy), 46, (125, 175, 225), -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (head_cx, head_cy - 10), (47, 36), 0, 180, 360, (35, 30, 30), -1, cv2.LINE_AA)
        cv2.circle(canvas, (head_cx, head_cy), 46, (255, 255, 255), 3, cv2.LINE_AA)
        # Kafa bandı ve uçuşan kurdelesi
        cv2.rectangle(canvas, (head_cx - 46, head_cy - 14), (head_cx + 46, head_cy - 2), (0, 230, 255), -1, cv2.LINE_AA)
        ribbon_x = head_cx - int(lean_score * 35)
        cv2.line(canvas, (head_cx, head_cy - 8), (ribbon_x - 22, head_cy + 18), (0, 230, 255), 6, cv2.LINE_AA)
        cv2.line(canvas, (head_cx, head_cy - 8), (ribbon_x + 18, head_cy + 22), (0, 200, 230), 5, cv2.LINE_AA)

        # 4. Maskotun Boks Eldivenleri (Sol ve Sağ Yumruklarımız)
        for side_lbl, gx, gy, gr, is_p in (
            ("SOL", l_gx, l_gy, l_gr, l_punching),
            ("SAG", r_gx, r_gy, r_gr, r_punching),
        ):
            g_col = (0, 255, 255) if is_p else (0, 210, 95)
            if is_p:
                cv2.circle(canvas, (gx, gy), gr + 16, (0, 255, 255), 3, cv2.LINE_AA)
            cv2.circle(canvas, (gx, gy), gr, g_col, -1, cv2.LINE_AA)
            cv2.circle(canvas, (gx, gy), gr, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.circle(canvas, (gx - gr // 4, gy - gr // 4), max(6, gr // 4), (190, 255, 210), -1, cv2.LINE_AA)
            lbl_txt = f"{side_lbl} YUMRUK!" if is_p else side_lbl
            cv2.putText(
                canvas,
                lbl_txt,
                (gx - 32, gy - gr - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52 if is_p else 0.45,
                (0, 255, 255) if is_p else (230, 255, 230),
                2,
                cv2.LINE_AA,
            )

    def _draw_mini_webcam_pip(
        self,
        canvas: np.ndarray,
        base_frame: np.ndarray,
        detected_hands: List[DetectedHand],
        hand_fingers_list: List[List[int]],
        pose: Optional[DetectedPose],
        lean_state: str,
        player_locked: bool = True,
    ) -> None:
        """Sağ alt köşeye iskelet çizimli minik canlı webcam görüntüsünü (PiP) yerleştirir."""
        cam_preview = base_frame.copy()
        draw_pose_skeleton(cam_preview, pose, game_mode=True)
        for idx, hand in enumerate(detected_hands):
            fingers = hand_fingers_list[idx] if idx < len(hand_fingers_list) else [0, 0, 0, 0, 0]
            draw_hand_skeleton(cam_preview, hand, fingers)

        pip_w, pip_h = 256, 144
        small_cam = cv2.resize(cam_preview, (pip_w, pip_h), interpolation=cv2.INTER_LINEAR)

        h, w = canvas.shape[:2]
        x2 = w - 18
        y2 = h - 16
        x1 = x2 - pip_w
        y1 = y2 - pip_h

        canvas[y1:y2, x1:x2] = small_cam
        border_col = (0, 255, 160) if player_locked else ((0, 255, 160) if lean_state in ("SOL", "SAG") else (0, 215, 255))
        cv2.rectangle(canvas, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), border_col, 2, cv2.LINE_AA)
        cv2.rectangle(canvas, (x1, y1 - 24), (x2, y1), (25, 25, 25), -1, cv2.LINE_AA)

        header_title = f"OYUNCU KILITLI | {lean_state}" if player_locked else f"KAMERA | {lean_state}"
        header_col = (0, 255, 160) if player_locked else border_col
        cv2.putText(
            canvas,
            header_title,
            (x1 + 8, y1 - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            header_col,
            1,
            cv2.LINE_AA,
        )

    def _draw_opponent(self, canvas: np.ndarray, cx: int, cy: int, now: float) -> None:
        """Karşıdaki rakip boksörü, eldivenlerini ve saldırı animasyonunu çizer."""
        if self.state == "VICTORY_KO":
            # Rakip nakavt olduysa aşağı düşme ve başı üzerinde dönen yıldızlar
            ko_y = min(cy + 160, int(canvas.shape[0] * 0.72))
            cv2.ellipse(canvas, (cx, ko_y + 50), (110, 45), 0, 0, 360, (40, 40, 140), -1, cv2.LINE_AA)
            cv2.circle(canvas, (cx, ko_y - 10), 58, (130, 165, 235), -1, cv2.LINE_AA)
            cv2.circle(canvas, (cx, ko_y - 10), 58, (255, 255, 255), 3, cv2.LINE_AA)
            # X_X gözler
            for ex in (cx - 22, cx + 22):
                ey = ko_y - 18
                cv2.line(canvas, (ex - 8, ey - 8), (ex + 8, ey + 8), (20, 20, 20), 3, cv2.LINE_AA)
                cv2.line(canvas, (ex + 8, ey - 8), (ex - 8, ey + 8), (20, 20, 20), 3, cv2.LINE_AA)
            # Dönen yıldızlar
            for i in range(4):
                ang = now * 4.5 + i * (math.pi / 2)
                sx = int(cx + math.cos(ang) * 75)
                sy = int(ko_y - 65 + math.sin(ang) * 18)
                cv2.circle(canvas, (sx, sy), 9, (0, 235, 255), -1, cv2.LINE_AA)
            return

        is_hurt = now < self.opp_hurt_until

        # Gövde / Omuzlar (Yarı saydam boksör silüeti)
        torso_col = (60, 70, 210) if not is_hurt else (120, 220, 255)
        pts_torso = np.array(
            [
                [cx - 115, cy + 40],
                [cx + 115, cy + 40],
                [cx + 85, cy + 230],
                [cx - 85, cy + 230],
            ],
            dtype=np.int32,
        )
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [pts_torso], torso_col)
        cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)
        cv2.polylines(canvas, [pts_torso], True, (255, 255, 255), 3, cv2.LINE_AA)

        # Kemer / Şort çizgisi
        cv2.rectangle(canvas, (cx - 88, cy + 190), (cx + 88, cy + 230), (30, 30, 30), -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"CHAMP R{self.round_num}",
            (cx - 58, cy + 217),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 215, 255),
            2,
            cv2.LINE_AA,
        )

        # Baş ve Boks Kaskı
        head_cy = cy - 28
        head_r = 62
        skin_col = (145, 180, 240) if not is_hurt else (120, 140, 255)
        helmet_col = (30, 40, 190)

        cv2.circle(canvas, (cx, head_cy), head_r + 8, helmet_col, -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, head_cy + 6), head_r - 6, skin_col, -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, head_cy), head_r + 8, (255, 255, 255), 3, cv2.LINE_AA)

        # Yüz İfadesi (Hasar alırken X_X, saldırırken kızgın kaşlar)
        if is_hurt:
            for ex in (cx - 22, cx + 22):
                ey = head_cy - 4
                cv2.line(canvas, (ex - 9, ey - 9), (ex + 9, ey + 9), (20, 20, 20), 3, cv2.LINE_AA)
                cv2.line(canvas, (ex + 9, ey - 9), (ex - 9, ey + 9), (20, 20, 20), 3, cv2.LINE_AA)
            cv2.circle(canvas, (cx, head_cy + 26), 14, (30, 30, 120), -1, cv2.LINE_AA)
        else:
            # Kızgın gözler ve kaşlar
            cv2.circle(canvas, (cx - 22, head_cy - 2), 8, (20, 20, 20), -1, cv2.LINE_AA)
            cv2.circle(canvas, (cx + 22, head_cy - 2), 8, (20, 20, 20), -1, cv2.LINE_AA)
            cv2.circle(canvas, (cx - 22, head_cy - 2), 3, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, (cx + 22, head_cy - 2), 3, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.line(canvas, (cx - 36, head_cy - 16), (cx - 10, head_cy - 8), (20, 20, 20), 4, cv2.LINE_AA)
            cv2.line(canvas, (cx + 36, head_cy - 16), (cx + 10, head_cy - 8), (20, 20, 20), 4, cv2.LINE_AA)
            # Dişlik
            cv2.rectangle(canvas, (cx - 20, head_cy + 22), (cx + 20, head_cy + 34), (240, 240, 240), -1, cv2.LINE_AA)

        # Karşı Atak (Counter) penceresi açıksa rakibin üzerinde sarı hedef halkası
        if now < self.counter_window_until:
            pulse_r = int(82 + math.sin(now * 18.0) * 10)
            cv2.circle(canvas, (cx, head_cy), pulse_r, (0, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(
                canvas,
                "ACIK YAKALADIN! VUR!",
                (cx - 115, head_cy - 82),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # Rakibin Sol ve Sağ Boks Eldivenleri (Saldırı durumuna göre 3B büyüme efekti)
        l_glove_x, l_glove_y, l_glove_r = cx - 115, cy + 55, 42
        r_glove_x, r_glove_y, r_glove_r = cx + 115, cy + 55, 42
        l_col = (0, 50, 210)
        r_col = (0, 50, 210)

        if self.opp_action == "WINDUP":
            prog = min(1.0, (now - self.attack_start_time) / max(1e-3, self.windup_duration))
            warn_col = (0, int(200 * (1.0 - prog)), 255)
            if self.attack_type == "SOL_KROSE":
                # Ekranın solundaki eldiven gerilip parlıyor
                l_glove_x = int(cx - 155 - prog * 35)
                l_glove_y = int(cy + 20 - prog * 25)
                l_glove_r = int(44 + prog * 22)
                l_col = warn_col
                cv2.circle(canvas, (l_glove_x, l_glove_y), int(l_glove_r + (1.0 - prog) * 45), (0, 255, 255), 3, cv2.LINE_AA)
            elif self.attack_type == "SAG_KROSE":
                # Ekranın sağındaki eldiven gerilip parlıyor
                r_glove_x = int(cx + 155 + prog * 35)
                r_glove_y = int(cy + 20 - prog * 25)
                r_glove_r = int(44 + prog * 22)
                r_col = warn_col
                cv2.circle(canvas, (r_glove_x, r_glove_y), int(r_glove_r + (1.0 - prog) * 45), (0, 255, 255), 3, cv2.LINE_AA)
            else:  # DUZ_MERKEZ
                l_glove_x = int(cx - 65)
                r_glove_x = int(cx + 65)
                r_glove_y = int(cy + 15)
                r_glove_r = int(46 + prog * 24)
                r_col = warn_col
                cv2.circle(canvas, (cx, cy + 15), int(60 + (1.0 - prog) * 55), (0, 255, 255), 3, cv2.LINE_AA)

        elif self.opp_action == "STRIKE":
            # Yumruk kameraya doğru devleşerek fırlar!
            if self.attack_type == "SOL_KROSE":
                l_glove_x, l_glove_y, l_glove_r = int(cx - 130), cy + 10, 115
                l_col = (0, 0, 255)
            elif self.attack_type == "SAG_KROSE":
                r_glove_x, r_glove_y, r_glove_r = int(cx + 130), cy + 10, 115
                r_col = (0, 0, 255)
            else:
                r_glove_x, r_glove_y, r_glove_r = cx, cy + 5, 125
                r_col = (0, 0, 255)

        # Eldivenleri çiz
        for gx, gy, gr, gcol in (
            (l_glove_x, l_glove_y, l_glove_r, l_col),
            (r_glove_x, r_glove_y, r_glove_r, r_col),
        ):
            cv2.circle(canvas, (gx, gy), gr, gcol, -1, cv2.LINE_AA)
            cv2.circle(canvas, (gx, gy), gr, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.circle(canvas, (gx - gr // 4, gy - gr // 4), max(5, gr // 4), (120, 160, 255), -1, cv2.LINE_AA)

    def _draw_game_hud(
        self,
        canvas: np.ndarray,
        lean_state: str,
        lean_score: float,
        fps: float,
        now: float,
    ) -> None:
        """Tam ekran boks oyunu üst/alt göstergelerini, can barlarını ve kaçış uyarılarını çizer."""
        h, w = canvas.shape[:2]

        # Üst bar arka planı
        top_bar = canvas[0:105, 0:w]
        black_bar = np.zeros_like(top_bar)
        cv2.addWeighted(top_bar, 0.30, black_bar, 0.70, 0, canvas[0:105, 0:w])

        # --- SOL ÜST: OYUNCU CAN BARI (SEN) ---
        cv2.putText(
            canvas,
            f"SEN (HP: {self.player_hp}/{self.player_max_hp})",
            (30, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (100, 255, 140),
            2,
            cv2.LINE_AA,
        )
        bar_w, bar_h = 410, 28
        cv2.rectangle(canvas, (30, 48), (30 + bar_w, 48 + bar_h), (60, 60, 60), -1, cv2.LINE_AA)
        p_fill = int(bar_w * max(0.0, self.player_hp / float(self.player_max_hp)))
        p_col = (0, 230, 100) if self.player_hp > 35 else (0, 80, 255)
        if p_fill > 0:
            cv2.rectangle(canvas, (30, 48), (30 + p_fill, 48 + bar_h), p_col, -1, cv2.LINE_AA)
        cv2.rectangle(canvas, (30, 48), (30 + bar_w, 48 + bar_h), (255, 255, 255), 2, cv2.LINE_AA)

        # --- SAĞ ÜST: RAKİP CAN BARI (RAKİP BOKSÖR) ---
        opp_title = f"RAKIP BOKSOR (HP: {self.opp_hp}/{self.opp_max_hp})"
        cv2.putText(
            canvas,
            opp_title,
            (w - 30 - bar_w, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (80, 150, 255),
            2,
            cv2.LINE_AA,
        )
        rx1 = w - 30 - bar_w
        cv2.rectangle(canvas, (rx1, 48), (rx1 + bar_w, 48 + bar_h), (60, 60, 60), -1, cv2.LINE_AA)
        o_fill = int(bar_w * max(0.0, self.opp_hp / float(self.opp_max_hp)))
        if o_fill > 0:
            cv2.rectangle(canvas, (rx1, 48), (rx1 + o_fill, 48 + bar_h), (40, 80, 255), -1, cv2.LINE_AA)
        cv2.rectangle(canvas, (rx1, 48), (rx1 + bar_w, 48 + bar_h), (255, 255, 255), 2, cv2.LINE_AA)

        # --- ORTA ÜST: RAUND, SKOR VE FPS ---
        cv2.putText(
            canvas,
            f"RAUND {self.round_num} | SKOR: {self.score}",
            (w // 2 - 140, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (0, 235, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"FPS: {int(fps)} | [R] Yeni Raund / Baslat | [ESC] Cikis",
            (w // 2 - 215, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

        # --- SALDIRI UYARI BAŞLIĞI (KAÇIŞ YÖNLENDİRMESİ) ---
        if self.state == "FIGHT" and self.opp_action == "WINDUP":
            if self.attack_type == "SOL_KROSE":
                warn_msg = "DIKKAT! RAKIP SOLDAN VURUYOR -> SAGA EGIL! >>>"
            elif self.attack_type == "SAG_KROSE":
                warn_msg = "<<< SOLA EGIL! <- RAKIP SAGDAN VURUYOR!"
            else:
                warn_msg = "<<< SOLA VEYA SAGA EGIL! >>> (DUZ YUMRUK GELIYOR!)"

            cv2.rectangle(canvas, (w // 2 - 410, 118), (w // 2 + 410, 174), (0, 0, 180), -1, cv2.LINE_AA)
            cv2.rectangle(canvas, (w // 2 - 410, 118), (w // 2 + 410, 174), (0, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(
                canvas,
                warn_msg,
                (w // 2 - 385, 155),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.82,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        elif now < self.banner_until:
            cv2.rectangle(canvas, (w // 2 - 390, 118), (w // 2 + 390, 168), (25, 25, 25), -1, cv2.LINE_AA)
            cv2.rectangle(canvas, (w // 2 - 390, 118), (w // 2 + 390, 168), self.banner_color, 2, cv2.LINE_AA)
            cv2.putText(
                canvas,
                self.banner_text,
                (w // 2 - 365, 151),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                self.banner_color,
                2,
                cv2.LINE_AA,
            )

        # --- ALT ORTA: VÜCUT EĞİLME RADARI (SOLA EĞİL | MERKEZ | SAĞA EĞİL) ---
        radar_w = 540
        radar_h = 56
        rx = (w - radar_w) // 2
        ry = h - 76

        cv2.rectangle(canvas, (rx, ry), (rx + radar_w, ry + radar_h), (30, 30, 30), -1, cv2.LINE_AA)

        # Sol, Merkez, Sağ bölgeleri
        zone_w = radar_w // 3
        sol_active = lean_state == "SOL"
        merkez_active = lean_state == "MERKEZ"
        sag_active = lean_state == "SAG"

        if sol_active:
            cv2.rectangle(canvas, (rx, ry), (rx + zone_w, ry + radar_h), (0, 180, 120), -1, cv2.LINE_AA)
        elif merkez_active:
            cv2.rectangle(canvas, (rx + zone_w, ry), (rx + 2 * zone_w, ry + radar_h), (90, 90, 90), -1, cv2.LINE_AA)
        elif sag_active:
            cv2.rectangle(canvas, (rx + 2 * zone_w, ry), (rx + radar_w, ry + radar_h), (0, 180, 120), -1, cv2.LINE_AA)

        cv2.rectangle(canvas, (rx, ry), (rx + radar_w, ry + radar_h), (220, 220, 220), 2, cv2.LINE_AA)
        cv2.line(canvas, (rx + zone_w, ry), (rx + zone_w, ry + radar_h), (160, 160, 160), 1, cv2.LINE_AA)
        cv2.line(canvas, (rx + 2 * zone_w, ry), (rx + 2 * zone_w, ry + radar_h), (160, 160, 160), 1, cv2.LINE_AA)

        cv2.putText(canvas, "<< SOLA EGIL", (rx + 22, ry + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "MERKEZ", (rx + zone_w + 52, ry + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "SAGA EGIL >>", (rx + 2 * zone_w + 22, ry + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

        # Oyuncunun anlık eğilme imleci (Canlı top)
        marker_x = int(rx + radar_w * 0.5 + (lean_score * (radar_w * 0.44)))
        marker_x = max(rx + 14, min(rx + radar_w - 14, marker_x))
        cv2.circle(canvas, (marker_x, ry + radar_h // 2), 14, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (marker_x, ry + radar_h // 2), 14, (20, 20, 20), 2, cv2.LINE_AA)

        # Zafer veya Yenilgi Tam Ekran Sonuç Kutusu
        if self.state in ("VICTORY_KO", "DEFEAT_KO"):
            bx1, by1, bx2, by2 = w // 2 - 390, h // 2 - 115, w // 2 + 390, h // 2 + 115
            sub = canvas[by1:by2, bx1:bx2]
            dark_box = np.zeros_like(sub)
            cv2.addWeighted(sub, 0.2, dark_box, 0.8, 0, canvas[by1:by2, bx1:bx2])
            border_c = (0, 255, 120) if self.state == "VICTORY_KO" else (0, 80, 255)
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), border_c, 4, cv2.LINE_AA)

            if self.state == "VICTORY_KO":
                title = f"NAKAVT (K.O.) - RAUND {self.round_num} KAZANILDI!"
                sub_txt = "Yeni Raunda baslamak icin [R] tusuna basin"
            else:
                title = "MAC BITTI - YERE SERILDIN!"
                sub_txt = "Tekrar baslamak icin [R] tusuna basin"

            cv2.putText(canvas, title, (bx1 + 35, by1 + 72), cv2.FONT_HERSHEY_SIMPLEX, 1.05, border_c, 3, cv2.LINE_AA)
            cv2.putText(canvas, sub_txt, (bx1 + 65, by1 + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.76, (255, 255, 255), 2, cv2.LINE_AA)


# ============================================================================
# GÖRSEL ÇİZİM VE ARAYÜZ (HUD) FONKSİYONLARI
# ============================================================================
FINGER_NAMES = ["Bas", "Isaret", "Orta", "Yuzuk", "Serce"]


def draw_hand_skeleton(
    frame: np.ndarray,
    hand: DetectedHand,
    fingers: List[int],
) -> None:
    """Elin 21 eklem noktasını ve bağlantı çizgilerini ekrana çizer."""
    h, w, _ = frame.shape
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand.landmarks]

    # Bağlantı çizgileri
    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(frame, pts[start_idx], pts[end_idx], (220, 220, 220), 2, cv2.LINE_AA)

    # Parmak uçları (4, 8, 12, 16, 20) açık/kapalı durumuna göre renklendirilir
    tip_indices = [4, 8, 12, 16, 20]
    for idx, (px, py) in enumerate(pts):
        if idx in tip_indices:
            f_idx = tip_indices.index(idx)
            is_open = fingers[f_idx] == 1
            color = (0, 255, 100) if is_open else (60, 70, 230)
            cv2.circle(frame, (px, py), 8, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 10, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.circle(frame, (px, py), 5, (255, 190, 0), -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 6, (30, 30, 30), 1, cv2.LINE_AA)


def draw_virtual_objects(frame: np.ndarray, objects: List[DraggableObject]) -> None:
    """Yumruk yaparak kaydırılabilen sanal kutuları çizer (sadece kutu bölgesini harmanlar)."""
    fh, fw = frame.shape[:2]
    for obj in objects:
        x1 = max(0, obj.x)
        y1 = max(0, obj.y)
        x2 = min(fw, obj.x + obj.w)
        y2 = min(fh, obj.y + obj.h)
        if x2 > x1 and y2 > y1:
            roi = frame[y1:y2, x1:x2]
            color_rect = np.full_like(roi, obj.color)
            cv2.addWeighted(color_rect, 0.55, roi, 0.45, 0, roi)

        border_col = (0, 255, 255) if obj.is_grabbed else (255, 255, 255)
        thickness = 3 if obj.is_grabbed else 2
        cv2.rectangle(
            frame,
            (obj.x, obj.y),
            (obj.x + obj.w, obj.y + obj.h),
            border_col,
            thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            obj.label,
            (obj.x + 18, obj.y + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        status_txt = "KAYDIRILIYOR" if obj.is_grabbed else "Yumrukla Tut"
        cv2.putText(
            frame,
            status_txt,
            (obj.x + 15, obj.y + 66),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )


def draw_hud(
    frame: np.ndarray,
    total_fingers: int,
    per_hand_details: List[Tuple[str, List[int]]],
    pose: Optional[DetectedPose],
    punch_tracker: PunchTracker,
    state: GestureMouseState,
    gesture_info: dict,
    fps: float,
) -> None:
    """Ekran üzerindeki bilgi panelini, parmak sayısını, vücut/yumruk durumunu ve aktif jest bilgisini çizer."""
    h, w, _ = frame.shape
    now = time.perf_counter()

    # Fare kontrol aktif çerçevesi
    if state.enabled:
        mx1 = int(state.margin_x * w)
        my1 = int(state.margin_y * h)
        mx2 = int((1.0 - state.margin_x) * w)
        my2 = int((1.0 - state.margin_y) * h)
        cv2.rectangle(frame, (mx1, my1), (mx2, my2), (100, 100, 100), 1, cv2.LINE_AA)
        cv2.putText(
            frame,
            "Fare Aktif Alani",
            (mx1 + 8, my2 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (140, 140, 140),
            1,
            cv2.LINE_AA,
        )

    # Pinch (Başparmak + İşaret) görsel çizgisi
    if gesture_info["thumb_px"] and gesture_info["index_px"]:
        t_px = gesture_info["thumb_px"]
        i_px = gesture_info["index_px"]
        p_px = gesture_info["pinch_px"]
        line_col = (0, 255, 255) if state.is_pinching else (180, 180, 180)
        cv2.line(frame, t_px, i_px, line_col, 2, cv2.LINE_AA)
        if state.is_pinching and p_px:
            cv2.circle(frame, p_px, 12, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, p_px, 5, (0, 255, 0), -1, cv2.LINE_AA)

    # Yumruk görsel halkası
    if state.is_fist and gesture_info["palm_px"]:
        cv2.circle(frame, gesture_info["palm_px"], 28, (255, 100, 255), 3, cv2.LINE_AA)

    # Üst Yarı-Saydam Bilgi Paneli
    panel_h = 122
    sub = frame[0:panel_h, 0:w]
    black_rect = np.zeros(sub.shape, dtype=np.uint8)
    cv2.addWeighted(sub, 0.35, black_rect, 0.65, 0, frame[0:panel_h, 0:w])

    # Sol Kutu: Açık Parmak Sayısı (Büyük Gösterge)
    cv2.rectangle(frame, (15, 12), (245, 108), (0, 190, 90), -1, cv2.LINE_AA)
    cv2.putText(
        frame,
        "ACIK PARMAK",
        (28, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (230, 255, 230),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        str(total_fingers),
        (105, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.9,
        (255, 255, 255),
        4,
        cv2.LINE_AA,
    )

    # Orta Bölüm: Açık/Kapalı Parmak Detayı, Vücut Eğilme ve Yumruk Durumu
    if per_hand_details:
        hand_label, f_list = per_hand_details[0]
        tr_label = "Sag El" if hand_label == "Right" else "Sol El"
        open_names = [FINGER_NAMES[i] for i, val in enumerate(f_list) if val == 1]
        detail_str = f"{tr_label}: " + (", ".join(open_names) if open_names else "Yumruk (0)")
    else:
        detail_str = "Kameraya elinizi gosterin..."

    cv2.putText(
        frame,
        detail_str,
        (260, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    body_str = f"Vucut: {pose.lean_state}" if pose is not None else "Vucut: Araniyor"
    punch_str = (
        punch_tracker.last_punch_text
        if now < punch_tracker.last_punch_until
        else f"Yumruk Sayisi: {punch_tracker.total_punches}"
    )
    cv2.putText(
        frame,
        f"Hareket: {gesture_info['gesture_name']} | {body_str}",
        (260, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (100, 235, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"Boks: {punch_str}  -->  [G] TAM EKRAN BOKS OYUNU!",
        (260, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 215, 255) if now < punch_tracker.last_punch_until else (120, 255, 180),
        2,
        cv2.LINE_AA,
    )

    # Son Aksiyon Bildirimi (Sol Tık, Sağ Tık, Kaydırma vb.)
    feedback_txt = (
        state.action_text
        if now < state.action_until
        else "Pinch: Imlec | Serce: Sol/Cift Tik | Orta: Sag Tik | [G] Boks Oyunu"
    )
    feedback_col = state.action_color if now < state.action_until else (190, 190, 190)
    cv2.putText(
        frame,
        feedback_txt,
        (260, 112),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        feedback_col,
        1,
        cv2.LINE_AA,
    )

    # Sağ Üst: FPS ve Kısayollar
    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (w - 130, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 180),
        2,
        cv2.LINE_AA,
    )
    mouse_status = "ACIK" if state.enabled else "KAPALI"
    cv2.putText(
        frame,
        f"[B] Fare: {mouse_status}",
        (w - 205, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 255, 100) if state.enabled else (80, 80, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "[G] Boks Oyunu (Tam Ekran)",
        (w - 225, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 235, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"[M] {state.fist_mode} [O] Kutu [R] Sifirla",
        (w - 245, 98),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "[Q / ESC] Cikis",
        (w - 165, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (170, 170, 170),
        1,
        cv2.LINE_AA,
    )


# ============================================================================
# ANA UYGULAMA DÖNGÜSÜ
# ============================================================================
WINDOW_NAME = "Proje 2 - Tam Ekran Boks Oyunu"


def main() -> None:
    # Masaüstüne geçildiğinde Windows'un süreci yavaşlatmasını engelle
    disable_windows_background_throttling()

    print("=" * 72)
    print("PROJE 2: TAM EKRAN BOKS OYUNU (ODAK: OYUN & TEK OYUNCU KILIDI)")
    print("- Oyun giriste otomatik olarak tam ekran baslatilir.")
    print("- Tek Oyuncu Kilidi: Oyuna baslayan kisi kilitlenir, arkadaki/yandaki baska eller algilanmaz!")
    print("- Kontroller:")
    print("  * Yumruk: Gercek zamanli sag/sol yumruk atarak rakibe vur.")
    print("  * Kacis: Rakip vurus hazirligindayken bas/govdenizi SAGA veya SOLA egerek kac.")
    print("  * Raund Gecisi: Raund bitince dinlenip hazir oldugunuzda [R] tusuna basarak yeni tura gecin.")
    print("  * Cikis: [ESC] veya [Q]")
    print("=" * 72)

    cap = ThreadedCamera(index=0, width=960, height=540)
    if not cap.isOpened():
        print("HATA: Webcam açılamadı! Lütfen kameranızın bağlı olduğundan emin olun.")
        return

    # max_hands=4: Sahnedeki diğer elleri de yakalayıp PlayerLockTracker ile sadece kilitli oyuncuyu seçiyoruz
    detector = HandDetector(max_hands=4, detection_conf=0.45, tracking_conf=0.40)
    pose_detector = PoseDetector(detection_conf=0.55, tracking_conf=0.55)
    player_lock = PlayerLockTracker()
    punch_tracker = PunchTracker()
    boxing_game = BoxingGame()
    boxing_game.active = True  # GİRİŞTE DİREKT OYUN BAŞLASIN
    boxing_game.reset_match(keep_round=False)

    mouse = WindowsMouseController()
    state = GestureMouseState()
    state.enabled = False  # Odak oyun olacak, fare kontrolü pasif
    objects: Optional[List[DraggableObject]] = None

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    focus_window(WINDOW_NAME)
    window_focused = False

    prev_time = time.perf_counter()
    fps = 30.0
    saved_mouse_enabled = state.enabled

    try:
        while True:
            success, frame = cap.read()
            if not success or frame is None:
                continue

            # Ayna görüntüsü (kullanıcının sağ/sol hareketiyle ekranın aynı yöne gitmesi için)
            frame = cv2.flip(frame, 1)
            frame_h, frame_w, _ = frame.shape
            if objects is None:
                objects = create_default_objects(frame_w)

            # 1. Elleri ve Vücudu (Pose) tespit et
            raw_hands = detector.detect(frame)
            detected_pose = pose_detector.detect(frame)

            # TEK OYUNCU KİLİDİ: Bir kişi oyuna başlayınca başka elleri tamamen ele!
            detected_hands = player_lock.filter_hands(raw_hands, detected_pose, time.perf_counter())

            total_open_fingers = 0
            per_hand_details: List[Tuple[str, List[int]]] = []
            hand_fingers_list: List[List[int]] = []
            primary_hand: Optional[DetectedHand] = None
            primary_fingers: List[int] = [0, 0, 0, 0, 0]

            for idx, hand in enumerate(detected_hands):
                fingers = count_open_fingers(hand)
                hand_fingers_list.append(fingers)
                total_open_fingers += sum(fingers)
                per_hand_details.append((hand.label, fingers))

                if idx == 0:
                    primary_hand = hand
                    primary_fingers = fingers

            # 2. Yumrukları ve vuruşları (punch) takip et
            fist_statuses, new_punches = punch_tracker.update(
                detected_hands,
                hand_fingers_list,
                detected_pose,
                frame_w,
                frame_h,
                game_mode=boxing_game.active,
            )

            # FPS hesapla
            curr_time = time.perf_counter()
            dt = max(1e-4, curr_time - prev_time)
            fps = 0.85 * fps + 0.15 * (1.0 / dt)
            prev_time = curr_time

            # 3. [G] BOKS OYUNU MODU VEYA NORMAL KAMERA/FARE MODU
            if boxing_game.active:
                display_frame = boxing_game.update_and_render(
                    frame,
                    detected_hands,
                    hand_fingers_list,
                    detected_pose,
                    fist_statuses,
                    new_punches,
                    punch_tracker,
                    fps,
                    player_locked=player_lock.is_locked,
                )
            else:
                # Vücut (Pose) iskeletini çiz
                draw_pose_skeleton(frame, detected_pose, game_mode=False)

                # El iskeletlerini çiz
                for idx, hand in enumerate(detected_hands):
                    draw_hand_skeleton(frame, hand, hand_fingers_list[idx])

                # Algılanan yumrukları ve vuruş efektlerini ekranda göster
                draw_fists_and_punches(frame, fist_statuses, punch_tracker, game_mode=False)

                # Fare imleci, Serçe Sol/Çift Tık, Orta Sağ Tık ve Yumrukla Kaydırma
                gesture_info = process_gestures(
                    primary_hand,
                    primary_fingers,
                    state,
                    mouse,
                    objects,
                    frame_w,
                    frame_h,
                )

                # Sanal nesneleri çiz
                if state.show_virtual_objects:
                    draw_virtual_objects(frame, objects)

                # Bilgi ekranını (HUD) çiz
                draw_hud(
                    frame,
                    total_open_fingers,
                    per_hand_details,
                    detected_pose,
                    punch_tracker,
                    state,
                    gesture_info,
                    fps,
                )
                display_frame = frame

            cv2.imshow(WINDOW_NAME, display_frame)
            if not window_focused:
                focus_window(WINDOW_NAME)
                window_focused = True

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):  # Q veya ESC
                break
            elif key in (ord("g"), ord("G")):
                boxing_game.active = not boxing_game.active
                mouse.stop_drag()
                if boxing_game.active:
                    saved_mouse_enabled = state.enabled
                    state.enabled = False
                    boxing_game.reset_match(keep_round=False)
                    cv2.setWindowProperty(
                        WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
                    )
                    focus_window(WINDOW_NAME)
                else:
                    state.enabled = saved_mouse_enabled
                    cv2.setWindowProperty(
                        WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL
                    )
                    cv2.resizeWindow(WINDOW_NAME, 960, 540)
                    state.set_feedback("Normal Moda Donuldu", (0, 255, 255), 1.2)
            elif key in (ord("b"), ord("B")):
                if not boxing_game.active:
                    state.enabled = not state.enabled
                    if not state.enabled:
                        mouse.stop_drag()
                    durum = "ACILDI" if state.enabled else "KAPATILDI"
                    state.set_feedback(f"Fare Kontrolu {durum}", (0, 255, 255), 1.2)
            elif key in (ord("m"), ord("M")):
                if not boxing_game.active:
                    mouse.stop_drag()
                    state.fist_mode = "SCROLL" if state.fist_mode == "DRAG" else "DRAG"
                    state.set_feedback(f"Yumruk Modu: {state.fist_mode}", (255, 200, 0), 1.2)
            elif key in (ord("o"), ord("O")):
                if not boxing_game.active:
                    state.show_virtual_objects = not state.show_virtual_objects
                    for obj in objects:
                        obj.is_grabbed = False
            elif key in (ord("r"), ord("R")):
                if boxing_game.active:
                    if boxing_game.state == "VICTORY_KO":
                        boxing_game.round_num += 1
                        boxing_game.reset_match(keep_round=True)
                    else:
                        boxing_game.reset_match(keep_round=False)
                else:
                    objects = create_default_objects(frame_w)
                    state.set_feedback("Sanal Nesneler Sifirlandi", (180, 255, 180), 1.0)

    finally:
        mouse.stop_drag()
        detector.close()
        pose_detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

