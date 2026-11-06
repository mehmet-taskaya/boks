# Proje 2: Kamerayla Tam Ekran Boks Oyunu, El & Vücut Takibi ve Tek Oyuncu Kilidi

`opencv-python` ve `mediapipe` kullanılarak geliştirilmiş, **girişte doğrudan tam ekran başlayan**, **tek oyuncu kilidine sahip** (başkalarının eli algılanmaz), klavyeye ihtiyaç bırakmayan interaktif Boks Oyunu ve hareket takibi uygulaması.

## Öne Çıkan Özellikler

### 1. Girişte Direkt Başlayan Tam Ekran Boks Oyunu
- Uygulama çalıştırıldığı anda doğrudan **Tam Ekran Boks Arenası (1280x720 HD)** modunda başlar.
- Windows pencere odağı otomatik olarak oyuna verilir.
- **Oyun Kontrolleri**:
  - Karşıdaki rakip yumruk hazırladığında vücudunuzla **SAĞA veya SOLA eğilerek** kaçın.
  - Kaçış sonrası açılan pencerede **2x Kritik Karşı Atak** vurun.
  - Kameraya doğru seri yumruklar atarak rakibin canını sıfırlayıp **NAKAVT (K.O.)** edin.
  - **Raund Geçişi**: Raunt bittiğinde siz hazır olup **`[R]` tuşuna basana kadar** yeni tur başlamaz. Dinlenip hazır olduğunuzda `[R]` tuşuna basarak devam edebilirsiniz.

### 2. Tek Oyuncu Kilidi (Player Lock)
- Oyuna başlayan kişi sistem tarafından otomatik olarak kilitlenir (`OYUNCU KILITLI`).
- **Yabancı El Filtresi**: Arkadan geçenlerin, yanınızda duranların veya kameraya uzatılan başka ellerin yumruk ya da hareket tetiklemesi engellenir.
- Vücut (Pose) iskeletine, bilek koordinatlarına ve gövde sınırlarına göre sadece oyunu oynayan tek kişinin sol ve sağ elleri dikkate alınır.

### 3. Vücut ve Canlı Kamera Önizlemesi (PiP)
- Ekranın sağ alt köşesinde minik canlı kamera penceresi (Picture-in-Picture) yer alır.
- Canlı pencerede oyuncunun kilit durumu (`OYUNCU KILITLI`), vücut eğilme açısı (`SOL`, `SAG`, `MERKEZ`) ve iskelet takibi gösterilir.

## Kurulum ve Çalıştırma

```bash
pip install -r requirements.txt
python main.py
```

## Kontroller

- **Yumruk**: Kameraya / rakibe doğru ileri el uzatarak yumruk atın.
- **Eğilme (Kaçış)**: Vücudunuzu sağa veya sola eğerek rakibin kroselerinden kaçın.
- **Raund Geçişi**: Raunt bitince yeni tura geçmek için `[R]` tuşuna basın.
- **Çıkış**: `[ESC]` veya `[Q]` ile oyunu kapatabilirsiniz.
