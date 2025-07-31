# 🚀 Kripto Analiz ve Sinyal Uygulaması

Modern ve kapsamlı kripto para analiz uygulaması. Binance API kullanarak gerçek zamanlı veri çeker ve gelişmiş teknik analiz göstergeleri ile alım-satım sinyalleri üretir.

## ✨ Özellikler

### 📊 Teknik Göstergeler
- **RSI (Relative Strength Index)** - Aşırı alım/satım seviyeleri
- **Bollinger Bantları** - Volatilite ve fiyat kanalları
- **MACD** - Momentum ve trend değişimleri
- **Stokastik Osilatör** - Momentum göstergesi
- **Williams %R** - Aşırı alım/satım seviyeleri
- **ATR (Average True Range)** - Volatilite göstergesi

### 🎯 Akıllı Sinyal Sistemi
- Çoklu gösterge analizi
- Güçlü/Zayıf sinyal ayrımı
- Otomatik strateji değerlendirmesi
- Sinyal geçmişi ve başarı oranı takibi

### 🎨 Modern Arayüz
- Responsive tasarım (mobil uyumlu)
- Bootstrap 5 ile modern görünüm
- Plotly ile interaktif grafikler
- Otomatik yenileme sistemi
- Favori coinler yönetimi

### 🔒 Güvenlik ve Performans
- Rate limiting (dakikada 30 istek)
- Güvenli HTTP istekleri
- Hata yönetimi ve loglama
- Optimize edilmiş grafik performansı

## 🛠️ Kurulum

### Gereksinimler
- Python 3.8+
- pip (Python paket yöneticisi)

### Adımlar

1. **Projeyi klonlayın:**
```bash
git clone <repository-url>
cd kripto-analiz
```

2. **Sanal ortam oluşturun (önerilen):**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Bağımlılıkları yükleyin:**
```bash
pip install -r requirements.txt
```

4. **Uygulamayı çalıştırın:**
```bash
python app.py
```

5. **Tarayıcıda açın:**
```
http://localhost:5000
```

## 📈 Kullanım

### Temel Kullanım
1. Coin sembolü girin (örn: BTCUSDT, ETHUSDT)
2. Zaman dilimi seçin (1m, 5m, 15m, 1h, 4h, 1d)
3. "Analiz Et" butonuna tıklayın
4. Sonuçları inceleyin

### Popüler Coinler
Sayfada önceden tanımlanmış popüler coinler bulunur:
- Bitcoin (BTC)
- Ethereum (ETH)
- Binance Coin (BNB)
- Solana (SOL)
- Ripple (XRP)
- Ve daha fazlası...

### Favoriler Sistemi
- ⭐ butonuna tıklayarak coinleri favorilere ekleyin
- Favori coinleriniz sayfanın üst kısmında görünür
- Hızlı erişim için favorilerinizi kullanın

### Otomatik Yenileme
- Sayfa otomatik olarak 30 saniyede bir yenilenir
- "Durdur/Başlat" butonu ile kontrol edebilirsiniz

## 🔧 Teknik Detaylar

### API Kullanımı
- **Binance API v3** kullanılır
- Rate limiting: Dakikada 30 istek
- Timeout: 10 saniye
- Retry mekanizması: 3 deneme

### Teknik Analiz Algoritmaları
- **RSI**: 14 periyot varsayılan
- **Bollinger Bantları**: 20 periyot, 2 standart sapma
- **MACD**: 12/26/9 parametreleri
- **Stokastik**: 14/3 parametreleri
- **Williams %R**: 14 periyot
- **ATR**: 14 periyot

### Sinyal Stratejisi
- **Güçlü AL**: 4+ gösterge AL sinyali
- **AL**: 3 gösterge AL sinyali
- **Zayıf AL**: 2 gösterge AL sinyali
- **BEKLE**: Yetersiz sinyal
- **Zayıf SAT**: 2 gösterge SAT sinyali
- **SAT**: 3 gösterge SAT sinyali
- **Güçlü SAT**: 4+ gösterge SAT sinyali

## 📁 Dosya Yapısı

```
kripto-analiz/
├── app.py                 # Ana uygulama dosyası
├── requirements.txt       # Python bağımlılıkları
├── README.md             # Bu dosya
└── templates/
    └── index.html        # HTML template
```

## 🚨 Önemli Notlar

### Risk Uyarısı
- Bu uygulama sadece eğitim amaçlıdır
- Yatırım kararları için profesyonel danışmanlık alın
- Geçmiş performans gelecekteki sonuçları garanti etmez

### API Limitleri
- Binance API rate limitlerine dikkat edin
- Uygulama dakikada maksimum 30 istek yapar
- Limit aşıldığında uyarı mesajı görünür

### Veri Güvenilirliği
- Veriler Binance API'den alınır
- İnternet bağlantısı gereklidir
- API kesintilerinde hata mesajları görünür

## 🐛 Sorun Giderme

### Yaygın Sorunlar

1. **"Veri alınamadı" hatası:**
   - İnternet bağlantınızı kontrol edin
   - Binance API durumunu kontrol edin
   - Rate limit aşılmış olabilir

2. **Grafik yüklenmiyor:**
   - JavaScript etkin olduğundan emin olun
   - Tarayıcı cache'ini temizleyin
   - Farklı tarayıcı deneyin

3. **Yavaş yükleme:**
   - İnternet bağlantınızı kontrol edin
   - Tarayıcı performansını kontrol edin
   - Otomatik yenilemeyi kapatın

### Log Dosyaları
Uygulama çalışırken konsol çıktısında hata mesajları görünür. Sorun yaşarsanız bu mesajları kontrol edin.

## 🔄 Güncellemeler

### v1.1 (Güncel)
- ✅ Hata yönetimi iyileştirildi
- ✅ Rate limiting eklendi
- ✅ Yeni teknik göstergeler (Williams %R, ATR)
- ✅ Performans optimizasyonu
- ✅ Güvenlik iyileştirmeleri
- ✅ Kod organizasyonu

### Gelecek Özellikler
- [ ] Daha fazla teknik gösterge
- [ ] Backtesting sistemi
- [ ] Email/Telegram bildirimleri
- [ ] Portföy takibi
- [ ] API key desteği

## 📞 Destek

Sorunlarınız için:
1. GitHub Issues kullanın
2. README dosyasını kontrol edin
3. Log mesajlarını inceleyin

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

---

**⚠️ Uyarı:** Bu uygulama sadece eğitim amaçlıdır. Yatırım kararları için profesyonel danışmanlık alın. 