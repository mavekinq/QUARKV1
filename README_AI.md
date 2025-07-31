# 🤖 AI Destekli Kripto Analiz Sistemi

Bu dokümantasyon, kripto analiz uygulamanızın yapay zeka ile geliştirilmiş versiyonunu açıklar.

## 🚀 Yeni AI Özellikleri

### 1. **Veri Kaydı Sistemi**
- Her sinyal otomatik olarak `signals.csv` dosyasına kaydedilir
- 30 dakika sonra sinyal başarısı otomatik hesaplanır
- Veri kalitesi analizi ve raporlama

### 2. **Yapay Zeka Modeli**
- **RandomForest** ve **GradientBoosting** modelleri
- 10 teknik gösterge ile eğitim
- Otomatik model seçimi (en iyi performans)
- Güven skoru hesaplama

### 3. **Akıllı Sinyal Üretimi**
- AI modeli varsa: AI tahminleri kullanılır
- AI modeli yoksa: Klasik strateji kullanılır
- Hibrit yaklaşım ile güvenilirlik artırılır

### 4. **Otomatik Model Güncelleme**
- Haftalık otomatik yeniden eğitim
- Veri kalitesi kontrolü
- Performans izleme ve loglama

## 📁 Dosya Yapısı

```
v1/
├── app.py                 # Ana Flask uygulaması (AI entegreli)
├── train_model.py         # AI model eğitim scripti
├── auto_train.py          # Otomatik eğitim scripti
├── requirements.txt       # Güncellenmiş bağımlılıklar
├── signals.csv           # Sinyal verileri (otomatik oluşur)
├── model.pkl             # Eğitilmiş AI modeli
├── scaler.pkl            # Özellik ölçeklendirici
├── model_info.txt        # Model bilgileri
├── auto_train.log        # Otomatik eğitim logları
└── templates/
    └── index.html        # Güncellenmiş template
```

## 🛠️ Kurulum ve Kullanım

### 1. Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt
```

### 2. Uygulamayı Başlatın
```bash
python app.py
```

### 3. İlk AI Modelini Eğitin
```bash
python train_model.py
```

### 4. Otomatik Eğitimi Başlatın (Opsiyonel)
```bash
python auto_train.py
```

## 📊 AI Model Detayları

### Özellikler (Features)
1. **RSI** - Relative Strength Index
2. **Bollinger Alt Bandı** - Alt destek seviyesi
3. **Bollinger Orta Bandı** - Hareketli ortalama
4. **Bollinger Üst Bandı** - Üst direnç seviyesi
5. **MACD** - MACD değeri
6. **MACD Sinyal** - MACD sinyal çizgisi
7. **Stochastic %K** - Stokastik K değeri
8. **Stochastic %D** - Stokastik D değeri
9. **Williams %R** - Williams %R değeri
10. **ATR** - Average True Range

### Hedef Değişken (Target)
- **0**: BEKLE
- **1**: AL (AL, AL Zayıf, AL Güçlü)
- **2**: SAT (SAT, SAT Zayıf, SAT Güçlü)

### Model Performansı
- **Cross-validation** ile doğrulama
- **Özellik önem** analizi
- **Sınıf dengesi** kontrolü
- **Güven skoru** hesaplama

## 🔄 Veri Akışı

### 1. Sinyal Üretimi
```
Kullanıcı İsteği → Teknik Analiz → AI Tahmin → Sinyal Üretimi → CSV Kaydı
```

### 2. Başarı Hesaplama
```
30 Dakika Bekleme → Güncel Fiyat → Kar/Zarar Hesaplama → Durum Güncelleme
```

### 3. Model Eğitimi
```
CSV Verisi → Veri Temizleme → Model Eğitimi → Performans Değerlendirme → Model Kaydetme
```

## 📈 Performans İzleme

### Veri Kalitesi Metrikleri
- Toplam sinyal sayısı
- Başarı oranı
- Sinyal türü dağılımı
- Coin çeşitliliği

### Model Performans Metrikleri
- Test accuracy
- Cross-validation skoru
- Sınıf bazında performans
- Özellik önem sırası

## ⚙️ Konfigürasyon

### Eğitim Parametreleri
```python
# train_model.py içinde
MIN_SIGNALS_FOR_TRAINING = 50  # Minimum sinyal sayısı
TEST_SIZE = 0.2                # Test seti oranı
CROSS_VALIDATION_FOLDS = 5     # CV katlama sayısı
```

### Otomatik Eğitim Zamanlaması
```python
# auto_train.py içinde
schedule.every().monday.at("02:00").do(scheduled_training)  # Her Pazartesi 02:00
```

## 🚨 Önemli Notlar

### Veri Gereksinimleri
- **Minimum 50 tamamlanmış sinyal** gerekli
- **Çeşitli coin** verileri önerilir
- **Farklı zaman dilimleri** önerilir

### Model Güncelleme
- Model her hafta otomatik güncellenir
- Yeni verilerle performans artar
- Eski model yedeklenir

### Güvenlik
- AI tahminleri %100 güvenilir değildir
- Risk yönetimi önemlidir
- Çoklu gösterge analizi yapın

## 🔧 Sorun Giderme

### Yaygın Sorunlar

1. **"AI model yüklenemedi" hatası:**
   ```bash
   python train_model.py  # Modeli eğitin
   ```

2. **"Yeterli veri yok" uyarısı:**
   - Daha fazla sinyal üretin
   - 30 dakika bekleyin
   - Farklı coinler deneyin

3. **Model performansı düşük:**
   - Daha fazla veri toplayın
   - Farklı zaman dilimleri kullanın
   - Teknik gösterge parametrelerini ayarlayın

### Log Dosyaları
- `auto_train.log`: Otomatik eğitim logları
- Konsol çıktısı: Gerçek zamanlı bilgiler
- `model_info.txt`: Model performans bilgileri

## 📊 Örnek Kullanım Senaryosu

### 1. İlk Kurulum
```bash
# Uygulamayı başlat
python app.py

# Birkaç sinyal üret (farklı coinler)
# 30 dakika bekle

# İlk modeli eğit
python train_model.py
```

### 2. Normal Kullanım
```bash
# Uygulamayı kullan (AI destekli)
python app.py

# Otomatik eğitimi başlat (arka planda)
python auto_train.py
```

### 3. Performans İzleme
```bash
# Veri kalitesini kontrol et
python train_model.py

# Model bilgilerini görüntüle
cat model_info.txt
```

## 🎯 Gelecek Geliştirmeler

- [ ] Daha gelişmiş AI modelleri (LSTM, Transformer)
- [ ] Duygu analizi entegrasyonu
- [ ] Portföy optimizasyonu
- [ ] Gerçek zamanlı öğrenme
- [ ] Webhook entegrasyonu
- [ ] Telegram bot entegrasyonu

---

**⚠️ Uyarı:** Bu AI sistemi eğitim amaçlıdır. Yatırım kararları için profesyonel danışmanlık alın. 