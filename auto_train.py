#!/usr/bin/env python3
"""
Otomatik Model Güncelleme Scripti
Belirli aralıklarla AI modelini otomatik olarak yeniden eğitir.
"""

import schedule
import time
import subprocess
import logging
import os
from datetime import datetime
import sys

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('auto_train.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def train_model():
    """Model eğitimini çalıştırır"""
    try:
        logger.info("=== OTOMATİK MODEL EĞİTİMİ BAŞLATILIYOR ===")
        
        # train_model.py scriptini çalıştır
        result = subprocess.run([sys.executable, 'train_model.py'], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            logger.info("Model eğitimi başarıyla tamamlandı!")
            logger.info("Çıktı: " + result.stdout)
        else:
            logger.error(f"Model eğitimi başarısız! Hata: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        logger.error("Model eğitimi zaman aşımına uğradı (5 dakika)")
    except Exception as e:
        logger.error(f"Model eğitimi hatası: {e}")

def check_data_availability():
    """Veri mevcutluğunu kontrol eder"""
    try:
        if not os.path.exists('signals.csv'):
            logger.warning("signals.csv dosyası bulunamadı!")
            return False
        
        # CSV dosyasını kontrol et
        import pandas as pd
        df = pd.read_csv('signals.csv')
        
        # Tamamlanmış sinyal sayısını kontrol et
        completed_signals = df[df['status'] != 'Beklemede']
        
        if len(completed_signals) < 50:
            logger.warning(f"Yeterli veri yok! Sadece {len(completed_signals)} tamamlanmış sinyal var.")
            return False
        
        logger.info(f"Yeterli veri mevcut: {len(completed_signals)} tamamlanmış sinyal")
        return True
        
    except Exception as e:
        logger.error(f"Veri kontrolü hatası: {e}")
        return False

def scheduled_training():
    """Zamanlanmış eğitim fonksiyonu"""
    logger.info("Zamanlanmış eğitim kontrolü başlatılıyor...")
    
    if check_data_availability():
        train_model()
    else:
        logger.info("Yeterli veri olmadığı için eğitim atlandı.")

def main():
    """Ana fonksiyon"""
    logger.info("=== OTOMATİK MODEL GÜNCELLEME SİSTEMİ BAŞLATILIYOR ===")
    
    # Eğitim zamanlaması (her hafta Pazartesi saat 02:00'de)
    schedule.every().monday.at("02:00").do(scheduled_training)
    
    # Test için: her 6 saatte bir (geliştirme aşamasında)
    # schedule.every(6).hours.do(scheduled_training)
    
    logger.info("Eğitim zamanlaması ayarlandı:")
    logger.info("- Her Pazartesi saat 02:00'de otomatik eğitim")
    logger.info("- Sistem sürekli çalışır durumda")
    logger.info("- Loglar 'auto_train.log' dosyasına kaydedilir")
    
    # İlk çalıştırmada veri kontrolü yap
    if check_data_availability():
        logger.info("İlk çalıştırma: Model eğitimi başlatılıyor...")
        train_model()
    else:
        logger.info("İlk çalıştırma: Yeterli veri yok, eğitim atlandı.")
    
    # Sürekli çalışan döngü
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)  # 1 dakika bekle
        except KeyboardInterrupt:
            logger.info("Sistem kullanıcı tarafından durduruldu.")
            break
        except Exception as e:
            logger.error(f"Beklenmeyen hata: {e}")
            time.sleep(300)  # 5 dakika bekle ve devam et

if __name__ == "__main__":
    main() 