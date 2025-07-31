#!/usr/bin/env python3
"""
AI Model Eğitim Scripti
Kripto Analiz ve Sinyal Uygulaması için yapay zeka modelini eğitir.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pickle
import os
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SIGNALS_FILE = 'signals.csv'
MODEL_FILE = 'model.pkl'
SCALER_FILE = 'scaler.pkl'

def load_and_prepare_data():
    """CSV dosyasından veriyi yükler ve hazırlar"""
    try:
        if not os.path.exists(SIGNALS_FILE):
            logger.error(f"{SIGNALS_FILE} dosyası bulunamadı!")
            return None, None, None
        
        # Veriyi yükle
        df = pd.read_csv(SIGNALS_FILE)
        logger.info(f"Toplam {len(df)} kayıt yüklendi.")
        
        # Sadece tamamlanmış sinyalleri al (status != 'Beklemede')
        completed_signals = df[df['status'] != 'Beklemede'].copy()
        logger.info(f"Tamamlanmış {len(completed_signals)} sinyal bulundu.")
        
        if len(completed_signals) < 50:
            logger.warning("Eğitim için yeterli veri yok! En az 50 tamamlanmış sinyal gerekli.")
            return None, None, None
        
        # Özellikler (features)
        feature_columns = [
            'rsi', 'bb_lower', 'bb_middle', 'bb_upper', 
            'macd', 'macd_signal', 'stoch_k', 'stoch_d', 
            'williams_r', 'atr'
        ]
        
        # Eksik değerleri temizle
        completed_signals = completed_signals.dropna(subset=feature_columns)
        logger.info(f"Eksik değerler temizlendikten sonra {len(completed_signals)} kayıt kaldı.")
        
        # Özellikler ve hedef değişken
        X = completed_signals[feature_columns].values
        y = completed_signals['signal'].values
        
        # Sinyal sınıflarını sayısal değerlere çevir
        signal_mapping = {
            'BEKLE': 0,
            'AL': 1,
            'AL (Zayıf)': 1,
            'AL (Güçlü)': 1,
            'SAT': 2,
            'SAT (Zayıf)': 2,
            'SAT (Güçlü)': 2
        }
        
        y_numeric = np.array([signal_mapping.get(signal, 0) for signal in y])
        
        # Sınıf dağılımını kontrol et
        unique, counts = np.unique(y_numeric, return_counts=True)
        logger.info(f"Sınıf dağılımı: {dict(zip(unique, counts))}")
        
        return X, y_numeric, feature_columns
        
    except Exception as e:
        logger.error(f"Veri yükleme hatası: {e}")
        return None, None, None

def train_model(X, y, feature_columns):
    """Modeli eğitir"""
    try:
        # Veriyi eğitim ve test setlerine ayır
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Eğitim seti: {len(X_train)} örnek")
        logger.info(f"Test seti: {len(X_test)} örnek")
        
        # Özellik ölçeklendirme
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Model seçimi ve eğitim
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                random_state=42,
                class_weight='balanced'
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
        }
        
        best_model = None
        best_score = 0
        best_model_name = None
        
        for name, model in models.items():
            logger.info(f"{name} modeli eğitiliyor...")
            
            # Modeli eğit
            model.fit(X_train_scaled, y_train)
            
            # Test seti üzerinde değerlendir
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            
            logger.info(f"{name} - Test Accuracy: {accuracy:.4f}")
            logger.info(f"{name} - CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # En iyi modeli seç
            if accuracy > best_score:
                best_score = accuracy
                best_model = model
                best_model_name = name
        
        logger.info(f"En iyi model: {best_model_name} (Accuracy: {best_score:.4f})")
        
        # En iyi modelin detaylı performansını göster
        y_pred_best = best_model.predict(X_test_scaled)
        
        logger.info("\n=== Sınıflandırma Raporu ===")
        logger.info(classification_report(y_test, y_pred_best, 
                                        target_names=['BEKLE', 'AL', 'SAT']))
        
        # Özellik önemini göster
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': feature_columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info("\n=== Özellik Önem Sırası ===")
            for idx, row in feature_importance.iterrows():
                logger.info(f"{row['feature']}: {row['importance']:.4f}")
        
        return best_model, scaler, best_score
        
    except Exception as e:
        logger.error(f"Model eğitim hatası: {e}")
        return None, None, None

def save_model(model, scaler, accuracy):
    """Modeli kaydeder"""
    try:
        # Model dosyasını kaydet
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(model, f)
        
        # Scaler dosyasını kaydet
        with open(SCALER_FILE, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Model bilgilerini kaydet
        model_info = {
            'accuracy': accuracy,
            'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_type': type(model).__name__,
            'feature_count': model.n_features_in_ if hasattr(model, 'n_features_in_') else 'Unknown'
        }
        
        with open('model_info.txt', 'w') as f:
            for key, value in model_info.items():
                f.write(f"{key}: {value}\n")
        
        logger.info(f"Model başarıyla kaydedildi: {MODEL_FILE}")
        logger.info(f"Model doğruluğu: {accuracy:.4f}")
        logger.info(f"Eğitim tarihi: {model_info['trained_at']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Model kaydetme hatası: {e}")
        return False

def analyze_data_quality():
    """Veri kalitesini analiz eder"""
    try:
        if not os.path.exists(SIGNALS_FILE):
            logger.error(f"{SIGNALS_FILE} dosyası bulunamadı!")
            return
        
        df = pd.read_csv(SIGNALS_FILE)
        
        logger.info("\n=== VERİ KALİTESİ ANALİZİ ===")
        logger.info(f"Toplam kayıt: {len(df)}")
        logger.info(f"Benzersiz coin sayısı: {df['symbol'].nunique()}")
        logger.info(f"Tarih aralığı: {df['timestamp'].min()} - {df['timestamp'].max()}")
        
        # Sinyal dağılımı
        signal_counts = df['signal'].value_counts()
        logger.info("\nSinyal dağılımı:")
        for signal, count in signal_counts.items():
            logger.info(f"  {signal}: {count}")
        
        # Durum dağılımı
        status_counts = df['status'].value_counts()
        logger.info("\nDurum dağılımı:")
        for status, count in status_counts.items():
            logger.info(f"  {status}: {count}")
        
        # Başarı oranı
        completed = df[df['status'] != 'Beklemede']
        if len(completed) > 0:
            success_rate = (completed['status'] == 'Başarılı').mean() * 100
            logger.info(f"\nGenel başarı oranı: %{success_rate:.2f}")
            
            # Sinyal türüne göre başarı oranı
            for signal in completed['signal'].unique():
                signal_data = completed[completed['signal'] == signal]
                if len(signal_data) > 0:
                    signal_success = (signal_data['status'] == 'Başarılı').mean() * 100
                    logger.info(f"  {signal}: %{signal_success:.2f} ({len(signal_data)} sinyal)")
        
    except Exception as e:
        logger.error(f"Veri analizi hatası: {e}")

def main():
    """Ana fonksiyon"""
    logger.info("=== AI MODEL EĞİTİMİ BAŞLATILIYOR ===")
    
    # Veri kalitesi analizi
    analyze_data_quality()
    
    # Veriyi yükle ve hazırla
    X, y, feature_columns = load_and_prepare_data()
    
    if X is None:
        logger.error("Veri yüklenemedi! Eğitim iptal edildi.")
        return
    
    # Modeli eğit
    model, scaler, accuracy = train_model(X, y, feature_columns)
    
    if model is None:
        logger.error("Model eğitilemedi!")
        return
    
    # Modeli kaydet
    if save_model(model, scaler, accuracy):
        logger.info("=== AI MODEL EĞİTİMİ BAŞARIYLA TAMAMLANDI ===")
    else:
        logger.error("Model kaydedilemedi!")

if __name__ == "__main__":
    main() 