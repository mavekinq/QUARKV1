#!/usr/bin/env python3
"""
Gelişmiş AI Modelleri
LSTM, Transformer ve Ensemble modelleri için
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
import pickle
import os
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AdvancedAIModels:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = [
            'rsi', 'bb_lower', 'bb_middle', 'bb_upper', 
            'macd', 'macd_signal', 'stoch_k', 'stoch_d', 
            'williams_r', 'atr'
        ]
        
    def prepare_lstm_data(self, df, sequence_length=10):
        """LSTM için veri hazırlama"""
        try:
            # Özellikleri normalize et
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df[self.feature_columns])
            
            X, y = [], []
            for i in range(sequence_length, len(scaled_data)):
                X.append(scaled_data[i-sequence_length:i])
                y.append(df['signal_numeric'].iloc[i])
            
            return np.array(X), np.array(y), scaler
            
        except Exception as e:
            logger.error(f"LSTM veri hazırlama hatası: {e}")
            return None, None, None
    
    def create_lstm_model(self, sequence_length=10, features=10):
        """LSTM modeli oluştur"""
        try:
            model = Sequential([
                Bidirectional(LSTM(50, return_sequences=True), input_shape=(sequence_length, features)),
                Dropout(0.2),
                Bidirectional(LSTM(50, return_sequences=False)),
                Dropout(0.2),
                Dense(25, activation='relu'),
                Dropout(0.2),
                Dense(3, activation='softmax')  # 3 sınıf: BEKLE, AL, SAT
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            return model
            
        except Exception as e:
            logger.error(f"LSTM model oluşturma hatası: {e}")
            return None
    
    def create_ensemble_model(self, X_train, y_train):
        """Ensemble model oluştur"""
        try:
            # Farklı modeller
            rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
            
            # Voting classifier
            ensemble = VotingClassifier(
                estimators=[('rf', rf), ('gb', gb)],
                voting='soft'
            )
            
            ensemble.fit(X_train, y_train)
            return ensemble
            
        except Exception as e:
            logger.error(f"Ensemble model oluşturma hatası: {e}")
            return None
    
    def train_advanced_models(self, signals_file='signals.csv'):
        """Gelişmiş modelleri eğit"""
        try:
            if not os.path.exists(signals_file):
                logger.error("Sinyal dosyası bulunamadı!")
                return False
            
            # Veriyi yükle
            df = pd.read_csv(signals_file)
            completed_signals = df[df['status'] != 'Beklemede'].copy()
            
            if len(completed_signals) < 100:
                logger.warning("LSTM için en az 100 sinyal gerekli!")
                return False
            
            # Sinyal sınıflarını sayısal değerlere çevir
            signal_mapping = {
                'BEKLE': 0,
                'AL': 1, 'AL (Zayıf)': 1, 'AL (Güçlü)': 1,
                'SAT': 2, 'SAT (Zayıf)': 2, 'SAT (Güçlü)': 2
            }
            
            completed_signals['signal_numeric'] = completed_signals['signal'].map(signal_mapping)
            completed_signals = completed_signals.dropna(subset=self.feature_columns + ['signal_numeric'])
            
            if len(completed_signals) < 100:
                logger.warning("Yeterli veri yok!")
                return False
            
            # Veriyi böl
            train_size = int(0.8 * len(completed_signals))
            train_data = completed_signals[:train_size]
            test_data = completed_signals[train_size:]
            
            logger.info(f"Eğitim verisi: {len(train_data)}, Test verisi: {len(test_data)}")
            
            # 1. Ensemble Model (Klasik ML)
            X_train = train_data[self.feature_columns].values
            y_train = train_data['signal_numeric'].values
            X_test = test_data[self.feature_columns].values
            y_test = test_data['signal_numeric'].values
            
            # Ölçeklendirme
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Ensemble model eğit
            ensemble_model = self.create_ensemble_model(X_train_scaled, y_train)
            if ensemble_model:
                ensemble_pred = ensemble_model.predict(X_test_scaled)
                ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
                logger.info(f"Ensemble Model Accuracy: {ensemble_accuracy:.4f}")
                
                self.models['ensemble'] = ensemble_model
                self.scalers['ensemble'] = scaler
            
            # 2. LSTM Model (Deep Learning)
            X_lstm_train, y_lstm_train, lstm_scaler = self.prepare_lstm_data(train_data)
            X_lstm_test, y_lstm_test, _ = self.prepare_lstm_data(test_data)
            
            if X_lstm_train is not None and len(X_lstm_train) > 0:
                lstm_model = self.create_lstm_model()
                if lstm_model:
                    # LSTM eğitimi
                    history = lstm_model.fit(
                        X_lstm_train, y_lstm_train,
                        epochs=50,
                        batch_size=32,
                        validation_split=0.2,
                        verbose=0
                    )
                    
                    # LSTM test
                    lstm_pred = np.argmax(lstm_model.predict(X_lstm_test), axis=1)
                    lstm_accuracy = accuracy_score(y_lstm_test, lstm_pred)
                    logger.info(f"LSTM Model Accuracy: {lstm_accuracy:.4f}")
                    
                    self.models['lstm'] = lstm_model
                    self.scalers['lstm'] = lstm_scaler
            
            # Modelleri kaydet
            self.save_models()
            
            return True
            
        except Exception as e:
            logger.error(f"Gelişmiş model eğitimi hatası: {e}")
            return False
    
    def predict_with_advanced_models(self, features, model_type='ensemble'):
        """Gelişmiş modellerle tahmin"""
        try:
            if model_type not in self.models:
                return None, "Model bulunamadı"
            
            model = self.models[model_type]
            scaler = self.scalers[model_type]
            
            if model_type == 'ensemble':
                # Ensemble model tahmini
                feature_values = []
                for feature in self.feature_columns:
                    feature_values.append(features.get(feature, 0))
                
                X_scaled = scaler.transform([feature_values])
                prediction = model.predict(X_scaled)[0]
                probabilities = model.predict_proba(X_scaled)[0]
                
            elif model_type == 'lstm':
                # LSTM model tahmini (sequence gerekli)
                # Bu kısım daha karmaşık, şimdilik ensemble kullan
                return self.predict_with_advanced_models(features, 'ensemble')
            
            # Sinyal dönüşümü
            signal_map = {0: 'BEKLE', 1: 'AL', 2: 'SAT'}
            signal = signal_map.get(prediction, 'BEKLE')
            
            # Güven skoru
            confidence = max(probabilities) * 100
            
            return signal, f"Advanced {model_type.upper()} (Güven: %{confidence:.1f})"
            
        except Exception as e:
            logger.error(f"Gelişmiş tahmin hatası: {e}")
            return None, f"Tahmin hatası: {e}"
    
    def save_models(self):
        """Modelleri kaydet"""
        try:
            # Klasik ML modelleri
            for name, model in self.models.items():
                if name == 'ensemble':
                    with open(f'advanced_{name}_model.pkl', 'wb') as f:
                        pickle.dump(model, f)
            
            # Scaler'ları kaydet
            for name, scaler in self.scalers.items():
                with open(f'advanced_{name}_scaler.pkl', 'wb') as f:
                    pickle.dump(scaler, f)
            
            # LSTM modeli (TensorFlow formatında)
            if 'lstm' in self.models:
                self.models['lstm'].save('advanced_lstm_model.h5')
            
            logger.info("Gelişmiş modeller kaydedildi!")
            
        except Exception as e:
            logger.error(f"Model kaydetme hatası: {e}")
    
    def load_models(self):
        """Modelleri yükle"""
        try:
            # Ensemble model
            if os.path.exists('advanced_ensemble_model.pkl'):
                with open('advanced_ensemble_model.pkl', 'rb') as f:
                    self.models['ensemble'] = pickle.load(f)
            
            # Scaler'lar
            if os.path.exists('advanced_ensemble_scaler.pkl'):
                with open('advanced_ensemble_scaler.pkl', 'rb') as f:
                    self.scalers['ensemble'] = pickle.load(f)
            
            # LSTM model
            if os.path.exists('advanced_lstm_model.h5'):
                self.models['lstm'] = tf.keras.models.load_model('advanced_lstm_model.h5')
            
            if os.path.exists('advanced_lstm_scaler.pkl'):
                with open('advanced_lstm_scaler.pkl', 'rb') as f:
                    self.scalers['lstm'] = pickle.load(f)
            
            logger.info("Gelişmiş modeller yüklendi!")
            return True
            
        except Exception as e:
            logger.error(f"Model yükleme hatası: {e}")
            return False

# Test fonksiyonu
def test_advanced_models():
    """Gelişmiş modelleri test et"""
    ai_models = AdvancedAIModels()
    
    # Modelleri eğit
    success = ai_models.train_advanced_models()
    
    if success:
        # Test tahmini
        test_features = {
            'rsi': 45.5,
            'bb_lower': 42000,
            'bb_middle': 45000,
            'bb_upper': 48000,
            'macd': 0.5,
            'macd_signal': 0.3,
            'stoch_k': 35.2,
            'stoch_d': 40.1,
            'williams_r': -65.3,
            'atr': 1200.5
        }
        
        signal, message = ai_models.predict_with_advanced_models(test_features, 'ensemble')
        print(f"Test Tahmini: {signal} - {message}")

if __name__ == "__main__":
    test_advanced_models() 