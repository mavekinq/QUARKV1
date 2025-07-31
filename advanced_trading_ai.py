#!/usr/bin/env python3
"""
Gelişmiş Trading AI Sistemi
Gerçek zamanlı al-sat tahminleri ve risk yönetimi
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
import logging
from datetime import datetime, timedelta
import json
import requests
from typing import Dict, List, Tuple, Optional
import talib

logger = logging.getLogger(__name__)

class AdvancedTradingAI:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.performance_history = []
        self.risk_config = {
            'max_position_size': 0.1,  # Portföyün %10'u
            'stop_loss': 0.05,         # %5 stop loss
            'take_profit': 0.15,       # %15 take profit
            'max_daily_trades': 10,    # Günlük maksimum işlem
            'min_confidence': 0.7      # Minimum güven skoru
        }
        
    def get_real_time_data(self, symbol: str, interval: str = '15m', limit: int = 100) -> pd.DataFrame:
        """Binance'den gerçek zamanlı veri al"""
        try:
            url = f"https://api.binance.com/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Veri tiplerini dönüştür
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Veri alma hatası: {e}")
            return pd.DataFrame()
    
    def calculate_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gelişmiş teknik indikatörler hesapla"""
        try:
            # Temel indikatörler
            df['rsi'] = talib.RSI(df['close'], timeperiod=14)
            df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'])
            df['stoch_k'], df['stoch_d'] = talib.STOCH(df['high'], df['low'], df['close'])
            df['williams_r'] = talib.WILLR(df['high'], df['low'], df['close'])
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'])
            
            # Gelişmiş indikatörler
            df['cci'] = talib.CCI(df['high'], df['low'], df['close'])
            df['obv'] = talib.OBV(df['close'], df['volume'])
            df['adx'] = talib.ADX(df['high'], df['low'], df['close'])
            df['sar'] = talib.SAR(df['high'], df['low'])
            
            # Momentum indikatörleri
            df['mfi'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'])
            df['roc'] = talib.ROC(df['close'])
            df['mom'] = talib.MOM(df['close'])
            
            # Volatilite indikatörleri
            df['natr'] = talib.NATR(df['high'], df['low'], df['close'])
            df['trange'] = talib.TRANGE(df['high'], df['low'], df['close'])
            
            # Hacim indikatörleri
            df['ad'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
            df['adosc'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'])
            
            return df
            
        except Exception as e:
            logger.error(f"İndikatör hesaplama hatası: {e}")
            return df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """AI için özellikler oluştur"""
        try:
            # Fiyat özellikleri
            df['price_change'] = df['close'].pct_change()
            df['price_change_2'] = df['close'].pct_change(2)
            df['price_change_5'] = df['close'].pct_change(5)
            
            # Hacim özellikleri
            df['volume_change'] = df['volume'].pct_change()
            df['volume_ma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            
            # Volatilite özellikleri
            df['volatility'] = df['close'].rolling(20).std()
            df['volatility_ratio'] = df['volatility'] / df['close']
            
            # Momentum özellikleri
            df['momentum_1'] = df['close'] - df['close'].shift(1)
            df['momentum_5'] = df['close'] - df['close'].shift(5)
            df['momentum_10'] = df['close'] - df['close'].shift(10)
            
            # Trend özellikleri
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            df['trend_20'] = (df['close'] - df['sma_20']) / df['sma_20']
            df['trend_50'] = (df['close'] - df['sma_50']) / df['sma_50']
            
            # İndikatör farkları
            df['rsi_diff'] = df['rsi'] - df['rsi'].shift(1)
            df['macd_diff'] = df['macd'] - df['macd_signal']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            return df
            
        except Exception as e:
            logger.error(f"Özellik oluşturma hatası: {e}")
            return df
    
    def generate_labels(self, df: pd.DataFrame, future_periods: int = 3) -> pd.Series:
        """Gelecek fiyat hareketlerine göre etiketler oluştur"""
        try:
            # Gelecek fiyat değişimi
            future_returns = df['close'].shift(-future_periods) / df['close'] - 1
            
            # Etiketler: 1 (AL), 0 (SAT), -1 (BEKLE)
            labels = pd.Series(index=df.index, dtype=int)
            
            # %2'den fazla artış = AL
            labels[future_returns > 0.02] = 1
            # %2'den fazla düşüş = SAT
            labels[future_returns < -0.02] = 0
            # Diğerleri = BEKLE
            labels[(future_returns >= -0.02) & (future_returns <= 0.02)] = -1
            
            return labels
            
        except Exception as e:
            logger.error(f"Etiket oluşturma hatası: {e}")
            return pd.Series()
    
    def train_models(self, symbol: str, interval: str = '15m'):
        """AI modellerini eğit"""
        try:
            logger.info(f"🔄 {symbol} için AI modelleri eğitiliyor...")
            
            # Veri al
            df = self.get_real_time_data(symbol, interval, 500)
            if df.empty:
                logger.error("Veri alınamadı")
                return False
            
            # İndikatörler hesapla
            df = self.calculate_advanced_indicators(df)
            
            # Özellikler oluştur
            df = self.create_features(df)
            
            # Etiketler oluştur
            labels = self.generate_labels(df)
            
            # NaN değerleri temizle
            df = df.dropna()
            labels = labels[df.index]
            
            if len(df) < 100:
                logger.error("Yeterli veri yok")
                return False
            
            # Özellik seçimi
            feature_columns = [
                'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_middle', 'bb_lower',
                'stoch_k', 'stoch_d', 'williams_r', 'atr', 'cci', 'obv', 'adx',
                'mfi', 'roc', 'mom', 'natr', 'price_change', 'price_change_2',
                'price_change_5', 'volume_change', 'volume_ratio', 'volatility_ratio',
                'momentum_1', 'momentum_5', 'momentum_10', 'trend_20', 'trend_50',
                'rsi_diff', 'macd_diff', 'bb_position'
            ]
            
            X = df[feature_columns].values
            y = labels.values
            
            # Veriyi eğitim ve test olarak böl
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Ölçeklendirme
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Modeller
            models = {
                'random_forest': RandomForestClassifier(n_estimators=200, random_state=42),
                'gradient_boosting': GradientBoostingClassifier(n_estimators=200, random_state=42),
                'neural_network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
            }
            
            # Eğitim ve değerlendirme
            for name, model in models.items():
                logger.info(f"📊 {name} modeli eğitiliyor...")
                
                # Eğitim
                model.fit(X_train_scaled, y_train)
                
                # Tahmin
                y_pred = model.predict(X_test_scaled)
                
                # Performans metrikleri
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                logger.info(f"✅ {name} - Doğruluk: {accuracy:.3f}, F1: {f1:.3f}")
                
                # Modeli kaydet
                self.models[name] = model
                self.scalers[name] = scaler
                
                # Performans kaydet
                performance = {
                    'model': name,
                    'symbol': symbol,
                    'interval': interval,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'timestamp': datetime.now().isoformat()
                }
                self.performance_history.append(performance)
            
            # Modelleri dosyaya kaydet
            self.save_models(symbol)
            
            logger.info(f"✅ {symbol} için AI modelleri eğitildi!")
            return True
            
        except Exception as e:
            logger.error(f"Model eğitme hatası: {e}")
            return False
    
    def predict_signal(self, symbol: str, interval: str = '15m') -> Dict:
        """Al-sat sinyali tahmin et"""
        try:
            # En son veriyi al
            df = self.get_real_time_data(symbol, interval, 100)
            if df.empty:
                return {'signal': 'HATA', 'confidence': 0, 'reason': 'Veri alınamadı'}
            
            # İndikatörler hesapla
            df = self.calculate_advanced_indicators(df)
            df = self.create_features(df)
            
            # Son veriyi al
            latest_data = df.iloc[-1:]
            
            # Özellikler
            feature_columns = [
                'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_middle', 'bb_lower',
                'stoch_k', 'stoch_d', 'williams_r', 'atr', 'cci', 'obv', 'adx',
                'mfi', 'roc', 'mom', 'natr', 'price_change', 'price_change_2',
                'price_change_5', 'volume_change', 'volume_ratio', 'volatility_ratio',
                'momentum_1', 'momentum_5', 'momentum_10', 'trend_20', 'trend_50',
                'rsi_diff', 'macd_diff', 'bb_position'
            ]
            
            X = latest_data[feature_columns].values
            
            # Ensemble tahmin
            predictions = []
            confidences = []
            
            for name, model in self.models.items():
                if name in self.scalers:
                    scaler = self.scalers[name]
                    X_scaled = scaler.transform(X)
                    
                    # Tahmin
                    pred = model.predict(X_scaled)[0]
                    prob = model.predict_proba(X_scaled)[0]
                    confidence = max(prob)
                    
                    predictions.append(pred)
                    confidences.append(confidence)
            
            if not predictions:
                return {'signal': 'HATA', 'confidence': 0, 'reason': 'Model yüklenemedi'}
            
            # Çoğunluk oyu
            final_prediction = max(set(predictions), key=predictions.count)
            avg_confidence = np.mean(confidences)
            
            # Sinyal dönüştürme
            signal_map = {1: 'AL', 0: 'SAT', -1: 'BEKLE'}
            signal = signal_map.get(final_prediction, 'BEKLE')
            
            # Güven skoru kontrolü
            if avg_confidence < self.risk_config['min_confidence']:
                signal = 'BEKLE'
                reason = f'Düşük güven skoru: {avg_confidence:.2f}'
            else:
                reason = f'Yüksek güven skoru: {avg_confidence:.2f}'
            
            return {
                'signal': signal,
                'confidence': avg_confidence,
                'reason': reason,
                'predictions': predictions,
                'confidences': confidences
            }
            
        except Exception as e:
            logger.error(f"Tahmin hatası: {e}")
            return {'signal': 'HATA', 'confidence': 0, 'reason': str(e)}
    
    def save_models(self, symbol: str):
        """Modelleri kaydet"""
        try:
            model_dir = f'models/{symbol}'
            os.makedirs(model_dir, exist_ok=True)
            
            for name, model in self.models.items():
                model_path = f'{model_dir}/{name}_model.pkl'
                scaler_path = f'{model_dir}/{name}_scaler.pkl'
                
                joblib.dump(model, model_path)
                joblib.dump(self.scalers[name], scaler_path)
            
            # Performans geçmişini kaydet
            with open(f'{model_dir}/performance.json', 'w') as f:
                json.dump(self.performance_history, f, indent=2)
                
            logger.info(f"✅ Modeller kaydedildi: {model_dir}")
            
        except Exception as e:
            logger.error(f"Model kaydetme hatası: {e}")
    
    def load_models(self, symbol: str) -> bool:
        """Modelleri yükle"""
        try:
            model_dir = f'models/{symbol}'
            
            if not os.path.exists(model_dir):
                return False
            
            model_names = ['random_forest', 'gradient_boosting', 'neural_network']
            
            for name in model_names:
                model_path = f'{model_dir}/{name}_model.pkl'
                scaler_path = f'{model_dir}/{name}_scaler.pkl'
                
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    self.models[name] = joblib.load(model_path)
                    self.scalers[name] = joblib.load(scaler_path)
            
            # Performans geçmişini yükle
            perf_path = f'{model_dir}/performance.json'
            if os.path.exists(perf_path):
                with open(perf_path, 'r') as f:
                    self.performance_history = json.load(f)
            
            logger.info(f"✅ Modeller yüklendi: {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Model yükleme hatası: {e}")
            return False

def test_advanced_trading_ai():
    """Test fonksiyonu"""
    ai = AdvancedTradingAI()
    
    # Test sembolü
    symbol = 'BTCUSDT'
    
    # Modelleri eğit
    success = ai.train_models(symbol)
    
    if success:
        # Tahmin yap
        prediction = ai.predict_signal(symbol)
        print(f"🎯 Tahmin: {prediction}")
    else:
        print("❌ Eğitim başarısız")

if __name__ == "__main__":
    test_advanced_trading_ai() 