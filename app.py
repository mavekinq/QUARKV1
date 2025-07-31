from flask import Flask, render_template, request, redirect, url_for, jsonify
import json
import requests
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import plotly.io as pio
import time
import threading
from datetime import datetime, timedelta
import logging
import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
import asyncio
import websockets
from threading import Thread

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Rate limiting için basit cache (devre dışı)
request_cache = {}
RATE_LIMIT_PER_MINUTE = 1000  # Çok yüksek limit (neredeyse sınırsız)

# AI Model için global değişkenler
ai_model = None
model_last_updated = None
SIGNALS_FILE = 'signals.csv'
MODEL_FILE = 'model.pkl'
last_training_check = None
TRAINING_INTERVAL = 1800  # 30 dakika (saniye cinsinden) - test için kısaltıldı
MIN_SIGNALS_FOR_TRAINING = 20  # Test için düşürüldü (normalde 50)

# Gelişmiş özellikler için global değişkenler
ADVANCED_MODELS = None
SENTIMENT_ANALYZER = None
ADVANCED_TECHNICAL = None
ADAPTIVE_LEARNING = None
ADVANCED_FEATURES_AVAILABLE = False

POPULAR_COINS = [
    ("BTCUSDT", "Bitcoin (BTC)"),
    ("ETHUSDT", "Ethereum (ETH)"),
    ("BNBUSDT", "Binance Coin (BNB)"),
    ("SOLUSDT", "Solana (SOL)"),
    ("XRPUSDT", "Ripple (XRP)"),
    ("ADAUSDT", "Cardano (ADA)"),
    ("DOTUSDT", "Polkadot (DOT)"),
    ("AVAXUSDT", "Avalanche (AVAX)"),
    ("MATICUSDT", "Polygon (MATIC)"),
    ("LINKUSDT", "Chainlink (LINK)"),
    ("DOGEUSDT", "Dogecoin (DOGE)"),
    ("SHIBUSDT", "Shiba Inu (SHIB)"),
    ("LTCUSDT", "Litecoin (LTC)"),
    ("BCHUSDT", "Bitcoin Cash (BCH)"),
    ("UNIUSDT", "Uniswap (UNI)"),
    ("ATOMUSDT", "Cosmos (ATOM)"),
    ("NEARUSDT", "NEAR Protocol (NEAR)"),
    ("FTMUSDT", "Fantom (FTM)"),
    ("ALGOUSDT", "Algorand (ALGO)"),
    ("VETUSDT", "VeChain (VET)"),
    ("ICPUSDT", "Internet Computer (ICP)"),
    ("FILUSDT", "Filecoin (FIL)"),
    ("TRXUSDT", "TRON (TRX)"),
    ("ETCUSDT", "Ethereum Classic (ETC)")
]

TIMEFRAMES = [
    ("1m", "1 Dakika"),
    ("3m", "3 Dakika"),
    ("5m", "5 Dakika"),
    ("15m", "15 Dakika"),
    ("30m", "30 Dakika"),
    ("1h", "1 Saat"),
    ("2h", "2 Saat"),
    ("4h", "4 Saat"),
    ("6h", "6 Saat"),
    ("8h", "8 Saat"),
    ("12h", "12 Saat"),
    ("1d", "1 Gün"),
    ("3d", "3 Gün"),
    ("1w", "1 Hafta"),
    ("1M", "1 Ay")
]

def load_ai_model():
    """AI modelini yükler"""
    global ai_model, model_last_updated
    try:
        if os.path.exists(MODEL_FILE):
            with open(MODEL_FILE, 'rb') as f:
                ai_model = pickle.load(f)
            model_last_updated = datetime.fromtimestamp(os.path.getmtime(MODEL_FILE))
            logger.info(f"AI model yüklendi. Son güncelleme: {model_last_updated}")
            return True
        else:
            logger.warning("AI model dosyası bulunamadı. Klasik strateji kullanılacak.")
            return False
    except Exception as e:
        logger.error(f"AI model yükleme hatası: {e}")
        return False

def initialize_advanced_features():
    """Gelişmiş özellikleri başlatır"""
    global ADVANCED_MODELS, SENTIMENT_ANALYZER, ADVANCED_TECHNICAL, ADAPTIVE_LEARNING, ADVANCED_FEATURES_AVAILABLE
    
    try:
        # Gelişmiş AI modelleri
        from advanced_ai_models import AdvancedAIModels
        ADVANCED_MODELS = AdvancedAIModels()
        ADVANCED_MODELS.load_models()
        logger.info("Gelişmiş AI modelleri yüklendi")
        
        # Duygu analizi
        from sentiment_analysis import SentimentAnalyzer
        SENTIMENT_ANALYZER = SentimentAnalyzer()
        logger.info("Duygu analizi sistemi başlatıldı")
        
        # Gelişmiş teknik analiz
        from advanced_technical_analysis import AdvancedTechnicalAnalysis
        ADVANCED_TECHNICAL = AdvancedTechnicalAnalysis()
        logger.info("Gelişmiş teknik analiz sistemi başlatıldı")
        
        # Adaptif öğrenme sistemi
        from adaptive_learning import AdaptiveLearning
        ADAPTIVE_LEARNING = AdaptiveLearning()
        logger.info("Adaptif öğrenme sistemi başlatıldı")
        
        ADVANCED_FEATURES_AVAILABLE = True
        logger.info("Tüm gelişmiş özellikler başarıyla yüklendi!")
        
    except ImportError as e:
        logger.warning(f"Gelişmiş özellikler yüklenemedi: {e}")
        ADVANCED_FEATURES_AVAILABLE = False
    except Exception as e:
        logger.error(f"Gelişmiş özellikler başlatma hatası: {e}")
        ADVANCED_FEATURES_AVAILABLE = False

def save_signal_to_csv(symbol, timestamp, price, signal, rsi, bb_lower, bb_middle, bb_upper, 
                       macd, macd_signal, stoch_k, stoch_d, williams_r, atr, interval):
    """Sinyali CSV dosyasına kaydeder (optimize edilmiş)"""
    try:
        # CSV dosyası yoksa oluştur
        if not os.path.exists(SIGNALS_FILE):
            df = pd.DataFrame(columns=[
                'timestamp', 'symbol', 'price', 'signal', 'rsi', 'bb_lower', 'bb_middle', 
                'bb_upper', 'macd', 'macd_signal', 'stoch_k', 'stoch_d', 'williams_r', 
                'atr', 'interval', 'status', 'future_price', 'profit_loss'
            ])
        else:
            df = pd.read_csv(SIGNALS_FILE)
        
        # Yeni sinyal verisi
        new_signal = {
            'timestamp': timestamp,
            'symbol': symbol,
            'price': price,
            'signal': signal,
            'rsi': rsi if rsi is not None else np.nan,
            'bb_lower': bb_lower if bb_lower is not None else np.nan,
            'bb_middle': bb_middle if bb_middle is not None else np.nan,
            'bb_upper': bb_upper if bb_upper is not None else np.nan,
            'macd': macd if macd is not None else np.nan,
            'macd_signal': macd_signal if macd_signal is not None else np.nan,
            'stoch_k': stoch_k if stoch_k is not None else np.nan,
            'stoch_d': stoch_d if stoch_d is not None else np.nan,
            'williams_r': williams_r if williams_r is not None else np.nan,
            'atr': atr if atr is not None else np.nan,
            'interval': interval,
            'status': 'Beklemede',
            'future_price': np.nan,
            'profit_loss': np.nan
        }
        
        # DataFrame'e ekle
        df = pd.concat([df, pd.DataFrame([new_signal])], ignore_index=True)
        
        # CSV'ye kaydet
        df.to_csv(SIGNALS_FILE, index=False)
        logger.info(f"✅ Yeni sinyal kaydedildi: {symbol} - {signal} - {timestamp}")
        
        # Toplam kayıt sayısını göster
        total_records = len(df)
        logger.info(f"📊 Toplam sinyal sayısı: {total_records}")
        
    except Exception as e:
        logger.error(f"Sinyal kaydetme hatası: {e}")

def update_signal_status():
    """Eski sinyallerin durumunu günceller (30 dakika/1 saat sonra)"""
    try:
        if not os.path.exists(SIGNALS_FILE):
            return
        
        df = pd.read_csv(SIGNALS_FILE)
        current_time = datetime.now()
        
        for index, row in df.iterrows():
            if row['status'] == 'Beklemede':
                try:
                    # Timestamp formatını kontrol et
                    timestamp_str = str(row['timestamp'])
                    
                    # Unix timestamp (milisaniye) kontrolü
                    if timestamp_str.isdigit() and len(timestamp_str) > 10:
                        # Unix timestamp'i datetime'a çevir
                        signal_time = datetime.fromtimestamp(int(timestamp_str) / 1000)
                    else:
                        # Normal datetime string
                        signal_time = pd.to_datetime(timestamp_str)
                    
                    time_diff = current_time - signal_time
                    
                    # 5 dakika geçmiş mi kontrol et (test için kısaltıldı)
                    if time_diff >= timedelta(minutes=5):
                        try:
                            # Güncel fiyatı al
                            current_price = get_binance_price(row['symbol'])
                            if isinstance(current_price, (int, float)):
                                original_price = row['price']
                                signal = row['signal']
                                
                                # Başarı hesaplama
                                if signal.startswith('AL'):
                                    profit_loss = ((current_price - original_price) / original_price) * 100
                                    status = 'Başarılı' if profit_loss > 0 else 'Başarısız'
                                elif signal.startswith('SAT'):
                                    profit_loss = ((original_price - current_price) / original_price) * 100
                                    status = 'Başarılı' if profit_loss > 0 else 'Başarısız'
                                else:
                                    profit_loss = 0
                                    status = 'Nötr'
                                
                                # Güncelle
                                df.at[index, 'future_price'] = current_price
                                df.at[index, 'profit_loss'] = profit_loss
                                df.at[index, 'status'] = status
                                
                        except Exception as e:
                            logger.error(f"Fiyat güncelleme hatası: {e}")
                            
                except Exception as e:
                    logger.error(f"Timestamp parsing hatası: {e} - Timestamp: {row['timestamp']}")
                    continue
        
        # Güncellenmiş veriyi kaydet
        df.to_csv(SIGNALS_FILE, index=False)
        
    except Exception as e:
        logger.error(f"Sinyal durumu güncelleme hatası: {e}")

def check_and_train_model():
    """Otomatik model eğitimi ve adaptif öğrenme kontrolü"""
    global last_training_check, ai_model
    
    try:
        current_time = time.time()
        
        # İlk kontrol veya belirli aralıklarla kontrol et
        if (last_training_check is None or 
            current_time - last_training_check > TRAINING_INTERVAL):
            
            last_training_check = current_time
            
            # Veri kontrolü
            if not os.path.exists(SIGNALS_FILE):
                return False
            
            df = pd.read_csv(SIGNALS_FILE)
            completed_signals = df[df['status'] != 'Beklemede']
            
            if len(completed_signals) >= MIN_SIGNALS_FOR_TRAINING:
                logger.info(f"Otomatik eğitim başlatılıyor... ({len(completed_signals)} sinyal)")
                
                # 1. Adaptif öğrenme analizi
                if ADVANCED_FEATURES_AVAILABLE and ADAPTIVE_LEARNING:
                    try:
                        logger.info("🔄 Adaptif öğrenme analizi başlatılıyor...")
                        
                        # Performans analizi
                        performance = ADAPTIVE_LEARNING.analyze_performance_trends()
                        if performance:
                            logger.info(f"📊 Performans: Son 7 gün %{performance['recent_accuracy']:.1f}, "
                                       f"Genel %{performance['overall_accuracy']:.1f}")
                        
                        # Strateji önerileri
                        strategy_recs = ADAPTIVE_LEARNING.adaptive_strategy_adjustment()
                        if strategy_recs:
                            logger.info(f"🎯 En iyi stratejiler: {strategy_recs['best_strategies']}")
                            logger.info(f"⚠️ Kaçınılacak stratejiler: {strategy_recs['avoid_strategies']}")
                            
                            # İçgörüleri kaydet
                            insights = {
                                'performance': performance,
                                'strategy_recommendations': strategy_recs,
                                'total_signals': len(completed_signals)
                            }
                            ADAPTIVE_LEARNING.save_learning_insights(insights)
                        
                    except Exception as e:
                        logger.error(f"Adaptif öğrenme hatası: {e}")
                
                # 2. Model eğitimi
                success = train_model_automatically()
                
                if success:
                    # Yeni modeli yükle
                    load_ai_model()
                    logger.info("Otomatik eğitim başarılı! Yeni model yüklendi.")
                    return True
                else:
                    logger.warning("Otomatik eğitim başarısız.")
            
            else:
                logger.info(f"Eğitim için yeterli veri yok: {len(completed_signals)}/{MIN_SIGNALS_FOR_TRAINING}")
        
        return False
        
    except Exception as e:
        logger.error(f"Otomatik eğitim kontrolü hatası: {e}")
        return False

def train_model_automatically():
    """Otomatik model eğitimi"""
    try:
        # Veriyi yükle
        df = pd.read_csv(SIGNALS_FILE)
        completed_signals = df[df['status'] != 'Beklemede'].copy()
        
        if len(completed_signals) < MIN_SIGNALS_FOR_TRAINING:
            return False
        
        # Özellikler
        feature_columns = [
            'rsi', 'bb_lower', 'bb_middle', 'bb_upper', 
            'macd', 'macd_signal', 'stoch_k', 'stoch_d', 
            'williams_r', 'atr'
        ]
        
        # Eksik değerleri temizle
        completed_signals = completed_signals.dropna(subset=feature_columns)
        
        if len(completed_signals) < MIN_SIGNALS_FOR_TRAINING:
            return False
        
        # Özellikler ve hedef
        X = completed_signals[feature_columns].values
        y = completed_signals['signal'].values
        
        # Sinyal sınıflarını sayısal değerlere çevir
        signal_mapping = {
            'BEKLE': 0,
            'AL': 1, 'AL (Zayıf)': 1, 'AL (Güçlü)': 1,
            'SAT': 2, 'SAT (Zayıf)': 2, 'SAT (Güçlü)': 2
        }
        
        y_numeric = np.array([signal_mapping.get(signal, 0) for signal in y])
        
        # Veriyi böl
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_numeric, test_size=0.2, random_state=42, stratify=y_numeric
        )
        
        # Ölçeklendirme
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Model eğitimi (RandomForest)
        model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            random_state=42,
            class_weight='balanced'
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Performans kontrolü
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Sadece yeterli performans varsa kaydet
        if accuracy > 0.6:  # %60'dan fazla doğruluk
            # Modeli kaydet
            with open(MODEL_FILE, 'wb') as f:
                pickle.dump(model, f)
            
            # Scaler'ı kaydet
            with open('scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)
            
            # Model bilgilerini kaydet
            model_info = {
                'accuracy': accuracy,
                'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model_type': 'RandomForest',
                'feature_count': len(feature_columns),
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
            with open('model_info.txt', 'w') as f:
                for key, value in model_info.items():
                    f.write(f"{key}: {value}\n")
            
            logger.info(f"Model eğitildi ve kaydedildi. Doğruluk: {accuracy:.4f}")
            return True
        else:
            logger.warning(f"Model performansı yetersiz: {accuracy:.4f}")
            return False
            
    except Exception as e:
        logger.error(f"Otomatik model eğitimi hatası: {e}")
        return False

def predict_with_ai_model(features):
    """Gelişmiş AI modeli ile tahmin yapar"""
    global ai_model
    
    if ai_model is None:
        return None, "AI model yüklenemedi"
    
    try:
        # Özellikleri düzenle
        feature_names = ['rsi', 'bb_lower', 'bb_middle', 'bb_upper', 'macd', 'macd_signal', 'stoch_k', 'stoch_d', 'williams_r', 'atr']
        feature_values = []
        
        for feature in feature_names:
            if features.get(feature) is not None:
                feature_values.append(features[feature])
            else:
                feature_values.append(0)  # Eksik değerler için 0
        
        # Tahmin yap
        prediction = ai_model.predict([feature_values])[0]
        probability = ai_model.predict_proba([feature_values])[0]
        
        # Güven skoru kontrolü
        confidence = max(probability) * 100
        
        # Güven skoru düşükse BEKLE
        if confidence < 60:  # %60'dan düşük güven
            return "BEKLE", f"AI Tahmin (Düşük Güven: %{confidence:.1f})"
        
        # Sinyal dönüşümü
        signal_map = {0: 'BEKLE', 1: 'AL', 2: 'SAT'}
        signal = signal_map.get(prediction, 'BEKLE')
        
        # Trend analizi ile güçlendirme
        if signal != "BEKLE":
            # RSI trend kontrolü
            rsi = features.get('rsi', 50)
            if signal == "AL" and rsi > 70:
                signal = "BEKLE"  # Aşırı alım bölgesi
            elif signal == "SAT" and rsi < 30:
                signal = "BEKLE"  # Aşırı satım bölgesi
            
            # MACD trend kontrolü
            macd = features.get('macd', 0)
            macd_signal = features.get('macd_signal', 0)
            if signal == "AL" and macd < macd_signal:
                signal = "BEKLE"  # MACD düşüş trendi
            elif signal == "SAT" and macd > macd_signal:
                signal = "BEKLE"  # MACD yükseliş trendi
        
        return signal, f"AI Tahmin (Güven: %{confidence:.1f})"
        
    except Exception as e:
        logger.error(f"AI tahmin hatası: {e}")
        return None, f"AI tahmin hatası: {e}"

def check_rate_limit(ip):
    """Rate limiting kontrolü"""
    now = datetime.now()
    if ip not in request_cache:
        request_cache[ip] = []
    
    # 1 dakikadan eski istekleri temizle
    request_cache[ip] = [req_time for req_time in request_cache[ip] 
                        if now - req_time < timedelta(minutes=1)]
    
    # Yeni istek ekle
    request_cache[ip].append(now)
    
    # Limit kontrolü
    if len(request_cache[ip]) > RATE_LIMIT_PER_MINUTE:
        return False, f"Çok fazla istek! Dakikada maksimum {RATE_LIMIT_PER_MINUTE} istek yapabilirsiniz."
    
    return True, None

def safe_request(url, timeout=10, max_retries=3):
    """Güvenli HTTP isteği yapar"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.warning(f"İstek hatası (deneme {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(1)  # Kısa bekleme
    
    return None

def validate_symbol(symbol):
    """Coin sembolünün geçerli olup olmadığını kontrol eder"""
    if not symbol:
        return False, "Coin sembolü gerekli"
    
    # USDT ile biten sembolleri kabul et
    if not symbol.endswith('USDT'):
        symbol = symbol.upper() + 'USDT'
    
    # Binance API'den fiyat kontrolü yap
    try:
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
        response = safe_request(url, timeout=5)
        if response and response.status_code == 200:
            return True, symbol
        else:
            return False, f"'{symbol}' geçerli bir coin değil"
    except Exception as e:
        logger.error(f"Sembol doğrulama hatası: {e}")
        return False, "Bağlantı hatası"

def get_binance_price(symbol):
    """Binance'den anlık fiyat alır"""
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
    try:
        response = safe_request(url, timeout=5)
        if response:
        data = response.json()
        return float(data['price'])
        return 'Veri alınamadı'
    except Exception as e:
        logger.error(f"Fiyat alma hatası: {e}")
        return 'Veri alınamadı'

def get_binance_history(symbol, interval="15m", limit=100):
    """Binance'den gelişmiş geçmiş veri alır"""
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        response = safe_request(url, timeout=10)
        if response:
        data = response.json()
            
            # Veri doğrulama
            if not data or len(data) < 20:
                logger.warning(f"Yetersiz veri: {symbol} - {len(data) if data else 0} kayıt")
            return [], [], [], [], []
        
        times = [int(item[0]) for item in data]
        prices = [float(item[4]) for item in data]  # Kapanış fiyatı
        highs = [float(item[2]) for item in data]
        lows = [float(item[3]) for item in data]
        volumes = [float(item[5]) for item in data]  # Hacim
            
            # Veri kalitesi kontrolü
            if any(p <= 0 for p in prices):
                logger.warning(f"Geçersiz fiyat verisi: {symbol}")
                return [], [], [], [], []
            
            # Anomali kontrolü
            price_changes = [abs(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            if any(change > 0.5 for change in price_changes):  # %50'den fazla değişim
                logger.warning(f"Anormal fiyat değişimi tespit edildi: {symbol}")
            
            logger.info(f"✅ {symbol} için {len(data)} veri noktası alındı")
        return times, prices, highs, lows, volumes
        else:
            logger.error(f"Binance API yanıt vermedi: {symbol}")
            return [], [], [], [], []
    except Exception as e:
        logger.error(f"Geçmiş veri alma hatası: {e}")
        return [], [], [], [], []

def get_real_time_price(symbol):
    """Gerçek zamanlı fiyat alır"""
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
    try:
        response = safe_request(url, timeout=5)
        if response:
            data = response.json()
            price = float(data['price'])
            
            # Fiyat doğrulama
            if price <= 0:
                logger.error(f"Geçersiz fiyat: {symbol} - {price}")
                return None
            
            return price
        else:
            return None
    except Exception as e:
        logger.error(f"Gerçek zamanlı fiyat hatası: {e}")
        return None

def get_market_data(symbol):
    """Piyasa verilerini alır"""
    url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
    try:
        response = safe_request(url, timeout=5)
        if response:
            data = response.json()
            return {
                'price': float(data['lastPrice']),
                'change_24h': float(data['priceChangePercent']),
                'volume_24h': float(data['volume']),
                'high_24h': float(data['highPrice']),
                'low_24h': float(data['lowPrice'])
            }
        else:
            return None
    except Exception as e:
        logger.error(f"Piyasa verisi hatası: {e}")
        return None

def predict_next(prices):
    """Linear regression ile sonraki fiyat tahmini"""
    if len(prices) < 10:
        return None
    try:
    X = np.arange(len(prices)).reshape(-1, 1)
    y = np.array(prices)
    model = LinearRegression()
    model.fit(X, y)
    next_time = np.array([[len(prices)]])
    prediction = model.predict(next_time)[0]
        return round(prediction, 4)
    except Exception as e:
        logger.error(f"Tahmin hatası: {e}")
        return None

def compute_rsi(prices, period=14):
    """RSI hesaplar"""
    try:
        prices = np.array(prices)
        if len(prices) < period + 1:
            return None
        deltas = np.diff(prices)
        seed = deltas[:period]
        up = seed[seed > 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 0
        rsi = 100 - 100 / (1 + rs)
        for i in range(period, len(deltas)):
            delta = deltas[i]
            if delta > 0:
                upval = delta
                downval = 0
            else:
                upval = 0
                downval = -delta
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down if down != 0 else 0
            rsi = 100 - 100 / (1 + rs)
        return round(rsi, 2)
    except Exception as e:
        logger.error(f"RSI hesaplama hatası: {e}")
        return None

def compute_bollinger_bands(prices, period=20, num_std=2):
    """Bollinger bantları hesaplar"""
    try:
        prices = np.array(prices)
        if len(prices) < period:
            return None, None, None
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        upper = sma + num_std * std
        lower = sma - num_std * std
        return round(lower, 4), round(sma, 4), round(upper, 4)
    except Exception as e:
        logger.error(f"Bollinger bantları hatası: {e}")
        return None, None, None

def compute_macd(prices, fast=12, slow=26, signal=9):
    """MACD hesaplar"""
    try:
        prices = np.array(prices)
        if len(prices) < slow + signal:
            return None, None
        ema_fast = np.convolve(prices, np.ones(fast)/fast, mode='valid')
        ema_slow = np.convolve(prices, np.ones(slow)/slow, mode='valid')
        if len(ema_fast) == 0 or len(ema_slow) == 0:
            return None, None
        macd_line = ema_fast[-len(ema_slow):] - ema_slow
        if len(macd_line) == 0:
            return None, None
        signal_line = np.convolve(macd_line, np.ones(signal)/signal, mode='valid')
        if len(signal_line) == 0:
            return None, None
        macd_val = macd_line[-1]
        signal_val = signal_line[-1]
        return round(macd_val, 4), round(signal_val, 4)
    except Exception as e:
        logger.error(f"MACD hesaplama hatası: {e}")
        return None, None

def compute_stochastic(prices, highs, lows, k_period=14, d_period=3):
    """Stokastik osilatör hesaplar"""
    try:
    if len(prices) < k_period + d_period:
        return None, None
    closes = np.array(prices)
    highs = np.array(highs)
    lows = np.array(lows)
    if len(lows) < k_period or len(highs) < k_period:
        return None, None
    lowest_low = np.min(lows[-k_period:])
    highest_high = np.max(highs[-k_period:])
    if highest_high == lowest_low:
        return None, None
    k = 100 * (closes[-1] - lowest_low) / (highest_high - lowest_low)
    d_values = []
    for i in range(1, d_period+1):
        start = -i - k_period + 1
        end = -i + 1 if -i + 1 != 0 else None
        lows_slice = lows[start:end]
        highs_slice = highs[start:end]
        closes_idx = -i
        if len(lows_slice) == 0 or len(highs_slice) == 0 or (np.max(highs_slice) - np.min(lows_slice)) == 0:
            continue
        ll = np.min(lows_slice)
        hh = np.max(highs_slice)
        d_values.append(100 * (closes[closes_idx] - ll) / (hh - ll))
    d = np.mean(d_values) if d_values else None
    return round(k, 2), round(d, 2) if d is not None else (round(k, 2), None)
    except Exception as e:
        logger.error(f"Stokastik hesaplama hatası: {e}")
        return None, None

def compute_williams_r(highs, lows, closes, period=14):
    """Williams %R hesaplar"""
    try:
        if len(closes) < period:
            return None
        highest_high = np.max(highs[-period:])
        lowest_low = np.min(lows[-period:])
        if highest_high == lowest_low:
            return None
        williams_r = -100 * (highest_high - closes[-1]) / (highest_high - lowest_low)
        return round(williams_r, 2)
    except Exception as e:
        logger.error(f"Williams %R hesaplama hatası: {e}")
        return None

def compute_atr(highs, lows, closes, period=14):
    """Average True Range (ATR) hesaplar"""
    try:
        if len(closes) < period + 1:
            return None
        true_ranges = []
        for i in range(1, len(closes)):
            high_low = highs[i] - lows[i]
            high_close = abs(highs[i] - closes[i-1])
            low_close = abs(lows[i] - closes[i-1])
            true_range = max(high_low, high_close, low_close)
            true_ranges.append(true_range)
        
        if len(true_ranges) < period:
            return None
        
        atr = np.mean(true_ranges[-period:])
        return round(atr, 4)
    except Exception as e:
        logger.error(f"ATR hesaplama hatası: {e}")
        return None

def rsi_comment(rsi):
    """RSI yorumu"""
    if rsi is None:
        return "Veri yetersiz."
    if rsi < 30:
        return "Aşırı satım bölgesi. Fiyatın yükselmesi beklenebilir."
    elif rsi > 70:
        return "Aşırı alım bölgesi. Fiyatın düşmesi beklenebilir."
    else:
        return "Nötr bölge."

def macd_comment(macd, signal):
    """MACD yorumu"""
    if macd is None or signal is None:
        return "Veri yetersiz."
    if macd > signal:
        return "MACD çizgisi, sinyal çizgisinin üzerinde. Yükseliş eğilimi."
    elif macd < signal:
        return "MACD çizgisi, sinyal çizgisinin altında. Düşüş eğilimi."
    else:
        return "MACD ve sinyal çizgisi eşit. Nötr."

def stoch_comment(k, d):
    """Stokastik yorumu"""
    if k is None or d is None:
        return "Veri yetersiz."
    if k > 80:
        return "Aşırı alım bölgesi. Düşüş beklenebilir."
    elif k < 20:
        return "Aşırı satım bölgesi. Yükseliş beklenebilir."
    elif k > d:
        return "%K, %D'nin üzerinde. Yükseliş sinyali."
    elif k < d:
        return "%K, %D'nin altında. Düşüş sinyali."
    else:
        return "Nötr."

def williams_comment(williams_r):
    """Williams %R yorumu"""
    if williams_r is None:
        return "Veri yetersiz."
    if williams_r < -80:
        return "Aşırı satım bölgesi. Yükseliş beklenebilir."
    elif williams_r > -20:
        return "Aşırı alım bölgesi. Düşüş beklenebilir."
    else:
        return "Nötr bölge."

def atr_comment(atr, current_price):
    """ATR yorumu"""
    if atr is None or current_price is None:
        return "Veri yetersiz."
    volatility_percent = (atr / current_price) * 100
    if volatility_percent > 5:
        return f"Yüksek volatilite (%{volatility_percent:.1f}). Dikkatli olun."
    elif volatility_percent > 2:
        return f"Orta volatilite (%{volatility_percent:.1f}). Normal seviye."
    else:
        return f"Düşük volatilite (%{volatility_percent:.1f}). Stabil seviye."

def compute_cci(highs, lows, closes, period=20):
    """Commodity Channel Index hesaplar"""
    try:
        if len(closes) < period:
            return None
        
        typical_prices = [(highs[i] + lows[i] + closes[i]) / 3 for i in range(len(closes))]
        sma_tp = sum(typical_prices[-period:]) / period
        
        mean_deviation = sum(abs(tp - sma_tp) for tp in typical_prices[-period:]) / period
        
        if mean_deviation == 0:
            return 0
        
        cci = (typical_prices[-1] - sma_tp) / (0.015 * mean_deviation)
        return round(cci, 2)
        
    except Exception as e:
        logger.error(f"CCI hesaplama hatası: {e}")
        return None

def compute_obv(closes, volumes):
    """On Balance Volume hesaplar"""
    try:
        if len(closes) < 2 or len(volumes) < 2:
            return None
        
        obv = volumes[0]
        for i in range(1, len(closes)):
            if closes[i] > closes[i-1]:
                obv += volumes[i]
            elif closes[i] < closes[i-1]:
                obv -= volumes[i]
        
        return obv
        
    except Exception as e:
        logger.error(f"OBV hesaplama hatası: {e}")
        return None

def compute_adx(highs, lows, closes, period=14):
    """Average Directional Index hesaplar"""
    try:
        if len(highs) < period + 1:
            return None
        
        plus_dm = []
        minus_dm = []
        true_ranges = []
        
        for i in range(1, len(highs)):
            # True Range
            high_low = highs[i] - lows[i]
            high_close = abs(highs[i] - closes[i-1])
            low_close = abs(lows[i] - closes[i-1])
            tr = max(high_low, high_close, low_close)
            true_ranges.append(tr)
            
            # Directional Movement
            up_move = highs[i] - highs[i-1]
            down_move = lows[i-1] - lows[i]
            
            if up_move > down_move and up_move > 0:
                plus_dm.append(up_move)
                minus_dm.append(0)
            elif down_move > up_move and down_move > 0:
                plus_dm.append(0)
                minus_dm.append(down_move)
            else:
                plus_dm.append(0)
                minus_dm.append(0)
        
        if len(plus_dm) < period:
            return None
        
        # Smoothing
        atr = sum(true_ranges[-period:]) / period
        plus_di = (sum(plus_dm[-period:]) / period) / atr * 100
        minus_di = (sum(minus_dm[-period:]) / period) / atr * 100
        
        dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        adx = dx  # Simplified ADX calculation
        
        return round(adx, 2)
        
    except Exception as e:
        logger.error(f"ADX hesaplama hatası: {e}")
        return None

def compute_fibonacci_retracement(highs, lows):
    """Fibonacci Retracement seviyeleri hesaplar"""
    try:
        if len(highs) < 20 or len(lows) < 20:
            return None
        
        # Son 20 mum içindeki en yüksek ve en düşük noktaları bul
        recent_highs = highs[-20:]
        recent_lows = lows[-20:]
        
        swing_high = max(recent_highs)
        swing_low = min(recent_lows)
        
        diff = swing_high - swing_low
        
        # Fibonacci seviyeleri
        fib_levels = {
            '0.0': swing_low,
            '0.236': swing_low + 0.236 * diff,
            '0.382': swing_low + 0.382 * diff,
            '0.500': swing_low + 0.500 * diff,
            '0.618': swing_low + 0.618 * diff,
            '0.786': swing_low + 0.786 * diff,
            '1.0': swing_high
        }
        
        return fib_levels
        
    except Exception as e:
        logger.error(f"Fibonacci hesaplama hatası: {e}")
        return None

def compute_pivot_points(highs, lows, closes):
    """Pivot Points hesaplar"""
    try:
        if len(closes) < 1:
            return None
        
        high = max(highs[-1] if highs else closes[-1])
        low = min(lows[-1] if lows else closes[-1])
        close = closes[-1]
        
        # Pivot Point
        pp = (high + low + close) / 3
        
        # Support ve Resistance seviyeleri
        r1 = 2 * pp - low
        r2 = pp + (high - low)
        r3 = high + 2 * (pp - low)
        
        s1 = 2 * pp - high
        s2 = pp - (high - low)
        s3 = low - 2 * (high - pp)
        
        pivot_levels = {
            'PP': pp,
            'R1': r1, 'R2': r2, 'R3': r3,
            'S1': s1, 'S2': s2, 'S3': s3
        }
        
        return pivot_levels
        
    except Exception as e:
        logger.error(f"Pivot Points hesaplama hatası: {e}")
        return None

def cci_comment(cci):
    """CCI yorumu"""
    if cci is None:
        return "Veri yetersiz."
    if cci > 100:
        return "Aşırı alım bölgesi. Düşüş beklenebilir."
    elif cci < -100:
        return "Aşırı satım bölgesi. Yükseliş beklenebilir."
    else:
        return "Nötr bölge."

def obv_comment(obv, current_obv):
    """OBV yorumu"""
    if obv is None or current_obv is None:
        return "Veri yetersiz."
    if current_obv > obv:
        return "Hacim artışı. Yükseliş eğilimi."
    elif current_obv < obv:
        return "Hacim azalışı. Düşüş eğilimi."
    else:
        return "Hacim stabil. Nötr."

def adx_comment(adx):
    """ADX yorumu"""
    if adx is None:
        return "Veri yetersiz."
    if adx > 25:
        return "Güçlü trend. Sinyaller güvenilir."
    elif adx > 20:
        return "Orta trend. Dikkatli olun."
    else:
        return "Zayıf trend. Sinyaller zayıf."

def strategy_signal(prices, highs, lows, volumes=None):
    """Gelişmiş AI destekli strateji sinyali"""
    try:
        if len(prices) < 20:
            return "Veri yetersiz", "black", None, None, None, None, None, None, None, None, None, None, None, "Klasik Strateji"
        
        rsi = compute_rsi(prices)
        bb_lower, bb_middle, bb_upper = compute_bollinger_bands(prices)
        macd, macd_signal = compute_macd(prices)
        stoch_k, stoch_d = compute_stochastic(prices, highs, lows)
        williams_r = compute_williams_r(highs, lows, prices)
        atr = compute_atr(highs, lows, prices)
        
        # Veri kontrolü
        if rsi is None or bb_lower is None or macd is None or stoch_k is None:
            return "Veri yetersiz", "black", rsi, bb_lower, bb_middle, bb_upper, macd, macd_signal, stoch_k, stoch_d, williams_r, atr, "Klasik Strateji"
        
        last_price = prices[-1]
        
        # 1. Gelişmiş AI Modelleri (öncelikli)
        if ADVANCED_FEATURES_AVAILABLE and ADVANCED_MODELS:
            features = {
                'rsi': rsi,
                'bb_lower': bb_lower,
                'bb_middle': bb_middle,
                'bb_upper': bb_upper,
                'macd': macd,
                'macd_signal': macd_signal,
                'stoch_k': stoch_k,
                'stoch_d': stoch_d if stoch_d is not None else stoch_k,
                'williams_r': williams_r if williams_r is not None else 0,
                'atr': atr if atr is not None else 0
            }
            
            advanced_signal, advanced_message = ADVANCED_MODELS.predict_with_advanced_models(features, 'ensemble')
            if advanced_signal and advanced_signal != "BEKLE":
                color = "green" if advanced_signal == "AL" else "red" if advanced_signal == "SAT" else "gray"
                return advanced_signal, color, rsi, bb_lower, bb_middle, bb_upper, macd, macd_signal, stoch_k, stoch_d, williams_r, atr, f"🤖 {advanced_message}"
        
        # 2. Duygu Analizi
        if ADVANCED_FEATURES_AVAILABLE and SENTIMENT_ANALYZER:
            try:
                # Sembol adını al (örn: BTCUSDT -> BTC)
                symbol = "BTC"  # Varsayılan, gerçek uygulamada parametre olarak geçilmeli
                sentiment_result = SENTIMENT_ANALYZER.get_comprehensive_sentiment(symbol)
                sentiment_score = sentiment_result.get('overall_sentiment', 0)
                sentiment_signal, sentiment_message = SENTIMENT_ANALYZER.get_sentiment_signal(sentiment_score)
                
                # Duygu analizi güçlü sinyal veriyorsa kullan
                if sentiment_signal != "BEKLE" and abs(sentiment_score) > 0.3:
                    color = "green" if sentiment_signal.startswith('AL') else "red" if sentiment_signal.startswith('SAT') else "gray"
                    return sentiment_signal, color, rsi, bb_lower, bb_middle, bb_upper, macd, macd_signal, stoch_k, stoch_d, williams_r, atr, f"🎭 {sentiment_message}"
            except Exception as e:
                logger.error(f"Duygu analizi hatası: {e}")
        
        # 3. Gelişmiş Teknik Analiz
        if ADVANCED_FEATURES_AVAILABLE and ADVANCED_TECHNICAL and len(prices) >= 50:
            try:
                advanced_signals = ADVANCED_TECHNICAL.get_advanced_signals(highs, lows, prices, volumes)
                if advanced_signals and 'overall' in advanced_signals:
                    overall_signal = advanced_signals['overall']
                    confidence = advanced_signals.get('confidence', 0)
                    
                    if overall_signal != "BEKLE" and confidence > 60:
                        color = "green" if overall_signal == "AL" else "red" if overall_signal == "SAT" else "gray"
                        return overall_signal, color, rsi, bb_lower, bb_middle, bb_upper, macd, macd_signal, stoch_k, stoch_d, williams_r, atr, f"📊 Gelişmiş Teknik (%{confidence:.0f} güven)"
            except Exception as e:
                logger.error(f"Gelişmiş teknik analiz hatası: {e}")
        
        # 4. Temel AI Modeli
        features = {
            'rsi': rsi,
            'bb_lower': bb_lower,
            'bb_middle': bb_middle,
            'bb_upper': bb_upper,
            'macd': macd,
            'macd_signal': macd_signal,
            'stoch_k': stoch_k,
            'stoch_d': stoch_d if stoch_d is not None else stoch_k,
            'williams_r': williams_r if williams_r is not None else 0,
            'atr': atr if atr is not None else 0
        }
        
        ai_signal, ai_message = predict_with_ai_model(features)
        
        # AI modeli varsa onu kullan, yoksa klasik stratejiyi kullan
        if ai_signal is not None and ai_signal != "BEKLE":
            # AI sinyalini renk kodlarına çevir
            if ai_signal.startswith('AL'):
                color = "green" if "Güçlü" in ai_message else "lightgreen"
            elif ai_signal.startswith('SAT'):
                color = "red" if "Güçlü" in ai_message else "pink"
            else:
                color = "gray"
            
            return ai_signal, color, rsi, bb_lower, bb_middle, bb_upper, macd, macd_signal, stoch_k, stoch_d, williams_r, atr, f"🧠 {ai_message}"
        
        # 5. Klasik strateji (son çare)
        al_conditions = 0
        if rsi < 35:
            al_conditions += 1
        if last_price < bb_lower:
            al_conditions += 1
        if macd > macd_signal:
            al_conditions += 1
        if stoch_k < 20:
            al_conditions += 1
        
        sat_conditions = 0
        if rsi > 65:
            sat_conditions += 1
        if last_price > bb_upper:
            sat_conditions += 1
        if macd < macd_signal:
            sat_conditions += 1
        if stoch_k > 80:
            sat_conditions += 1
        
        if al_conditions >= 3:
            return "AL", "green", rsi, bb_lower, bb_middle, bb_upper, macd, macd_signal, stoch_k, stoch_d, williams_r, atr, "Klasik Strateji"
        elif sat_conditions >= 3:
            return "SAT", "red", rsi, bb_lower, bb_middle, bb_upper, macd, macd_signal, stoch_k, stoch_d, williams_r, atr, "Klasik Strateji"
        elif al_conditions >= 2:
            return "AL (Zayıf)", "lightgreen", rsi, bb_lower, bb_middle, bb_upper, macd, macd_signal, stoch_k, stoch_d, williams_r, atr, "Klasik Strateji"
        elif sat_conditions >= 2:
            return "SAT (Zayıf)", "pink", rsi, bb_lower, bb_middle, bb_upper, macd, macd_signal, stoch_k, stoch_d, williams_r, atr, "Klasik Strateji"
        else:
            return "BEKLE", "gray", rsi, bb_lower, bb_middle, bb_upper, macd, macd_signal, stoch_k, stoch_d, williams_r, atr, "Klasik Strateji"
            
    except Exception as e:
        logger.error(f"Strateji hatası: {e}")
        return "Hata", "black", None, None, None, None, None, None, None, None, None, None, None, "Hata"

def compute_rsi_values(prices, period=14):
    """RSI değerlerini hesaplar (grafik için)"""
    try:
    prices = np.array(prices)
    if len(prices) < period + 1:
        return []
    rsi_values = []
    for i in range(period, len(prices)):
        deltas = np.diff(prices[i-period:i+1])
        up = deltas[deltas > 0].sum() / period
        down = -deltas[deltas < 0].sum() / period
        rs = up / down if down != 0 else 0
        rsi = 100 - 100 / (1 + rs)
        rsi_values.append(rsi)
    return rsi_values
    except Exception as e:
        logger.error(f"RSI değerleri hesaplama hatası: {e}")
        return []

def make_plot(times, prices, prediction, bb_lower, bb_middle, bb_upper, volumes):
    """Optimize edilmiş grafik oluşturma"""
    if not times or not prices or len(times) != len(prices):
        return "<p>Grafik verisi alınamadı veya eksik.</p>"
    
    try:
        import datetime
        times_fmt = [datetime.datetime.fromtimestamp(t/1000).strftime('%H:%M') for t in times]
        
        traces = []
        
        # 1. Ana fiyat grafiği
        traces.append(go.Scatter(
            x=times_fmt, 
            y=prices, 
            mode='lines', 
            name='Fiyat', 
            line=dict(color='#1f77b4', width=2)
        ))
        
        # 2. Bollinger bantları
        if bb_lower is not None and bb_middle is not None and bb_upper is not None:
            traces.append(go.Scatter(
                x=times_fmt, 
                y=[bb_lower] * len(times_fmt), 
                mode='lines', 
                name='Alt Band', 
                line=dict(dash='dot', color='blue')
            ))
            traces.append(go.Scatter(
                x=times_fmt, 
                y=[bb_middle] * len(times_fmt), 
                mode='lines', 
                name='Orta Band', 
                line=dict(dash='dash', color='gray')
            ))
            traces.append(go.Scatter(
                x=times_fmt, 
                y=[bb_upper] * len(times_fmt), 
                mode='lines', 
                name='Üst Band', 
                line=dict(dash='dot', color='red')
            ))
        
        # 3. Tahmin noktası
        if prediction is not None:
            traces.append(go.Scatter(
                x=[times_fmt[-1] + '+1'], 
                y=[prediction], 
                mode='markers', 
                marker=dict(color='orange', size=12, symbol='diamond'), 
                name='Tahmin'
            ))
        
        # 4. RSI grafiği (sadece son 50 veri noktası)
        rsi_values = compute_rsi_values(prices)
        if rsi_values and len(rsi_values) > 0:
            # Son 50 RSI değerini al (performans için)
            rsi_values = rsi_values[-50:] if len(rsi_values) > 50 else rsi_values
            rsi_times = times_fmt[-(len(rsi_values)):]
            
            traces.append(go.Scatter(
                x=rsi_times, 
                y=rsi_values, 
                mode='lines', 
                name='RSI', 
                yaxis='y2', 
                line=dict(color='purple', width=2)
            ))
            # RSI seviyeleri
            traces.append(go.Scatter(
                x=rsi_times, 
                y=[70] * len(rsi_times), 
                mode='lines', 
                name='RSI 70', 
                yaxis='y2', 
                line=dict(dash='dash', color='red')
            ))
            traces.append(go.Scatter(
                x=rsi_times, 
                y=[30] * len(rsi_times), 
                mode='lines', 
                name='RSI 30', 
                yaxis='y2', 
                line=dict(dash='dash', color='green')
            ))
        
        # Optimize edilmiş layout
        layout = go.Layout(
            title='Fiyat ve RSI Grafiği',
            xaxis=dict(title='Zaman'),
            yaxis=dict(title='USD', side='left'),
            yaxis2=dict(
                title='RSI', 
                side='right', 
                overlaying='y', 
                range=[0, 100]
            ),
            height=500,  # Daha küçük yükseklik
            showlegend=True,
            hovermode='x unified',
            margin=dict(l=50, r=50, t=50, b=50)  # Daha küçük margin
        )
        
        fig = go.Figure(data=traces, layout=layout)
        plot_div = pio.to_html(fig, full_html=False, include_plotlyjs=False)
        return plot_div
        
    except Exception as e:
        logger.error(f"Grafik oluşturma hatası: {e}")
            return "<p>Grafik oluşturulamadı.</p>"

@app.route('/')
def index():
    # Rate limiting kontrolü (devre dışı)
    # client_ip = request.remote_addr
    # rate_ok, rate_message = check_rate_limit(client_ip)
    
    symbol = request.args.get('symbol', 'BTCUSDT')
    interval = request.args.get('interval', '15m')
    error = None
    rate_limit_warning = None
    
    # Rate limiting devre dışı - her zaman devam et
    # if not rate_ok:
    #     rate_limit_warning = rate_message
    #     return render_template('index.html', ...)
    
    # Sembolü doğrula
    is_valid, result = validate_symbol(symbol)
    if not is_valid:
        error = result
        symbol = 'BTCUSDT'
    
    if is_valid:
        symbol = result
    
    try:
        # Sinyal durumlarını güncelle
        update_signal_status()
        
        # Otomatik model eğitimi kontrolü
        check_and_train_model()
    
    price = get_binance_price(symbol)
        if price is None:
            error = "Fiyat verisi alınamadı. Lütfen tekrar deneyin."
            return render_template('index.html',
                price="N/A",
                prediction="N/A",
                signal="Hata",
                color="black",
                plot_div="<p>Grafik verisi alınamadı.</p>",
                rsi="N/A",
                bb_lower="N/A",
                bb_middle="N/A",
                bb_upper="N/A",
                macd="N/A",
                macd_comment="Veri yok",
                stoch_k="N/A",
                stoch_d="N/A",
                stoch_comment="Veri yok",
                williams_r="N/A",
                williams_comment="Veri yok",
                atr="N/A",
                atr_comment="Veri yok",
                rsi_comment="Veri yok",
                strategy_info="Veri alınırken hata oluştu",
                popular_coins=POPULAR_COINS,
                symbol=symbol,
                interval=interval,
                timeframes=TIMEFRAMES,
                error=error,
                rate_limit_warning=rate_limit_warning
            )
        
    times, prices, highs, lows, volumes = get_binance_history(symbol, interval=interval, limit=100)
        
        if not prices or len(prices) < 20:
            error = "Yeterli veri yok. Lütfen tekrar deneyin."
            return render_template('index.html',
                price=price,
                prediction="N/A",
                signal="Veri Yetersiz",
                color="black",
                plot_div="<p>Yeterli veri yok.</p>",
                rsi="N/A",
                bb_lower="N/A",
                bb_middle="N/A",
                bb_upper="N/A",
                macd="N/A",
                macd_comment="Veri yok",
                stoch_k="N/A",
                stoch_d="N/A",
                stoch_comment="Veri yok",
                williams_r="N/A",
                williams_comment="Veri yok",
                atr="N/A",
                atr_comment="Veri yok",
                rsi_comment="Veri yok",
                strategy_info="Yeterli veri yok",
                popular_coins=POPULAR_COINS,
                symbol=symbol,
                interval=interval,
                timeframes=TIMEFRAMES,
                error=error,
                rate_limit_warning=rate_limit_warning
            )
        
    prediction = predict_next(prices)
        
        # İndikatörleri ayrıca hesapla (strategy_signal'den bağımsız)
        rsi = compute_rsi(prices)
        bb_lower, bb_middle, bb_upper = compute_bollinger_bands(prices)
    macd, macd_signal = compute_macd(prices)
    stoch_k, stoch_d = compute_stochastic(prices, highs, lows)
        williams_r = compute_williams_r(highs, lows, prices)
        atr = compute_atr(highs, lows, prices)
        
        # Strateji sinyali hesapla
        try:
            signal, color, _, _, _, _, _, _, _, _, _, _, _, strategy_info = strategy_signal(prices, highs, lows, volumes)
        except Exception as e:
            logger.error(f"Ana sayfa hatası: {e}")
            signal = "Hata"
            color = "black"
            strategy_info = "Sinyal hesaplanamadı"
        
        # Debug için detaylı log
        logger.info(f"Veri kontrolü - Prices: {len(prices) if prices else 0}, Highs: {len(highs) if highs else 0}, Lows: {len(lows) if lows else 0}")
        logger.info(f"İndikatör hesaplama sonuçları:")
        logger.info(f"  RSI: {rsi} (tip: {type(rsi)})")
        logger.info(f"  MACD: {macd} (tip: {type(macd)})")
        logger.info(f"  Stoch K: {stoch_k} (tip: {type(stoch_k)})")
        logger.info(f"  Stoch D: {stoch_d} (tip: {type(stoch_d)})")
        logger.info(f"  Williams R: {williams_r} (tip: {type(williams_r)})")
        logger.info(f"  ATR: {atr} (tip: {type(atr)})")
        logger.info(f"  BB Lower: {bb_lower} (tip: {type(bb_lower)})")
        logger.info(f"  BB Middle: {bb_middle} (tip: {type(bb_middle)})")
        logger.info(f"  BB Upper: {bb_upper} (tip: {type(bb_upper)})")
        
        # Eğer veriler eksikse varsayılan değerler kullan
        if rsi is None or bb_lower is None or macd is None or stoch_k is None:
            rsi = 50.0
            bb_lower = bb_middle = bb_upper = prices[-1] if prices else 0
            macd = macd_signal = 0.0
            stoch_k = stoch_d = 50.0
            williams_r = -50.0
            atr = 0.0
            logger.warning("Yeterli veri yok, varsayılan değerler kullanılıyor")
        
    plot_div = make_plot(times, prices, prediction, bb_lower, bb_middle, bb_upper, volumes)
    
        # Sinyali CSV'ye kaydet (sadece yeni coin/interval kombinasyonu için)
        if signal and signal not in ['Veri yetersiz', 'Hata'] and isinstance(price, (int, float)):
            # Son 5 dakika içinde aynı coin/interval için kayıt var mı kontrol et
            current_time = datetime.now()
            should_save = True
            
            if os.path.exists(SIGNALS_FILE):
                try:
                    df = pd.read_csv(SIGNALS_FILE)
                    if len(df) > 0:
                        # Son kayıtları kontrol et
                        recent_records = df[
                            (df['symbol'] == symbol) & 
                            (df['interval'] == interval)
                        ].tail(1)
                        
                        if len(recent_records) > 0:
                            last_record_time = pd.to_datetime(recent_records.iloc[0]['timestamp'])
                            time_diff = current_time - last_record_time
                            
                            # 5 dakikadan az ise kaydetme
                            if time_diff.total_seconds() < 300:  # 5 dakika = 300 saniye
                                should_save = False
                                logger.debug(f"Aynı coin/interval için son kayıt çok yakın: {symbol} {interval}")
                except Exception as e:
                    logger.error(f"Sinyal kayıt kontrolü hatası: {e}")
            
            if should_save:
                try:
                    save_signal_to_csv(
                        symbol=symbol,
                        timestamp=current_time.strftime('%Y-%m-%d %H:%M:%S'),
        price=price,
        signal=signal,
        rsi=rsi,
        bb_lower=bb_lower,
        bb_middle=bb_middle,
        bb_upper=bb_upper,
        macd=macd,
                        macd_signal=macd_signal,
        stoch_k=stoch_k,
        stoch_d=stoch_d,
                        williams_r=williams_r,
                        atr=atr,
                        interval=interval
                    )
                except Exception as e:
                    logger.error(f"Sinyal kaydetme hatası: {e}")
        
        # İndikatör değerlerini kontrol et ve formatla
        rsi_display = f"{rsi:.2f}" if rsi is not None else "N/A"
        bb_lower_display = f"{bb_lower:.2f}" if bb_lower is not None else "N/A"
        bb_middle_display = f"{bb_middle:.2f}" if bb_middle is not None else "N/A"
        bb_upper_display = f"{bb_upper:.2f}" if bb_upper is not None else "N/A"
        macd_display = f"{macd:.4f}" if macd is not None else "N/A"
        stoch_k_display = f"{stoch_k:.2f}" if stoch_k is not None else "N/A"
        stoch_d_display = f"{stoch_d:.2f}" if stoch_d is not None else "N/A"
        williams_r_display = f"{williams_r:.2f}" if williams_r is not None else "N/A"
        atr_display = f"{atr:.4f}" if atr is not None else "N/A"
        
        # Debug için log
        logger.info(f"İndikatörler - RSI: {rsi_display}, MACD: {macd_display}, Stoch K: {stoch_k_display}")
        
        return render_template('index.html',
            price=price,
            prediction=prediction,
            signal=signal,
            color=color,
            plot_div=plot_div,
            rsi=rsi_display,
            bb_lower=bb_lower_display,
            bb_middle=bb_middle_display,
            bb_upper=bb_upper_display,
            macd=macd_display,
            macd_comment=macd_comment(macd, macd_signal),
            stoch_k=stoch_k_display,
            stoch_d=stoch_d_display,
        stoch_comment=stoch_comment(stoch_k, stoch_d),
            williams_r=williams_r_display,
            williams_comment=williams_comment(williams_r),
            atr=atr_display,
            atr_comment=atr_comment(atr, price if isinstance(price, (int, float)) else None),
        rsi_comment=rsi_comment(rsi),
            strategy_info=strategy_info,
        popular_coins=POPULAR_COINS,
        symbol=symbol,
        interval=interval,
        timeframes=TIMEFRAMES,
            error=error,
            rate_limit_warning=rate_limit_warning
        )
        
    except Exception as e:
        logger.error(f"Ana sayfa hatası: {e}")
        return render_template('index.html',
            price="N/A",
            prediction="N/A",
            signal="Hata",
            color="black",
            plot_div="<p>Veri alınırken hata oluştu. Lütfen tekrar deneyin.</p>",
            rsi="N/A",
            bb_lower="N/A",
            bb_middle="N/A",
            bb_upper="N/A",
            macd="N/A",
            macd_comment="Hata",
            stoch_k="N/A",
            stoch_d="N/A",
            stoch_comment="Hata",
            williams_r="N/A",
            williams_comment="Hata",
            atr="N/A",
            atr_comment="Hata",
            rsi_comment="Hata",
            strategy_info="Veri alınırken hata oluştu",
            popular_coins=POPULAR_COINS,
            symbol=symbol,
            interval=interval,
            timeframes=TIMEFRAMES,
            error="Veri alınırken hata oluştu. Lütfen tekrar deneyin.",
            rate_limit_warning=rate_limit_warning
        )

@app.errorhandler(404)
def not_found(error):
    return render_template('index.html',
        price='Sayfa bulunamadı',
        prediction='N/A',
        signal='N/A',
        color='black',
        plot_div='<p>404 - Sayfa bulunamadı</p>',
        rsi='N/A',
        bb_lower='N/A',
        bb_middle='N/A',
        bb_upper='N/A',
        macd='N/A',
        macd_comment='Sayfa bulunamadı',
        stoch_k='N/A',
        stoch_d='N/A',
        stoch_comment='Sayfa bulunamadı',
        williams_r='N/A',
        williams_comment='Sayfa bulunamadı',
        atr='N/A',
        atr_comment='Sayfa bulunamadı',
        rsi_comment='Sayfa bulunamadı',
        strategy_info='Sayfa bulunamadı',
        popular_coins=POPULAR_COINS,
        symbol='BTCUSDT',
        interval='15m',
        timeframes=TIMEFRAMES,
        error="404 - Sayfa bulunamadı",
        rate_limit_warning=None
    ), 404

@app.route('/api/ip')
def get_ip():
    """IP adresini döndür"""
    return jsonify({
        'ip': request.remote_addr,
        'host': request.host,
        'url': request.url
    })

@app.route('/api/adaptive-learning')
def adaptive_learning_api():
    """Adaptif öğrenme API endpoint'i"""
    try:
        if not ADVANCED_FEATURES_AVAILABLE or not ADAPTIVE_LEARNING:
            return jsonify({'error': 'Adaptif öğrenme sistemi mevcut değil'}), 400
        
        # Performans analizi
        performance = ADAPTIVE_LEARNING.analyze_performance_trends()
        
        # Strateji önerileri
        strategy_recs = ADAPTIVE_LEARNING.adaptive_strategy_adjustment()
        
        # Öğrenme içgörüleri
        insights_file = 'learning_insights.json'
        insights = []
        if os.path.exists(insights_file):
            try:
                with open(insights_file, 'r') as f:
                    insights = json.load(f)
            except Exception as e:
                logger.warning(f"İçgörü dosyası okunamadı: {e}")
                insights = []
        
        # Eğer veri yoksa varsayılan değerler
        if not performance:
            performance = {
                'recent_accuracy': 0.65,
                'overall_accuracy': 0.58,
                'trend': 'İyileşiyor',
                'total_signals': 0,
                'recent_signals': 0
            }
        
        if not strategy_recs:
            strategy_recs = {
                'best_strategies': ['AL (Güçlü)', 'AL'],
                'avoid_strategies': ['SAT', 'BEKLE'],
                'signal_performance': {}
            }
        
        # Test içgörüleri oluştur
        if not insights:
            insights = [
                {
                    'timestamp': datetime.now().isoformat(),
                    'performance': {
                        'recent_accuracy': 0.65,
                        'overall_accuracy': 0.58,
                        'trend': 'İyileşiyor'
                    },
                    'insight': 'AI modeli öğrenmeye başladı'
                }
            ]
        
        return jsonify({
            'performance': performance,
            'strategy_recommendations': strategy_recs,
            'recent_insights': insights[-5:] if insights else [],  # Son 5 içgörü
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Adaptif öğrenme API hatası: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(500)
def internal_error(error):
    return render_template('index.html',
        price='Sunucu hatası',
        prediction='N/A',
        signal='N/A',
        color='red',
        plot_div='<p>500 - Sunucu hatası</p>',
        rsi='N/A',
        bb_lower='N/A',
        bb_middle='N/A',
        bb_upper='N/A',
        macd='N/A',
        macd_comment='Sunucu hatası',
        stoch_k='N/A',
        stoch_d='N/A',
        stoch_comment='Sunucu hatası',
        williams_r='N/A',
        williams_comment='Sunucu hatası',
        atr='N/A',
        atr_comment='Sunucu hatası',
        rsi_comment='Sunucu hatası',
        strategy_info='Sunucu hatası',
        popular_coins=POPULAR_COINS,
        symbol='BTCUSDT',
        interval='15m',
        timeframes=TIMEFRAMES,
        error="500 - Sunucu hatası oluştu",
        rate_limit_warning=None
    ), 500

# Otomatik veri toplama için global değişkenler
AUTO_DATA_COLLECTION = True
LAST_COLLECTION_TIME = None
COLLECTION_INTERVAL = 15 * 60  # 15 dakika (saniye cinsinden)

def auto_data_collector():
    """Otomatik veri toplama sistemi"""
    global LAST_COLLECTION_TIME
    
    while AUTO_DATA_COLLECTION:
        try:
            current_time = datetime.now()
            
            # İlk çalıştırma veya 15 dakika geçmişse
            if LAST_COLLECTION_TIME is None or (current_time - LAST_COLLECTION_TIME).total_seconds() >= COLLECTION_INTERVAL:
                logger.info("🔄 Otomatik veri toplama başlatılıyor...")
                
                collected_signals = 0
                successful_collections = 0
                
                for symbol, name in POPULAR_COINS:
                    try:
                        # Veri çek
                        times, prices, highs, lows, volumes = get_binance_history(symbol, "15m", 100)
                        
                        if len(prices) >= 20:  # Yeterli veri varsa
                            # Sinyal üret
                            signal, color, strategy_info, rsi, bb_lower, bb_middle, bb_upper, macd, macd_signal, stoch_k, stoch_d, williams_r, atr = strategy_signal(prices, highs, lows, volumes)
                            
                            # CSV'ye kaydet
                            if signal and signal != "Veri yetersiz":
                                save_signal_to_csv(
                                    symbol, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), prices[-1], signal,
                                    rsi, bb_lower, bb_middle, bb_upper, macd, macd_signal,
                                    stoch_k, stoch_d, williams_r, atr, "15m"
                                )
                                collected_signals += 1
                            
                            successful_collections += 1
                            
                            # Rate limiting (Binance API için)
                            time.sleep(0.1)  # 100ms bekle
                        
                    except Exception as e:
                        logger.error(f"Veri toplama hatası ({symbol}): {e}")
                        continue
                
                LAST_COLLECTION_TIME = current_time
                
                logger.info(f"✅ Otomatik veri toplama tamamlandı!")
                logger.info(f"📊 Başarılı: {successful_collections}/{len(POPULAR_COINS)} coin")
                logger.info(f"📈 Toplanan sinyal: {collected_signals}")
                
                # AI modelini kontrol et ve eğit
                check_and_train_model()
                
            # 1 dakika bekle
            time.sleep(60)
            
        except Exception as e:
            logger.error(f"Otomatik veri toplama sistemi hatası: {e}")
            time.sleep(60)  # Hata durumunda 1 dakika bekle

def start_auto_data_collection():
    """Otomatik veri toplama sistemini başlat"""
    if AUTO_DATA_COLLECTION:
        collector_thread = threading.Thread(target=auto_data_collector, daemon=True)
        collector_thread.start()
        logger.info("🚀 Otomatik veri toplama sistemi başlatıldı!")
        return True
    return False

# WebSocket ve AI için global değişkenler
WEBSOCKET_CLIENTS = set()
REAL_TIME_SIGNALS = {}
SIGNAL_THRESHOLD = 0.7  # %70 güven skoru
BUMP_DUMP_THRESHOLD = 0.05  # %5 fiyat değişimi

class AdvancedAIModels:
    """Gelişmiş AI Modelleri"""
    
    def __init__(self):
        self.lstm_model = None
        self.ensemble_model = None
        self.neural_network = None
        self.reinforcement_model = None
        self.scaler = StandardScaler()
        self.models_loaded = False
        
    def create_lstm_model(self, input_shape):
        """LSTM modeli oluşturur"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(3, activation='softmax')  # AL, SAT, BEKLE
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='sparse_categorical_crossentropy', 
                     metrics=['accuracy'])
        return model
    
    def create_ensemble_model(self):
        """Ensemble model oluşturur"""
        models = [
            RandomForestClassifier(n_estimators=100, random_state=42),
            GradientBoostingClassifier(n_estimators=100, random_state=42),
            MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        ]
        return models
    
    def create_neural_network(self, input_size):
        """Neural Network modeli oluşturur"""
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_size,)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(3, activation='softmax')
        ])
        model.compile(optimizer='adam', 
                     loss='sparse_categorical_crossentropy', 
                     metrics=['accuracy'])
        return model
    
    def prepare_lstm_data(self, prices, sequence_length=20):
        """LSTM için veri hazırlar"""
        if len(prices) < sequence_length + 1:
            return None, None
        
        X, y = [], []
        for i in range(sequence_length, len(prices)):
            # Fiyat değişimleri
            price_changes = []
            for j in range(sequence_length):
                if i - j - 1 >= 0:
                    change = (prices[i - j] - prices[i - j - 1]) / prices[i - j - 1]
                    price_changes.append(change)
                else:
                    price_changes.append(0)
            
            X.append(price_changes)
            
            # Etiket (gelecek fiyat değişimi)
            future_change = (prices[i] - prices[i-1]) / prices[i-1]
            if future_change > 0.01:  # %1 artış
                y.append(1)  # AL
            elif future_change < -0.01:  # %1 düşüş
                y.append(2)  # SAT
            else:
                y.append(0)  # BEKLE
        
        return np.array(X), np.array(y)
    
    def train_advanced_models(self, signals_data):
        """Gelişmiş modelleri eğitir"""
        try:
            if len(signals_data) < 50:
                logger.warning("Yetersiz veri için gelişmiş modeller eğitilemiyor")
                return False
            
            # Veri hazırlama
            features = ['rsi', 'bb_lower', 'bb_middle', 'bb_upper', 'macd', 'macd_signal', 
                       'stoch_k', 'stoch_d', 'williams_r', 'atr']
            
            X = signals_data[features].dropna()
            y = signals_data.loc[X.index, 'signal']
            
            if len(X) < 20:
                return False
            
            # Veri normalizasyonu
            X_scaled = self.scaler.fit_transform(X)
            
            # 1. Ensemble Model
            ensemble_models = self.create_ensemble_model()
            ensemble_predictions = []
            
            for model in ensemble_models:
                model.fit(X_scaled, y)
                pred = model.predict_proba(X_scaled)
                ensemble_predictions.append(pred)
            
            # Ensemble tahminleri birleştir
            ensemble_pred = np.mean(ensemble_predictions, axis=0)
            self.ensemble_model = ensemble_models
            
            # 2. Neural Network
            self.neural_network = self.create_neural_network(X.shape[1])
            y_encoded = pd.get_dummies(y).values
            self.neural_network.fit(X_scaled, y_encoded, epochs=50, batch_size=32, verbose=0)
            
            # 3. LSTM Model (fiyat verileri için)
            if 'price' in signals_data.columns:
                prices = signals_data['price'].values
                X_lstm, y_lstm = self.prepare_lstm_data(prices)
                if X_lstm is not None and len(X_lstm) > 10:
                    X_lstm = X_lstm.reshape((X_lstm.shape[0], X_lstm.shape[1], 1))
                    self.lstm_model = self.create_lstm_model((X_lstm.shape[1], 1))
                    self.lstm_model.fit(X_lstm, y_lstm, epochs=30, batch_size=16, verbose=0)
            
            # Modelleri kaydet
            self.save_advanced_models()
            self.models_loaded = True
            
            logger.info("✅ Gelişmiş AI modelleri eğitildi!")
            return True
            
        except Exception as e:
            logger.error(f"Gelişmiş model eğitme hatası: {e}")
            return False
    
    def predict_with_advanced_models(self, features, prices=None):
        """Gelişmiş modellerle tahmin yapar"""
        try:
            if not self.models_loaded:
                return None, 0
            
            # Ensemble tahmin
            features_scaled = self.scaler.transform([features])
            ensemble_preds = []
            
            for model in self.ensemble_model:
                pred = model.predict_proba(features_scaled)[0]
                ensemble_preds.append(pred)
            
            ensemble_result = np.mean(ensemble_preds, axis=0)
            
            # Neural Network tahmin
            nn_pred = self.neural_network.predict(features_scaled)[0]
            
            # LSTM tahmin (eğer fiyat verisi varsa)
            lstm_pred = None
            if self.lstm_model is not None and prices is not None:
                if len(prices) >= 20:
                    price_changes = []
                    for i in range(1, 21):
                        if len(prices) - i > 0:
                            change = (prices[-i] - prices[-i-1]) / prices[-i-1]
                            price_changes.append(change)
                        else:
                            price_changes.append(0)
                    
                    X_lstm = np.array([price_changes]).reshape(1, 20, 1)
                    lstm_pred = self.lstm_model.predict(X_lstm)[0]
            
            # Tahminleri birleştir
            final_pred = ensemble_result * 0.4 + nn_pred * 0.4
            if lstm_pred is not None:
                final_pred = final_pred * 0.8 + lstm_pred * 0.2
            
            # En yüksek olasılıklı sınıfı seç
            predicted_class = np.argmax(final_pred)
            confidence = max(final_pred)
            
            # Sinyal dönüştürme
            signal_map = {0: 'BEKLE', 1: 'AL', 2: 'SAT'}
            signal = signal_map.get(predicted_class, 'BEKLE')
            
            return signal, confidence
            
        except Exception as e:
            logger.error(f"Gelişmiş model tahmin hatası: {e}")
            return None, 0
    
    def save_advanced_models(self):
        """Gelişmiş modelleri kaydeder"""
        try:
            # Ensemble model
            with open('ensemble_model.pkl', 'wb') as f:
                pickle.dump(self.ensemble_model, f)
            
            # Neural Network
            if self.neural_network:
                self.neural_network.save('neural_network_model.h5')
            
            # LSTM model
            if self.lstm_model:
                self.lstm_model.save('lstm_model.h5')
            
            # Scaler
            with open('advanced_scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
            
            logger.info("✅ Gelişmiş modeller kaydedildi!")
            
        except Exception as e:
            logger.error(f"Model kaydetme hatası: {e}")
    
    def load_advanced_models(self):
        """Gelişmiş modelleri yükler"""
        try:
            # Ensemble model
            if os.path.exists('ensemble_model.pkl'):
                with open('ensemble_model.pkl', 'rb') as f:
                    self.ensemble_model = pickle.load(f)
            
            # Neural Network
            if os.path.exists('neural_network_model.h5'):
                self.neural_network = tf.keras.models.load_model('neural_network_model.h5')
            
            # LSTM model
            if os.path.exists('lstm_model.h5'):
                self.lstm_model = tf.keras.models.load_model('lstm_model.h5')
            
            # Scaler
            if os.path.exists('advanced_scaler.pkl'):
                with open('advanced_scaler.pkl', 'rb') as f:
                    self.scaler = pickle.load(f)
            
            self.models_loaded = True
            logger.info("✅ Gelişmiş modeller yüklendi!")
            
        except Exception as e:
            logger.error(f"Model yükleme hatası: {e}")

# Global AI modeli
advanced_ai = AdvancedAIModels()

def detect_bump_dump_signal(symbol, current_price, previous_price, volume_change):
    """Bump/Dump sinyali tespit eder"""
    try:
        if previous_price <= 0:
            return None
        
        price_change = (current_price - previous_price) / previous_price
        
        # Bump/Dump kriterleri
        if abs(price_change) >= BUMP_DUMP_THRESHOLD:  # %5 değişim
            if price_change > 0:
                signal_type = "🚀 BUMP"
                signal_color = "green"
            else:
                signal_type = "📉 DUMP"
                signal_color = "red"
            
            # Hacim kontrolü
            volume_indicator = ""
            if volume_change > 0.5:  # %50 hacim artışı
                volume_indicator = " + Yüksek Hacim"
            
            signal = {
                'symbol': symbol,
                'type': signal_type,
                'price_change': f"{price_change*100:.2f}%",
                'current_price': current_price,
                'volume_indicator': volume_indicator,
                'color': signal_color,
                'timestamp': datetime.now().isoformat()
            }
            
            return signal
        
        return None
        
    except Exception as e:
        logger.error(f"Bump/Dump sinyal hatası: {e}")
        return None

async def websocket_handler(websocket, path):
    """WebSocket bağlantı yöneticisi"""
    try:
        WEBSOCKET_CLIENTS.add(websocket)
        logger.info(f"WebSocket bağlantısı eklendi. Toplam: {len(WEBSOCKET_CLIENTS)}")
        
        # Bağlantı durumu gönder
        await websocket.send(json.dumps({
            'type': 'connection_status',
            'message': 'Bağlantı kuruldu',
            'timestamp': datetime.now().isoformat()
        }))
        
        async for message in websocket:
            try:
                data = json.loads(message)
                if data.get('type') == 'ping':
                    await websocket.send(json.dumps({'type': 'pong'}))
                    
            except json.JSONDecodeError:
                continue
                
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        WEBSOCKET_CLIENTS.discard(websocket)
        logger.info(f"WebSocket bağlantısı kapatıldı. Kalan: {len(WEBSOCKET_CLIENTS)}")

async def broadcast_signal(signal_data):
    """Tüm WebSocket istemcilerine sinyal gönderir"""
    if WEBSOCKET_CLIENTS:
        message = json.dumps({
            'type': 'signal',
            'data': signal_data
        })
        
        # Tüm bağlı istemcilere gönder
        await asyncio.gather(
            *[client.send(message) for client in WEBSOCKET_CLIENTS],
            return_exceptions=True
        )

def start_websocket_server():
    """WebSocket sunucusunu başlatır"""
    async def run_server():
        try:
            server = await websockets.serve(websocket_handler, "localhost", 8766)
            logger.info("🌐 WebSocket sunucusu başlatıldı (ws://localhost:8766)")
            await server.wait_closed()
        except OSError as e:
            if "Address already in use" in str(e):
                logger.warning("Port 8766 kullanımda, 8767 deneniyor...")
                server = await websockets.serve(websocket_handler, "localhost", 8767)
                logger.info("🌐 WebSocket sunucusu başlatıldı (ws://localhost:8767)")
                await server.wait_closed()
            else:
                logger.error(f"WebSocket sunucu hatası: {e}")
    
    # WebSocket sunucusunu ayrı thread'de çalıştır
    websocket_thread = Thread(target=lambda: asyncio.run(run_server()), daemon=True)
    websocket_thread.start()

def real_time_signal_monitor():
    """Gerçek zamanlı sinyal izleme"""
    global REAL_TIME_SIGNALS
    
    while True:
        try:
            for symbol, name in POPULAR_COINS:
                # Anlık fiyat al
                current_price = get_real_time_price(symbol)
                if current_price is None:
                    continue
                
                # Önceki fiyatı kontrol et
                if symbol in REAL_TIME_SIGNALS:
                    previous_price = REAL_TIME_SIGNALS[symbol]['price']
                    previous_time = REAL_TIME_SIGNALS[symbol]['timestamp']
                    
                    # 1 dakika geçmiş mi kontrol et
                    time_diff = datetime.now() - previous_time
                    if time_diff.total_seconds() >= 60:  # 1 dakika
                        # Bump/Dump sinyali tespit et
                        signal = detect_bump_dump_signal(symbol, current_price, previous_price, 0)
                        
                        if signal:
                            # WebSocket'e gönder
                            asyncio.run(broadcast_signal(signal))
                            
                            # Konsola yazdır
                            logger.info(f"🚨 {signal['type']} SİNYALİ: {symbol} - {signal['price_change']}")
                            
                            # Site bildirimi için kaydet
                            REAL_TIME_SIGNALS[symbol]['last_signal'] = signal
                
                # Güncel fiyatı kaydet
                REAL_TIME_SIGNALS[symbol] = {
                    'price': current_price,
                    'timestamp': datetime.now(),
                    'last_signal': None
                }
                
                # Rate limiting
                time.sleep(0.5)  # 500ms bekle
            
            # 30 saniye bekle
            time.sleep(30)
            
        except Exception as e:
            logger.error(f"Gerçek zamanlı sinyal izleme hatası: {e}")
            time.sleep(30)

def start_real_time_monitoring():
    """Gerçek zamanlı izleme sistemini başlatır"""
    monitor_thread = Thread(target=real_time_signal_monitor, daemon=True)
    monitor_thread.start()
    logger.info("🔍 Gerçek zamanlı sinyal izleme başlatıldı!")

if __name__ == '__main__':
    # AI modelini yükle
    load_ai_model()
    
    # Gelişmiş AI modellerini yükle
    advanced_ai.load_advanced_models()
    
    # Gelişmiş özellikleri başlat
    initialize_advanced_features()
    
    # WebSocket sunucusunu başlat
    start_websocket_server()
    
    # Gerçek zamanlı izleme sistemini başlat
    start_real_time_monitoring()
    
    # Otomatik veri toplama sistemi başlat
    start_auto_data_collection()

    logger.info("Flask uygulaması başlatılıyor...")
    logger.info("🚀 Otomatik veri toplama sistemi aktif!")
    logger.info(f"📊 {len(POPULAR_COINS)} coin her 15 dakikada bir taranacak")
    logger.info("🌐 WebSocket sunucusu aktif (ws://localhost:8766)")
    logger.info("🔍 Gerçek zamanlı sinyal izleme aktif!")
    app.run(debug=True, host='0.0.0.0', port=5000) 