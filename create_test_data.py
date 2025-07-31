#!/usr/bin/env python3
"""
Test verisi oluşturucu
signals.csv için örnek veri üretir
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def create_test_signals():
    """Test sinyalleri oluştur"""
    
    # Test verisi parametreleri
    num_signals = 200  # 200 test sinyali
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT']
    intervals = ['15m', '1h', '4h']
    signals = ['AL', 'SAT', 'BEKLE', 'AL (Zayıf)', 'SAT (Zayıf)', 'AL (Güçlü)', 'SAT (Güçlü)']
    statuses = ['Başarılı', 'Başarısız', 'Beklemede']
    
    # Test verisi oluştur
    test_data = []
    base_time = datetime.now() - timedelta(days=30)  # 30 gün öncesinden başla
    
    for i in range(num_signals):
        # Rastgele değerler
        symbol = random.choice(symbols)
        interval = random.choice(intervals)
        signal = random.choice(signals)
        status = random.choice(statuses)
        
        # Fiyat aralıkları (sembol bazlı)
        if symbol == 'BTCUSDT':
            price_range = (40000, 50000)
        elif symbol == 'ETHUSDT':
            price_range = (2000, 3000)
        else:
            price_range = (100, 1000)
        
        current_price = random.uniform(*price_range)
        
        # Gelecek fiyat (başarılı/başarısız durumuna göre)
        if status == 'Başarılı':
            if 'AL' in signal:
                future_price = current_price * random.uniform(1.02, 1.10)  # %2-10 artış
            else:
                future_price = current_price * random.uniform(0.90, 0.98)  # %2-10 düşüş
        elif status == 'Başarısız':
            if 'AL' in signal:
                future_price = current_price * random.uniform(0.90, 0.98)  # %2-10 düşüş
            else:
                future_price = current_price * random.uniform(1.02, 1.10)  # %2-10 artış
        else:
            future_price = current_price  # Beklemede
        
        # Kar/zarar hesapla
        if 'AL' in signal:
            profit_loss = ((future_price - current_price) / current_price) * 100
        else:
            profit_loss = ((current_price - future_price) / current_price) * 100
        
        # Teknik göstergeler (gerçekçi değerler)
        rsi = random.uniform(20, 80)
        bb_lower = current_price * random.uniform(0.95, 0.98)
        bb_middle = current_price * random.uniform(0.98, 1.02)
        bb_upper = current_price * random.uniform(1.02, 1.05)
        macd = random.uniform(-100, 100)
        macd_signal = macd + random.uniform(-20, 20)
        stoch_k = random.uniform(10, 90)
        stoch_d = stoch_k + random.uniform(-10, 10)
        williams_r = random.uniform(-90, -10)
        atr = current_price * random.uniform(0.01, 0.05)
        
        # Zaman damgası
        timestamp = base_time + timedelta(hours=i*2)  # 2 saat aralıklarla
        
        test_data.append({
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': symbol,
            'price': round(current_price, 2),
            'signal': signal,
            'rsi': round(rsi, 2),
            'bb_lower': round(bb_lower, 2),
            'bb_middle': round(bb_middle, 2),
            'bb_upper': round(bb_upper, 2),
            'macd': round(macd, 2),
            'macd_signal': round(macd_signal, 2),
            'stoch_k': round(stoch_k, 2),
            'stoch_d': round(stoch_d, 2),
            'williams_r': round(williams_r, 2),
            'atr': round(atr, 2),
            'interval': interval,
            'status': status,
            'future_price': round(future_price, 2),
            'profit_loss': round(profit_loss, 2)
        })
    
    # DataFrame oluştur
    df = pd.DataFrame(test_data)
    
    # CSV'ye kaydet
    df.to_csv('signals.csv', index=False)
    
    print(f"✅ {len(test_data)} test sinyali oluşturuldu!")
    print(f"📊 Sinyal dağılımı:")
    print(df['signal'].value_counts())
    print(f"📈 Durum dağılımı:")
    print(df['status'].value_counts())
    print(f"💰 Ortalama kar/zarar: %{df['profit_loss'].mean():.2f}")
    
    return df

if __name__ == "__main__":
    create_test_signals() 