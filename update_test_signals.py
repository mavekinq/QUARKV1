#!/usr/bin/env python3
"""
Test sinyallerini güncelle
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def update_test_signals():
    """Test sinyallerini güncelle"""
    
    try:
        # CSV dosyasını oku
        df = pd.read_csv('signals.csv')
        
        # Son 10 sinyali güncelle
        for i in range(min(10, len(df))):
            signal = df.iloc[i]
            
            # Rastgele başarı durumu
            success = random.choice([True, False])
            profit_loss = random.uniform(-5, 10) if success else random.uniform(-10, -1)
            
            # Fiyat değişimi hesapla
            original_price = signal['price']
            if signal['signal'].startswith('AL'):
                future_price = original_price * (1 + profit_loss/100)
            elif signal['signal'].startswith('SAT'):
                future_price = original_price * (1 - profit_loss/100)
            else:
                future_price = original_price
            
            # Durumu güncelle
            status = 'Başarılı' if profit_loss > 0 else 'Başarısız'
            
            df.iloc[i, df.columns.get_loc('future_price')] = future_price
            df.iloc[i, df.columns.get_loc('profit_loss')] = profit_loss
            df.iloc[i, df.columns.get_loc('status')] = status
        
        # CSV'ye kaydet
        df.to_csv('signals.csv', index=False)
        
        print("✅ Test sinyalleri güncellendi!")
        print(f"📊 {min(10, len(df))} sinyal güncellendi")
        
        # İstatistikleri göster
        completed = df[df['status'] != 'Beklemede']
        successful = completed[completed['status'] == 'Başarılı']
        
        print(f"📈 Tamamlanan: {len(completed)}")
        print(f"✅ Başarılı: {len(successful)}")
        print(f"❌ Başarısız: {len(completed) - len(successful)}")
        
        if len(completed) > 0:
            success_rate = (len(successful) / len(completed)) * 100
            print(f"🎯 Başarı Oranı: %{success_rate:.1f}")
        
    except Exception as e:
        print(f"❌ Hata: {e}")

if __name__ == "__main__":
    update_test_signals() 