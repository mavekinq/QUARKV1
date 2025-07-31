#!/usr/bin/env python3
"""
Gelişmiş Teknik Analiz
Ichimoku, Parabolic SAR, CCI, OBV ve diğer göstergeler
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, List, Tuple, Optional
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AdvancedTechnicalAnalysis:
    def __init__(self):
        self.indicators = {}
    
    def calculate_ichimoku(self, high: List[float], low: List[float], close: List[float]) -> Dict:
        """Ichimoku Cloud hesaplama"""
        try:
            high = np.array(high)
            low = np.array(low)
            close = np.array(close)
            
            # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
            period9_high = pd.Series(high).rolling(window=9).max()
            period9_low = pd.Series(low).rolling(window=9).min()
            tenkan_sen = (period9_high + period9_low) / 2
            
            # Kijun-sen (Base Line): (26-period high + 26-period low)/2
            period26_high = pd.Series(high).rolling(window=26).max()
            period26_low = pd.Series(low).rolling(window=26).min()
            kijun_sen = (period26_high + period26_low) / 2
            
            # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
            
            # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
            period52_high = pd.Series(high).rolling(window=52).max()
            period52_low = pd.Series(low).rolling(window=52).min()
            senkou_span_b = ((period52_high + period52_low) / 2).shift(26)
            
            # Chikou Span (Lagging Span): Close price shifted back 26 periods
            chikou_span = pd.Series(close).shift(-26)
            
            return {
                'tenkan_sen': tenkan_sen.iloc[-1] if not tenkan_sen.empty else None,
                'kijun_sen': kijun_sen.iloc[-1] if not kijun_sen.empty else None,
                'senkou_span_a': senkou_span_a.iloc[-1] if not senkou_span_a.empty else None,
                'senkou_span_b': senkou_span_b.iloc[-1] if not senkou_span_b.empty else None,
                'chikou_span': chikou_span.iloc[-1] if not chikou_span.empty else None,
                'current_price': close[-1] if len(close) > 0 else None
            }
            
        except Exception as e:
            logger.error(f"Ichimoku hesaplama hatası: {e}")
            return {}
    
    def calculate_parabolic_sar(self, high: List[float], low: List[float], close: List[float], 
                               acceleration: float = 0.02, maximum: float = 0.2) -> Dict:
        """Parabolic SAR hesaplama"""
        try:
            high = np.array(high)
            low = np.array(low)
            close = np.array(close)
            
            # TALib kullanarak Parabolic SAR hesapla
            sar = talib.SAR(high, low, acceleration=acceleration, maximum=maximum)
            
            current_sar = sar[-1] if not np.isnan(sar[-1]) else None
            current_price = close[-1] if len(close) > 0 else None
            
            # Trend belirleme
            trend = "BULLISH" if current_price > current_sar else "BEARISH" if current_price < current_sar else "NEUTRAL"
            
            return {
                'sar': current_sar,
                'trend': trend,
                'current_price': current_price
            }
            
        except Exception as e:
            logger.error(f"Parabolic SAR hesaplama hatası: {e}")
            return {}
    
    def calculate_cci(self, high: List[float], low: List[float], close: List[float], period: int = 20) -> Dict:
        """Commodity Channel Index hesaplama"""
        try:
            high = np.array(high)
            low = np.array(low)
            close = np.array(close)
            
            # TALib kullanarak CCI hesapla
            cci = talib.CCI(high, low, close, timeperiod=period)
            
            current_cci = cci[-1] if not np.isnan(cci[-1]) else None
            
            # CCI sinyalleri
            if current_cci is not None:
                if current_cci > 100:
                    signal = "AŞIRI ALIM"
                elif current_cci < -100:
                    signal = "AŞIRI SATIM"
                elif current_cci > 0:
                    signal = "POZİTİF"
                else:
                    signal = "NEGATİF"
            else:
                signal = "NÖTR"
            
            return {
                'cci': current_cci,
                'signal': signal,
                'period': period
            }
            
        except Exception as e:
            logger.error(f"CCI hesaplama hatası: {e}")
            return {}
    
    def calculate_obv(self, close: List[float], volume: List[float]) -> Dict:
        """On-Balance Volume hesaplama"""
        try:
            close = np.array(close)
            volume = np.array(volume)
            
            # TALib kullanarak OBV hesapla
            obv = talib.OBV(close, volume)
            
            current_obv = obv[-1] if not np.isnan(obv[-1]) else None
            
            # OBV trend analizi
            if len(obv) >= 20:
                obv_ma = np.mean(obv[-20:])  # 20 periyot ortalaması
                if current_obv > obv_ma:
                    trend = "YÜKSELEN"
                elif current_obv < obv_ma:
                    trend = "DÜŞEN"
                else:
                    trend = "YATAY"
            else:
                trend = "YETERSİZ VERİ"
            
            return {
                'obv': current_obv,
                'trend': trend
            }
            
        except Exception as e:
            logger.error(f"OBV hesaplama hatası: {e}")
            return {}
    
    def calculate_fibonacci_retracement(self, high: List[float], low: List[float]) -> Dict:
        """Fibonacci Retracement seviyeleri"""
        try:
            if len(high) < 2 or len(low) < 2:
                return {}
            
            # Son swing high ve low
            swing_high = max(high[-20:])  # Son 20 periyot
            swing_low = min(low[-20:])
            
            # Fark
            diff = swing_high - swing_low
            
            # Fibonacci seviyeleri
            levels = {
                '0.0': swing_low,
                '0.236': swing_low + 0.236 * diff,
                '0.382': swing_low + 0.382 * diff,
                '0.5': swing_low + 0.5 * diff,
                '0.618': swing_low + 0.618 * diff,
                '0.786': swing_low + 0.786 * diff,
                '1.0': swing_high
            }
            
            current_price = high[-1] if len(high) > 0 else None
            
            # Hangi seviyede olduğunu belirle
            current_level = "YUKARI"
            for level_name, level_price in sorted(levels.items(), key=lambda x: float(x[0])):
                if current_price <= level_price:
                    current_level = f"FIB {level_name}"
                    break
            
            return {
                'levels': levels,
                'current_price': current_price,
                'current_level': current_level,
                'swing_high': swing_high,
                'swing_low': swing_low
            }
            
        except Exception as e:
            logger.error(f"Fibonacci hesaplama hatası: {e}")
            return {}
    
    def calculate_pivot_points(self, high: List[float], low: List[float], close: List[float]) -> Dict:
        """Pivot Point seviyeleri"""
        try:
            if len(high) < 1 or len(low) < 1 or len(close) < 1:
                return {}
            
            # Son günün verileri
            prev_high = high[-1]
            prev_low = low[-1]
            prev_close = close[-1]
            
            # Pivot Point
            pivot = (prev_high + prev_low + prev_close) / 3
            
            # Support ve Resistance seviyeleri
            r1 = 2 * pivot - prev_low
            r2 = pivot + (prev_high - prev_low)
            r3 = prev_high + 2 * (pivot - prev_low)
            
            s1 = 2 * pivot - prev_high
            s2 = pivot - (prev_high - prev_low)
            s3 = prev_low - 2 * (prev_high - pivot)
            
            current_price = close[-1] if len(close) > 0 else None
            
            # Pozisyon belirleme
            if current_price > r1:
                position = "R1 ÜSTÜ"
            elif current_price > pivot:
                position = "PIVOT-R1 ARASI"
            elif current_price > s1:
                position = "S1-PIVOT ARASI"
            else:
                position = "S1 ALTINDA"
            
            return {
                'pivot': pivot,
                'r1': r1, 'r2': r2, 'r3': r3,
                's1': s1, 's2': s2, 's3': s3,
                'current_price': current_price,
                'position': position
            }
            
        except Exception as e:
            logger.error(f"Pivot Point hesaplama hatası: {e}")
            return {}
    
    def calculate_volume_profile(self, close: List[float], volume: List[float], bins: int = 10) -> Dict:
        """Volume Profile analizi"""
        try:
            if len(close) < bins or len(volume) < bins:
                return {}
            
            # Son N periyot
            recent_close = close[-bins:]
            recent_volume = volume[-bins:]
            
            # Fiyat aralıkları
            price_min = min(recent_close)
            price_max = max(recent_close)
            price_range = price_max - price_min
            
            if price_range == 0:
                return {}
            
            # Volume profile hesapla
            volume_profile = {}
            for i in range(bins):
                price_level = price_min + (i * price_range / bins)
                volume_at_level = sum(recent_volume[j] for j in range(len(recent_close)) 
                                    if abs(recent_close[j] - price_level) <= price_range / (2 * bins))
                volume_profile[f"level_{i+1}"] = {
                    'price': price_level,
                    'volume': volume_at_level
                }
            
            # POC (Point of Control) - En yüksek hacimli seviye
            poc_level = max(volume_profile.items(), key=lambda x: x[1]['volume'])
            
            current_price = close[-1] if len(close) > 0 else None
            
            return {
                'volume_profile': volume_profile,
                'poc': poc_level[1]['price'],
                'current_price': current_price,
                'bins': bins
            }
            
        except Exception as e:
            logger.error(f"Volume Profile hesaplama hatası: {e}")
            return {}
    
    def get_advanced_signals(self, high: List[float], low: List[float], close: List[float], 
                           volume: List[float] = None) -> Dict:
        """Tüm gelişmiş göstergeleri hesapla ve sinyal üret"""
        try:
            signals = {}
            
            # Ichimoku
            ichimoku = self.calculate_ichimoku(high, low, close)
            if ichimoku:
                current_price = ichimoku['current_price']
                tenkan = ichimoku['tenkan_sen']
                kijun = ichimoku['kijun_sen']
                
                if current_price and tenkan and kijun:
                    if current_price > tenkan > kijun:
                        signals['ichimoku'] = 'AL (Güçlü)'
                    elif current_price > tenkan and tenkan < kijun:
                        signals['ichimoku'] = 'AL (Zayıf)'
                    elif current_price < tenkan < kijun:
                        signals['ichimoku'] = 'SAT (Güçlü)'
                    elif current_price < tenkan and tenkan > kijun:
                        signals['ichimoku'] = 'SAT (Zayıf)'
                    else:
                        signals['ichimoku'] = 'BEKLE'
            
            # Parabolic SAR
            sar_data = self.calculate_parabolic_sar(high, low, close)
            if sar_data:
                signals['parabolic_sar'] = 'AL' if sar_data['trend'] == 'BULLISH' else 'SAT' if sar_data['trend'] == 'BEARISH' else 'BEKLE'
            
            # CCI
            cci_data = self.calculate_cci(high, low, close)
            if cci_data:
                if cci_data['signal'] == 'AŞIRI SATIM':
                    signals['cci'] = 'AL'
                elif cci_data['signal'] == 'AŞIRI ALIM':
                    signals['cci'] = 'SAT'
                else:
                    signals['cci'] = 'BEKLE'
            
            # OBV (volume varsa)
            if volume and len(volume) > 0:
                obv_data = self.calculate_obv(close, volume)
                if obv_data:
                    signals['obv'] = 'AL' if obv_data['trend'] == 'YÜKSELEN' else 'SAT' if obv_data['trend'] == 'DÜŞEN' else 'BEKLE'
            
            # Fibonacci
            fib_data = self.calculate_fibonacci_retracement(high, low)
            if fib_data:
                current_price = fib_data['current_price']
                levels = fib_data['levels']
                
                if current_price <= levels['0.236']:
                    signals['fibonacci'] = 'AL'
                elif current_price >= levels['0.786']:
                    signals['fibonacci'] = 'SAT'
                else:
                    signals['fibonacci'] = 'BEKLE'
            
            # Pivot Points
            pivot_data = self.calculate_pivot_points(high, low, close)
            if pivot_data:
                current_price = pivot_data['current_price']
                pivot = pivot_data['pivot']
                
                if current_price > pivot:
                    signals['pivot'] = 'AL'
                elif current_price < pivot:
                    signals['pivot'] = 'SAT'
                else:
                    signals['pivot'] = 'BEKLE'
            
            # Genel sinyal (çoğunluk oylaması)
            if signals:
                buy_count = sum(1 for signal in signals.values() if 'AL' in signal)
                sell_count = sum(1 for signal in signals.values() if 'SAT' in signal)
                wait_count = sum(1 for signal in signals.values() if signal == 'BEKLE')
                
                total_signals = len(signals)
                
                if buy_count > total_signals / 2:
                    overall_signal = 'AL'
                elif sell_count > total_signals / 2:
                    overall_signal = 'SAT'
                else:
                    overall_signal = 'BEKLE'
                
                signals['overall'] = overall_signal
                signals['confidence'] = max(buy_count, sell_count, wait_count) / total_signals * 100
            
            return signals
            
        except Exception as e:
            logger.error(f"Gelişmiş sinyal hesaplama hatası: {e}")
            return {}

# Test fonksiyonu
def test_advanced_analysis():
    """Gelişmiş teknik analiz test"""
    analyzer = AdvancedTechnicalAnalysis()
    
    # Test verisi
    high = [45000, 45500, 46000, 45800, 46200, 46500, 46300, 46700, 47000, 46800]
    low = [44000, 44500, 45000, 44800, 45200, 45500, 45300, 45700, 46000, 45800]
    close = [44500, 45200, 45500, 45000, 46000, 46200, 45800, 46500, 46800, 46500]
    volume = [1000, 1200, 1100, 900, 1300, 1400, 1200, 1500, 1600, 1400]
    
    # Gelişmiş sinyaller
    signals = analyzer.get_advanced_signals(high, low, close, volume)
    
    print("=== Gelişmiş Teknik Analiz Sonuçları ===")
    for indicator, signal in signals.items():
        print(f"{indicator.upper()}: {signal}")
    
    # Detaylı analizler
    print("\n=== Detaylı Analizler ===")
    
    ichimoku = analyzer.calculate_ichimoku(high, low, close)
    print(f"Ichimoku: {ichimoku}")
    
    sar = analyzer.calculate_parabolic_sar(high, low, close)
    print(f"Parabolic SAR: {sar}")
    
    cci = analyzer.calculate_cci(high, low, close)
    print(f"CCI: {cci}")
    
    fib = analyzer.calculate_fibonacci_retracement(high, low)
    print(f"Fibonacci: {fib}")

if __name__ == "__main__":
    test_advanced_analysis() 