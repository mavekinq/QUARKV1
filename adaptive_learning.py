#!/usr/bin/env python3
"""
Adaptif Öğrenme Sistemi
AI modellerinin otomatik kendini iyileştirmesi
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pickle
import os
import logging
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class AdaptiveLearning:
    def __init__(self):
        self.performance_history = []
        self.model_versions = []
        self.learning_config = {
            'min_accuracy_threshold': 0.65,  # %65 minimum doğruluk
            'improvement_threshold': 0.02,   # %2 iyileştirme beklentisi
            'max_retrain_attempts': 3,       # Maksimum 3 deneme
            'feature_importance_threshold': 0.05  # %5'den az önemli özellikleri çıkar
        }

    def analyze_performance_trends(self, signals_file='signals.csv'):
        """Performans trendlerini analiz et"""
        try:
            if not os.path.exists(signals_file):
                return None

            df = pd.read_csv(signals_file)
            completed_signals = df[df['status'] != 'Beklemede'].copy()

            if len(completed_signals) < 20:
                return None

            # Hatalı timestamp'leri otomatik olarak NaT yap ve logla
            original_count = len(completed_signals)
            completed_signals['timestamp'] = pd.to_datetime(completed_signals['timestamp'], errors='coerce')
            bad_ts = completed_signals['timestamp'].isnull().sum()
            if bad_ts > 0:
                logger.warning(f"{bad_ts} adet hatalı timestamp verisi atlandı.")
            completed_signals = completed_signals.dropna(subset=['timestamp'])
            logger.info(f"{original_count} satırdan {len(completed_signals)} satır işleniyor (timestamp hatası hariç).")

            completed_signals = completed_signals.sort_values('timestamp')

            # Son 7 günlük performans
            week_ago = datetime.now() - timedelta(days=7)
            recent_signals = completed_signals[completed_signals['timestamp'] > week_ago]

            if len(recent_signals) < 10:
                return None

            # Performans metrikleri
            recent_accuracy = (recent_signals['profit_loss'] > 0).mean()
            overall_accuracy = (completed_signals['profit_loss'] > 0).mean()

            # Trend analizi
            trend = "İyileşiyor" if recent_accuracy > overall_accuracy else "Kötüleşiyor"

            return {
                'recent_accuracy': recent_accuracy,
                'overall_accuracy': overall_accuracy,
                'trend': trend,
                'total_signals': len(completed_signals),
                'recent_signals': len(recent_signals)
            }

        except Exception as e:
            logger.error(f"Performans analizi hatası: {e}")
            return None

    def adaptive_feature_selection(self, X, y):
        """Adaptif özellik seçimi"""
        try:
            # Mevcut model ile özellik önemini hesapla
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)

            # Özellik önem skorları
            feature_importance = model.feature_importances_
            feature_names = X.columns

            # Düşük önemli özellikleri filtrele
            important_features = []
            for i, importance in enumerate(feature_importance):
                if importance > self.learning_config['feature_importance_threshold']:
                    important_features.append(feature_names[i])

            logger.info(f"Özellik seçimi: {len(feature_names)} -> {len(important_features)}")
            return important_features

        except Exception as e:
            logger.error(f"Özellik seçimi hatası: {e}")
            return list(X.columns)

    def adaptive_hyperparameter_tuning(self, X, y, current_accuracy):
        """Adaptif hiperparametre ayarlama"""
        try:
            best_params = None
            best_accuracy = current_accuracy

            # Farklı parametre kombinasyonları dene
            param_combinations = [
                {'n_estimators': 100, 'max_depth': 10},
                {'n_estimators': 150, 'max_depth': 15},
                {'n_estimators': 200, 'max_depth': 8},
                {'n_estimators': 100, 'max_depth': 12, 'min_samples_split': 5},
                {'n_estimators': 150, 'max_depth': 10, 'min_samples_leaf': 2}
            ]

            for params in param_combinations:
                model = RandomForestClassifier(**params, random_state=42)
                scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
                avg_accuracy = scores.mean()

                if avg_accuracy > best_accuracy:
                    best_accuracy = avg_accuracy
                    best_params = params

            if best_params and (best_accuracy - current_accuracy) > self.learning_config['improvement_threshold']:
                logger.info(f"Hiperparametre iyileştirmesi: {current_accuracy:.3f} -> {best_accuracy:.3f}")
                return best_params

            return None

        except Exception as e:
            logger.error(f"Hiperparametre ayarlama hatası: {e}")
            return None

    def adaptive_strategy_adjustment(self, signals_file='signals.csv'):
        """Adaptif strateji ayarlaması"""
        try:
            if not os.path.exists(signals_file):
                return None

            df = pd.read_csv(signals_file)
            completed_signals = df[df['status'] != 'Beklemede'].copy()

            if len(completed_signals) < 30:
                return None

            # Sinyal türü bazlı başarı analizi
            signal_performance = {}
            for signal_type in completed_signals['signal'].unique():
                signal_data = completed_signals[completed_signals['signal'] == signal_type]
                if len(signal_data) >= 5:  # En az 5 sinyal
                    success_rate = (signal_data['profit_loss'] > 0).mean()
                    avg_profit = signal_data['profit_loss'].mean()
                    signal_performance[signal_type] = {
                        'success_rate': success_rate,
                        'avg_profit': avg_profit,
                        'count': len(signal_data)
                    }

            # En başarılı stratejileri belirle
            best_strategies = []
            for signal, perf in signal_performance.items():
                if perf['success_rate'] > 0.6 and perf['avg_profit'] > 1.0:
                    best_strategies.append(signal)

            # Strateji önerileri
            recommendations = {
                'best_strategies': best_strategies,
                'avoid_strategies': [s for s, p in signal_performance.items()
                                     if p['success_rate'] < 0.4],
                'signal_performance': signal_performance
            }

            return recommendations

        except Exception as e:
            logger.error(f"Strateji ayarlama hatası: {e}")
            return None

    def adaptive_learning_cycle(self, signals_file='signals.csv'):
        """Tam adaptif öğrenme döngüsü"""
        try:
            logger.info("🔄 Adaptif öğrenme döngüsü başlatılıyor...")

            # 1. Performans analizi
            performance = self.analyze_performance_trends(signals_file)
            if not performance:
                logger.warning("Yeterli veri yok, adaptif öğrenme atlanıyor")
                return False

            logger.info(f"📊 Performans: Son 7 gün %{performance['recent_accuracy']:.1f}, "
                        f"Genel %{performance['overall_accuracy']:.1f}")

            # 2. Strateji önerileri
            strategy_recs = self.adaptive_strategy_adjustment(signals_file)
            if strategy_recs:
                logger.info(f"🎯 En iyi stratejiler: {strategy_recs['best_strategies']}")
                logger.info(f"⚠️ Kaçınılacak stratejiler: {strategy_recs['avoid_strategies']}")

            # 3. Model iyileştirme önerileri
            if performance['recent_accuracy'] < self.learning_config['min_accuracy_threshold']:
                logger.warning("⚠️ Performans düşük, model iyileştirmesi gerekli")

                # Özellik seçimi ve hiperparametre ayarlama
                # (Bu kısım mevcut model eğitimi ile entegre edilecek)

                return True  # İyileştirme gerekli

            logger.info("✅ Performans yeterli, iyileştirme gerekmiyor")
            return False

        except Exception as e:
            logger.error(f"Adaptif öğrenme döngüsü hatası: {e}")
            return False

    def save_learning_insights(self, insights):
        """Öğrenme içgörülerini kaydet"""
        try:
            insights['timestamp'] = datetime.now().isoformat()

            # Mevcut içgörüleri yükle
            if os.path.exists('learning_insights.json'):
                with open('learning_insights.json', 'r') as f:
                    all_insights = json.load(f)
            else:
                all_insights = []

            # Yeni içgörüyü ekle
            all_insights.append(insights)

            # Son 100 içgörüyü tut
            if len(all_insights) > 100:
                all_insights = all_insights[-100:]

            # Kaydet
            with open('learning_insights.json', 'w') as f:
                json.dump(all_insights, f, indent=2)

            logger.info("💾 Öğrenme içgörüleri kaydedildi")

        except Exception as e:
            logger.error(f"İçgörü kaydetme hatası: {e}")

# Test fonksiyonu
def test_adaptive_learning():
    """Adaptif öğrenme sistemini test et"""
    adaptive = AdaptiveLearning()

    # Adaptif öğrenme döngüsünü çalıştır
    needs_improvement = adaptive.adaptive_learning_cycle()

    if needs_improvement:
        print("🔄 Model iyileştirmesi gerekli!")
    else:
        print("✅ Model performansı yeterli")

    # Strateji önerilerini al
    strategy_recs = adaptive.adaptive_strategy_adjustment()
    if strategy_recs:
        print(f"🎯 En iyi stratejiler: {strategy_recs['best_strategies']}")
        print(f"⚠️ Kaçınılacak stratejiler: {strategy_recs['avoid_strategies']}")

if __name__ == "__main__":
    test_adaptive_learning()
