#!/usr/bin/env python3
"""
Test içgörüleri oluştur
"""

import json
from datetime import datetime, timedelta
import random

def create_test_insights():
    """Test içgörüleri oluştur"""
    
    insights = []
    
    # Son 5 gün için test verileri
    for i in range(5):
        date = datetime.now() - timedelta(days=i)
        
        # Rastgele performans değerleri
        recent_accuracy = random.uniform(0.55, 0.75)
        overall_accuracy = random.uniform(0.50, 0.70)
        trend = "İyileşiyor" if recent_accuracy > overall_accuracy else "Kötüleşiyor"
        
        insight = {
            'timestamp': date.isoformat(),
            'performance': {
                'recent_accuracy': recent_accuracy,
                'overall_accuracy': overall_accuracy,
                'trend': trend
            },
            'insight': f'AI modeli gün {i+1} öğrenme döngüsü'
        }
        
        insights.append(insight)
    
    # JSON dosyasına kaydet
    with open('learning_insights.json', 'w') as f:
        json.dump(insights, f, indent=2)
    
    print("✅ Test içgörüleri oluşturuldu!")
    print(f"📊 {len(insights)} adet içgörü kaydedildi")

if __name__ == "__main__":
    create_test_insights() 