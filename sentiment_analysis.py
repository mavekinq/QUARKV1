#!/usr/bin/env python3
"""
Duygu Analizi Sistemi
Twitter, haber ve sosyal medya duygu analizi
"""

import requests
import json
import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import tweepy
import time
import logging
from datetime import datetime, timedelta
import os
import re
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.sentiment_cache = {}
        self.cache_duration = 300  # 5 dakika cache
        
        # API anahtarları (güvenlik için environment variables kullanın)
        self.twitter_api = None
        self.news_api_key = None
        
        # Duygu skorları
        self.sentiment_scores = {
            'very_positive': 0.8,
            'positive': 0.4,
            'neutral': 0.0,
            'negative': -0.4,
            'very_negative': -0.8
        }
    
    def analyze_text_sentiment(self, text):
        """Metin duygu analizi"""
        try:
            if not text or len(text.strip()) < 10:
                return 0.0, 'neutral'
            
            # VADER sentiment
            vader_scores = self.vader_analyzer.polarity_scores(text)
            vader_compound = vader_scores['compound']
            
            # TextBlob sentiment
            blob = TextBlob(text)
            textblob_polarity = blob.sentiment.polarity
            
            # Ortalama sentiment skoru
            avg_sentiment = (vader_compound + textblob_polarity) / 2
            
            # Sentiment kategorisi
            if avg_sentiment >= 0.5:
                category = 'very_positive'
            elif avg_sentiment >= 0.1:
                category = 'positive'
            elif avg_sentiment >= -0.1:
                category = 'neutral'
            elif avg_sentiment >= -0.5:
                category = 'negative'
            else:
                category = 'very_negative'
            
            return avg_sentiment, category
            
        except Exception as e:
            logger.error(f"Metin duygu analizi hatası: {e}")
            return 0.0, 'neutral'
    
    def get_crypto_news_sentiment(self, symbol='BTC', limit=10):
        """Kripto haber duygu analizi"""
        try:
            # Basit haber API (gerçek uygulamada NewsAPI kullanın)
            news_data = self._fetch_crypto_news(symbol, limit)
            
            if not news_data:
                return 0.0, 'neutral', []
            
            sentiments = []
            total_sentiment = 0.0
            
            for news in news_data:
                title = news.get('title', '')
                description = news.get('description', '')
                text = f"{title} {description}"
                
                sentiment_score, category = self.analyze_text_sentiment(text)
                sentiments.append({
                    'title': title,
                    'sentiment': sentiment_score,
                    'category': category,
                    'source': news.get('source', 'Unknown'),
                    'published_at': news.get('publishedAt', '')
                })
                
                total_sentiment += sentiment_score
            
            avg_sentiment = total_sentiment / len(sentiments) if sentiments else 0.0
            
            # Genel kategori
            if avg_sentiment >= 0.3:
                overall_category = 'positive'
            elif avg_sentiment >= -0.3:
                overall_category = 'neutral'
            else:
                overall_category = 'negative'
            
            return avg_sentiment, overall_category, sentiments
            
        except Exception as e:
            logger.error(f"Haber duygu analizi hatası: {e}")
            return 0.0, 'neutral', []
    
    def get_social_sentiment(self, symbol='BTC', limit=50):
        """Sosyal medya duygu analizi"""
        try:
            # Twitter benzeri veri (gerçek uygulamada Twitter API kullanın)
            social_data = self._fetch_social_data(symbol, limit)
            
            if not social_data:
                return 0.0, 'neutral', []
            
            sentiments = []
            total_sentiment = 0.0
            
            for post in social_data:
                text = post.get('text', '')
                sentiment_score, category = self.analyze_text_sentiment(text)
                
                sentiments.append({
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'sentiment': sentiment_score,
                    'category': category,
                    'likes': post.get('likes', 0),
                    'retweets': post.get('retweets', 0),
                    'user': post.get('user', 'Unknown')
                })
                
                # Ağırlıklı sentiment (likes ve retweets ile)
                weight = 1 + (post.get('likes', 0) + post.get('retweets', 0)) / 100
                total_sentiment += sentiment_score * weight
            
            avg_sentiment = total_sentiment / len(sentiments) if sentiments else 0.0
            
            # Genel kategori
            if avg_sentiment >= 0.2:
                overall_category = 'positive'
            elif avg_sentiment >= -0.2:
                overall_category = 'neutral'
            else:
                overall_category = 'negative'
            
            return avg_sentiment, overall_category, sentiments
            
        except Exception as e:
            logger.error(f"Sosyal medya duygu analizi hatası: {e}")
            return 0.0, 'neutral', []
    
    def get_fear_greed_index(self):
        """Fear & Greed Index"""
        try:
            # Basit fear & greed hesaplama
            # Gerçek uygulamada alternative.me API kullanın
            
            # Simüle edilmiş veri
            fear_greed_data = {
                'value': 65,  # 0-100 arası
                'classification': 'Greed',
                'timestamp': datetime.now().isoformat()
            }
            
            return fear_greed_data
            
        except Exception as e:
            logger.error(f"Fear & Greed Index hatası: {e}")
            return {'value': 50, 'classification': 'Neutral', 'timestamp': datetime.now().isoformat()}
    
    def get_comprehensive_sentiment(self, symbol='BTC'):
        """Kapsamlı duygu analizi"""
        try:
            # Cache kontrolü
            cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H')}"
            if cache_key in self.sentiment_cache:
                cache_time, data = self.sentiment_cache[cache_key]
                if time.time() - cache_time < self.cache_duration:
                    return data
            
            # 1. Haber duygu analizi
            news_sentiment, news_category, news_details = self.get_crypto_news_sentiment(symbol)
            
            # 2. Sosyal medya duygu analizi
            social_sentiment, social_category, social_details = self.get_social_sentiment(symbol)
            
            # 3. Fear & Greed Index
            fear_greed = self.get_fear_greed_index()
            
            # 4. Ağırlıklı ortalama sentiment
            # Haberler %40, Sosyal medya %40, Fear & Greed %20
            weighted_sentiment = (
                news_sentiment * 0.4 +
                social_sentiment * 0.4 +
                (fear_greed['value'] - 50) / 50 * 0.2  # 0-100'ü -1 ile 1'e çevir
            )
            
            # Genel kategori
            if weighted_sentiment >= 0.3:
                overall_category = 'very_positive'
            elif weighted_sentiment >= 0.1:
                overall_category = 'positive'
            elif weighted_sentiment >= -0.1:
                overall_category = 'neutral'
            elif weighted_sentiment >= -0.3:
                overall_category = 'negative'
            else:
                overall_category = 'very_negative'
            
            # Sonuç
            result = {
                'symbol': symbol,
                'overall_sentiment': weighted_sentiment,
                'overall_category': overall_category,
                'news_sentiment': news_sentiment,
                'news_category': news_category,
                'social_sentiment': social_sentiment,
                'social_category': social_category,
                'fear_greed': fear_greed,
                'news_details': news_details[:5],  # İlk 5 haber
                'social_details': social_details[:5],  # İlk 5 post
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache'e kaydet
            self.sentiment_cache[cache_key] = (time.time(), result)
            
            return result
            
        except Exception as e:
            logger.error(f"Kapsamlı duygu analizi hatası: {e}")
            return {
                'symbol': symbol,
                'overall_sentiment': 0.0,
                'overall_category': 'neutral',
                'error': str(e)
            }
    
    def _fetch_crypto_news(self, symbol, limit):
        """Kripto haberlerini getir (simüle edilmiş)"""
        try:
            # Gerçek uygulamada NewsAPI kullanın
            # Şimdilik simüle edilmiş veri
            sample_news = [
                {
                    'title': f'{symbol} price reaches new highs',
                    'description': f'Positive news about {symbol} market performance',
                    'source': 'CryptoNews',
                    'publishedAt': datetime.now().isoformat()
                },
                {
                    'title': f'{symbol} adoption increases',
                    'description': f'More companies adopting {symbol} technology',
                    'source': 'BlockchainNews',
                    'publishedAt': datetime.now().isoformat()
                }
            ]
            
            return sample_news[:limit]
            
        except Exception as e:
            logger.error(f"Haber getirme hatası: {e}")
            return []
    
    def _fetch_social_data(self, symbol, limit):
        """Sosyal medya verilerini getir (simüle edilmiş)"""
        try:
            # Gerçek uygulamada Twitter API kullanın
            # Şimdilik simüle edilmiş veri
            sample_posts = [
                {
                    'text': f'{symbol} is going to the moon! 🚀',
                    'likes': 150,
                    'retweets': 25,
                    'user': 'crypto_enthusiast'
                },
                {
                    'text': f'I believe in {symbol} long term potential',
                    'likes': 89,
                    'retweets': 12,
                    'user': 'investor_pro'
                }
            ]
            
            return sample_posts[:limit]
            
        except Exception as e:
            logger.error(f"Sosyal medya veri getirme hatası: {e}")
            return []
    
    def get_sentiment_signal(self, sentiment_score):
        """Duygu skoruna göre sinyal üret"""
        try:
            if sentiment_score >= 0.5:
                return 'AL', 'Çok Pozitif Duygu'
            elif sentiment_score >= 0.2:
                return 'AL (Zayıf)', 'Pozitif Duygu'
            elif sentiment_score >= -0.2:
                return 'BEKLE', 'Nötr Duygu'
            elif sentiment_score >= -0.5:
                return 'SAT (Zayıf)', 'Negatif Duygu'
            else:
                return 'SAT', 'Çok Negatif Duygu'
                
        except Exception as e:
            logger.error(f"Duygu sinyali hatası: {e}")
            return 'BEKLE', 'Duygu Analizi Hatası'

# Test fonksiyonu
def test_sentiment_analysis():
    """Duygu analizi test"""
    analyzer = SentimentAnalyzer()
    
    # Kapsamlı duygu analizi
    result = analyzer.get_comprehensive_sentiment('BTC')
    
    print("=== Duygu Analizi Sonuçları ===")
    print(f"Genel Duygu: {result['overall_sentiment']:.3f} ({result['overall_category']})")
    print(f"Haber Duygu: {result['news_sentiment']:.3f} ({result['news_category']})")
    print(f"Sosyal Duygu: {result['social_sentiment']:.3f} ({result['social_category']})")
    print(f"Fear & Greed: {result['fear_greed']['value']} ({result['fear_greed']['classification']})")
    
    # Sinyal üret
    signal, message = analyzer.get_sentiment_signal(result['overall_sentiment'])
    print(f"Duygu Sinyali: {signal} - {message}")

if __name__ == "__main__":
    test_sentiment_analysis() 