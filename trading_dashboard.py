#!/usr/bin/env python3
"""
Gelişmiş Trading Dashboard
Gerçek zamanlı trading analizi ve yönetimi
"""

from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from advanced_trading_ai import AdvancedTradingAI
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global değişkenler
trading_ai = AdvancedTradingAI()
active_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
portfolio = {
    'balance': 10000,  # USD
    'positions': {},
    'trades': [],
    'performance': {
        'total_return': 0,
        'win_rate': 0,
        'total_trades': 0
    }
}

@app.route('/')
def dashboard():
    """Ana trading dashboard"""
    return render_template('trading_dashboard.html')

@app.route('/api/portfolio')
def get_portfolio():
    """Portföy bilgilerini döndür"""
    return jsonify(portfolio)

@app.route('/api/signal/<symbol>')
def get_signal(symbol):
    """AI sinyali al"""
    try:
        # Modelleri yükle (eğer yoksa eğit)
        if not trading_ai.load_models(symbol):
            logger.info(f"Modeller yüklenemedi, {symbol} için eğitim başlatılıyor...")
            trading_ai.train_models(symbol)
        
        # Sinyal tahmin et
        prediction = trading_ai.predict_signal(symbol)
        
        return jsonify({
            'symbol': symbol,
            'prediction': prediction,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Sinyal hatası: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/execute_trade', methods=['POST'])
def execute_trade():
    """İşlem gerçekleştir"""
    try:
        data = request.json
        symbol = data['symbol']
        signal = data['signal']
        price = float(data['price'])
        confidence = float(data['confidence'])
        
        # Risk kontrolü
        if confidence < trading_ai.risk_config['min_confidence']:
            return jsonify({'error': 'Düşük güven skoru'}), 400
        
        # İşlem miktarı hesapla
        position_size = portfolio['balance'] * trading_ai.risk_config['max_position_size']
        
        # İşlem kaydet
        trade = {
            'id': len(portfolio['trades']) + 1,
            'symbol': symbol,
            'signal': signal,
            'price': price,
            'amount': position_size,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'status': 'OPEN'
        }
        
        portfolio['trades'].append(trade)
        
        # Portföy güncelle
        if signal == 'AL':
            portfolio['positions'][symbol] = {
                'entry_price': price,
                'amount': position_size,
                'entry_time': datetime.now().isoformat()
            }
            portfolio['balance'] -= position_size
        
        return jsonify({'success': True, 'trade': trade})
        
    except Exception as e:
        logger.error(f"İşlem hatası: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/performance')
def get_performance():
    """Performans analizi"""
    try:
        # Son 30 günlük performans
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        recent_trades = [
            trade for trade in portfolio['trades']
            if datetime.fromisoformat(trade['timestamp']) >= start_date
        ]
        
        if recent_trades:
            winning_trades = [t for t in recent_trades if t.get('profit', 0) > 0]
            win_rate = len(winning_trades) / len(recent_trades)
            
            total_profit = sum(t.get('profit', 0) for t in recent_trades)
            total_return = (total_profit / portfolio['balance']) * 100
        else:
            win_rate = 0
            total_return = 0
        
        return jsonify({
            'win_rate': win_rate,
            'total_return': total_return,
            'total_trades': len(recent_trades),
            'balance': portfolio['balance']
        })
        
    except Exception as e:
        logger.error(f"Performans analizi hatası: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chart/<symbol>')
def get_chart_data(symbol):
    """Grafik verisi"""
    try:
        # Son 100 veri noktası
        df = trading_ai.get_real_time_data(symbol, '15m', 100)
        
        if df.empty:
            return jsonify({'error': 'Veri alınamadı'}), 500
        
        # İndikatörler hesapla
        df = trading_ai.calculate_advanced_indicators(df)
        
        # Grafik verisi
        chart_data = {
            'timestamps': df.index.strftime('%Y-%m-%d %H:%M').tolist(),
            'prices': df['close'].tolist(),
            'volumes': df['volume'].tolist(),
            'rsi': df['rsi'].tolist(),
            'macd': df['macd'].tolist(),
            'bb_upper': df['bb_upper'].tolist(),
            'bb_middle': df['bb_middle'].tolist(),
            'bb_lower': df['bb_lower'].tolist()
        }
        
        return jsonify(chart_data)
        
    except Exception as e:
        logger.error(f"Grafik verisi hatası: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/train/<symbol>')
def train_model(symbol):
    """Model eğitimi"""
    try:
        success = trading_ai.train_models(symbol)
        
        if success:
            return jsonify({'success': True, 'message': f'{symbol} modeli eğitildi'})
        else:
            return jsonify({'error': 'Eğitim başarısız'}), 500
            
    except Exception as e:
        logger.error(f"Eğitim hatası: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("🚀 Gelişmiş Trading Dashboard başlatılıyor...")
    app.run(debug=True, host='0.0.0.0', port=5001) 