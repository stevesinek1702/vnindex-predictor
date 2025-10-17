import pandas as pd
import numpy as np
import requests
import xgboost as xgb
import joblib
from datetime import datetime, timedelta
from flask import Flask, request, jsonify

# --- Tải "bộ não" đã được huấn luyện vào bộ nhớ ---
# Chương trình chỉ cần làm việc này một lần khi khởi động
print("Đang tải 'bộ não' AI vào bộ nhớ...")
try:
    model = joblib.load('vnindex_model.pkl')
    print("'Bộ não' AI đã sẵn sàng!")
except FileNotFoundError:
    print("\n!!! LỖI !!!")
    print("Không tìm thấy file 'vnindex_model.pkl'.")
    print("Vui lòng chạy file 'train.py' trước để tạo ra 'bộ não'.")
    exit() # Thoát chương trình nếu không có model

# --- Khởi tạo Web Server ---
app = Flask(__name__)

# --- Các hàm xử lý dữ liệu (giống hệt file train.py) ---
def fetch_latest_data(symbol):
    """Tải dữ liệu 60 ngày gần nhất để tính toán features."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60) # Cần đủ dữ liệu để tính MA20, RSI...
    
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    url = f"https://www.fireant.vn/api/Data/Markets/HistoricalQuotes?symbol={symbol}&startDate={start_date_str}&endDate={end_date_str}"
    
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by='Date', ascending=True)
        df.set_index('Date', inplace=True)
        
        df = df.rename(columns={'Close': 'DongCua', 'Volume': 'KhoiLuong'})
        return df[['DongCua', 'KhoiLuong']]
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu mới nhất: {e}")
        return None

def create_features_for_prediction(df):
    """Tạo features cho dữ liệu mới nhất."""
    # Features về giá trong quá khứ (lags)
    for i in range(1, 6):
        df[f'price_lag_{i}'] = df['DongCua'].shift(i)
    # Features về khối lượng trong quá khứ
    for i in range(1, 4):
        df[f'volume_lag_{i}'] = df['KhoiLuong'].shift(i)
    # Features về trung bình động (Moving Averages)
    df['ma_5'] = df['DongCua'].rolling(window=5).mean()
    df['ma_10'] = df['DongCua'].rolling(window=10).mean()
    df['ma_20'] = df['DongCua'].rolling(window=20).mean()
    # Feature: Chỉ số RSI
    delta = df['DongCua'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Chỉ lấy dòng cuối cùng (dữ liệu mới nhất) để dự báo
    latest_features = df.iloc[[-1]]
    
    # Lấy đúng danh sách các cột đã dùng để train model
    model_columns = model.get_booster().feature_names
    latest_features = latest_features[model_columns]
    
    return latest_features

# --- Tạo "Cổng Giao Tiếp" cho Google Sheet ---
@app.route('/predict', methods=['POST'])
def predict():
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Nhận được yêu cầu dự báo từ Google Sheet.")
    try:
        # 1. Lấy dữ liệu mới nhất từ FireAnt
        latest_data_df = fetch_latest_data("HOSTC")
        if latest_data_df is None:
            return jsonify({'error': 'Không thể tải dữ liệu từ FireAnt'}), 500
        
        # 2. Tạo features cho dữ liệu mới nhất đó
        features_to_predict = create_features_for_prediction(latest_data_df.copy())
        
        # 3. Đưa vào "bộ não" để dự báo
        prediction_raw = model.predict(features_to_predict)
        
        # 4. Làm tròn và định dạng kết quả
        prediction_final = round(float(prediction_raw[0]), 2)
        
        print(f"Đã tính toán xong. Kết quả dự báo: {prediction_final}")
        
        # 5. Trả kết quả về cho Google Sheet
        return jsonify({'prediction': prediction_final})

    except Exception as e:
        print(f"Lỗi nghiêm trọng khi xử lý yêu cầu: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    print("\n--- MÁY CHỦ DỰ BÁO ĐÃ SẴN SÀNG ---")
    print("Server đang chạy tại http://127.0.0.1:5000")
    print("Hãy để cửa sổ này chạy ngầm và chuyển sang bước tiếp theo...")
    app.run(host='0.0.0.0', port=5000)
