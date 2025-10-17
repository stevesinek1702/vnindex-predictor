import pandas as pd
import numpy as np
import requests
import xgboost as xgb
import joblib
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
import os

# --- Cấu hình ---
MODEL_FILENAME = 'vnindex_model_v2.pkl'
fiin_key = os.environ.get('FIIN_KEY', 'default_key')
fiin_seed = os.environ.get('FIIN_SEED', 'default_seed')

# Headers
fitrade_headers = {
    "accept": "application/json", "origin": "https://portal.fidt.vn", "referer": "https://portal.fidt.vn/",
    "user-agent": "Mozilla/5.0", "x-fiin-key": fiin_key, "x-fiin-seed": fiin_seed,
    "x-fiin-user-id": "c4c89b7c-6ddb-44c8-9e46-ed23e7983f2a@@"
}
fireant_headers = {'User-Agent': 'Mozilla/5.0'}

# --- Tải "bộ não" ---
print("Đang tải 'bộ não' AI V3.0 vào bộ nhớ...")
try:
    model, model_columns = joblib.load(MODEL_FILENAME)
    print("'Bộ não' V3.0 đã sẵn sàng!")
except FileNotFoundError:
    print(f"\n!!! LỖI !!! Không tìm thấy file '{MODEL_FILENAME}'. Vui lòng chạy 'train.py' trước.")
    exit()

# --- Khởi tạo Web Server ---
app = Flask(__name__)

# --- Hàm lấy dữ liệu ---
def get_prediction_data():
    print("Đang tải dữ liệu mới nhất từ các nguồn...")
    
    # 1. FireAnt Data
    end_date_str = datetime.now().strftime('%Y-%m-%d')
    start_date_str = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    url_vnindex = f"https://www.fireant.vn/api/Data/Markets/HistoricalQuotes?symbol=HOSTC&startDate={start_date_str}&endDate={end_date_str}"
    url_vn30 = f"https://www.fireant.vn/api/Data/Markets/HistoricalQuotes?symbol=VN30&startDate={start_date_str}&endDate={end_date_str}"
    
    df_vnindex = pd.DataFrame(requests.get(url_vnindex, headers=fireant_headers).json())
    df_vn30 = pd.DataFrame(requests.get(url_vn30, headers=fireant_headers).json())
    
    df_vnindex['Date'] = pd.to_datetime(df_vnindex['Date']).dt.date
    df_vn30['Date'] = pd.to_datetime(df_vn30['Date']).dt.date
    
    df_vnindex = df_vnindex.set_index('Date')[['Close', 'Volume']].rename(columns={'Close': 'vnindex_close', 'Volume': 'vnindex_volume'})
    df_vn30 = df_vn30.set_index('Date')[['Close']].rename(columns={'Close': 'vn30_close'})
    
    # 2. FITRADE Investor Chart Data
    url_investor = "https://wl-market.fiintrade.vn/MoneyFlow/GetStatisticInvestorChart?language=vi&Code=VNINDEX&Frequently=Daily"
    df_investor = pd.DataFrame(requests.get(url_investor, headers=fitrade_headers).json()['items'])
    
    df_investor['Date'] = pd.to_datetime(df_investor['tradingDate']).dt.date
    df_investor = df_investor.set_index('Date')
    df_investor['foreign_net'] = (df_investor['foreignBuyValueMatched'] - df_investor['foreignSellValueMatched']) / 1e9
    df_investor['prop_net'] = (df_investor['proprietaryTotalMatchBuyTradeValue'] - df_investor['proprietaryTotalMatchSellTradeValue']) / 1e9
    df_investor['individual_net'] = (df_investor['localIndividualBuyMatchValue'] - df_investor['localIndividualSellMatchValue']) / 1e9
    df_investor['institution_net'] = -(df_investor['foreign_net'] + df_investor['prop_net'] + df_investor['individual_net'])
    df_investor = df_investor[['foreign_net', 'prop_net', 'individual_net', 'institution_net']]

    # Ghép tất cả lại
    master_df = df_vnindex.join(df_vn30, how='inner').join(df_investor, how='left')
    master_df = master_df.fillna(method='ffill').fillna(0)

    # Tạo features
    features_df = master_df.copy()
    for col in master_df.columns:
        if col != 'target':
            for i in range(1, 6): features_df[f'{col}_lag_{i}'] = features_df[col].shift(i)
            features_df[f'{col}_roll_mean_5'] = features_df[col].rolling(window=5).mean()

    # Chuẩn bị để dự báo
    latest_features = features_df.iloc[[-1]]
    for col in model_columns:
        if col not in latest_features.columns: latest_features[col] = 0
    latest_features = latest_features[model_columns]
    
    return latest_features

# --- "Cổng Giao Tiếp" ---
@app.route('/predict', methods=['POST'])
def predict():
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Nhận được yêu cầu dự báo V3.0.")
    try:
        features_to_predict = get_prediction_data()
        prediction_raw = model.predict(features_to_predict)
        prediction_final = round(float(prediction_raw[0]), 2)
        print(f"Đã tính toán xong. Kết quả dự báo: {prediction_final}")
        return jsonify({'prediction': prediction_final})
    except Exception as e:
        print(f"Lỗi nghiêm trọng khi xử lý yêu cầu: {e}")
        return jsonify({'error': str(e)}), 500