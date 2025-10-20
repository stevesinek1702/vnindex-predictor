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
print("Đang tải 'bộ não' AI V3.2 vào bộ nhớ...")
try:
    model, model_columns = joblib.load(MODEL_FILENAME)
    print("'Bộ não' V3.2 đã sẵn sàng!")
except FileNotFoundError:
    print(f"\n!!! LỖI !!! Không tìm thấy file '{MODEL_FILENAME}'.")
    exit()

# --- Khởi tạo Web Server ---
app = Flask(__name__)

# --- Các hàm xử lý dữ liệu ---
def get_daily_data(days_to_fetch=120): # Lấy nhiều dữ liệu hơn cho backtest
    # ... (Hàm này giữ nguyên không đổi) ...
    # 1. FireAnt Data
    end_date_str = datetime.now().strftime('%Y-%m-%d')
    start_date_str = (datetime.now() - timedelta(days=days_to_fetch)).strftime('%Y-%m-%d')
    url_vnindex = f"https://www.fireant.vn/api/Data/Markets/HistoricalQuotes?symbol=HOSTC&startDate={start_date_str}&endDate={end_date_str}"
    df_vnindex = pd.DataFrame(requests.get(url_vnindex, headers=fireant_headers).json())
    df_vnindex['Date'] = pd.to_datetime(df_vnindex['Date']).dt.date
    df_vnindex = df_vnindex.set_index('Date')[['Close', 'Volume']].rename(columns={'Close': 'vnindex_close', 'Volume': 'vnindex_volume'})
    
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
    
    master_df = df_vnindex.join(df_investor, how='left').fillna(method='ffill').fillna(0)
    return master_df

def create_features_from_df(df):
    features_df = df.copy()
    for col in df.columns:
        if col != 'target':
            for i in range(1, 6): features_df[f'{col}_lag_{i}'] = features_df[col].shift(i)
            features_df[f'{col}_roll_mean_5'] = features_df[col].rolling(window=5).mean()
    return features_df

# --- Cổng Giao Tiếp 1: Dự báo T+1 ---
@app.route('/predict', methods=['POST'])
def predict():
    # ... (Hàm này giữ nguyên không đổi) ...
    try:
        master_df = get_daily_data(days_to_fetch=90)
        features_df = create_features_from_df(master_df)
        latest_features = features_df.iloc[[-1]].copy()
        latest_features = latest_features.reindex(columns=model_columns, fill_value=0)
        prediction_raw = model.predict(latest_features)
        prediction_final = round(float(prediction_raw[0]), 2)
        return jsonify({'prediction': prediction_final})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- Cổng Giao Tiếp 2: Backtest & Forecast (Đã nâng cấp) ---
@app.route('/backtest_forecast', methods=['POST'])
def backtest_forecast():
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Nhận được yêu cầu Backtest & Forecast V3.2.")
    try:
        master_df = get_daily_data(days_to_fetch=120)
        
        # 1. Backtest
        backtest_results = []
        backtest_range = 20
        all_errors = []
        for i in range(backtest_range, 0, -1):
            snapshot_df = master_df.iloc[:-i]
            features_df = create_features_from_df(snapshot_df)
            features_to_predict = features_df.iloc[[-1]].copy().reindex(columns=model_columns, fill_value=0)
            prediction = model.predict(features_to_predict)[0]
            actual_date = master_df.index[-i]
            actual_price = master_df.iloc[-i]['vnindex_close']
            error_pct = (prediction - actual_price) / actual_price if actual_price != 0 else 0
            all_errors.append(abs(error_pct))
            backtest_results.append({
                'date': actual_date.strftime('%Y-%m-%d'),
                'actual': actual_price,
                'predicted': round(float(prediction), 2)
            })
            
        # Tính toán thống kê độ chính xác
        mape = np.mean(all_errors) * 100 if all_errors else 0
        accuracy = 100 - mape
        
        # 2. Forecast
        forecast_results = []
        future_df = master_df.copy()
        for i in range(10): # Dự báo 10 ngày
            features_df = create_features_from_df(future_df)
            features_to_predict = features_df.iloc[[-1]].copy().reindex(columns=model_columns, fill_value=0)
            prediction = model.predict(features_to_predict)[0]
            
            next_date = future_df.index[-1] + timedelta(days=1)
            forecast_results.append({
                'date': next_date.strftime('%Y-%m-%d'),
                'actual': None,
                'predicted': round(float(prediction), 2)
            })
            
            new_row_data = {col: 0 for col in future_df.columns}
            new_row_data['vnindex_close'] = prediction
            new_row = pd.DataFrame(new_row_data, index=[next_date])
            future_df = pd.concat([future_df, new_row])
            future_df = future_df.fillna(method='ffill')

        print("Hoàn tất Backtest & Forecast.")
        return jsonify({
            'backtest': backtest_results,
            'forecast': forecast_results,
            'stats': {
                'mape': round(mape, 2),
                'accuracy': round(accuracy, 2)
            }
        })

    except Exception as e:
        print(f"Lỗi nghiêm trọng khi thực hiện Backtest/Forecast: {e}")
        return jsonify({'error': str(e)}), 500