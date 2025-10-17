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
fitrade_headers_4_groups = {
    "accept": "application/json",
    "accept-encoding": "gzip, deflate, br, zstd",
    "accept-language": "en-US,en-GB;q=0.9,en;q=0.8",
    "cache-control": "no-cache",
    "connection": "keep-alive",
    "content-type": "application/json",
    "host": "wl-market.fiintrade.vn",
    "origin": "https://portal.fidt.vn",
    "pragma": "no-cache",
    "referer": "https://portal.fidt.vn/",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    "x-fiin-key": fiin_key,
    "x-fiin-seed": fiin_seed,
    "x-fiin-user-id": "c4c89b7c-6ddb-44c8-9e46-ed23e7983f2a@@"
}
fireant_headers = {'User-Agent': 'Mozilla/5.0'}

# --- Tải "bộ não" V2.3 vào bộ nhớ ---
print("Đang tải 'bộ não' AI V2.3 vào bộ nhớ...")
try:
    model, model_columns = joblib.load(MODEL_FILENAME)
    print("'Bộ não' V2.3 đã sẵn sàng!")
except FileNotFoundError:
    print(f"\n!!! LỖI !!! Không tìm thấy file '{MODEL_FILENAME}'. Vui lòng chạy 'train.py' trước.")
    exit()

# --- Khởi tạo Web Server ---
app = Flask(__name__)

# --- Các hàm lấy dữ liệu "lite" ---
# ... (Toàn bộ các hàm fetch_... và create_... giữ nguyên không đổi) ...
def fetch_fireant_data_lite(symbol):
    end_date_str = datetime.now().strftime('%Y-%m-%d')
    start_date_str = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    url = f"https://www.fireant.vn/api/Data/Markets/HistoricalQuotes?symbol={symbol}&startDate={start_date_str}&endDate={end_date_str}"
    response = requests.get(url, headers=fireant_headers)
    response.raise_for_status()
    df = pd.DataFrame(response.json())
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    return df.set_index('Date')[['Close', 'Volume']]

def fetch_fitrade_industry_flow_lite():
    end_date_str = datetime.now().strftime('%d-%m-%Y')
    start_date_str = (datetime.now() - timedelta(days=90)).strftime('%d-%m-%Y')
    ind_codes = "8350,8630,8770,2350,1750,9530,3350,6570,8980,3720,2710,1350,5370,4570,2770,3780,3530,2730,3570,2720,7570,2750,3760,2790,5550,6530,4530,1730,1770,5330,0530,0570,3740"
    url = f"https://apigw.fitrade.vn/pbapi/api/indActiveBuySell?indCode={ind_codes}&FromDate={start_date_str}&Todate={end_date_str}"
    response = requests.get(url, headers={'User-Agent': 'Mozilla/50'})
    response.raise_for_status()
    data = response.json()['data']
    df = pd.DataFrame(data)
    df['tradeDate'] = pd.to_datetime(df['tradeDate']).dt.date
    df = df.sort_values(by=['indName', 'tradeDate'])
    df['netActiveBuy_daily'] = df.groupby('indName')['netActiveBuy'].diff().fillna(df['netActiveBuy']) / 1e9
    pivot_df = df.pivot_table(index='tradeDate', columns='indName', values='netActiveBuy_daily', aggfunc='sum')
    pivot_df.index.name = 'Date'
    top_industries_from_model = [col.replace('industry_flow_', '') for col in model_columns if 'industry_flow_' in col]
    cols_to_use = [col for col in top_industries_from_model if col in pivot_df.columns]
    pivot_df = pivot_df[cols_to_use]
    return pivot_df.add_prefix('industry_flow_')

def fetch_fitrade_investor_flow_lite():
    investor_types = {'foreign': 'ForeignMatch', 'prop': 'ProprietaryMatch', 'individual': 'LocalIndividualMatch', 'institution': 'LocalInstitutionMatch'}
    net_values = {}
    for name, type_code in investor_types.items():
        url = f"https://wl-market.fiintrade.vn/MoneyFlow/GetStatisticInvestor?language=vi&comGroupCode=VNINDEX&investorType={type_code}"
        try:
            response = requests.get(url, headers=fitrade_headers_4_groups, timeout=15)
            response.raise_for_status()
            data = response.json().get('items', [{}])[0].get('today', {})
            if name == 'foreign': net_values[f'foreign_net_flow'] = data.get('foreignNetValue', 0) / 1e9
            elif name == 'prop': net_values[f'prop_net_flow'] = data.get('proprietaryNetValue', 0) / 1e9
            elif name == 'institution':
                buy = data.get('localInstitutionBuyValue', 0); sell = data.get('localInstitutionSellValue', 0)
                net_values[f'institution_net_flow'] = (buy - sell) / 1e9
            elif name == 'individual':
                total_buy = data.get('foreignBuyValue', 0); total_sell = data.get('foreignSellValue', 0)
                other_buy = data.get('proprietaryBuyValue', 0) + data.get('localInstitutionBuyValue', 0) + data.get('foreignBuyValue', 0)
                other_sell = data.get('proprietarySellValue', 0) + data.get('localInstitutionSellValue', 0) + data.get('foreignSellValue', 0)
                net_values[f'individual_net_flow'] = ((total_buy - other_buy) - (total_sell - other_sell)) / 1e9
        except Exception: net_values[f'{name}_net_flow'] = 0
    return net_values

def fetch_fireant_intraday_lite(symbol):
    url = f"https://www.fireant.vn/api/Data/Markets/IntradayMarketStatistic?symbol={symbol}"
    try:
        response = requests.get(url, headers=fireant_headers, timeout=10); response.raise_for_status()
        return response.json()
    except Exception: return None

def create_intraday_features(intraday_data):
    if not intraday_data or len(intraday_data) == 0: return {}
    df = pd.DataFrame(intraday_data)
    df['luc_cau'] = df['TotalActiveBuyVolume'] / (df['TotalActiveBuyVolume'] + df['TotalActiveSellVolume'])
    df['luc_cau'] = df['luc_cau'].fillna(0.5)
    features = {}
    features['lc_highest'] = df['luc_cau'].max(); features['lc_lowest'] = df['luc_cau'].min()
    features['lc_average'] = df['luc_cau'].mean(); features['lc_close'] = df['luc_cau'].iloc[-1]
    bounced = 1 if 0.35 <= features['lc_lowest'] <= 0.37 and features['lc_close'] > features['lc_lowest'] + 0.02 else 0
    features['bounced_from_bottom_zone'] = bounced
    sentiment_conditions = [ features['lc_average'] > 0.47, features['lc_average'] >= 0.42, features['lc_average'] < 0.37]
    sentiment_choices = [2, 1, -2]
    features['market_sentiment_score'] = np.select(sentiment_conditions, sentiment_choices, default=-1)
    features['time_in_positive_zone_percent'] = (df['luc_cau'] > 0.47).sum() / len(df)
    return features

# --- "Cổng Giao Tiếp" ---
@app.route('/predict', methods=['POST'])
def predict():
    # ... (Hàm này giữ nguyên logic, không đổi) ...
    try:
        df_vnindex = fetch_fireant_data_lite("HOSTC"); df_vn30 = fetch_fireant_data_lite("VN30")
        df_industry_flow = fetch_fitrade_industry_flow_lite()
        investor_flow_today = fetch_fitrade_investor_flow_lite()
        intraday_vnindex_data = fetch_fireant_intraday_lite("HOSTC")
        intraday_vn30_data = fetch_fireant_intraday_lite("VN30")
        vnindex_intraday_features = create_intraday_features(intraday_vnindex_data)
        vn30_intraday_features = create_intraday_features(intraday_vn30_data)
        df_vnindex = df_vnindex.rename(columns={'Close': 'vnindex_close', 'Volume': 'vnindex_volume'})
        df_vn30 = df_vn30[['Close']].rename(columns={'Close': 'vn30_close'})
        master_df = df_vnindex.join(df_vn30, how='inner').join(df_industry_flow, how='left')
        last_date = master_df.index[-1]
        for key, value in investor_flow_today.items():
            if key not in master_df.columns: master_df[key] = 0.0
            master_df.loc[last_date, key] = value
        master_df = master_df.fillna(method='ffill').fillna(0)
        features_df = master_df.copy()
        for col in master_df.columns:
            for i in range(1, 4): features_df[f'{col}_lag_{i}'] = features_df[col].shift(i)
            features_df[f'{col}_roll_mean_5'] = features_df[col].rolling(window=5).mean()
            features_df[f'{col}_roll_mean_10'] = features_df[col].rolling(window=10).mean()
        latest_features = features_df.iloc[[-1]].copy()
        for key, value in vnindex_intraday_features.items(): latest_features[f'vnindex_{key}'] = value
        for key, value in vn30_intraday_features.items(): latest_features[f'vn30_{key}'] = value
        if 'vnindex_lc_average' in latest_features.columns and 'vn30_lc_average' in latest_features.columns:
            latest_features['lc_divergence'] = latest_features['vn30_lc_average'] - latest_features['vnindex_lc_average']
        for col in model_columns:
            if col not in latest_features.columns: latest_features[col] = 0
        latest_features = latest_features[model_columns]
        prediction_raw = model.predict(latest_features)
        prediction_final = round(float(prediction_raw[0]), 2)
        print(f"Dự báo: {prediction_final}")
        return jsonify({'prediction': prediction_final})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Dòng này đã bị xóa đi để Render tự quản lý
# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=5000)