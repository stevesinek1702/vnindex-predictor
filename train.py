import pandas as pd
import numpy as np
import requests
import xgboost as xgb
import joblib
from datetime import datetime, timedelta
import os

# --- Cấu hình ---
MODEL_FILENAME = 'vnindex_model_v2.pkl'
fiin_key = os.environ.get('FIIN_KEY', 'default_key')
fiin_seed = os.environ.get('FIIN_SEED', 'default_seed')
YEARS_OF_DATA = 5

# Headers
fitrade_headers = {
    "accept": "application/json", "origin": "https://portal.fidt.vn", "referer": "https://portal.fidt.vn/",
    "user-agent": "Mozilla/5.0", "x-fiin-key": fiin_key, "x-fiin-seed": fiin_seed,
    "x-fiin-user-id": "c4c89b7c-6ddb-44c8-9e46-ed23e7983f2a@@"
}
fireant_headers = {'User-Agent': 'Mozilla/5.0'}

def fetch_fireant_data(symbol, years):
    print(f"Đang tải {years} năm dữ liệu giá cho {symbol} từ FireAnt...")
    end_date_str = datetime.now().strftime('%Y-%m-%d')
    start_date_str = (datetime.now() - timedelta(days=years * 365)).strftime('%Y-%m-%d')
    url = f"https://www.fireant.vn/api/Data/Markets/HistoricalQuotes?symbol={symbol}&startDate={start_date_str}&endDate={end_date_str}"
    try:
        response = requests.get(url, headers=fireant_headers, timeout=30)
        response.raise_for_status()
        df = pd.DataFrame(response.json())
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        df = df.set_index('Date')
        return df[['Close', 'Volume']]
    except Exception as e:
        print(f"Lỗi FireAnt cho {symbol}: {e}")
        return pd.DataFrame()

def fetch_investor_chart_data():
    print("Đang tải dữ liệu lịch sử dòng tiền 4 nhóm NĐT từ FITRADE...")
    url = "https://wl-market.fiintrade.vn/MoneyFlow/GetStatisticInvestorChart?language=vi&Code=VNINDEX&Frequently=Daily"
    try:
        response = requests.get(url, headers=fitrade_headers, timeout=30)
        response.raise_for_status()
        data = response.json().get('items', [])
        if not data: return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['tradingDate']).dt.date
        df = df.set_index('Date')
        
        df['foreign_net'] = (df['foreignBuyValueMatched'] - df['foreignSellValueMatched']) / 1e9
        df['prop_net'] = (df['proprietaryTotalMatchBuyTradeValue'] - df['proprietaryTotalMatchSellTradeValue']) / 1e9
        df['individual_net'] = (df['localIndividualBuyMatchValue'] - df['localIndividualSellMatchValue']) / 1e9
        df['institution_net'] = -(df['foreign_net'] + df['prop_net'] + df['individual_net'])
        
        return df[['foreign_net', 'prop_net', 'individual_net', 'institution_net']]
    except Exception as e:
        print(f"Lỗi FITRADE (4 nhóm NĐT): {e}")
        return pd.DataFrame()

def fetch_fitrade_industry_flow(years):
    print(f"Đang tải {years} năm dữ liệu dòng tiền ngành từ FITRADE...")
    end_date_str = datetime.now().strftime('%d-%m-%Y')
    start_date_str = (datetime.now() - timedelta(days=years * 365)).strftime('%d-%m-%Y')
    ind_codes = "8350,8630,8770,2350,1750,9530,3350,6570,8980,3720,2710,1350,5370,4570,2770,3780,3530,2730,3570,2720,7570,2750,3760,2790,5550,6530,4530,1730,1770,5330,0530,0570,3740"
    url = f"https://apigw.fitrade.vn/pbapi/api/indActiveBuySell?indCode={ind_codes}&FromDate={start_date_str}&Todate={end_date_str}"
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/50'}, timeout=30)
        response.raise_for_status()
        data = response.json()['data']
        df = pd.DataFrame(data)
        df['tradeDate'] = pd.to_datetime(df['tradeDate']).dt.date
        
        df = df.sort_values(by=['indName', 'tradeDate'])
        df['netActiveBuy_daily'] = df.groupby('indName')['netActiveBuy'].diff().fillna(df['netActiveBuy']) / 1e9
        
        pivot_df = df.pivot_table(index='tradeDate', columns='indName', values='netActiveBuy_daily', aggfunc='sum')
        pivot_df.index.name = 'Date'
        
        top_5_industries = pivot_df.abs().sum().nlargest(5).index
        pivot_df = pivot_df[top_5_industries].add_prefix('industry_flow_')
        return pivot_df
    except Exception as e:
        print(f"Lỗi FITRADE (Ngành): {e}")
        return pd.DataFrame()

def create_features(df):
    print("Đang tạo features cho mô hình...")
    df['target'] = df['vnindex_close'].shift(-1)
    
    for col in df.columns:
        if 'target' not in col:
            for i in range(1, 6): df[f'{col}_lag_{i}'] = df[col].shift(i)
            df[f'{col}_roll_mean_5'] = df[col].rolling(window=5).mean()

    df = df.dropna()
    features = [col for col in df.columns if col != 'target']
    X = df[features]
    y = df['target']
    
    print(f"Tạo features thành công. Kích thước dữ liệu huấn luyện: {X.shape}")
    return X, y, features

def train_model(X, y):
    print("Bắt đầu huấn luyện mô hình XGBoost V3.0...")
    model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, objective='reg:squarederror', random_state=42, n_jobs=-1, early_stopping_rounds=50)
    X_train, X_val, y_train, y_val = X[:-200], X[-200:], y[:-200], y[-200:]
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print("Huấn luyện thành công!")
    return model

if __name__ == "__main__":
    df_vnindex = fetch_fireant_data("HOSTC", YEARS_OF_DATA)
    df_vn30 = fetch_fireant_data("VN30", YEARS_OF_DATA)
    df_investor = fetch_investor_chart_data()
    df_industry = fetch_fitrade_industry_flow(YEARS_OF_DATA)

    if not df_vnindex.empty:
        df_vnindex = df_vnindex.rename(columns={'Close': 'vnindex_close', 'Volume': 'vnindex_volume'})
        df_vn30 = df_vn30[['Close']].rename(columns={'Close': 'vn30_close'})
        
        master_df = df_vnindex.join(df_vn30, how='inner')
        if not df_investor.empty: master_df = master_df.join(df_investor, how='left')
        if not df_industry.empty: master_df = master_df.join(df_industry, how='left')
        
        master_df = master_df.fillna(method='ffill').fillna(0)
        print(f"Ghép nối dữ liệu thành công. Tổng số dòng: {len(master_df)}")

        X, y, model_columns = create_features(master_df.copy())
        
        if len(X) < 300:
             print(f"LỖI: Dữ liệu sau khi xử lý quá ít ({len(X)} dòng). Dừng huấn luyện.")
        else:
            trained_model = train_model(X, y)
            joblib.dump((trained_model, model_columns), MODEL_FILENAME)
            print(f"\n--- HOÀN TẤT ---\nĐã lưu 'bộ não' V3.0 vào file '{MODEL_FILENAME}'!")
    else:
        print("Lỗi nghiêm trọng: Không tải được dữ liệu giá VNINDEX. Dừng chương trình.")