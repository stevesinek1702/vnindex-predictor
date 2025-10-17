import pandas as pd
import numpy as np
import requests
import xgboost as xgb
import joblib
from datetime import datetime, timedelta
import os

# --- Cấu hình ---
API_SYMBOL_INDEX = "HOSTC"
API_SYMBOL_VN30 = "VN30"
YEARS_OF_DATA = 5
MODEL_FILENAME = 'vnindex_model_v2.pkl' # Vẫn giữ tên file này để Render tự nhận diện

# Header cho API của FireAnt
fireant_headers = {'User-Agent': 'Mozilla/5.0'}

def fetch_fireant_data(symbol, years):
    print(f"Đang tải {years} năm dữ liệu giá cho {symbol} từ FireAnt...")
    end_date_str = datetime.now().strftime('%Y-%m-%d')
    start_date_str = (datetime.now() - timedelta(days=years * 365)).strftime('%Y-%m-%d')
    url = f"https://www.fireant.vn/api/Data/Markets/HistoricalQuotes?symbol={symbol}&startDate={start_date_str}&endDate={end_date_str}"
    try:
        response = requests.get(url, headers=fireant_headers)
        response.raise_for_status()
        df = pd.DataFrame(response.json())
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        df = df.set_index('Date')
        print(f"Tải thành công {len(df)} phiên giá cho {symbol}.")
        return df[['Close', 'Volume']]
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu FireAnt cho {symbol}: {e}")
        return pd.DataFrame()

def fetch_fitrade_industry_flow(years):
    print(f"Đang tải {years} năm dữ liệu dòng tiền ngành từ FITRADE...")
    end_date_str = datetime.now().strftime('%d-%m-%Y')
    start_date_str = (datetime.now() - timedelta(days=years * 365)).strftime('%d-%m-%Y')
    # Danh sách các mã ngành đã lọc
    ind_codes = "8350,8630,8770,2350,1750,9530,3350,6570,8980,3720,2710,1350,5370,4570,2770,3780,3530,2730,3570,2720,7570,2750,3760,2790,5550,6530,4530,1730,1770,5330,0530,0570,3740"
    url = f"https://apigw.fitrade.vn/pbapi/api/indActiveBuySell?indCode={ind_codes}&FromDate={start_date_str}&Todate={end_date_str}"
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/50'})
        response.raise_for_status()
        data = response.json()['data']
        df = pd.DataFrame(data)
        df['tradeDate'] = pd.to_datetime(df['tradeDate']).dt.date
        
        df = df.sort_values(by=['indName', 'tradeDate'])
        df['netActiveBuy_daily'] = df.groupby('indName')['netActiveBuy'].diff().fillna(df['netActiveBuy']) / 1e9 # Convert to billions
        
        pivot_df = df.pivot_table(index='tradeDate', columns='indName', values='netActiveBuy_daily', aggfunc='sum')
        pivot_df.index.name = 'Date'
        
        # Chỉ giữ lại 5 ngành có biến động dòng tiền lớn nhất trong_khoảng_thời gian huấn luyện
        top_5_industries = pivot_df.abs().sum().nlargest(5).index
        pivot_df = pivot_df[top_5_industries]
        pivot_df = pivot_df.add_prefix('industry_flow_')
        
        print(f"Tải và xử lý thành công dòng tiền cho top 5 ngành.")
        return pivot_df
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu dòng tiền ngành FITRADE: {e}")
        return pd.DataFrame()

def create_features(df):
    print("Đang tạo features (đặc điểm) cho mô hình...")
    df['target'] = df['vnindex_close'].shift(-1)
    
    # Tạo features từ tất cả các cột dữ liệu
    for col in df.columns:
        if 'target' not in col:
            # Lag features (dữ liệu của các ngày trước đó)
            for i in range(1, 4):
                df[f'{col}_lag_{i}'] = df[col].shift(i)
            # Rolling features (trung bình trượt)
            df[f'{col}_roll_mean_5'] = df[col].rolling(window=5).mean()
            df[f'{col}_roll_mean_10'] = df[col].rolling(window=10).mean()

    df = df.dropna()
    features = [col for col in df.columns if col != 'target']
    X = df[features]
    y = df['target']
    
    print(f"Tạo features thành công. Kích thước dữ liệu huấn luyện: {X.shape}")
    return X, y, features

def train_model(X, y):
    print("Bắt đầu huấn luyện mô hình XGBoost V2.3... (Quá trình này có thể mất vài phút)")
    model = xgb.XGBRegressor(
        n_estimators=1000, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8,
        objective='reg:squarederror', random_state=42, n_jobs=-1,
        early_stopping_rounds=50 # Dừng sớm nếu mô hình không cải thiện
    )
    
    # Chia dữ liệu để có tập validation cho early stopping
    X_train, X_val = X[:-200], X[-200:]
    y_train, y_val = y[:-200], y[-200:]
    
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print("Huấn luyện thành công!")
    return model

if __name__ == "__main__":
    df_vnindex = fetch_fireant_data(API_SYMBOL_INDEX, YEARS_OF_DATA)
    df_vn30 = fetch_fireant_data(API_SYMBOL_VN30, YEARS_OF_DATA)
    df_industry_flow = fetch_fitrade_industry_flow(YEARS_OF_DATA)
    
    df_vnindex = df_vnindex.rename(columns={'Close': 'vnindex_close', 'Volume': 'vnindex_volume'})
    df_vn30 = df_vn30[['Close']].rename(columns={'Close': 'vn30_close'})
    
    master_df = df_vnindex.join(df_vn30, how='inner')
    master_df = master_df.join(df_industry_flow, how='left')
    master_df = master_df.fillna(0)

    if not master_df.empty:
        X, y, model_columns = create_features(master_df.copy())
        trained_model = train_model(X, y)
        
        # Lưu cả mô hình và danh sách các cột đã dùng để huấn luyện
        joblib.dump((trained_model, model_columns), MODEL_FILENAME)
        print(f"\n--- HOÀN TẤT ---")
        print(f"Đã huấn luyện và lưu 'bộ não' V2.3 vào file '{MODEL_FILENAME}'!")
    else:
        print("Không có dữ liệu để huấn luyện. Vui lòng kiểm tra lại.")