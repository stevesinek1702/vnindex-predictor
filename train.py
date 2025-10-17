import pandas as pd
import numpy as np
import requests
import xgboost as xgb
import joblib
from datetime import datetime, timedelta

# --- Cấu hình ---
API_SYMBOL = "HOSTC"  # HOSTC là mã của VNINDEX trên FireAnt
YEARS_OF_DATA = 5     # Lấy dữ liệu 5 năm để huấn luyện

def fetch_data(symbol, years):
    """Tải dữ liệu lịch sử từ API FireAnt."""
    print(f"Bắt đầu tải dữ liệu lịch sử {years} năm cho mã {symbol}...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    url = f"https://www.fireant.vn/api/Data/Markets/HistoricalQuotes?symbol={symbol}&startDate={start_date_str}&endDate={end_date_str}"
    
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()  # Báo lỗi nếu request không thành công
        data = response.json()
        
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by='Date', ascending=True)
        df.set_index('Date', inplace=True)
        
        # Đổi tên cột cho dễ sử dụng
        df = df.rename(columns={
            'Close': 'DongCua',
            'Volume': 'KhoiLuong'
        })
        print(f"Tải thành công {len(df)} phiên giao dịch.")
        return df[['DongCua', 'KhoiLuong']]
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu: {e}")
        return None

def create_features(df):
    """Tạo các feature (đặc điểm) cho mô hình học."""
    print("Đang xử lý dữ liệu và tạo features...")
    df['target'] = df['DongCua'].shift(-1) # Mục tiêu là dự đoán giá đóng cửa ngày mai
    
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

    # Feature: Chỉ số RSI (Relative Strength Index)
    delta = df['DongCua'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Xóa các dòng có giá trị rỗng (do tính lag và rolling mean)
    df = df.dropna()
    
    features = [col for col in df.columns if col != 'target']
    X = df[features]
    y = df['target']
    
    print(f"Tạo features thành công. Kích thước dữ liệu huấn luyện: {X.shape}")
    return X, y

def train_model(X, y):
    """Huấn luyện mô hình XGBoost và lưu lại."""
    print("Bắt đầu huấn luyện mô hình XGBoost... (Quá trình này có thể mất vài phút)")
    
    model = xgb.XGBRegressor(
        n_estimators=1000,      # Số lượng "cây" mô hình sẽ xây dựng
        learning_rate=0.05,     # Tốc độ học
        max_depth=5,            # Độ sâu tối đa của mỗi cây
        subsample=0.8,          # Tỷ lệ dữ liệu dùng cho mỗi cây
        colsample_bytree=0.8,   # Tỷ lệ feature dùng cho mỗi cây
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1               # Sử dụng tất cả CPU để tăng tốc
    )
    
    model.fit(X, y)
    
    print("Huấn luyện thành công!")
    return model

if __name__ == "__main__":
    # 1. Tải dữ liệu
    vnindex_df = fetch_data(API_SYMBOL, YEARS_OF_DATA)
    
    if vnindex_df is not None:
        # 2. Tạo features
        X, y = create_features(vnindex_df.copy())
        
        # 3. Huấn luyện mô hình
        trained_model = train_model(X, y)
        
        # 4. Lưu mô hình đã huấn luyện ra file
        joblib.dump(trained_model, 'vnindex_model.pkl')
        print("\n--- HOÀN TẤT ---")
        print("Đã huấn luyện và lưu 'bộ não' AI vào file 'vnindex_model.pkl' thành công!")
        print("Bạn đã sẵn sàng để chuyển sang bước tiếp theo.")