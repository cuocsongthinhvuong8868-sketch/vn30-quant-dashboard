import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from vnstock import Quote
import time
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from datetime import date, datetime, timedelta
import os
from scipy.stats import t

st.set_page_config(page_title="VN30 Quant Monitor", layout="wide", page_icon="📈")

# ==========================================
# KHỞI TẠO SESSION STATE
# ==========================================
if 'df_c_raw' not in st.session_state:
    st.session_state.df_c_raw = pd.DataFrame()
if 'df_v_raw' not in st.session_state:
    st.session_state.df_v_raw = pd.DataFrame()
if 'is_loaded' not in st.session_state:
    st.session_state.is_loaded = False

# ==========================================
# HÀM TẢI DỮ LIỆU
# ==========================================
def load_market_data(start, end):
    api_key = "vnstock_17b56a86b930db526e25e8de447a0bfd"
    os.environ['VNSTOCK_API_KEY'] = api_key
    
    symbols = ['VN30', 'ACB', 'BCM', 'BID', 'BVH', 'CTG', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG', 
               'MBB', 'MSN', 'MWG', 'PLX', 'POW', 'SAB', 'SHB', 'SSB', 'SSI', 'STB', 'TCB', 
               'TPB', 'VCB', 'VHM', 'VIB', 'VIC', 'VJC', 'VNM', 'VPB', 'VRE']
    
    df_c = pd.DataFrame()
    df_v = pd.DataFrame()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, sym in enumerate(symbols):
        status_text.text(f"Đang tải dữ liệu: {sym} ({i+1}/{len(symbols)})...")
        try:
            q = Quote(symbol=sym, source='KBS') 
            df_hist = q.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
            if df_hist is not None and not df_hist.empty:
                df_hist['time'] = pd.to_datetime(df_hist['time'])
                df_hist.set_index('time', inplace=True)
                df_c[sym] = df_hist['close']
                df_v[sym] = df_hist['volume']
            time.sleep(3.5) 
        except Exception:
            time.sleep(3.5)
            pass
        progress_bar.progress((i + 1) / len(symbols))
        
    status_text.empty()
    progress_bar.empty()
    return df_c.ffill(), df_v.fillna(0)

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.header("📥 1. Tải Dữ liệu")
start_date = st.sidebar.date_input("Ngày bắt đầu", date(2018, 1, 1))
end_date = st.sidebar.date_input("Ngày kết thúc", date.today() + timedelta(days=1))

if st.sidebar.button("🚀 Tải dữ liệu (Chỉ bấm 1 lần)"):
    with st.spinner("Đang kéo dữ liệu từ VNStock..."):
        df_c, df_v = load_market_data(start_date, end_date)
        st.session_state.df_c_raw = df_c
        st.session_state.df_v_raw = df_v
        st.session_state.is_loaded = True
        st.success("Tải dữ liệu thành công!")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ 2. Kỹ thuật")

with st.sidebar.form("tech_params_form"):
    stdev_window = st.number_input("Cửa sổ Stdev / Số phiên giả lập MC", min_value=10, max_value=200, value=60, step=5)
    ma_window = st.number_input("Cửa sổ MA", min_value=10, max_value=500, value=125, step=5)
    ma_band_pct = st.number_input("Biên độ MA (± %)", min_value=1.0, max_value=30.0, value=5.0, step=0.5)
    rank_window = st.number_input("Cửa sổ Percent Rank", min_value=100, max_value=2000, value=750, step=50)
    submit_tech = st.form_submit_button("🔄 Cập nhật")

# ==========================================
# HÀM TÍNH TOÁN
# ==========================================
def process_data(df_close, df_volume, scenario_pct=0.0, stdev_win=60, vol_noise=1.5):
    df_c = df_close.copy()
    df_v = df_volume.copy()
    
    # GIẢ LẬP MONTE CARLO
    if scenario_pct != 0.0:
        last_date = df_c.index[-1]
        last_price = df_c['VN30'].iloc[-1]
        hist_returns = df_c['VN30'].pct_change().dropna()
        hist_vol = hist_returns.std()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=stdev_win, freq='B')
        
        np.random.seed(42) 
        noise = t.rvs(df=3, loc=0, scale=hist_vol * vol_noise, size=stdev_win) 
        noise = noise - np.mean(noise)
        daily_drift = (scenario_pct / 100.0) / stdev_win
        sim_returns = daily_drift + noise
        
        future_prices = []
        current_price = last_price
        for r in sim_returns:
            current_price *= (1 + r)
            future_prices.append(current_price)
            
        correction_factor = (last_price * (1 + scenario_pct / 100.0)) / future_prices[-1]
        lin_correction = np.linspace(1, correction_factor, stdev_win) 
        future_prices = future_prices * lin_correction

        avg_vol = df_v['VN30'].iloc[-stdev_win:].mean()
        sim_vols = avg_vol + t.rvs(df=3, loc=0, scale=df_v['VN30'].iloc[-stdev_win:].std(), size=stdev_win)
        sim_vols = np.maximum(sim_vols, avg_vol * 0.1) 
        
        for i, d in enumerate(future_dates):
            df_c.loc[d, 'VN30'] = future_prices[i]
            df_v.loc[d, 'VN30'] = sim_vols[i]
            
    df = pd.DataFrame(index=df_c.index)
    df['VN30'] = df_c['VN30']
    
    # QUAY LẠI SỬ DỤNG ROLLING (MƯỢT MÀ)
    df['MA'] = df['VN30'].rolling(window=ma_window).mean()
    df['Upper_Band'] = df['MA'] * (1 + ma_band_pct / 100)
    df['Lower_Band'] = df['MA'] * (1 - ma_band_pct / 100)
    df['Rank'] = df['VN30'].rolling(window=rank_window, min_periods=252).apply(lambda x: (x <= x[-1]).mean(), raw=True)
    
    df['Volatility'] = df['VN30'].pct_change().rolling(window=stdev_win).std()
    df['Volume_Std'] = df_v['VN30'].rolling(window=stdev_win).std()
    
    df['MA252'] = df['VN30'].rolling(window=252).mean()
    df['Dist_MA252'] = (df['VN30'] / df['MA252']) - 1
    
    window = 252 
    def rolling_zscore(series):
        return (series - series.rolling(window).mean()) / (series.rolling(window).std() + 1e-9)
        
    df['Z_Volat'] = rolling_zscore(df['Volatility'])
    df['Z_Vol'] = rolling_zscore(df['Volume_Std'])
    df['Z_Dist'] = rolling_zscore(df['Dist_MA252'])
    
    df = df.dropna()
    if df.empty: return df
    
    # 1. ISOLATION FOREST
    if_features = ['Z_Volat', 'Z_Vol']
    iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    iso_forest.fit(df[if_features]) 
    df['IF_Score'] = iso_forest.decision_function(df[if_features])
    
    # 2. AUTOENCODER
    ae_features = ['Z_Volat', 'Z_Vol', 'Z_Dist', 'Rank']
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[ae_features])
    
    autoencoder = MLPRegressor(hidden_layer_sizes=(4, 2, 4), activation='relu', solver='adam', max_iter=1000, random_state=42)
    autoencoder.fit(scaled_features, scaled_features)
    
    pred_features = autoencoder.predict(scaled_features)
    df['AE_Error'] = np.mean(np.square(scaled_features - pred_features), axis=1)
    
    # 🎯 FIX LỖI BÓP BIỂU ĐỒ: Cắt gọt các đỉnh quá dị biệt (Capping at 99.5th percentile)
    clip_threshold = df['AE_Error'].quantile(0.995)
    df['AE_Error'] = df['AE_Error'].clip(upper=clip_threshold)
    
    return df

# ==========================================
# HÀM VẼ BIỂU ĐỒ (UI NỀN TRẮNG TƯƠNG PHẢN CAO)
# ==========================================
def plot_chart_plotly(df, is_simulated=False):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.4, 0.2, 0.2, 0.2],
                        subplot_titles=("<b>Giá VN30 & Kênh xu hướng</b>", 
                                        "<b>Isolation Forest (Z-Vol & Z-Volat)</b>", 
                                        "<b>Autoencoder (Rủi ro cấu trúc & Khoảng cách cốt lõi)</b>", 
                                        "<b>Percent Rank (Vùng giá)</b>"))

    # --- 1. BIỂU ĐỒ GIÁ ---
    fig.add_trace(go.Scatter(x=df.index, y=df['VN30'], name='VN30 Index', line=dict(color='black', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA'], name='MA', line=dict(color='blue', width=1, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Upper_Band'], name='Upper Band', line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Lower_Band'], name='MA Band', fill='tonexty', fillcolor='rgba(0, 0, 255, 0.1)', line=dict(width=0), hoverinfo='skip'), row=1, col=1)

    if is_simulated:
        sim_start = df.index[-stdev_window]
        fig.add_trace(go.Scatter(x=df.index[-stdev_window:], y=df['VN30'].iloc[-stdev_window:], name='Giả lập MC', line=dict(color='red', width=2.5)), row=1, col=1)
        fig.add_vrect(x0=sim_start, x1=df.index[-1], fillcolor="yellow", opacity=0.2, layer="below", line_width=0)

    # --- 2. ISOLATION FOREST ---
    fig.add_trace(go.Scatter(x=df.index, y=df['IF_Score'], name='IF Score', line=dict(color='purple', width=1.2)), row=2, col=1)
    for y_val, color, dash in [(0.17, 'orange', 'dash'), (0.15, 'orange', 'dash'), (0.02, 'red', 'dot'), (-0.02, 'red', 'dot')]:
        fig.add_hline(y=y_val, line_dash=dash, line_color=color, line_width=1, row=2, col=1)
    fig.add_hrect(y0=0.15, y1=0.17, fillcolor="orange", opacity=0.15, layer="below", row=2, col=1)

    # --- 3. AUTOENCODER ---
    fig.add_trace(go.Scatter(x=df.index, y=df['AE_Error'], name='AE Error (MSE)', line=dict(color='crimson', width=1.2)), row=3, col=1)
    ae_threshold = df['AE_Error'].quantile(0.95)
    fig.add_hline(y=ae_threshold, line_dash="dash", line_color="red", line_width=1.5, annotation_text="<b>Ngưỡng 5% Đỉnh</b>", annotation_font_color="black", row=3, col=1)

    # --- 4. PERCENT RANK ---
    fig.add_trace(go.Scatter(x=df.index, y=df['Rank'], name='Percent Rank', line=dict(color='teal', width=1.5)), row=4, col=1)
    fig.add_hline(y=0.8, line_dash="dot", line_color="red", line_width=1, row=4, col=1)
    fig.add_hline(y=0.2, line_dash="dot", line_color="green", line_width=1, row=4, col=1)
    fig.add_hrect(y0=0.8, y1=1.0, fillcolor="red", opacity=0.1, layer="below", row=4, col=1)
    fig.add_hrect(y0=0.0, y1=0.2, fillcolor="green", opacity=0.1, layer="below", row=4, col=1)

    # --- ÉP GIAO DIỆN TRẮNG, CHỮ ĐEN TƯƠNG PHẢN CAO ---
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color="black", size=12),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="white", font_size=13, font_family="Arial", font_color="black"),
        height=950, 
        margin=dict(l=10, r=10, t=50, b=10),
        showlegend=False
    )
    
    fig.update_annotations(font_color="black", font_size=14)
    fig.update_xaxes(showgrid=True, gridcolor='lightgray', showline=True, linecolor='black', tickfont=dict(color='black', size=11), showspikes=True, spikemode="across", spikesnap="cursor")
    fig.update_yaxes(showgrid=True, gridcolor='lightgray', showline=True, linecolor='black', tickfont=dict(color='black', size=11))
    fig.update_xaxes(title_text="<b>Thời gian (Ngày giao dịch)</b>", title_font=dict(color="black", size=13), row=4, col=1)

    return fig

# ==========================================
# KHU VỰC CHÍNH (TABS)
# ==========================================
st.title("📊 VN30 Price Momentium Monitor : Structural Break Warrning")

if not st.session_state.is_loaded:
    st.info("👈 Vui lòng cấu hình ngày tháng và bấm nút 'Tải dữ liệu' ở thanh bên trái để bắt đầu.")
else:
    df_c_raw = st.session_state.df_c_raw
    df_v_raw = st.session_state.df_v_raw

    tab1, tab2 = st.tabs(["🔴 B. Go Live (Giám sát Real-time)", "🧪 C. Scenario Testing"])

    with tab1:
        if not df_c_raw.empty:
            df_live = process_data(df_c_raw, df_v_raw, scenario_pct=0.0, stdev_win=stdev_window)
            current_price = df_live['VN30'].iloc[-1]
            st.metric(label="VN30 Index", value=f"{current_price:.2f}")
            
            fig_live = plot_chart_plotly(df_live, is_simulated=False)
            st.plotly_chart(fig_live, use_container_width=True)

    with tab2:
        if not df_c_raw.empty:
            last_actual_price = df_c_raw['VN30'].iloc[-1]
            
            with st.form("scenario_form"):
                col1, col2 = st.columns(2)
                with col1:
                    scenario_val = st.slider(f"Xu hướng giá mục tiêu sau {stdev_window} phiên (%):", min_value=-30.0, max_value=30.0, value=0.0, step=5.0)
                with col2:
                    vol_noise_val = st.slider("Độ nhiễu biến động (Volatility Noise):", min_value=0.1, max_value=5.0, value=1.5, step=0.1)
                submit_scenario = st.form_submit_button("🚀 Chạy Giả lập")
            
            sim_price = last_actual_price * (1 + scenario_val/100)
            st.metric(label=f"Mục tiêu VN30", value=f"{sim_price:.2f}", delta=f"{scenario_val}%")
            
            if scenario_val != 0.0:
                df_sim = process_data(df_c_raw, df_v_raw, scenario_pct=scenario_val, stdev_win=stdev_window, vol_noise=vol_noise_val)
                fig_sim = plot_chart_plotly(df_sim, is_simulated=True)

                st.plotly_chart(fig_sim, use_container_width=True)
