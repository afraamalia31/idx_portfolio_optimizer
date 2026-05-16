"""
IDX Portfolio Optimizer - Main Application
Menggunakan Markowitz Efficient Frontier dengan PyPortfolioOpt dan Yahoo Finance
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import json
import io
import zipfile
import tempfile

# =========================
# CONFIG MLFLOW
# =========================
from dotenv import load_dotenv
import os
#setting dagshub
import dagshub
import mlflow

# load env
load_dotenv()

# (opsional) cek
print(os.getenv("MLFLOW_TRACKING_USERNAME"))

#  init dagshub (auto pakai env)
dagshub.init(
    repo_name="idx_portfolio_optimizer",
    repo_owner="numpangdesign4",
    mlflow=True
)

# cek tracking uri
print("TRACKING URI :", mlflow.get_tracking_uri())

mlflow.set_experiment("IDX Portfolio Optimizer")

# =========================
# Import modules lokal
from src.data_fetcher import fetch_stock_data, get_stock_info
from src.portfolio_optimizer import PortfolioOptimizer
from src.visualizer import (
    plot_efficient_frontier,
    plot_portfolio_weights,
    plot_correlation_heatmap,
    plot_cumulative_returns,
    plot_risk_return_scatter
)
from src.utils import (
    format_percentage,
    format_currency,
    calculate_portfolio_stats,
    SAHAM_IDX_POPULER
)
from src.eda import tampilkan_eda

# ─── Konfigurasi Halaman ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="IDX Portfolio Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

    :root {
        --bg-primary: #0a0e1a;
        --bg-card: #111827;
        --bg-card2: #1a2235;
        --accent-green: #00d4aa;
        --accent-blue: #3b82f6;
        --accent-gold: #f59e0b;
        --accent-red: #ef4444;
        --text-primary: #f1f5f9;
        --text-muted: #64748b;
        --border: #1e2d3d;
    }

    .stApp {
        background: var(--bg-primary);
        font-family: 'DM Sans', sans-serif;
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #0a0e1a 0%, #111827 50%, #0d1b2a 100%);
        border: 1px solid #1e3a5f;
        border-radius: 16px;
        padding: 32px 40px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(0,212,170,0.06) 0%, transparent 70%);
        pointer-events: none;
    }
    .main-header h1 {
        font-family: 'Space Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--accent-green);
        margin: 0;
        letter-spacing: -1px;
    }
    .main-header p {
        color: var(--text-muted);
        font-size: 0.95rem;
        margin: 8px 0 0 0;
        font-weight: 400;
    }
    .badge {
        display: inline-block;
        background: rgba(0,212,170,0.1);
        border: 1px solid rgba(0,212,170,0.3);
        color: var(--accent-green);
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem;
        padding: 3px 10px;
        border-radius: 20px;
        margin-right: 8px;
        margin-bottom: 14px;
    }

    /* Metric Cards */
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 16px;
    }
    .metric-label {
        font-size: 0.75rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .metric-value {
        font-family: 'Space Mono', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1;
    }
    .metric-value.positive { color: var(--accent-green); }
    .metric-value.negative { color: var(--accent-red); }
    .metric-value.neutral { color: var(--accent-blue); }

    /* Section Headers */
    .section-header {
        font-family: 'Space Mono', monospace;
        font-size: 0.8rem;
        color: var(--accent-green);
        text-transform: uppercase;
        letter-spacing: 2px;
        border-bottom: 1px solid var(--border);
        padding-bottom: 10px;
        margin: 24px 0 16px 0;
    }

    /* Portfolio Table */
    .portfolio-table {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }

    /* Info Box */
    .info-box {
        background: rgba(59,130,246,0.08);
        border: 1px solid rgba(59,130,246,0.25);
        border-radius: 10px;
        padding: 14px 18px;
        font-size: 0.85rem;
        color: #93c5fd;
        margin: 12px 0;
    }

    /* Warning Box */
    .warning-box {
        background: rgba(245,158,11,0.08);
        border: 1px solid rgba(245,158,11,0.25);
        border-radius: 10px;
        padding: 14px 18px;
        font-size: 0.85rem;
        color: #fcd34d;
        margin: 12px 0;
    }

    /* Split Info Box */
    .split-box {
        background: rgba(0,212,170,0.06);
        border: 1px solid rgba(0,212,170,0.25);
        border-radius: 10px;
        padding: 14px 18px;
        font-size: 0.85rem;
        color: #6ee7d4;
        margin: 12px 0;
        font-family: 'Space Mono', monospace;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--bg-card) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: var(--text-primary);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00d4aa, #0099ff);
        color: #0a0e1a;
        font-family: 'Space Mono', monospace;
        font-weight: 700;
        font-size: 0.85rem;
        border: none;
        border-radius: 8px;
        padding: 12px 28px;
        width: 100%;
        transition: all 0.2s;
        letter-spacing: 0.5px;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 25px rgba(0,212,170,0.3);
    }

    /* Selectbox & Inputs */
    .stMultiSelect [data-baseweb="select"] {
        background: var(--bg-card2) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-card);
        border-radius: 10px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
        color: var(--text-muted);
    }
    .stTabs [aria-selected="true"] {
        background: var(--accent-green) !important;
        color: #0a0e1a !important;
        border-radius: 8px;
    }

    /* Plotly Chart Container */
    .js-plotly-plot {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Streamlit overrides */
    .block-container { padding-top: 1rem; }
    h1, h2, h3 { color: var(--text-primary) !important; }
    .stSlider label, .stSelectbox label { color: var(--text-muted) !important; font-size: 0.85rem !important; }

    /* Footer */
    .footer {
        text-align: center;
        padding: 24px;
        color: var(--text-muted);
        font-size: 0.75rem;
        border-top: 1px solid var(--border);
        margin-top: 40px;
    }
    .footer span { color: var(--accent-green); }
</style>
""", unsafe_allow_html=True)


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div>
        <span class="badge">MARKOWITZ</span>
        <span class="badge">EFFICIENT FRONTIER</span>
        <span class="badge">IDX STOCKS</span>
    </div>
    <h1>📈 IDX Portfolio Optimizer</h1>
    <p>Optimasi portofolio saham Bursa Efek Indonesia menggunakan teori Modern Portfolio Theory (MPT) — 
    powered by PyPortfolioOpt & Yahoo Finance</p>
</div>
""", unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Konfigurasi")
    st.markdown("---")

    # Pemilihan Saham
    st.markdown("**📋 Pilih Saham IDX**")
    
    kategori = st.selectbox(
        "Kategori Saham",
        ["Semua Populer", "Blue Chip LQ45", "Perbankan", "Konsumer", "Energi", "Teknologi"]
    )
    
    saham_kategori = {
        "Semua Populer": SAHAM_IDX_POPULER["semua"],
        "Blue Chip LQ45": SAHAM_IDX_POPULER["lq45"],
        "Perbankan": SAHAM_IDX_POPULER["perbankan"],
        "Konsumer": SAHAM_IDX_POPULER["konsumer"],
        "Energi": SAHAM_IDX_POPULER["energi"],
        "Teknologi": SAHAM_IDX_POPULER["teknologi"],
    }
    
    default_saham = saham_kategori[kategori][:5]
    
    selected_stocks = st.multiselect(
        "Pilih saham (min. 3, maks. 15)",
        options=saham_kategori[kategori],
        default=default_saham,
        format_func=lambda x: x.replace(".JK", "")
    )

    # Tambah saham custom
    custom_stock = st.text_input("Atau ketik kode saham (cth: TLKM.JK)", "")
    if custom_stock and custom_stock not in selected_stocks:
        if not custom_stock.endswith(".JK"):
            custom_stock = custom_stock.upper() + ".JK"
        if st.button("➕ Tambahkan"):
            selected_stocks.append(custom_stock)

    st.markdown("---")
    
    # Periode Data
    st.markdown("**📅 Periode Data**")
    periode = st.selectbox(
        "Rentang Waktu",
        ["1 Tahun", "2 Tahun", "3 Tahun", "5 Tahun"],
        index=1
    )
    
    periode_map = {"1 Tahun": 365, "2 Tahun": 730, "3 Tahun": 1095, "5 Tahun": 1825}
    hari = periode_map[periode]
    end_date = datetime.today()
    start_date = end_date - timedelta(days=hari)

    st.markdown("---")
    
    # Parameter Optimasi
    st.markdown("**🎯 Parameter Optimasi**")
    
    risk_free_rate = st.slider(
        "Risk-Free Rate (BI Rate) %",
        min_value=0.0, max_value=10.0, value=5.75, step=0.25
    ) / 100
    
    metode_optimasi = st.radio(
        "Metode Optimasi",
        ["Max Sharpe Ratio", "Min Volatilitas", "Efficient Return", "Efficient Risk"],
        index=0
    )
    
    target_return = None
    target_risk = None
    
    if metode_optimasi == "Efficient Return":
        target_return = st.slider("Target Return (%/tahun)", 5, 50, 20, 1) / 100
    elif metode_optimasi == "Efficient Risk":
        target_risk = st.slider("Target Risiko (%/tahun)", 5, 40, 15, 1) / 100

    st.markdown("---")
    
    # Bobot Constraints
    st.markdown("**⚖️ Batasan Bobot**")
    min_weight = st.slider("Bobot Minimum per Saham (%)", 0, 20, 5, 1) / 100
    max_weight = st.slider("Bobot Maksimum per Saham (%)", 20, 100, 40, 5) / 100
    
    st.markdown("---")
    
    # Simulasi Monte Carlo
    n_portfolios = st.select_slider(
        "Simulasi Monte Carlo",
        options=[1000, 3000, 5000, 10000],
        value=5000
    )
    
    # Tombol Optimasi
    st.markdown("")
    run_optimizer = st.button("🚀 JALANKAN OPTIMASI", use_container_width=True)


# ─── Validasi Input ───────────────────────────────────────────────────────────
if len(selected_stocks) < 3:
    st.markdown("""
    <div class="info-box">
        ℹ️ <strong>Pilih minimal 3 saham</strong> dari sidebar untuk memulai optimasi portofolio.
        Teori Markowitz memerlukan diversifikasi antar aset.
    </div>
    """, unsafe_allow_html=True)

    # Tampilkan saham populer sebagai referensi
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">SAHAM IDX POPULER</div>', unsafe_allow_html=True)
        saham_display = pd.DataFrame({
            "Kode": ["BBCA", "BBRI", "TLKM", "ASII", "UNVR", "BMRI", "ICBP", "KLBF"],
            "Nama": ["Bank Central Asia", "Bank Rakyat Indonesia", "Telkom Indonesia",
                     "Astra International", "Unilever Indonesia", "Bank Mandiri",
                     "Indofood CBP", "Kalbe Farma"],
            "Sektor": ["Perbankan", "Perbankan", "Telekomunikasi", "Otomotif",
                       "Konsumer", "Perbankan", "Konsumer", "Farmasi"]
        })
        st.dataframe(saham_display, hide_index=True, use_container_width=True)
    with col2:
        st.markdown('<div class="section-header">CARA PENGGUNAAN</div>', unsafe_allow_html=True)
        st.markdown("""
        1. **Pilih Kategori** saham dari sidebar
        2. **Select** minimal 3 saham IDX
        3. **Atur parameter** optimasi sesuai preferensi
        4. Klik **🚀 JALANKAN OPTIMASI**
        5. Analisis hasil **Efficient Frontier**
        """)
    st.stop()


# ─── Fetch Data ───────────────────────────────────────────────────────────────
if run_optimizer or "portfolio_data" in st.session_state:
    
    if run_optimizer:
        with st.spinner("📡 Mengunduh data & menjalankan optimasi..."):
            try:
                # PERBAIKAN: unpack 3 nilai (price_data, failed_stocks, split)
                price_data, failed_stocks, split = fetch_stock_data(
                    selected_stocks,
                    start_date,
                    end_date,
                    train_ratio=0.70,
                    val_ratio=0.15
                )
                
                if failed_stocks:
                    st.markdown(f"""
                    <div class="warning-box">
                        ⚠️ Saham berikut tidak dapat diunduh dan dilewati: 
                        <strong>{', '.join([s.replace('.JK','') for s in failed_stocks])}</strong>
                    </div>
                    """, unsafe_allow_html=True)
                
                if price_data is None or price_data.shape[1] < 3:
                    st.error("❌ Data tidak cukup. Pastikan minimal 3 saham valid terpilih.")
                    st.stop()

                # Tampilkan info split jika berhasil
                if split is not None:
                    st.markdown(f"""
                    <div class="split-box">
                        ✂️ <strong>Split Data Berhasil</strong> &nbsp;|&nbsp;
                        TRAIN: {len(split.train)} hari ({split.train_start.date()} → {split.train_end.date()}) &nbsp;|&nbsp;
                        VAL: {len(split.val)} hari ({split.val_start.date()} → {split.val_end.date()}) &nbsp;|&nbsp;
                        TEST: {len(split.test)} hari ({split.test_start.date()} → {split.test_end.date()})
                    </div>
                    """, unsafe_allow_html=True)
                
                # ==========================================
                # MLFLOW TRACKING — satu run mencakup semua
                # ==========================================
                if mlflow.active_run():
                    mlflow.end_run()

                with mlflow.start_run(
                    run_name=f"{metode_optimasi}_{datetime.now().strftime('%H%M%S')}"
                ):
                    # ── PARAMS ──────────────────────────────
                    mlflow.log_param("selected_stocks",   ",".join(selected_stocks))
                    mlflow.log_param("periode",           periode)
                    mlflow.log_param("risk_free_rate",    risk_free_rate)
                    mlflow.log_param("optimization_method", metode_optimasi)
                    mlflow.log_param("min_weight",        min_weight)
                    mlflow.log_param("max_weight",        max_weight)
                    mlflow.log_param("n_portfolios",      n_portfolios)

                    # ── DATASET METRICS ──────────────────────
                    mlflow.log_metric("dataset_num_rows",    price_data.shape[0])
                    mlflow.log_metric("dataset_num_columns", price_data.shape[1])

                    # ── SPLIT METRICS ────────────────────────
                    if split is not None:
                        mlflow.log_metric("split_train_rows", len(split.train))
                        mlflow.log_metric("split_val_rows",   len(split.val))
                        mlflow.log_metric("split_test_rows",  len(split.test))

                    # ── DATASET ARTIFACT ─────────────────────
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        dataset_path = os.path.join(tmp_dir, "price_data.csv")
                        price_data.to_csv(dataset_path)
                        mlflow.log_artifact(dataset_path, artifact_path="dataset")

                    # ── OPTIMASI ─────────────────────────────
                    metode_map = {
                        "Max Sharpe Ratio": "max_sharpe",
                        "Min Volatilitas":  "min_volatility",
                        "Efficient Return": "efficient_return",
                        "Efficient Risk":   "efficient_risk",
                    }

                    data_untuk_optimasi = (
                        split.train if split is not None else price_data
                    )
                    optimizer = PortfolioOptimizer(
                        price_data=data_untuk_optimasi,
                        risk_free_rate=risk_free_rate,
                    )
                    mc_results = optimizer.monte_carlo_simulation(n_portfolios=n_portfolios)
                    optimal_weights, optimal_stats = optimizer.optimize(
                        method=metode_map[metode_optimasi],
                        weight_bounds=(min_weight, max_weight),
                        target_return=target_return,
                        target_volatility=target_risk,
                    )
                    frontier_df = optimizer.get_efficient_frontier_points(
                        n_points=100,
                        weight_bounds=(min_weight, max_weight),
                    )

                    # ── OPTIMIZATION METRICS ─────────────────
                    mlflow.log_metric("opt_expected_return",   optimal_stats["expected_return"])
                    mlflow.log_metric("opt_volatility",        optimal_stats["volatility"])
                    mlflow.log_metric("opt_sharpe_ratio",      optimal_stats["sharpe_ratio"])
                    if "max_drawdown" in optimal_stats:
                        mlflow.log_metric("opt_max_drawdown", optimal_stats["max_drawdown"])
                    if "sortino_ratio" in optimal_stats:
                        mlflow.log_metric("opt_sortino_ratio", optimal_stats["sortino_ratio"])
                    if "calmar_ratio" in optimal_stats:
                        mlflow.log_metric("opt_calmar_ratio",  optimal_stats["calmar_ratio"])
                    if "var_95" in optimal_stats:
                        mlflow.log_metric("opt_var_95",        optimal_stats["var_95"])

                    # Monte Carlo best portfolio
                    mc_best = mc_results.loc[mc_results["sharpe"].idxmax()]
                    mlflow.log_metric("mc_best_return",     mc_best["return"])
                    mlflow.log_metric("mc_best_volatility", mc_best["volatility"])
                    mlflow.log_metric("mc_best_sharpe",     mc_best["sharpe"])
                    mlflow.log_metric("frontier_max_sharpe", frontier_df["sharpe"].max())
                    mlflow.log_metric(
                        "sharpe_improvement",
                        optimal_stats["sharpe_ratio"] - mc_best["sharpe"],
                    )

                    # ── WEIGHTS ARTIFACT ─────────────────────
                    weights_df_mlf = pd.DataFrame(
                        {"Stock": list(optimal_weights.keys()),
                         "Weight": list(optimal_weights.values())}
                    )
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        weights_path = os.path.join(tmp_dir, "optimal_weights.csv")
                        weights_df_mlf.to_csv(weights_path, index=False)
                        mlflow.log_artifact(weights_path, artifact_path="weights")

                st.session_state.optimizer       = optimizer
                st.session_state.mc_results      = mc_results
                st.session_state.optimal_weights = optimal_weights
                st.session_state.optimal_stats   = optimal_stats
                st.session_state.frontier_df     = frontier_df
                st.session_state.price_data      = price_data
                st.session_state.split           = split
                st.session_state.config         = {
                    "risk_free_rate": risk_free_rate,
                    "metode":         metode_optimasi,
                    "target_return":  target_return,
                    "target_risk":    target_risk,
                    "min_weight":     min_weight,
                    "max_weight":     max_weight,
                    "n_portfolios":   n_portfolios,
                    "periode":        periode,
                    "start_date":     start_date,
                    "end_date":       end_date,
                }
                st.session_state.portfolio_data = True

            except Exception as e:
                st.error(f"❌ Error saat fetch data atau optimasi: {str(e)}")
                st.stop()
    
    # Ambil dari session state
    price_data     = st.session_state.price_data
    split          = st.session_state.get("split", None)  # ✅ aman jika belum ada
    config         = st.session_state.config
    optimizer      = st.session_state.optimizer
    mc_results     = st.session_state.mc_results
    optimal_weights = st.session_state.optimal_weights
    optimal_stats  = st.session_state.optimal_stats
    frontier_df    = st.session_state.frontier_df
    
    # ─── Metric Summary ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">RINGKASAN PORTOFOLIO OPTIMAL</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    metrics = [
        ("Expected Return", optimal_stats["expected_return"], "%", "positive"),
        ("Volatilitas", optimal_stats["volatility"], "%", "neutral"),
        ("Sharpe Ratio", optimal_stats["sharpe_ratio"], "x", "positive" if optimal_stats["sharpe_ratio"] > 1 else "neutral"),
        ("Max Drawdown", optimal_stats.get("max_drawdown", 0), "%", "negative"),
        ("Jumlah Saham", len([k for k, v in optimal_weights.items() if v > 0.01]), "", "neutral"),
    ]
    
    for col, (label, value, unit, cls) in zip([col1, col2, col3, col4, col5], metrics):
        with col:
            if unit == "%":
                display_val = f"{value*100:.2f}%"
            elif unit == "x":
                display_val = f"{value:.3f}"
            else:
                display_val = f"{int(value)}"
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value {cls}">{display_val}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # ─── Tabs Utama ───────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🎯 Efficient Frontier",
        "🥧 Alokasi Portofolio",
        "📊 Analisis Risiko",
        "📈 Kinerja Historis",
        "✂️ Split Data",   # tab5
        "🔍 EDA",           # tab6
        "📋 Data & Statistik"  # tab7
    ])
    
    with tab1:
        st.markdown("#### Markowitz Efficient Frontier")
        st.markdown("""
        <div class="info-box">
            Setiap titik pada grafik mewakili sebuah portofolio acak. 
            <strong>Efficient Frontier</strong> (garis hijau) adalah batas tertinggi return untuk setiap level risiko.
            Bintang ⭐ menunjukkan portofolio optimal yang dipilih.
        </div>
        """, unsafe_allow_html=True)
        
        fig_frontier = plot_efficient_frontier(
            mc_results=mc_results,
            frontier_df=frontier_df,
            optimal_stats=optimal_stats,
            risk_free_rate=config["risk_free_rate"]
        )
        st.plotly_chart(fig_frontier, use_container_width=True)
    
    with tab2:
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.markdown("#### Alokasi Bobot Optimal")
            fig_weights = plot_portfolio_weights(optimal_weights)
            st.plotly_chart(fig_weights, use_container_width=True)
        
        with col_right:
            st.markdown("#### Detail Alokasi")
            
            # Tabel alokasi
            weights_df = pd.DataFrame([
                {
                    "Saham": k.replace(".JK", ""),
                    "Bobot (%)": f"{v*100:.2f}%",
                    "Nilai (Rp 100jt)": format_currency(v * 100_000_000),
                    "Bar": v
                }
                for k, v in sorted(optimal_weights.items(), key=lambda x: x[1], reverse=True)
                if v > 0.001
            ])
            
            st.dataframe(
                weights_df.drop("Bar", axis=1),
                hide_index=True,
                use_container_width=True
            )
            
            # Rekomendasi investasi
            st.markdown("**💡 Simulasi Investasi**")
            modal = st.number_input(
                "Modal Investasi (Rp)",
                min_value=1_000_000,
                max_value=10_000_000_000,
                value=10_000_000,
                step=1_000_000,
                format="%d"
            )
            
            st.markdown("**Alokasi Per Saham:**")
            for k, v in sorted(optimal_weights.items(), key=lambda x: x[1], reverse=True):
                if v > 0.001:
                    alokasi = v * modal
                    st.markdown(f"`{k.replace('.JK','')}` → **{format_currency(alokasi)}** ({v*100:.1f}%)")
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Matriks Korelasi")
            fig_corr = plot_correlation_heatmap(price_data)
            st.plotly_chart(fig_corr, use_container_width=True)
        
        with col2:
            st.markdown("#### Risk-Return per Saham")
            fig_scatter = plot_risk_return_scatter(price_data, optimal_weights)
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Statistik per saham
        st.markdown("#### Statistik Individual Saham")
        individual_stats = []
        returns = price_data.pct_change().dropna()
        
        for col in price_data.columns:
            ret = returns[col]
            ann_return = (1 + ret.mean()) ** 252 - 1
            ann_vol = ret.std() * np.sqrt(252)
            sharpe = (ann_return - config["risk_free_rate"]) / ann_vol if ann_vol > 0 else 0
            max_dd = ((price_data[col] / price_data[col].cummax()) - 1).min()
            
            individual_stats.append({
                "Saham": col.replace(".JK", ""),
                "Return/Tahun": f"{ann_return*100:.2f}%",
                "Volatilitas/Tahun": f"{ann_vol*100:.2f}%",
                "Sharpe Ratio": f"{sharpe:.3f}",
                "Max Drawdown": f"{max_dd*100:.2f}%",
                "Bobot Optimal": f"{optimal_weights.get(col, 0)*100:.2f}%"
            })
        
        df_stats = pd.DataFrame(individual_stats)
        st.dataframe(df_stats, hide_index=True, use_container_width=True)
    
    with tab4:
        st.markdown("#### Kinerja Historis Portofolio vs Benchmark")
        
        fig_returns = plot_cumulative_returns(price_data, optimal_weights)
        st.plotly_chart(fig_returns, use_container_width=True)
        
        # Rolling stats
        st.markdown("#### Rolling Sharpe Ratio (252 Hari)")
        returns_port = price_data.pct_change().dropna()
        port_daily_returns = sum(
            returns_port[col] * optimal_weights.get(col, 0) 
            for col in returns_port.columns
        )
        
        rolling_sharpe = port_daily_returns.rolling(252).apply(
            lambda x: (x.mean() * 252 - config["risk_free_rate"]) / (x.std() * np.sqrt(252))
            if x.std() > 0 else 0
        )
        
        fig_rolling = go.Figure()
        fig_rolling.add_trace(go.Scatter(
            x=rolling_sharpe.index,
            y=rolling_sharpe.values,
            fill='tozeroy',
            fillcolor='rgba(0,212,170,0.1)',
            line=dict(color='#00d4aa', width=2),
            name='Rolling Sharpe (252d)'
        ))
        fig_rolling.add_hline(y=1, line_dash="dash", line_color="#f59e0b",
                              annotation_text="Sharpe = 1.0")
        fig_rolling.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(17,24,39,1)',
            plot_bgcolor='rgba(17,24,39,1)',
            height=350,
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis=dict(gridcolor='#1e2d3d'),
            yaxis=dict(gridcolor='#1e2d3d', title="Sharpe Ratio"),
            font=dict(family='DM Sans', color='#64748b')
        )
        st.plotly_chart(fig_rolling, use_container_width=True)

    # ─── Tab Baru: Split Data ─────────────────────────────────────────────────
    with tab5:
        st.markdown("#### ✂️ Informasi Split Data (70% / 15% / 15%)")

        if split is None:
            st.markdown("""
            <div class="warning-box">
                ⚠️ Split data tidak tersedia. Data historis mungkin terlalu sedikit 
                (minimal 100 hari perdagangan diperlukan).
            </div>
            """, unsafe_allow_html=True)
        else:
            # Ringkasan split
            col_a, col_b, col_c = st.columns(3)
            for col_ui, nama, data_split, pct in [
                (col_a, "🟢 TRAIN", split.train, "70%"),
                (col_b, "🟡 VALIDATION", split.val, "15%"),
                (col_c, "🔴 TEST", split.test, "15%"),
            ]:
                with col_ui:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{nama} ({pct})</div>
                        <div class="metric-value neutral">{len(data_split)}</div>
                        <div style="color:#64748b; font-size:0.8rem; margin-top:6px;">
                            hari perdagangan<br>
                            {data_split.index[0].date()} → {data_split.index[-1].date()}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")

            # Perbandingan statistik per split
            st.markdown("#### Perbandingan Statistik Return Antar Split")
            split_stats = []
            for nama, data_split in [("Train (70%)", split.train), ("Val (15%)", split.val), ("Test (15%)", split.test)]:
                ret = data_split.pct_change().dropna()
                port_ret = sum(
                    ret[col] * optimal_weights.get(col, 0)
                    for col in ret.columns
                    if col in optimal_weights
                )
                ann_ret   = (1 + port_ret.mean()) ** 252 - 1
                ann_vol   = port_ret.std() * np.sqrt(252)
                sharpe    = (ann_ret - config["risk_free_rate"]) / ann_vol if ann_vol > 0 else 0
                max_dd    = ((( 1 + port_ret).cumprod() / (1 + port_ret).cumprod().cummax()) - 1).min()

                split_stats.append({
                    "Split": nama,
                    "Periode": f"{data_split.index[0].date()} → {data_split.index[-1].date()}",
                    "Jumlah Hari": len(data_split),
                    "Ann. Return": f"{ann_ret*100:.2f}%",
                    "Volatilitas": f"{ann_vol*100:.2f}%",
                    "Sharpe Ratio": f"{sharpe:.3f}",
                    "Max Drawdown": f"{max_dd*100:.2f}%",
                })

            df_split_stats = pd.DataFrame(split_stats)
            st.dataframe(df_split_stats, hide_index=True, use_container_width=True)

            st.markdown("""
            <div class="info-box">
                💡 <strong>Cara membaca tabel ini:</strong> Optimasi portofolio dilakukan 
                menggunakan data <strong>TRAIN</strong>. Kolom <strong>VAL</strong> dan 
                <strong>TEST</strong> menunjukkan bagaimana bobot optimal tersebut bekerja 
                pada data yang belum pernah dilihat sebelumnya — semakin konsisten hasilnya, 
                semakin andal strategi ini.
            </div>
            """, unsafe_allow_html=True)

    # ─── Tab EDA ─────────────────────────────────────────────────────────────
    with tab6:
        tampilkan_eda(price_data, risk_free_rate=config["risk_free_rate"])

    with tab7:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Harga Historis (Normalized)")
            price_norm = price_data / price_data.iloc[0] * 100
            fig_price = go.Figure()
            colors = px.colors.qualitative.Set3
            for i, col in enumerate(price_norm.columns):
                fig_price.add_trace(go.Scatter(
                    x=price_norm.index,
                    y=price_norm[col],
                    name=col.replace(".JK", ""),
                    line=dict(color=colors[i % len(colors)], width=1.5)
                ))
            fig_price.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(17,24,39,1)',
                plot_bgcolor='rgba(17,24,39,1)',
                height=400,
                margin=dict(l=0, r=0, t=20, b=0),
                xaxis=dict(gridcolor='#1e2d3d'),
                yaxis=dict(gridcolor='#1e2d3d', title="Harga (Base=100)"),
                font=dict(family='DM Sans', color='#64748b'),
                legend=dict(bgcolor='rgba(0,0,0,0)')
            )
            st.plotly_chart(fig_price, use_container_width=True)
        
        with col2:
            st.markdown("#### Raw Data Harga")
            st.dataframe(
                price_data.tail(30).round(2),
                use_container_width=True,
                height=400
            )
        
        # ─── Download Data ────────────────────────────────────────────────────
        st.markdown("#### ⬇️ Download Data")
        col_d1, col_d2 = st.columns(2)
        
        with col_d1:
            csv_price = price_data.to_csv().encode("utf-8")
            st.download_button(
                "📥 Download Harga (CSV)",
                data=csv_price,
                file_name="idx_harga_historis.csv",
                mime="text/csv"
            )
        
        with col_d2:
            weights_export = pd.DataFrame([
                {"Saham": k, "Bobot": v} 
                for k, v in optimal_weights.items()
            ])
            csv_weights = weights_export.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Bobot Optimal (CSV)",
                data=csv_weights,
                file_name="idx_bobot_optimal.csv",
                mime="text/csv"
            )

        st.markdown("---")

        # ─── Download Model Weights (JSON) ────────────────────────────────────
        st.markdown("#### 🤖 Download Model")

        col_m1, col_m2, col_m3 = st.columns(3)

        with col_m1:
            # Export bobot optimal sebagai JSON
            weights_json = json.dumps(
                {k: round(v, 6) for k, v in optimal_weights.items()},
                indent=2
            )
            st.download_button(
                label="📥 Model Weights (JSON)",
                data=weights_json,
                file_name="model_weights_optimal.json",
                mime="application/json",
                help="Bobot optimal portofolio hasil optimasi"
            )

        with col_m2:
            # Export statistik model sebagai JSON
            stats_export = {
                "method": metode_optimasi,
                "risk_free_rate": config["risk_free_rate"],
                "optimal_weights": {k: round(v, 6) for k, v in optimal_weights.items()},
                "performance": {
                    "expected_return": round(optimal_stats["expected_return"], 6),
                    "volatility": round(optimal_stats["volatility"], 6),
                    "sharpe_ratio": round(optimal_stats["sharpe_ratio"], 6),
                    "max_drawdown": round(optimal_stats.get("max_drawdown", 0), 6),
                    "sortino_ratio": round(optimal_stats.get("sortino_ratio", 0), 6),
                    "calmar_ratio": round(optimal_stats.get("calmar_ratio", 0), 6),
                    "var_95": round(optimal_stats.get("var_95", 0), 6),
                },
                "meta": {
                    "periode": periode,
                    "start_date": str(start_date.date()),
                    "end_date": str(end_date.date()),
                    "n_stocks": len([v for v in optimal_weights.values() if v > 0.01]),
                    "stocks": [k for k, v in optimal_weights.items() if v > 0.01],
                    "generated_at": datetime.now().isoformat(),
                }
            }
            stats_json = json.dumps(stats_export, indent=2)
            st.download_button(
                label="📊 Model Report (JSON)",
                data=stats_json,
                file_name="model_report.json",
                mime="application/json",
                help="Laporan lengkap model termasuk statistik dan metadata"
            )

        with col_m3:
            # Export semua file kode sumber sebagai ZIP
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                # Tambahkan source code utama
                for fname in ["app.py", "portfolio_optimizer.py",
                               "requirements.txt", "README.md"]:
                    if os.path.exists(fname):
                        zf.write(fname)

                # Tambahkan folder src/ jika ada
                if os.path.isdir("src"):
                    for root, dirs, files in os.walk("src"):
                        for file in files:
                            fp = os.path.join(root, file)
                            zf.write(fp)

                # Sertakan model_report.json di dalam ZIP
                zf.writestr("model_report.json", stats_json)
                zf.writestr("model_weights_optimal.json", weights_json)

            zip_buf.seek(0)
            st.download_button(
                label="📦 Download Semua (ZIP)",
                data=zip_buf,
                file_name="idx_portfolio_optimizer.zip",
                mime="application/zip",
                help="Source code + model weights dalam satu file ZIP"
            )

        st.markdown("---")

        # ─── Download MLflow Artifacts ────────────────────────────────────────
        st.markdown("#### 🧪 Download MLflow Artifacts")
        try:
            experiment = mlflow.get_experiment_by_name("IDX Portfolio Optimizer")
            if experiment:
                runs = mlflow.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["start_time DESC"],
                    max_results=1
                )
                if not runs.empty:
                    run_id = runs.iloc[0]["run_id"]
                    st.markdown(f"""
                    <div class="split-box">
                        📌 MLflow Run ID: <strong>{run_id}</strong><br>
                        🕐 Started: {runs.iloc[0]['start_time']}
                    </div>
                    """, unsafe_allow_html=True)

                    # Cari path artifact lokal
                    artifact_uri = runs.iloc[0].get("artifact_uri", "")
                    artifact_path = artifact_uri.replace("file://", "").strip()

                    col_mlf1, col_mlf2 = st.columns(2)

                    with col_mlf1:
                        if os.path.exists(artifact_path) and os.listdir(artifact_path):
                            mlf_zip_buf = io.BytesIO()
                            with zipfile.ZipFile(mlf_zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                                for root, dirs, files in os.walk(artifact_path):
                                    for file in files:
                                        fp = os.path.join(root, file)
                                        arcname = os.path.relpath(fp, artifact_path)
                                        zf.write(fp, arcname)
                            mlf_zip_buf.seek(0)
                            st.download_button(
                                label="📥 Download MLflow Artifacts (ZIP)",
                                data=mlf_zip_buf,
                                file_name=f"mlflow_artifacts_{run_id[:8]}.zip",
                                mime="application/zip",
                                help="Artifacts dari MLflow run terakhir"
                            )
                        else:
                            st.markdown("""
                            <div class="warning-box">
                                ⚠️ Artifact lokal tidak ditemukan. 
                                Jalankan ulang optimasi agar MLflow menyimpan artifact.
                            </div>
                            """, unsafe_allow_html=True)

                    with col_mlf2:
                        # Export run metrics sebagai CSV
                        metric_cols = [c for c in runs.columns if c.startswith("metrics.")]
                        param_cols  = [c for c in runs.columns if c.startswith("params.")]
                        mlf_export  = runs[["run_id", "start_time", "status"] + metric_cols + param_cols].copy()
                        mlf_export.columns = [c.replace("metrics.", "").replace("params.", "") for c in mlf_export.columns]
                        mlf_csv = mlf_export.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="📊 Download Run Metrics (CSV)",
                            data=mlf_csv,
                            file_name=f"mlflow_run_{run_id[:8]}.csv",
                            mime="text/csv",
                            help="Metrics dan parameter dari MLflow run"
                        )
                else:
                    st.markdown("""
                    <div class="warning-box">
                        ⚠️ Belum ada MLflow run. Jalankan optimasi terlebih dahulu.
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="info-box">
                    ℹ️ Experiment MLflow belum dibuat. 
                    Jalankan optimasi untuk membuat experiment.
                </div>
                """, unsafe_allow_html=True)
        except Exception as e_mlf:
            st.markdown(f"""
            <div class="warning-box">
                ⚠️ Tidak dapat mengakses MLflow: <code>{e_mlf}</code>
            </div>
            """, unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    IDX Portfolio Optimizer · Markowitz Efficient Frontier · 
    <span>PyPortfolioOpt</span> + <span>Yahoo Finance</span> + <span>Streamlit</span>
    <br>⚠️ Bukan saran investasi. Selalu lakukan due diligence sebelum berinvestasi.
</div>
""", unsafe_allow_html=True)
