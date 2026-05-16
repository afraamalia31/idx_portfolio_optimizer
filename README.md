# 📈 IDX Portfolio Optimizer
### Optimasi Portofolio Saham BEI dengan Markowitz Efficient Frontier

> Proyek portfolio optimizer berbasis teori Modern Portfolio Theory (MPT) untuk saham-saham 
> Bursa Efek Indonesia (IDX), menggunakan PyPortfolioOpt, Yahoo Finance, dan Streamlit.

---

## 🏗️ Struktur Proyek

```
idx_portfolio_optimizer/
│
├── app.py                      # Aplikasi Streamlit utama
├── requirements.txt            # Dependencies Python
├── .streamlit/
│   └── config.toml             # Konfigurasi tema Streamlit
│
└── src/                        # Module-module inti
    ├── __init__.py
    ├── data_fetcher.py         # Unduh data saham dari Yahoo Finance
    ├── portfolio_optimizer.py  # Implementasi Markowitz & PyPortfolioOpt
    ├── visualizer.py           # Grafik interaktif dengan Plotly
    └── utils.py                # Fungsi pembantu & konstanta
```

---

## 🧠 Teori: Modern Portfolio Theory (Markowitz)

### Konsep Dasar
Harry Markowitz (1952) memperkenalkan teori bahwa investor yang rasional akan memilih 
portofolio yang **memaksimalkan return untuk tingkat risiko tertentu**, atau 
**meminimalkan risiko untuk tingkat return tertentu**.

### Formula Utama

**1. Return Portofolio:**
```
E(Rp) = Σ wi × E(Ri)
```
di mana:
- `E(Rp)` = Expected return portofolio
- `wi` = Bobot aset ke-i
- `E(Ri)` = Expected return aset ke-i

**2. Varians Portofolio (Risiko):**
```
σ²p = Σᵢ Σⱼ wi × wj × σij
```
di mana:
- `σ²p` = Varians portofolio
- `σij` = Kovarians antara aset i dan j

**3. Sharpe Ratio:**
```
SR = (E(Rp) - Rf) / σp
```
di mana:
- `Rf` = Risk-free rate (BI Rate)
- `σp` = Standar deviasi portofolio

**4. Efficient Frontier:**
Kumpulan portofolio yang memberikan:
- Return maksimum untuk setiap level risiko, ATAU
- Risiko minimum untuk setiap level return

---

## ⚙️ Instalasi

### Prerequisites
- Python 3.10+
- pip atau conda

### Langkah Instalasi

```bash
# 1. Clone atau buat direktori proyek
mkdir idx_portfolio_optimizer
cd idx_portfolio_optimizer

# 2. Buat virtual environment (sangat disarankan)
python -m venv venv

# Aktivasi di Windows
venv\Scripts\activate

# Aktivasi di macOS/Linux
source venv/bin/activate

# 3. Install semua dependencies
pip install -r requirements.txt

# 4. Jalankan aplikasi
streamlit run app.py
```

### Alternatif dengan Conda
```bash
conda create -n idx_portfolio python=3.11
conda activate idx_portfolio
pip install -r requirements.txt
streamlit run app.py
```

---

## 🚀 Cara Penggunaan

### Langkah 1: Pilih Saham
- Pilih **kategori saham** (LQ45, Perbankan, dll.) dari sidebar
- **Select** minimal 3 saham IDX dari daftar
- Atau ketik kode saham manual (format: `BBCA.JK`)

### Langkah 2: Atur Parameter
| Parameter | Deskripsi | Default |
|-----------|-----------|---------|
| Risk-Free Rate | BI Rate saat ini | 5.75% |
| Periode Data | Rentang data historis | 2 Tahun |
| Metode Optimasi | Max Sharpe / Min Vol / dll. | Max Sharpe |
| Bobot Min/Max | Batasan bobot per saham | 5%-40% |
| Monte Carlo | Jumlah simulasi portofolio acak | 5.000 |

### Langkah 3: Jalankan Optimasi
Klik **🚀 JALANKAN OPTIMASI** dan tunggu hasil analisis.

### Langkah 4: Interpretasi Hasil

**Tab Efficient Frontier:**
- Titik berwarna = portofolio acak dari Monte Carlo
- Garis hijau = Efficient Frontier (batas optimal)
- Garis emas = Capital Market Line
- Bintang merah ⭐ = Portofolio optimal yang dipilih

**Tab Alokasi Portofolio:**
- Donut chart distribusi bobot
- Tabel detail alokasi
- Simulasi investasi berdasarkan modal

**Tab Analisis Risiko:**
- Matriks korelasi antar saham
- Risk-return scatter per saham
- Statistik individual (Sharpe, Max DD, dll.)

**Tab Kinerja Historis:**
- Kumulatif return portofolio vs saham individual
- Rolling Sharpe Ratio (252 hari)

---

## 📚 Penjelasan Modul

### `src/data_fetcher.py`

```python
from src.data_fetcher import fetch_stock_data, get_stock_info

# Unduh data historis
price_data, failed = fetch_stock_data(
    tickers=["BBCA.JK", "TLKM.JK", "ASII.JK"],
    start_date=datetime(2022, 1, 1),
    end_date=datetime(2024, 1, 1)
)

# Info saham
info = get_stock_info("BBCA.JK")
print(info["nama"])  # "Bank Central Asia"
```

**Fitur:**
- Retry otomatis jika koneksi gagal
- Forward fill untuk hari libur pasar
- Drop saham dengan data < 80%

### `src/portfolio_optimizer.py`

```python
from src.portfolio_optimizer import PortfolioOptimizer

optimizer = PortfolioOptimizer(
    price_data=price_data,
    risk_free_rate=0.0575  # 5.75% BI Rate
)

# Estimasi parameter
# mu = expected returns (Ledoit-Wolf shrinkage)
# S  = covariance matrix

# 1. Optimasi Max Sharpe Ratio
weights, stats = optimizer.optimize(
    method="max_sharpe",
    weight_bounds=(0.05, 0.40)  # Min 5%, Max 40% per saham
)

# 2. Optimasi Min Volatilitas
weights, stats = optimizer.optimize(method="min_volatility")

# 3. Target Return Spesifik
weights, stats = optimizer.optimize(
    method="efficient_return",
    target_return=0.20  # 20% per tahun
)

# 4. Target Risiko Spesifik
weights, stats = optimizer.optimize(
    method="efficient_risk",
    target_volatility=0.15  # 15% volatilitas per tahun
)

# Simulasi Monte Carlo
mc_df = optimizer.monte_carlo_simulation(n_portfolios=5000)

# Titik Efficient Frontier
frontier = optimizer.get_efficient_frontier_points(n_points=100)

# Alokasi lot saham
allocation, leftover = optimizer.get_discrete_allocation(
    weights=weights,
    total_portfolio_value=10_000_000  # Rp 10 juta
)
```

**Output `stats`:**
```python
{
    "expected_return": 0.1850,    # 18.5% per tahun
    "volatility": 0.1420,         # 14.2% per tahun
    "sharpe_ratio": 0.898,        # (return - Rf) / vol
    "max_drawdown": -0.1850,      # Penurunan maksimum
    "var_95": -0.0142,            # Value at Risk 95%
    "sortino_ratio": 1.245,       # Sharpe berbasis downside
    "calmar_ratio": 1.001,        # Return / |Max DD|
    "n_assets": 6                 # Jumlah saham aktif
}
```

### `src/visualizer.py`

```python
from src.visualizer import (
    plot_efficient_frontier,
    plot_portfolio_weights,
    plot_correlation_heatmap,
    plot_cumulative_returns,
    plot_risk_return_scatter
)

# Semua fungsi mengembalikan go.Figure (Plotly)
fig = plot_efficient_frontier(mc_results, frontier_df, optimal_stats)
st.plotly_chart(fig, use_container_width=True)
```

### `src/utils.py`

```python
from src.utils import (
    SAHAM_IDX_POPULER,      # Dict kategori saham
    format_currency,         # "Rp 10.50jt"
    format_percentage,       # "15.25%"
    calculate_portfolio_stats,  # 15+ metrik statistik
    check_diversification,   # Evaluasi HHI
    get_sector_allocation    # Alokasi per sektor
)
```

---

## 📊 Metrik yang Dihitung

### Return Metrics
| Metrik | Formula | Keterangan |
|--------|---------|------------|
| Expected Return | E(Rp) = Σwi×E(Ri) | Return tahunan yang diharapkan |
| Total Return | (Pakhir - Pawal) / Pawal | Return kumulatif selama periode |

### Risk Metrics
| Metrik | Formula | Keterangan |
|--------|---------|------------|
| Volatilitas | σp×√252 | Standar deviasi tahunan |
| Max Drawdown | min(Pt/max(P0..Pt) - 1) | Penurunan puncak ke lembah |
| VaR 95% | Percentil ke-5 dari return | Kerugian maksimum 95% CI |
| CVaR/ES | E[R \| R < VaR] | Expected Shortfall |

### Risk-Adjusted Returns
| Metrik | Formula | Keterangan |
|--------|---------|------------|
| Sharpe Ratio | (Rp - Rf) / σp | Return per unit risiko total |
| Sortino Ratio | (Rp - Rf) / σdownside | Return per unit risiko negatif |
| Calmar Ratio | Rp / \|Max DD\| | Return vs drawdown maksimum |
| Information Ratio | (Rp - Rm) / TE | Aktif return vs tracking error |

---

## 🛠️ Konfigurasi Tambahan

### `.streamlit/config.toml`
```toml
[theme]
base = "dark"
backgroundColor = "#0a0e1a"
secondaryBackgroundColor = "#111827"
primaryColor = "#00d4aa"
textColor = "#f1f5f9"

[server]
maxUploadSize = 200

[browser]
gatherUsageStats = false
```

### Environment Variables (Opsional)
```bash
# .env
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=true
```

---

## ⚡ Tips Optimasi Performa

1. **Cache Data**: Data Yahoo Finance di-cache via `st.cache_data` selama 1 jam
2. **Solver**: Gunakan `CLARABEL` (default) untuk stabilitas, atau `ECOS` untuk kecepatan
3. **Monte Carlo**: Kurangi ke 1000 untuk respons lebih cepat saat development
4. **Lookback Period**: 2-3 tahun memberikan keseimbangan antara relevansi dan data cukup

---

## ⚠️ Disclaimer

> Aplikasi ini dibuat untuk **tujuan edukasi dan penelitian** saja.  
> Hasil optimasi **bukan merupakan saran investasi**.  
> Kinerja masa lalu tidak menjamin kinerja di masa mendatang.  
> Selalu lakukan analisis mendalam dan konsultasikan dengan profesional keuangan.

---

## 📖 Referensi

- Markowitz, H. (1952). "Portfolio Selection." *Journal of Finance*, 7(1), 77-91.
- Sharpe, W.F. (1966). "Mutual Fund Performance." *Journal of Business*, 39(1), 119-138.
- [PyPortfolioOpt Documentation](https://pyportfolioopt.readthedocs.io/)
- [Yahoo Finance API (yfinance)](https://pypi.org/project/yfinance/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [BEI (Indonesia Stock Exchange)](https://www.idx.co.id/)

---

## 🤝 Kontribusi

Pull request dan issue sangat disambut! Beberapa area pengembangan:
- [ ] Backtest dengan rolling rebalancing
- [ ] Black-Litterman model
- [ ] Factor investing (Fama-French)
- [ ] Integrasi dengan API broker Indonesia
- [ ] Notifikasi rebalancing otomatis
- [ ] Export laporan PDF

---

*Dibuat dengan ❤️ untuk komunitas investor Indonesia*
