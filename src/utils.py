"""
Utils Module
Fungsi pembantu dan konstanta untuk IDX Portfolio Optimizer
"""

import pandas as pd
import numpy as np
from typing import Any


# ─── Daftar Saham IDX ─────────────────────────────────────────────────────────
SAHAM_IDX_POPULER = {
    "semua": [
        "BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "BNGA.JK",
        "TLKM.JK", "ASII.JK", "UNVR.JK", "ICBP.JK", "KLBF.JK",
        "HMSP.JK", "GGRM.JK", "INDF.JK", "MNCN.JK", "SMGR.JK",
        "PTBA.JK", "ADRO.JK", "ANTM.JK", "TINS.JK", "INCO.JK",
        "EMTK.JK", "GOTO.JK", "BUKA.JK", "DEWA.JK", "MDKA.JK",
        "PGAS.JK", "MEDC.JK", "ELSA.JK", "AKRA.JK", "LPPF.JK"
    ],
    "lq45": [
        "BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK",
        "TLKM.JK", "ASII.JK", "UNVR.JK", "ICBP.JK",
        "KLBF.JK", "HMSP.JK", "INDF.JK", "SMGR.JK",
        "PTBA.JK", "ADRO.JK", "ANTM.JK", "PGAS.JK",
        "MEDC.JK", "AKRA.JK", "JSMR.JK", "WTON.JK"
    ],
    "perbankan": [
        "BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK",
        "BNGA.JK", "NISP.JK", "BDMN.JK", "BTPS.JK",
        "BJBR.JK", "BJTM.JK", "AGRO.JK", "MEGA.JK"
    ],
    "konsumer": [
        "UNVR.JK", "ICBP.JK", "INDF.JK", "KLBF.JK",
        "HMSP.JK", "GGRM.JK", "SIDO.JK", "MYOR.JK",
        "ULTJ.JK", "CPIN.JK", "JPFA.JK", "ACES.JK"
    ],
    "energi": [
        "ADRO.JK", "PTBA.JK", "PGAS.JK", "MEDC.JK",
        "INCO.JK", "ANTM.JK", "TINS.JK", "MDKA.JK",
        "ELSA.JK", "BSSR.JK", "ITMG.JK", "HRUM.JK"
    ],
    "teknologi": [
        "EMTK.JK", "GOTO.JK", "BUKA.JK", "DEWA.JK",
        "MNCN.JK", "MTDL.JK", "MLPT.JK", "MCAS.JK"
    ]
}

# Nama lengkap saham
NAMA_SAHAM = {
    "BBCA.JK": "Bank Central Asia",
    "BBRI.JK": "Bank Rakyat Indonesia",
    "BMRI.JK": "Bank Mandiri",
    "BBNI.JK": "Bank Negara Indonesia",
    "BNGA.JK": "Bank CIMB Niaga",
    "TLKM.JK": "Telkom Indonesia",
    "ASII.JK": "Astra International",
    "UNVR.JK": "Unilever Indonesia",
    "ICBP.JK": "Indofood CBP",
    "KLBF.JK": "Kalbe Farma",
    "HMSP.JK": "HM Sampoerna",
    "GGRM.JK": "Gudang Garam",
    "INDF.JK": "Indofood Sukses Makmur",
    "SMGR.JK": "Semen Indonesia",
    "PTBA.JK": "Bukit Asam",
    "ADRO.JK": "Adaro Energy",
    "ANTM.JK": "Aneka Tambang",
    "PGAS.JK": "Perusahaan Gas Negara",
    "MEDC.JK": "Medco Energi",
    "GOTO.JK": "GoTo Gojek Tokopedia",
    "EMTK.JK": "Elang Mahkota Teknologi",
    "BUKA.JK": "Bukalapak",
    "MDKA.JK": "Merdeka Copper Gold",
    "INCO.JK": "Vale Indonesia",
    "TINS.JK": "Timah"
}


# ─── Format Helpers ───────────────────────────────────────────────────────────
def format_percentage(value: float, decimals: int = 2) -> str:
    """Format angka desimal menjadi string persentase."""
    return f"{value * 100:.{decimals}f}%"


def format_currency(value: float, currency: str = "Rp") -> str:
    """Format angka menjadi string mata uang Indonesia."""
    if abs(value) >= 1_000_000_000:
        return f"{currency} {value/1_000_000_000:.2f}M"
    elif abs(value) >= 1_000_000:
        return f"{currency} {value/1_000_000:.2f}jt"
    elif abs(value) >= 1_000:
        return f"{currency} {value/1_000:.1f}rb"
    else:
        return f"{currency} {value:.0f}"


def format_large_number(value: float) -> str:
    """Format angka besar dengan singkatan."""
    if abs(value) >= 1e12:
        return f"{value/1e12:.2f}T"
    elif abs(value) >= 1e9:
        return f"{value/1e9:.2f}B"
    elif abs(value) >= 1e6:
        return f"{value/1e6:.2f}M"
    elif abs(value) >= 1e3:
        return f"{value/1e3:.2f}K"
    return str(value)


# ─── Statistik Portofolio ─────────────────────────────────────────────────────
def calculate_portfolio_stats(
    price_data: pd.DataFrame,
    weights: dict,
    risk_free_rate: float = 0.0575
) -> dict:
    """
    Hitung berbagai metrik statistik untuk portofolio.
    
    Parameters:
    -----------
    price_data : pd.DataFrame
        Data harga historis
    weights : dict
        Bobot per saham
    risk_free_rate : float
        Tingkat bunga bebas risiko
    
    Returns:
    --------
    dict : Berbagai metrik portofolio
    """
    returns = price_data.pct_change().dropna()
    
    # Return harian portofolio
    port_daily = sum(
        returns[col] * weights.get(col, 0) 
        for col in returns.columns
        if col in weights
    )
    
    # Return tahunan
    ann_return = (1 + port_daily.mean()) ** 252 - 1
    
    # Volatilitas tahunan
    ann_vol = port_daily.std() * np.sqrt(252)
    
    # Sharpe Ratio
    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0
    
    # Kumulatif return
    cumulative = (1 + port_daily).cumprod()
    
    # Max Drawdown
    rolling_max = cumulative.cummax()
    drawdown = cumulative / rolling_max - 1
    max_dd = drawdown.min()
    
    # Value at Risk (95% CI, harian)
    var_95_daily = np.percentile(port_daily, 5)
    var_95_annual = var_95_daily * np.sqrt(252)
    
    # Conditional VaR (Expected Shortfall)
    cvar_95 = port_daily[port_daily <= var_95_daily].mean()
    
    # Sortino Ratio
    downside = port_daily[port_daily < 0].std() * np.sqrt(252)
    sortino = (ann_return - risk_free_rate) / downside if downside > 0 else 0
    
    # Calmar Ratio
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0
    
    # Beta terhadap IHSG (menggunakan proxy sebagai equal-weight index)
    equal_weight = returns.mean(axis=1)
    covariance = port_daily.cov(equal_weight)
    variance = equal_weight.var()
    beta = covariance / variance if variance > 0 else 1.0
    
    # Alpha (Jensen's Alpha)
    mkt_return = (1 + equal_weight.mean()) ** 252 - 1
    alpha = ann_return - (risk_free_rate + beta * (mkt_return - risk_free_rate))
    
    # Tracking Error
    tracking_error = (port_daily - equal_weight).std() * np.sqrt(252)
    
    # Information Ratio
    active_return = ann_return - mkt_return
    info_ratio = active_return / tracking_error if tracking_error > 0 else 0
    
    # Win Rate
    win_rate = (port_daily > 0).mean()
    
    # Average Win & Loss
    avg_win = port_daily[port_daily > 0].mean() if (port_daily > 0).any() else 0
    avg_loss = port_daily[port_daily < 0].mean() if (port_daily < 0).any() else 0
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    
    return {
        # Return Metrics
        "ann_return": ann_return,
        "total_return": cumulative.iloc[-1] - 1,
        
        # Risk Metrics
        "ann_volatility": ann_vol,
        "max_drawdown": max_dd,
        "var_95_daily": var_95_daily,
        "var_95_annual": var_95_annual,
        "cvar_95": cvar_95,
        "downside_volatility": downside,
        
        # Risk-Adjusted Returns
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "info_ratio": info_ratio,
        
        # Market Stats
        "beta": beta,
        "alpha": alpha,
        "tracking_error": tracking_error,
        
        # Trading Stats
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        
        # Portfolio Info
        "n_assets": len([v for v in weights.values() if v > 0.01]),
        "herfindahl_index": sum(v**2 for v in weights.values()),  # Konsentrasi
    }


def calculate_rebalancing_cost(
    old_weights: dict,
    new_weights: dict,
    transaction_cost: float = 0.0015  # 0.15% typical IDX cost
) -> float:
    """
    Estimasi biaya rebalancing portofolio.
    
    Parameters:
    -----------
    old_weights : dict
        Bobot lama
    new_weights : dict
        Bobot baru
    transaction_cost : float
        Biaya transaksi per sisi (beli atau jual)
    
    Returns:
    --------
    float : Total biaya rebalancing sebagai persentase
    """
    all_tickers = set(old_weights.keys()) | set(new_weights.keys())
    total_turnover = sum(
        abs(new_weights.get(t, 0) - old_weights.get(t, 0))
        for t in all_tickers
    )
    return total_turnover * transaction_cost


def check_diversification(weights: dict, threshold: float = 0.3) -> dict:
    """
    Evaluasi diversifikasi portofolio.
    
    Parameters:
    -----------
    weights : dict
        Bobot per saham
    threshold : float
        Batas konsentrasi (default: 30%)
    
    Returns:
    --------
    dict : Evaluasi diversifikasi
    """
    n_assets = len([v for v in weights.values() if v > 0.01])
    hhi = sum(v**2 for v in weights.values())  # Herfindahl-Hirschman Index
    max_weight = max(weights.values())
    
    # Effective N (jumlah aset efektif)
    effective_n = 1 / hhi if hhi > 0 else 0
    
    return {
        "n_assets": n_assets,
        "hhi": hhi,
        "effective_n": effective_n,
        "max_weight": max_weight,
        "is_concentrated": max_weight > threshold,
        "diversification_score": min(effective_n / n_assets, 1.0) if n_assets > 0 else 0,
        "warning": max_weight > threshold
    }


def get_sector_allocation(weights: dict) -> pd.DataFrame:
    """
    Hitung alokasi per sektor berdasarkan bobot portofolio.
    
    Parameters:
    -----------
    weights : dict
        Bobot per saham
    
    Returns:
    --------
    pd.DataFrame : Alokasi per sektor
    """
    sektor_map = {
        "BBCA.JK": "Perbankan", "BBRI.JK": "Perbankan",
        "BMRI.JK": "Perbankan", "BBNI.JK": "Perbankan",
        "BNGA.JK": "Perbankan", "NISP.JK": "Perbankan",
        "TLKM.JK": "Telekomunikasi",
        "ASII.JK": "Otomotif & Komponen",
        "UNVR.JK": "Konsumer", "ICBP.JK": "Konsumer",
        "INDF.JK": "Konsumer", "KLBF.JK": "Farmasi",
        "HMSP.JK": "Rokok", "GGRM.JK": "Rokok",
        "SMGR.JK": "Material", "PTBA.JK": "Energi",
        "ADRO.JK": "Energi", "ANTM.JK": "Pertambangan",
        "INCO.JK": "Pertambangan", "MDKA.JK": "Pertambangan",
        "PGAS.JK": "Energi", "MEDC.JK": "Energi",
        "GOTO.JK": "Teknologi", "EMTK.JK": "Teknologi",
        "BUKA.JK": "Teknologi"
    }
    
    sektor_alloc = {}
    for ticker, weight in weights.items():
        sektor = sektor_map.get(ticker, "Lainnya")
        sektor_alloc[sektor] = sektor_alloc.get(sektor, 0) + weight
    
    df = pd.DataFrame([
        {"Sektor": k, "Bobot": v, "Bobot (%)": f"{v*100:.2f}%"}
        for k, v in sorted(sektor_alloc.items(), key=lambda x: x[1], reverse=True)
        if v > 0.001
    ])
    
    return df
