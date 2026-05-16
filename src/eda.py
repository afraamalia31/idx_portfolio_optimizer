"""
EDA Module — Exploratory Data Analysis
Analisis eksploratif data saham IDX sebelum optimasi portofolio.

Modul ini menjawab pertanyaan:
  1. Seberapa lengkap dan bersih data kita?
  2. Bagaimana karakteristik return tiap saham?
  3. Saham mana yang paling untung / paling berisiko?
  4. Apakah ada pola atau anomali yang perlu diketahui?
  5. Seberapa terhubung antar saham?
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats as scipy_stats

# ─── Tema Warna (konsisten dengan visualizer.py) ──────────────────────────────
COLORS = {
    "bg":       "rgba(17,24,39,1)",
    "bg_paper": "rgba(10,14,26,1)",
    "grid":     "#1e2d3d",
    "text":     "#64748b",
    "green":    "#00d4aa",
    "blue":     "#3b82f6",
    "gold":     "#f59e0b",
    "red":      "#ef4444",
    "purple":   "#8b5cf6",
    "white":    "#f1f5f9",
    "orange":   "#f97316",
}

PALETTE = [
    "#00d4aa", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6",
    "#ec4899", "#06b6d4", "#84cc16", "#f97316", "#14b8a6"
]

def _base_layout(**kwargs) -> dict:
    base = dict(
        template="plotly_dark",
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        font=dict(family="DM Sans, sans-serif", color=COLORS["text"]),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    base.update(kwargs)
    return base

def _axis(title: str) -> dict:
    return dict(
        title=dict(text=title, font=dict(color=COLORS["text"])),
        gridcolor=COLORS["grid"],
        linecolor=COLORS["grid"],
    )


# ===========================================================================
# BAGIAN 1 — KUALITAS DATA
# ===========================================================================

def eda_kualitas_data(price_data: pd.DataFrame) -> dict:
    """
    Periksa kelengkapan dan kebersihan data harga saham.

    Parameters:
    -----------
    price_data : pd.DataFrame
        Data harga historis saham

    Returns:
    --------
    dict : Ringkasan kualitas data per saham
    """
    hasil = {}

    for col in price_data.columns:
        seri = price_data[col]
        nama = col.replace(".JK", "")

        total_hari     = len(seri)
        hari_kosong    = seri.isnull().sum()
        pct_lengkap    = (1 - hari_kosong / total_hari) * 100
        harga_min      = seri.min()
        harga_max      = seri.max()
        harga_terakhir = seri.dropna().iloc[-1]

        # Deteksi harga tidak wajar
        harga_nol_neg  = (seri <= 0).sum()
        # Deteksi lonjakan ekstrem (>50% dalam sehari)
        ret_harian     = seri.pct_change().dropna()
        lonjakan       = (ret_harian.abs() > 0.50).sum()

        hasil[nama] = {
            "Total Hari"        : total_hari,
            "Data Kosong"       : hari_kosong,
            "Kelengkapan (%)"   : round(pct_lengkap, 2),
            "Harga Min"         : round(harga_min, 2),
            "Harga Max"         : round(harga_max, 2),
            "Harga Terakhir"    : round(harga_terakhir, 2),
            "Harga Nol/Negatif" : harga_nol_neg,
            "Lonjakan Ekstrem"  : lonjakan,
            "Status"            : "✅ Baik" if pct_lengkap >= 95 and harga_nol_neg == 0
                                  else "⚠️ Perlu Cek",
        }

    return hasil


def plot_kelengkapan_data(price_data: pd.DataFrame) -> go.Figure:
    """Bar chart kelengkapan data (%) tiap saham."""
    kualitas = eda_kualitas_data(price_data)
    saham    = list(kualitas.keys())
    pct      = [kualitas[s]["Kelengkapan (%)"] for s in saham]
    warna    = [COLORS["green"] if p >= 95 else COLORS["gold"] for p in pct]

    fig = go.Figure(go.Bar(
        x=saham,
        y=pct,
        marker=dict(color=warna, line=dict(width=0)),
        text=[f"{p:.1f}%" for p in pct],
        textposition="outside",
        textfont=dict(color=COLORS["white"], size=11),
        hovertemplate="<b>%{x}</b><br>Kelengkapan: %{y:.2f}%<extra></extra>"
    ))

    fig.add_hline(y=95, line_dash="dash", line_color=COLORS["gold"],
                  annotation_text="Batas 95%", annotation_font_color=COLORS["gold"])

    fig.update_layout(**_base_layout(
        title=dict(text="Kelengkapan Data per Saham",
                   font=dict(size=14, color=COLORS["white"]), x=0.01),
        xaxis=_axis("Saham"),
        yaxis=dict(**_axis("Kelengkapan (%)"), range=[80, 102]),
        height=350
    ))
    return fig


# ===========================================================================
# BAGIAN 2 — STATISTIK DESKRIPTIF RETURN
# ===========================================================================

def eda_statistik_return(price_data: pd.DataFrame,
                          risk_free_rate: float = 0.0575) -> pd.DataFrame:
    """
    Hitung statistik deskriptif return harian dan tahunan tiap saham.

    Returns:
    --------
    pd.DataFrame : Tabel statistik lengkap
    """
    returns = price_data.pct_change().dropna()
    rows    = []

    for col in returns.columns:
        ret  = returns[col]
        nama = col.replace(".JK", "")

        ann_return  = (1 + ret.mean()) ** 252 - 1
        ann_vol     = ret.std() * np.sqrt(252)
        sharpe      = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0
        skewness    = ret.skew()
        kurtosis    = ret.kurt()
        max_dd      = ((price_data[col] / price_data[col].cummax()) - 1).min()
        best_day    = ret.max()
        worst_day   = ret.min()
        win_rate    = (ret > 0).mean()

        rows.append({
            "Saham"           : nama,
            "Return/Tahun"    : f"{ann_return*100:.2f}%",
            "Volatilitas/Tahun": f"{ann_vol*100:.2f}%",
            "Sharpe Ratio"    : f"{sharpe:.3f}",
            "Max Drawdown"    : f"{max_dd*100:.2f}%",
            "Hari Terbaik"    : f"{best_day*100:.2f}%",
            "Hari Terburuk"   : f"{worst_day*100:.2f}%",
            "Win Rate"        : f"{win_rate*100:.1f}%",
            "Skewness"        : f"{skewness:.3f}",
            "Kurtosis"        : f"{kurtosis:.3f}",
            # nilai mentah untuk sorting/plotting
            "_ann_return"     : ann_return,
            "_ann_vol"        : ann_vol,
            "_sharpe"         : sharpe,
            "_max_dd"         : max_dd,
        })

    return pd.DataFrame(rows)


def plot_return_vs_risiko(df_stats: pd.DataFrame,
                           risk_free_rate: float = 0.0575) -> go.Figure:
    """
    Scatter plot Return vs Volatilitas tiap saham.
    Ukuran titik = Sharpe Ratio.
    """
    fig = go.Figure()

    for i, row in df_stats.iterrows():
        sharpe   = row["_sharpe"]
        warna    = COLORS["green"] if sharpe > 1 else \
                   COLORS["blue"]  if sharpe > 0.5 else COLORS["red"]
        ukuran   = max(abs(sharpe) * 25, 12)

        fig.add_trace(go.Scatter(
            x=[row["_ann_vol"] * 100],
            y=[row["_ann_return"] * 100],
            mode="markers+text",
            text=[row["Saham"]],
            textposition="top center",
            textfont=dict(size=10, color=COLORS["white"]),
            marker=dict(size=ukuran, color=warna, opacity=0.85,
                        line=dict(color=COLORS["bg"], width=1.5)),
            name=row["Saham"],
            showlegend=False,
            hovertemplate=(
                f"<b>{row['Saham']}</b><br>"
                f"Return: {row['Return/Tahun']}<br>"
                f"Volatilitas: {row['Volatilitas/Tahun']}<br>"
                f"Sharpe: {row['Sharpe Ratio']}<extra></extra>"
            )
        ))

    fig.add_hline(y=risk_free_rate * 100, line_dash="dash",
                  line_color=COLORS["gold"], opacity=0.5,
                  annotation_text=f"BI Rate {risk_free_rate*100:.1f}%",
                  annotation_font_color=COLORS["gold"])

    fig.update_layout(**_base_layout(
        title=dict(text="Risk vs Return per Saham (Ukuran = |Sharpe Ratio|)",
                   font=dict(size=14, color=COLORS["white"]), x=0.01),
        xaxis=_axis("Volatilitas/Tahun (%)"),
        yaxis=_axis("Return/Tahun (%)"),
        height=420
    ))
    return fig


def plot_ranking_return(df_stats: pd.DataFrame) -> go.Figure:
    """Bar chart ranking return tahunan tiap saham."""
    df_sorted = df_stats.sort_values("_ann_return", ascending=True)
    warna     = [COLORS["green"] if v >= 0 else COLORS["red"]
                 for v in df_sorted["_ann_return"]]

    fig = go.Figure(go.Bar(
        x=df_sorted["_ann_return"] * 100,
        y=df_sorted["Saham"],
        orientation="h",
        marker=dict(color=warna, line=dict(width=0)),
        text=[f"{v*100:.2f}%" for v in df_sorted["_ann_return"]],
        textposition="outside",
        textfont=dict(color=COLORS["white"], size=10),
        hovertemplate="<b>%{y}</b><br>Return: %{x:.2f}%<extra></extra>"
    ))

    fig.add_vline(x=0, line_color=COLORS["text"], line_width=1)

    fig.update_layout(**_base_layout(
        title=dict(text="Ranking Return Tahunan",
                   font=dict(size=14, color=COLORS["white"]), x=0.01),
        xaxis=_axis("Return/Tahun (%)"),
        yaxis=dict(gridcolor=COLORS["grid"], linecolor=COLORS["grid"]),
        height=400
    ))
    return fig


# ===========================================================================
# BAGIAN 3 — DISTRIBUSI RETURN
# ===========================================================================

def plot_distribusi_return(price_data: pd.DataFrame) -> go.Figure:
    """
    Histogram distribusi return harian tiap saham
    dengan kurva normal sebagai pembanding.
    """
    returns = price_data.pct_change().dropna()
    n_cols  = min(len(returns.columns), 2)
    n_rows  = (len(returns.columns) + 1) // 2

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[c.replace(".JK", "") for c in returns.columns]
    )

    for idx, col in enumerate(returns.columns):
        row = idx // n_cols + 1
        col_pos = idx % n_cols + 1
        ret  = returns[col].dropna()
        nama = col.replace(".JK", "")

        # Histogram
        fig.add_trace(go.Histogram(
            x=ret * 100,
            nbinsx=60,
            name=nama,
            marker=dict(color=PALETTE[idx % len(PALETTE)], opacity=0.7,
                        line=dict(width=0)),
            showlegend=False,
            hovertemplate=f"{nama}<br>Return: %{{x:.2f}}%<br>Frekuensi: %{{y}}<extra></extra>",
            histnorm="probability density"
        ), row=row, col=col_pos)

        # Kurva normal teoritis
        mu_val  = ret.mean() * 100
        std_val = ret.std() * 100
        x_range = np.linspace(mu_val - 4 * std_val, mu_val + 4 * std_val, 200)
        y_norm  = scipy_stats.norm.pdf(x_range, mu_val, std_val)

        fig.add_trace(go.Scatter(
            x=x_range, y=y_norm,
            mode="lines",
            line=dict(color=COLORS["gold"], width=1.5),
            showlegend=False,
            hoverinfo="skip"
        ), row=row, col=col_pos)

    fig.update_layout(**_base_layout(
        title=dict(text="Distribusi Return Harian (Garis Emas = Normal Teoritis)",
                   font=dict(size=14, color=COLORS["white"]), x=0.01),
        height=max(300 * n_rows, 400),
        showlegend=False
    ))
    fig.update_xaxes(gridcolor=COLORS["grid"], linecolor=COLORS["grid"])
    fig.update_yaxes(gridcolor=COLORS["grid"], linecolor=COLORS["grid"])

    return fig


def plot_boxplot_return(price_data: pd.DataFrame) -> go.Figure:
    """Box plot return harian — deteksi outlier sekaligus."""
    returns = price_data.pct_change().dropna()

    fig = go.Figure()
    for i, col in enumerate(returns.columns):
        fig.add_trace(go.Box(
            y=returns[col] * 100,
            name=col.replace(".JK", ""),
            marker=dict(color=PALETTE[i % len(PALETTE)], size=3),
            line=dict(color=PALETTE[i % len(PALETTE)]),
            boxmean=True,
            hovertemplate="%{y:.3f}%<extra></extra>"
        ))

    fig.add_hline(y=0, line_color=COLORS["text"], line_width=1, opacity=0.5)

    fig.update_layout(**_base_layout(
        title=dict(text="Box Plot Return Harian (Titik di luar kotak = Outlier)",
                   font=dict(size=14, color=COLORS["white"]), x=0.01),
        xaxis=_axis("Saham"),
        yaxis=_axis("Return Harian (%)"),
        height=420,
        showlegend=False
    ))
    return fig


# ===========================================================================
# BAGIAN 4 — TREN & POLA WAKTU
# ===========================================================================

def plot_harga_normalized(price_data: pd.DataFrame) -> go.Figure:
    """Harga ternormalisasi — semua mulai dari 100."""
    norm = price_data / price_data.iloc[0] * 100

    fig = go.Figure()
    for i, col in enumerate(norm.columns):
        fig.add_trace(go.Scatter(
            x=norm.index,
            y=norm[col],
            name=col.replace(".JK", ""),
            line=dict(color=PALETTE[i % len(PALETTE)], width=1.8),
            hovertemplate=f"{col.replace('.JK','')}: %{{y:.1f}}<extra></extra>"
        ))

    fig.add_hline(y=100, line_dash="dot", line_color=COLORS["text"], opacity=0.4)

    fig.update_layout(**_base_layout(
        title=dict(text="Harga Ternormalisasi (Base = 100 di awal periode)",
                   font=dict(size=14, color=COLORS["white"]), x=0.01),
        xaxis=_axis("Tanggal"),
        yaxis=_axis("Harga (Base = 100)"),
        legend=dict(bgcolor="rgba(17,24,39,0.8)", bordercolor=COLORS["grid"],
                    borderwidth=1, font=dict(size=10)),
        height=420,
        hovermode="x unified"
    ))
    return fig


def plot_return_kumulatif(price_data: pd.DataFrame) -> go.Figure:
    """Return kumulatif tiap saham."""
    returns = price_data.pct_change().dropna()
    cum_ret = (1 + returns).cumprod()

    fig = go.Figure()
    for i, col in enumerate(cum_ret.columns):
        fig.add_trace(go.Scatter(
            x=cum_ret.index,
            y=cum_ret[col],
            name=col.replace(".JK", ""),
            line=dict(color=PALETTE[i % len(PALETTE)], width=1.8),
            hovertemplate=f"{col.replace('.JK','')}: %{{y:.3f}}x<extra></extra>"
        ))

    fig.add_hline(y=1, line_dash="dot", line_color=COLORS["gold"], opacity=0.5,
                  annotation_text="Modal Awal", annotation_font_color=COLORS["gold"])

    fig.update_layout(**_base_layout(
        title=dict(text="Return Kumulatif per Saham (1.0x = Modal Awal)",
                   font=dict(size=14, color=COLORS["white"]), x=0.01),
        xaxis=_axis("Tanggal"),
        yaxis=_axis("Return Kumulatif (x)"),
        legend=dict(bgcolor="rgba(17,24,39,0.8)", bordercolor=COLORS["grid"],
                    borderwidth=1, font=dict(size=10)),
        height=420,
        hovermode="x unified"
    ))
    return fig


def plot_rolling_volatilitas(price_data: pd.DataFrame,
                              window: int = 30) -> go.Figure:
    """Volatilitas bergulir (rolling) — lihat perubahan risiko dari waktu ke waktu."""
    returns  = price_data.pct_change().dropna()
    roll_vol = returns.rolling(window).std() * np.sqrt(252) * 100

    fig = go.Figure()
    for i, col in enumerate(roll_vol.columns):
        fig.add_trace(go.Scatter(
            x=roll_vol.index,
            y=roll_vol[col],
            name=col.replace(".JK", ""),
            line=dict(color=PALETTE[i % len(PALETTE)], width=1.5),
            opacity=0.8,
            hovertemplate=f"{col.replace('.JK','')}: %{{y:.2f}}%<extra></extra>"
        ))

    fig.update_layout(**_base_layout(
        title=dict(text=f"Rolling Volatilitas {window} Hari (Tahunan %)",
                   font=dict(size=14, color=COLORS["white"]), x=0.01),
        xaxis=_axis("Tanggal"),
        yaxis=_axis("Volatilitas (%)"),
        legend=dict(bgcolor="rgba(17,24,39,0.8)", bordercolor=COLORS["grid"],
                    borderwidth=1, font=dict(size=10)),
        height=380,
        hovermode="x unified"
    ))
    return fig


def plot_return_bulanan(price_data: pd.DataFrame) -> go.Figure:
    """
    Heatmap return bulanan per saham —
    deteksi pola musiman (bulan apa biasanya bagus/jelek).
    """
    returns = price_data.pct_change().dropna()

    # Rata-rata return per bulan
    monthly = returns.copy()
    monthly.index = pd.to_datetime(monthly.index)
    monthly_avg = monthly.groupby(monthly.index.month).mean() * 100

    bulan_label = ["Jan","Feb","Mar","Apr","Mei","Jun",
                   "Jul","Agu","Sep","Okt","Nov","Des"]
    saham_label = [c.replace(".JK","") for c in monthly_avg.columns]

    fig = go.Figure(go.Heatmap(
        z=monthly_avg.values.T,
        x=[bulan_label[m-1] for m in monthly_avg.index],
        y=saham_label,
        colorscale=[
            [0.0,  "#ef4444"],
            [0.5,  "#1a2235"],
            [1.0,  "#00d4aa"],
        ],
        zmid=0,
        text=np.round(monthly_avg.values.T, 2),
        texttemplate="%{text}%",
        textfont=dict(size=9, color="white"),
        colorbar=dict(
            title=dict(text="Return (%)", font=dict(color=COLORS["text"])),
            tickfont=dict(color=COLORS["text"]),
            thickness=12
        ),
        hovertemplate="%{y} — %{x}<br>Rata-rata Return: %{z:.2f}%<extra></extra>"
    ))

    fig.update_layout(**_base_layout(
        title=dict(text="Rata-rata Return Bulanan (Pola Musiman)",
                   font=dict(size=14, color=COLORS["white"]), x=0.01),
        xaxis=dict(gridcolor=COLORS["grid"], linecolor=COLORS["grid"]),
        yaxis=dict(gridcolor=COLORS["grid"], linecolor=COLORS["grid"],
                   autorange="reversed"),
        height=380
    ))
    return fig


# ===========================================================================
# BAGIAN 5 — DETEKSI ANOMALI & OUTLIER
# ===========================================================================

def eda_hari_ekstrem(price_data: pd.DataFrame,
                     threshold: float = 0.05) -> pd.DataFrame:
    """
    Temukan hari-hari dengan pergerakan harga ekstrem (> threshold).

    Parameters:
    -----------
    threshold : float
        Batas pergerakan ekstrem (default 5%)

    Returns:
    --------
    pd.DataFrame : Daftar hari ekstrem beserta detailnya
    """
    returns = price_data.pct_change().dropna()
    rows    = []

    for col in returns.columns:
        ret  = returns[col]
        nama = col.replace(".JK", "")

        # Hari dengan pergerakan > threshold
        ekstrem = ret[ret.abs() > threshold]
        for tanggal, nilai in ekstrem.items():
            rows.append({
                "Saham"  : nama,
                "Tanggal": tanggal.date(),
                "Return" : f"{nilai*100:.2f}%",
                "Jenis"  : "🚀 Rally" if nilai > 0 else "💥 Crash",
                "_abs"   : abs(nilai)
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("_abs", ascending=False).drop("_abs", axis=1)
    return df.reset_index(drop=True)


def plot_outlier_timeline(price_data: pd.DataFrame,
                          threshold: float = 0.05) -> go.Figure:
    """
    Timeline yang menandai hari-hari ekstrem di setiap saham.
    """
    returns = price_data.pct_change().dropna()

    fig = go.Figure()
    for i, col in enumerate(returns.columns):
        ret  = returns[col]
        nama = col.replace(".JK", "")

        # Return normal
        fig.add_trace(go.Scatter(
            x=ret.index, y=ret * 100,
            mode="lines",
            name=nama,
            line=dict(color=PALETTE[i % len(PALETTE)], width=0.8),
            opacity=0.4,
            showlegend=True,
            hovertemplate=f"{nama}: %{{y:.2f}}%<extra></extra>"
        ))

        # Tandai hari ekstrem
        ekstrem = ret[ret.abs() > threshold]
        if not ekstrem.empty:
            warna_ext = [COLORS["green"] if v > 0 else COLORS["red"]
                         for v in ekstrem]
            fig.add_trace(go.Scatter(
                x=ekstrem.index,
                y=ekstrem * 100,
                mode="markers",
                marker=dict(color=warna_ext, size=7,
                            symbol="circle",
                            line=dict(color="white", width=1)),
                name=f"{nama} Ekstrem",
                showlegend=False,
                hovertemplate=f"<b>{nama}</b><br>%{{x}}<br>Return: %{{y:.2f}}%<extra></extra>"
            ))

    fig.add_hline(y=threshold * 100, line_dash="dash",
                  line_color=COLORS["gold"], opacity=0.4)
    fig.add_hline(y=-threshold * 100, line_dash="dash",
                  line_color=COLORS["gold"], opacity=0.4)

    fig.update_layout(**_base_layout(
        title=dict(text=f"Timeline Pergerakan Harian (Titik = Ekstrem >{threshold*100:.0f}%)",
                   font=dict(size=14, color=COLORS["white"]), x=0.01),
        xaxis=_axis("Tanggal"),
        yaxis=_axis("Return Harian (%)"),
        height=420,
        hovermode="x unified",
        legend=dict(bgcolor="rgba(17,24,39,0.8)", bordercolor=COLORS["grid"],
                    borderwidth=1, font=dict(size=9))
    ))
    return fig


# ===========================================================================
# BAGIAN 6 — ANALISIS KORELASI
# ===========================================================================

def plot_heatmap_korelasi(price_data: pd.DataFrame) -> go.Figure:
    """Heatmap korelasi antar saham."""
    returns = price_data.pct_change().dropna()
    corr    = returns.corr()
    labels  = [c.replace(".JK", "") for c in corr.columns]

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=labels, y=labels,
        colorscale=[
            [0.0, "#ef4444"],
            [0.5, "#1a2235"],
            [1.0, "#00d4aa"],
        ],
        zmid=0, zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=9, color="white"),
        colorbar=dict(
            title=dict(text="Korelasi", font=dict(color=COLORS["text"])),
            tickfont=dict(color=COLORS["text"]), thickness=12
        ),
        hovertemplate="%{x} vs %{y}<br>Korelasi: %{z:.3f}<extra></extra>"
    ))

    fig.update_layout(**_base_layout(
        title=dict(text="Matriks Korelasi Antar Saham",
                   font=dict(size=14, color=COLORS["white"]), x=0.01),
        xaxis=dict(side="bottom", tickfont=dict(size=10),
                   gridcolor=COLORS["grid"], linecolor=COLORS["grid"]),
        yaxis=dict(tickfont=dict(size=10), autorange="reversed",
                   gridcolor=COLORS["grid"], linecolor=COLORS["grid"]),
        height=420
    ))
    return fig


def eda_ringkasan_korelasi(price_data: pd.DataFrame) -> pd.DataFrame:
    """
    Ringkasan pasangan saham dengan korelasi tertinggi dan terendah.
    Berguna untuk memilih saham yang saling melengkapi (low correlation).
    """
    returns = price_data.pct_change().dropna()
    corr    = returns.corr()
    rows    = []

    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            rows.append({
                "Saham A" : cols[i].replace(".JK", ""),
                "Saham B" : cols[j].replace(".JK", ""),
                "Korelasi": round(corr.iloc[i, j], 4),
                "Interpretasi": (
                    "🔴 Sangat Tinggi"  if corr.iloc[i, j] > 0.8  else
                    "🟡 Tinggi"         if corr.iloc[i, j] > 0.5  else
                    "🟢 Rendah"         if corr.iloc[i, j] > 0.2  else
                    "🔵 Sangat Rendah / Negatif"
                )
            })

    df = pd.DataFrame(rows).sort_values("Korelasi", ascending=False)
    return df.reset_index(drop=True)


# ===========================================================================
# FUNGSI UTAMA — Tampilkan semua EDA di Streamlit
# ===========================================================================

def tampilkan_eda(price_data: pd.DataFrame,
                  risk_free_rate: float = 0.0575) -> None:
    """
    Tampilkan seluruh hasil EDA di aplikasi Streamlit.
    Dipanggil dari app.py dengan:
        from src.eda import tampilkan_eda
        tampilkan_eda(price_data, risk_free_rate)

    Parameters:
    -----------
    price_data      : pd.DataFrame — data harga historis
    risk_free_rate  : float        — BI Rate
    """

    st.markdown("## 🔍 Exploratory Data Analysis (EDA)")
    st.markdown("""
    <div style='background:rgba(59,130,246,0.08);border:1px solid rgba(59,130,246,0.25);
    border-radius:10px;padding:14px 18px;font-size:0.85rem;color:#93c5fd;margin:12px 0;'>
        EDA dilakukan <strong>sebelum optimasi</strong> untuk memahami karakteristik data,
        mendeteksi anomali, dan memastikan data layak digunakan. Hasil EDA menentukan
        apakah data perlu dibersihkan lebih lanjut sebelum masuk ke PortfolioOptimizer.
    </div>
    """, unsafe_allow_html=True)

    # ── Tab EDA ──────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "✅ Kualitas Data",
        "📊 Statistik Return",
        "📈 Tren & Pola",
        "💥 Anomali & Outlier",
        "🔗 Korelasi",
    ])

    # ── Tab 1: Kualitas Data ─────────────────────────────────────────────────
    with tab1:
        st.markdown("#### Kelengkapan & Kebersihan Data")

        fig_lengkap = plot_kelengkapan_data(price_data)
        st.plotly_chart(fig_lengkap, use_container_width=True)

        kualitas = eda_kualitas_data(price_data)
        df_kualitas = pd.DataFrame(kualitas).T.reset_index()
        df_kualitas.rename(columns={"index": "Saham"}, inplace=True)
        df_kualitas = df_kualitas.drop(columns=[
            c for c in df_kualitas.columns if c.startswith("_")
        ], errors="ignore")
        st.dataframe(df_kualitas, hide_index=True, use_container_width=True)

        n_masalah = sum(1 for v in kualitas.values() if "⚠️" in v["Status"])
        if n_masalah == 0:
            st.success("✅ Semua saham memiliki kualitas data yang baik.")
        else:
            st.warning(f"⚠️ {n_masalah} saham perlu diperiksa lebih lanjut.")

    # ── Tab 2: Statistik Return ──────────────────────────────────────────────
    with tab2:
        st.markdown("#### Statistik Deskriptif Return")

        df_stats = eda_statistik_return(price_data, risk_free_rate)

        col1, col2 = st.columns(2)
        with col1:
            fig_rr = plot_return_vs_risiko(df_stats, risk_free_rate)
            st.plotly_chart(fig_rr, use_container_width=True)
        with col2:
            fig_rank = plot_ranking_return(df_stats)
            st.plotly_chart(fig_rank, use_container_width=True)

        # Tabel statistik lengkap (sembunyikan kolom internal)
        kolom_tampil = [c for c in df_stats.columns if not c.startswith("_")]
        st.dataframe(df_stats[kolom_tampil], hide_index=True, use_container_width=True)

        st.markdown("""
        <div style='background:rgba(0,212,170,0.06);border:1px solid rgba(0,212,170,0.2);
        border-radius:8px;padding:12px 16px;font-size:0.82rem;color:#6ee7d4;margin-top:12px;'>
        💡 <b>Cara baca:</b>
        Saham ideal ada di <b>pojok kiri atas</b> grafik scatter (return tinggi, volatilitas rendah).
        Saham di bawah garis BI Rate tidak lebih menguntungkan dari deposito bank.
        Skewness negatif = lebih sering turun tajam. Kurtosis tinggi = outlier lebih sering terjadi.
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 3: Tren & Pola ───────────────────────────────────────────────────
    with tab3:
        st.markdown("#### Pergerakan Harga & Pola Waktu")

        fig_norm = plot_harga_normalized(price_data)
        st.plotly_chart(fig_norm, use_container_width=True)

        fig_cum = plot_return_kumulatif(price_data)
        st.plotly_chart(fig_cum, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            window = st.slider("Window Rolling Volatilitas (hari)", 10, 90, 30, 5)
            fig_roll = plot_rolling_volatilitas(price_data, window)
            st.plotly_chart(fig_roll, use_container_width=True)
        with col2:
            fig_bln = plot_return_bulanan(price_data)
            st.plotly_chart(fig_bln, use_container_width=True)

    # ── Tab 4: Anomali & Outlier ─────────────────────────────────────────────
    with tab4:
        st.markdown("#### Deteksi Hari Ekstrem & Anomali")

        threshold = st.slider("Batas Pergerakan Ekstrem (%)", 3, 15, 5, 1) / 100

        fig_dist = plot_distribusi_return(price_data)
        st.plotly_chart(fig_dist, use_container_width=True)

        fig_box = plot_boxplot_return(price_data)
        st.plotly_chart(fig_box, use_container_width=True)

        fig_out = plot_outlier_timeline(price_data, threshold)
        st.plotly_chart(fig_out, use_container_width=True)

        st.markdown(f"#### Daftar Hari Ekstrem (pergerakan > {threshold*100:.0f}%)")
        df_ekstrem = eda_hari_ekstrem(price_data, threshold)
        if df_ekstrem.empty:
            st.info("Tidak ada hari ekstrem dengan threshold ini.")
        else:
            st.dataframe(df_ekstrem.head(50), hide_index=True, use_container_width=True)

    # ── Tab 5: Korelasi ──────────────────────────────────────────────────────
    with tab5:
        st.markdown("#### Analisis Korelasi Antar Saham")

        fig_corr = plot_heatmap_korelasi(price_data)
        st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown("#### Ringkasan Pasangan Korelasi")
        df_corr = eda_ringkasan_korelasi(price_data)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🔴 Korelasi Tertinggi** (kurang ideal untuk diversifikasi)")
            st.dataframe(df_corr.head(10), hide_index=True, use_container_width=True)
        with col2:
            st.markdown("**🟢 Korelasi Terendah** (ideal untuk diversifikasi)")
            st.dataframe(df_corr.tail(10).iloc[::-1], hide_index=True,
                         use_container_width=True)

        st.markdown("""
        <div style='background:rgba(0,212,170,0.06);border:1px solid rgba(0,212,170,0.2);
        border-radius:8px;padding:12px 16px;font-size:0.82rem;color:#6ee7d4;margin-top:12px;'>
        💡 <b>Cara baca:</b>
        Korelasi mendekati <b>+1</b> = dua saham bergerak searah (kurang bagus untuk diversifikasi).
        Korelasi mendekati <b>0</b> atau <b>negatif</b> = dua saham bergerak tidak berhubungan
        (bagus! — kerugian satu bisa dikompensasi saham lain).
        </div>
        """, unsafe_allow_html=True)
