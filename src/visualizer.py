"""
Visualizer Module
Membuat grafik interaktif menggunakan Plotly untuk analisis portofolio
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


# ─── Tema Warna ───────────────────────────────────────────────────────────────
COLORS = {
    "bg": "rgba(17,24,39,1)",
    "bg_paper": "rgba(10,14,26,1)",
    "grid": "#1e2d3d",
    "text": "#64748b",
    "green": "#00d4aa",
    "blue": "#3b82f6",
    "gold": "#f59e0b",
    "red": "#ef4444",
    "purple": "#8b5cf6",
    "white": "#f1f5f9",
}

PLOTLY_TEMPLATE = dict(
    template="plotly_dark",
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["bg"],
    font=dict(family="DM Sans, sans-serif", color=COLORS["text"]),
    xaxis=dict(gridcolor=COLORS["grid"], linecolor=COLORS["grid"]),
    yaxis=dict(gridcolor=COLORS["grid"], linecolor=COLORS["grid"]),
    margin=dict(l=10, r=10, t=40, b=10),
)


def _base_layout(**kwargs) -> dict:
    """Helper: gabungkan PLOTLY_TEMPLATE dengan override tanpa duplikat key."""
    base = {
        "template": PLOTLY_TEMPLATE["template"],
        "paper_bgcolor": PLOTLY_TEMPLATE["paper_bgcolor"],
        "plot_bgcolor": PLOTLY_TEMPLATE["plot_bgcolor"],
        "font": PLOTLY_TEMPLATE["font"],
        "margin": PLOTLY_TEMPLATE["margin"],
    }
    base.update(kwargs)
    return base


def _axis(title_text: str) -> dict:
    """Helper: buat dict axis dengan title dan grid dari PLOTLY_TEMPLATE."""
    return dict(
        title=dict(text=title_text, font=dict(color=COLORS["text"])),
        gridcolor=COLORS["grid"],
        linecolor=COLORS["grid"],
    )


def plot_efficient_frontier(
    mc_results: pd.DataFrame,
    frontier_df: pd.DataFrame,
    optimal_stats: dict,
    risk_free_rate: float = 0.0575
) -> go.Figure:
    """
    Buat grafik Efficient Frontier dengan overlay Monte Carlo simulation.
    
    Parameters:
    -----------
    mc_results : pd.DataFrame
        Hasil simulasi Monte Carlo (volatility, return, sharpe)
    frontier_df : pd.DataFrame
        Titik-titik Efficient Frontier
    optimal_stats : dict
        Statistik portofolio optimal
    risk_free_rate : float
        Tingkat bunga bebas risiko untuk Capital Market Line
    
    Returns:
    --------
    go.Figure : Grafik Plotly interaktif
    """
    fig = go.Figure()
    
    # 1. Scatter plot Monte Carlo
    fig.add_trace(go.Scatter(
        x=mc_results["volatility"] * 100,
        y=mc_results["return"] * 100,
        mode="markers",
        marker=dict(
            size=4,
            color=mc_results["sharpe"],
            colorscale=[
                [0, "#1a2235"],
                [0.3, "#1e3a5f"],
                [0.6, "#0ea5e9"],
                [1.0, "#00d4aa"]
            ],
            colorbar=dict(
                title=dict(
                    text="Sharpe<br>Ratio",
                    font=dict(color=COLORS["text"])
                ),
                tickfont=dict(color=COLORS["text"]),
                thickness=12,
                len=0.6,
                x=1.02
            ),
            opacity=0.6,
            line=dict(width=0)
        ),
        name="Portofolio Acak",
        hovertemplate=(
            "<b>Portofolio</b><br>"
            "Volatilitas: %{x:.2f}%<br>"
            "Return: %{y:.2f}%<br>"
            "Sharpe: %{marker.color:.3f}<extra></extra>"
        )
    ))
    
    # 2. Efficient Frontier line
    if not frontier_df.empty:
        fig.add_trace(go.Scatter(
            x=frontier_df["volatility"] * 100,
            y=frontier_df["return"] * 100,
            mode="lines",
            line=dict(color=COLORS["green"], width=3),
            name="Efficient Frontier",
            hovertemplate=(
                "<b>Efficient Frontier</b><br>"
                "Volatilitas: %{x:.2f}%<br>"
                "Return: %{y:.2f}%<extra></extra>"
            )
        ))
    
    # 3. Capital Market Line (CML)
    if not frontier_df.empty:
        # Cari titik tangent (max sharpe pada frontier)
        tangent_vol = optimal_stats["volatility"] * 100
        tangent_ret = optimal_stats["expected_return"] * 100
        rf_ret = risk_free_rate * 100
        
        cml_vols = np.linspace(0, tangent_vol * 1.5, 50)
        slope = (tangent_ret - rf_ret) / tangent_vol
        cml_rets = rf_ret + slope * cml_vols
        
        fig.add_trace(go.Scatter(
            x=cml_vols,
            y=cml_rets,
            mode="lines",
            line=dict(color=COLORS["gold"], width=1.5, dash="dash"),
            name="Capital Market Line",
            hoverinfo="skip"
        ))
        
        # Risk-free point
        fig.add_trace(go.Scatter(
            x=[0],
            y=[rf_ret],
            mode="markers+text",
            marker=dict(size=8, color=COLORS["gold"], symbol="diamond"),
            text=["Rf"],
            textposition="top right",
            textfont=dict(color=COLORS["gold"], size=11),
            name=f"Risk-Free ({rf_ret:.1f}%)",
            hovertemplate=f"<b>Risk-Free Rate</b><br>{rf_ret:.2f}%<extra></extra>"
        ))
    
    # 4. Optimal Portfolio (Star)
    fig.add_trace(go.Scatter(
        x=[optimal_stats["volatility"] * 100],
        y=[optimal_stats["expected_return"] * 100],
        mode="markers+text",
        marker=dict(
            size=20,
            color=COLORS["red"],
            symbol="star",
            line=dict(color="white", width=1.5)
        ),
        text=["Optimal"],
        textposition="top center",
        textfont=dict(color=COLORS["white"], size=12, family="Space Mono, monospace"),
        name=f"Portofolio Optimal<br>Sharpe: {optimal_stats['sharpe_ratio']:.3f}",
        hovertemplate=(
            "<b>⭐ Portofolio Optimal</b><br>"
            f"Return: {optimal_stats['expected_return']*100:.2f}%<br>"
            f"Volatilitas: {optimal_stats['volatility']*100:.2f}%<br>"
            f"Sharpe: {optimal_stats['sharpe_ratio']:.3f}<extra></extra>"
        )
    ))
    
    # Layout
    fig.update_layout(**_base_layout(
        title=dict(
            text="Markowitz Efficient Frontier",
            font=dict(size=16, color=COLORS["white"], family="Space Mono, monospace"),
            x=0.01
        ),
        xaxis=_axis("Volatilitas / Risiko (%/tahun)"),
        yaxis=_axis("Expected Return (%/tahun)"),
        legend=dict(
            bgcolor="rgba(17,24,39,0.8)",
            bordercolor=COLORS["grid"],
            borderwidth=1,
            font=dict(size=11)
        ),
        height=500,
        hovermode="closest"
    ))
    
    return fig


def plot_portfolio_weights(weights: dict) -> go.Figure:
    """
    Buat donut chart untuk visualisasi bobot portofolio.
    
    Parameters:
    -----------
    weights : dict
        Bobot per saham (ticker: weight)
    
    Returns:
    --------
    go.Figure
    """
    # Filter bobot > 1%
    filtered = {k.replace(".JK", ""): v for k, v in weights.items() if v > 0.01}
    
    # Custom color palette
    colors = [
        "#00d4aa", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6",
        "#ec4899", "#06b6d4", "#84cc16", "#f97316", "#14b8a6"
    ]
    
    fig = go.Figure(go.Pie(
        labels=list(filtered.keys()),
        values=list(filtered.values()),
        hole=0.5,
        marker=dict(
            colors=colors[:len(filtered)],
            line=dict(color=COLORS["bg"], width=3)
        ),
        textinfo="label+percent",
        textfont=dict(size=12, color=COLORS["white"]),
        hovertemplate="<b>%{label}</b><br>Bobot: %{value:.2%}<extra></extra>",
        sort=True,
        direction="clockwise"
    ))
    
    fig.update_layout(
        **PLOTLY_TEMPLATE,
        title=dict(
            text="Distribusi Bobot Portofolio",
            font=dict(size=14, color=COLORS["white"], family="Space Mono, monospace"),
            x=0.01
        ),
        annotations=[dict(
            text=f"<b>{len(filtered)}</b><br>Saham",
            x=0.5, y=0.5,
            font=dict(size=14, color=COLORS["white"]),
            showarrow=False
        )],
        showlegend=False,
        height=380
    )
    
    return fig


def plot_correlation_heatmap(price_data: pd.DataFrame) -> go.Figure:
    """
    Buat heatmap korelasi antar saham.
    
    Parameters:
    -----------
    price_data : pd.DataFrame
        Data harga historis saham
    
    Returns:
    --------
    go.Figure
    """
    returns = price_data.pct_change().dropna()
    corr = returns.corr()
    
    # Bersihkan nama ticker
    labels = [col.replace(".JK", "") for col in corr.columns]
    
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=labels,
        y=labels,
        colorscale=[
            [0.0, "#ef4444"],    # Negatif = merah
            [0.5, "#1a2235"],    # Nol = biru gelap
            [1.0, "#00d4aa"],    # Positif = hijau
        ],
        zmid=0,
        zmin=-1,
        zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=10, color="white"),
        colorbar=dict(
            title=dict(
                text="Korelasi",
                font=dict(color=COLORS["text"])
            ),
            tickfont=dict(color=COLORS["text"]),
            thickness=12
        ),
        hovertemplate="%{x} vs %{y}<br>Korelasi: %{z:.3f}<extra></extra>"
    ))
    
    fig.update_layout(**_base_layout(
        title=dict(
            text="Matriks Korelasi",
            font=dict(size=14, color=COLORS["white"], family="Space Mono, monospace"),
            x=0.01
        ),
        height=380,
        xaxis=dict(side="bottom", tickfont=dict(size=10), gridcolor=COLORS["grid"], linecolor=COLORS["grid"]),
        yaxis=dict(tickfont=dict(size=10), autorange="reversed", gridcolor=COLORS["grid"], linecolor=COLORS["grid"])
    ))
    
    return fig


def plot_cumulative_returns(
    price_data: pd.DataFrame,
    weights: dict
) -> go.Figure:
    """
    Plot kumulatif return portofolio optimal vs saham individual vs IHSG.
    
    Parameters:
    -----------
    price_data : pd.DataFrame
        Data harga historis
    weights : dict
        Bobot portofolio optimal
    
    Returns:
    --------
    go.Figure
    """
    returns = price_data.pct_change().dropna()
    
    # Return portofolio optimal
    port_returns = pd.Series(0.0, index=returns.index)
    for ticker, weight in weights.items():
        if ticker in returns.columns and weight > 0.01:
            port_returns += returns[ticker] * weight
    
    cumulative_port = (1 + port_returns).cumprod()
    
    fig = go.Figure()
    
    # Plot saham individual (tipis, abu-abu)
    for col in price_data.columns:
        cum_ret = (1 + returns[col]).cumprod()
        fig.add_trace(go.Scatter(
            x=cum_ret.index,
            y=cum_ret.values,
            mode="lines",
            name=col.replace(".JK", ""),
            line=dict(width=1, color="rgba(100,116,139,0.4)"),
            showlegend=True,
            hovertemplate=f"{col.replace('.JK','')}: %{{y:.3f}}x<extra></extra>"
        ))
    
    # Plot portofolio optimal (tebal, hijau)
    fig.add_trace(go.Scatter(
        x=cumulative_port.index,
        y=cumulative_port.values,
        mode="lines",
        name="Portofolio Optimal",
        line=dict(color=COLORS["green"], width=3),
        hovertemplate="Portofolio: %{y:.3f}x<extra></extra>"
    ))
    
    # Garis baseline
    fig.add_hline(y=1, line_dash="dot", line_color=COLORS["gold"], opacity=0.5)
    
    fig.update_layout(**_base_layout(
        title=dict(
            text="Kinerja Kumulatif (Base = 1.0)",
            font=dict(size=14, color=COLORS["white"], family="Space Mono, monospace"),
            x=0.01
        ),
        xaxis=_axis("Tanggal"),
        yaxis=_axis("Return Kumulatif (x)"),
        legend=dict(
            bgcolor="rgba(17,24,39,0.8)",
            bordercolor=COLORS["grid"],
            borderwidth=1,
            font=dict(size=10)
        ),
        height=420,
        hovermode="x unified"
    ))
    
    return fig


def plot_risk_return_scatter(
    price_data: pd.DataFrame,
    optimal_weights: dict,
    risk_free_rate: float = 0.0575
) -> go.Figure:
    """
    Scatter plot Risk-Return per saham individual dengan bubble size = bobot optimal.
    
    Parameters:
    -----------
    price_data : pd.DataFrame
        Data harga historis
    optimal_weights : dict
        Bobot portofolio optimal
    risk_free_rate : float
        Tingkat bunga bebas risiko
    
    Returns:
    --------
    go.Figure
    """
    returns = price_data.pct_change().dropna()
    
    data_points = []
    for col in price_data.columns:
        ret = returns[col]
        ann_return = (1 + ret.mean()) ** 252 - 1
        ann_vol = ret.std() * np.sqrt(252)
        weight = optimal_weights.get(col, 0)
        name = col.replace(".JK", "")
        
        data_points.append({
            "name": name,
            "return": ann_return * 100,
            "volatility": ann_vol * 100,
            "weight": weight * 100,
            "size": max(weight * 800, 80)  # Minimum size
        })
    
    df_plot = pd.DataFrame(data_points)
    
    colors_individual = [
        "#00d4aa" if row["weight"] > 5 else "#3b82f6" 
        for _, row in df_plot.iterrows()
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_plot["volatility"],
        y=df_plot["return"],
        mode="markers+text",
        text=df_plot["name"],
        textposition="top center",
        textfont=dict(size=10, color=COLORS["white"]),
        marker=dict(
            size=df_plot["size"] / 10,
            color=colors_individual,
            opacity=0.85,
            line=dict(color=COLORS["bg"], width=1.5)
        ),
        customdata=df_plot[["weight"]],
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Return: %{y:.2f}%<br>"
            "Volatilitas: %{x:.2f}%<br>"
            "Bobot: %{customdata[0]:.1f}%<extra></extra>"
        ),
        name="Saham Individual"
    ))
    
    # Risk-free annotation
    fig.add_annotation(
        x=0, y=risk_free_rate * 100,
        text=f"Risk-Free: {risk_free_rate*100:.1f}%",
        showarrow=False,
        font=dict(color=COLORS["gold"], size=10),
        xanchor="left"
    )
    fig.add_hline(
        y=risk_free_rate * 100,
        line_dash="dash",
        line_color=COLORS["gold"],
        opacity=0.4
    )
    
    fig.update_layout(**_base_layout(
        title=dict(
            text="Risk-Return per Saham (Ukuran = Bobot Optimal)",
            font=dict(size=14, color=COLORS["white"], family="Space Mono, monospace"),
            x=0.01
        ),
        xaxis=_axis("Volatilitas/Tahun (%)"),
        yaxis=_axis("Expected Return/Tahun (%)"),
        showlegend=False,
        height=380
    ))
    
    return fig


def plot_drawdown(port_returns: pd.Series) -> go.Figure:
    """
    Plot underwater / drawdown chart portofolio.
    
    Parameters:
    -----------
    port_returns : pd.Series
        Return harian portofolio
    
    Returns:
    --------
    go.Figure
    """
    cumulative = (1 + port_returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative / rolling_max - 1) * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values,
        mode="lines",
        fill="tozeroy",
        fillcolor="rgba(239,68,68,0.15)",
        line=dict(color=COLORS["red"], width=1.5),
        name="Drawdown",
        hovertemplate="Drawdown: %{y:.2f}%<extra></extra>"
    ))
    
    fig.update_layout(**_base_layout(
        title=dict(
            text="Portfolio Drawdown",
            font=dict(size=14, color=COLORS["white"], family="Space Mono, monospace"),
            x=0.01
        ),
        xaxis=_axis("Tanggal"),
        yaxis=_axis("Drawdown (%)"),
        height=280
    ))
    
    return fig
