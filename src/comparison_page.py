"""
Halaman Perbandingan Model — Streamlit UI
=========================================
Menampilkan hasil perbandingan:
  Markowitz Efficient Frontier vs HRP vs Equal Weight

Dipanggil dari app.py atau dijalankan mandiri:
    streamlit run src/comparison_page.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os

# Import model comparison
from model_comparison import (
    fetch_data, split_data,
    model_markowitz, model_hrp, model_equal_weight,
    hitung_metrik,
    TICKERS, START_DATE, END_DATE, RISK_FREE,
    TRAIN_RATIO, VAL_RATIO
)

# ─── Warna ────────────────────────────────────────────────────────────────────
WARNA = {
    "Markowitz Efficient Frontier": "#00d4aa",   # hijau tosca
    "HRP"          : "#3b82f6",   # biru
    "Equal Weight" : "#f59e0b",   # emas
}

LAYOUT = dict(
    template    = "plotly_dark",
    paper_bgcolor = "rgba(17,24,39,1)",
    plot_bgcolor  = "rgba(17,24,39,1)",
    font        = dict(family="DM Sans", color="#64748b"),
    margin      = dict(l=10, r=10, t=40, b=10),
)


# ===========================================================================
# FUNGSI UTAMA
# ===========================================================================

@st.cache_data(ttl=3600)
def jalankan_semua_model():
    """
    Unduh data, split, latih 3 model, evaluasi.
    Di-cache 1 jam agar tidak diulang setiap refresh.
    """
    price_data = fetch_data(TICKERS, START_DATE, END_DATE)
    train, val, test = split_data(price_data, TRAIN_RATIO, VAL_RATIO)

    # Latih model (menggunakan data TRAIN saja)
    w_mpt = model_markowitz(train, risk_free=RISK_FREE)
    w_hrp = model_hrp(train)
    w_eqw = model_equal_weight(list(train.columns))

    models = {
        "Markowitz Efficient Frontier": w_mpt,
        "HRP"          : w_hrp,
        "Equal Weight" : w_eqw,
    }

    # Evaluasi di semua split
    hasil = {}
    for nama, weights in models.items():
        hasil[nama] = {
            "weights": weights,
            "train"  : hitung_metrik(train, weights, RISK_FREE),
            "val"    : hitung_metrik(val, weights, RISK_FREE),
            "test"   : hitung_metrik(test, weights, RISK_FREE),
        }

    return hasil, price_data, train, val, test

def hex_to_rgba(hex_color, alpha=0.15):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def plot_radar(hasil: dict, split: str = "test") -> go.Figure:
    """Radar chart perbandingan 5 dimensi antar model."""

    kategori = [
        "Sharpe Ratio", "Return", "Rendah Volatilitas",
        "Sortino", "Win Rate"
    ]

    fig = go.Figure()

    for nama, data in hasil.items():
        m       = data[split]
        # Normalisasi 0-1 untuk radar
        sharpe  = min(max(m["sharpe_ratio"] / 2, 0), 1)
        ret     = min(max(m["ann_return"] / 0.5, 0), 1)
        low_vol = min(max(1 - m["ann_vol"] / 0.5, 0), 1)
        sortino = min(max(m["sortino_ratio"] / 3, 0), 1)
        winrate = m["win_rate"]

        values  = [sharpe, ret, low_vol, sortino, winrate]
        values += [values[0]]  # tutup radar

        fig.add_trace(go.Scatterpolar(
    r         = values,
    theta     = kategori + [kategori[0]],
    fill      = "toself",
    name      = nama,
    line      = dict(color=WARNA[nama], width=2),
    fillcolor = hex_to_rgba(WARNA[nama], 0.15),  # ✅ FIX
    opacity   = 0.8,
))

    fig.update_layout(
        **LAYOUT,
        polar = dict(
            radialaxis = dict(
                visible    = True,
                range      = [0, 1],
                gridcolor  = "#1e2d3d",
                linecolor  = "#1e2d3d",
            ),
            angularaxis = dict(
                gridcolor  = "#1e2d3d",
                linecolor  = "#1e2d3d",
                tickfont   = dict(color="#f1f5f9", size=11),
            ),
            bgcolor = "rgba(17,24,39,1)",
        ),
        title  = dict(
            text = f"Radar Chart Perbandingan Model ({split.upper()})",
            font = dict(size=14, color="#f1f5f9"),
            x    = 0.01
        ),
        legend = dict(
            bgcolor     = "rgba(17,24,39,0.8)",
            bordercolor = "#1e2d3d",
            borderwidth = 1,
        ),
        height = 420,
    )
    return fig


def plot_performa_kumulatif(hasil: dict,
                             price_data: pd.DataFrame) -> go.Figure:
    """Grafik return kumulatif ketiga model sepanjang waktu."""
    returns = price_data.pct_change().dropna()
    fig     = go.Figure()

    for nama, data in hasil.items():
        weights = data["weights"]
        port_ret = sum(
            returns[col] * weights.get(col, 0)
            for col in returns.columns
            if col in weights
        )
        cum_ret = (1 + port_ret).cumprod()

        fig.add_trace(go.Scatter(
            x           = cum_ret.index,
            y           = cum_ret.values,
            name        = nama,
            line        = dict(color=WARNA[nama], width=2.5),
            hovertemplate = f"{nama}: %{{y:.3f}}x<extra></extra>"
        ))

    fig.add_hline(
        y=1, line_dash="dot",
        line_color="#f59e0b", opacity=0.4,
        annotation_text="Modal Awal",
        annotation_font_color="#f59e0b"
    )

    # Garis pemisah Train/Val/Test
    n          = len(price_data)
    train_end  = int(n * TRAIN_RATIO)
    val_end    = train_end + int(n * VAL_RATIO)
    idx        = price_data.index

    for garis_idx, label, warna in [
        (train_end, "TRAIN | VAL", "#64748b"),
        (val_end,   "VAL | TEST",  "#64748b"),
    ]:
        if garis_idx < len(idx):
            # Konversi Timestamp ke string agar kompatibel dengan Plotly
            x_str = str(idx[garis_idx].date())
            fig.add_shape(
                type   = "line",
                x0     = x_str, x1 = x_str,
                y0     = 0,     y1 = 1,
                xref   = "x",   yref = "paper",
                line   = dict(color=warna, dash="dash", width=1.5),
                opacity= 0.5,
            )
            fig.add_annotation(
                x         = x_str,
                y         = 1.02,
                xref      = "x",
                yref      = "paper",
                text      = label,
                showarrow = False,
                font      = dict(color=warna, size=10),
                bgcolor   = "rgba(17,24,39,0.8)",
                bordercolor = warna,
                borderwidth = 1,
            )

    fig.update_layout(
        **LAYOUT,
        title  = dict(
            text = "Return Kumulatif Ketiga Model (1.0x = Modal Awal)",
            font = dict(size=14, color="#f1f5f9"),
            x    = 0.01
        ),
        xaxis  = dict(gridcolor="#1e2d3d", linecolor="#1e2d3d"),
        yaxis  = dict(
            gridcolor="#1e2d3d", linecolor="#1e2d3d",
            title="Return Kumulatif (x)"
        ),
        legend = dict(
            bgcolor="rgba(17,24,39,0.8)",
            bordercolor="#1e2d3d", borderwidth=1
        ),
        height       = 420,
        hovermode    = "x unified",
    )
    return fig


def plot_bobot_grouped(hasil: dict) -> go.Figure:
    """Bar chart bobot tiap saham per model."""
    tickers_bersih = [t.replace(".JK", "") for t in TICKERS]
    fig = go.Figure()

    for nama, data in hasil.items():
        bobot = [
            data["weights"].get(t, 0) * 100
            for t in TICKERS
        ]
        fig.add_trace(go.Bar(
            name            = nama,
            x               = tickers_bersih,
            y               = bobot,
            marker_color    = WARNA[nama],
            text            = [f"{b:.1f}%" for b in bobot],
            textposition    = "outside",
            textfont        = dict(size=10, color="#f1f5f9"),
        ))

    fig.update_layout(
        **LAYOUT,
        title       = dict(
            text="Perbandingan Bobot Saham per Model",
            font=dict(size=14, color="#f1f5f9"), x=0.01
        ),
        barmode     = "group",
        xaxis       = dict(gridcolor="#1e2d3d", linecolor="#1e2d3d"),
        yaxis       = dict(
            gridcolor="#1e2d3d", linecolor="#1e2d3d",
            title="Bobot (%)"
        ),
        legend      = dict(
            bgcolor="rgba(17,24,39,0.8)",
            bordercolor="#1e2d3d", borderwidth=1
        ),
        height      = 380,
    )
    return fig


def plot_metrik_grouped(hasil: dict, split: str = "test") -> go.Figure:
    """Bar chart grouped untuk setiap metrik per model."""

    metrik_list = [
        ("Sharpe Ratio",   "sharpe_ratio",  1,   False),
        ("Return (%)",     "ann_return",    100,  False),
        ("Volatilitas (%)", "ann_vol",      100,  True),
        ("Max DD (%)",     "max_drawdown",  100,  True),
        ("Sortino",        "sortino_ratio", 1,    False),
        ("Win Rate (%)",   "win_rate",      100,  False),
    ]

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[m[0] for m in metrik_list]
    )

    for idx, (label, key, mult, lower_better) in enumerate(metrik_list):
        row = idx // 3 + 1
        col = idx % 3 + 1

        nilais = {
            nama: data[split][key] * mult
            for nama, data in hasil.items()
        }

        if lower_better:
            best = min(nilais.values())
        else:
            best = max(nilais.values())

        for nama, nilai in nilais.items():
            is_best = abs(nilai - best) < 0.0001
            fig.add_trace(go.Bar(
                name            = nama,
                x               = [nama.replace(" ", "<br>")],
                y               = [abs(nilai)],
                marker_color    = WARNA[nama],
                marker_line     = dict(
                    color="#f1f5f9" if is_best else "rgba(0,0,0,0)",
                    width=2 if is_best else 0
                ),
                text            = [f"{'★ ' if is_best else ''}{nilai:.3f}"],
                textposition    = "outside",
                textfont        = dict(size=9, color="#f1f5f9"),
                showlegend      = idx == 0,
            ), row=row, col=col)

    fig.update_layout(
        **LAYOUT,
        title   = dict(
            text=f"Perbandingan Metrik — {split.upper()} SET (★ = Terbaik)",
            font=dict(size=14, color="#f1f5f9"), x=0.01
        ),
        height  = 500,
        barmode = "group",
        showlegend = True,
        legend  = dict(
            bgcolor="rgba(17,24,39,0.8)",
            bordercolor="#1e2d3d", borderwidth=1
        ),
    )
    fig.update_xaxes(gridcolor="#1e2d3d", linecolor="#1e2d3d")
    fig.update_yaxes(gridcolor="#1e2d3d", linecolor="#1e2d3d")
    return fig


# ===========================================================================
# HALAMAN STREAMLIT
# ===========================================================================

def tampilkan_halaman_perbandingan():
    """
    Fungsi utama yang dipanggil dari app.py.
    Menampilkan seluruh UI perbandingan model.
    """
    st.markdown("## ⚖️ Perbandingan Model: Markowitz vs HRP vs Equal Weight")

    st.markdown("""
    <div style='background:rgba(59,130,246,0.08);border:1px solid rgba(59,130,246,0.25);
    border-radius:10px;padding:14px 18px;font-size:0.85rem;color:#93c5fd;margin:12px 0;'>
        Halaman ini membandingkan <strong>3 model optimasi portofolio</strong> secara objektif
        menggunakan data yang sama dan split yang sama (70/15/15), sehingga hasilnya bisa
        dibandingkan secara adil. Model dilatih menggunakan data TRAIN dan dievaluasi
        di VAL dan TEST.
    </div>
    """, unsafe_allow_html=True)

    # Penjelasan singkat tiap model
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style='background:#111827;border:1px solid #00d4aa44;border-radius:10px;
        padding:16px;height:160px;'>
            <div style='color:#00d4aa;font-weight:700;font-size:0.9rem;'>
                🔵 Markowitz Efficient Frontier
            </div>
            <div style='color:#94a3b8;font-size:0.8rem;margin-top:8px;'>
                Memaksimalkan Sharpe Ratio menggunakan optimasi kuadratik.
                Butuh estimasi expected return & kovariansi.
                <br><br><em>Cocok untuk: investor yang ingin return/risiko optimal.</em>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='background:#111827;border:1px solid #3b82f644;border-radius:10px;
        padding:16px;height:160px;'>
            <div style='color:#3b82f6;font-weight:700;font-size:0.9rem;'>
                🟢 HRP (Hierarchical Risk Parity)
            </div>
            <div style='color:#94a3b8;font-size:0.8rem;margin-top:8px;'>
                Menggunakan clustering hierarkis untuk membagi risiko merata.
                Tidak butuh inversi matriks, lebih stabil.
                <br><br><em>Cocok untuk: investor yang ingin diversifikasi risiko merata.</em>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style='background:#111827;border:1px solid #f59e0b44;border-radius:10px;
        padding:16px;height:160px;'>
            <div style='color:#f59e0b;font-weight:700;font-size:0.9rem;'>
                ⚪ Equal Weight (Baseline)
            </div>
            <div style='color:#94a3b8;font-size:0.8rem;margin-top:8px;'>
                Bagi modal sama rata ke semua saham.
                Tidak ada optimasi sama sekali.
                <br><br><em>Dipakai sebagai pembanding: apakah model lebih baik dari bagi rata?</em>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Jalankan model
    with st.spinner("🔬 Melatih dan membandingkan 3 model..."):
        try:
            hasil, price_data, train, val, test = jalankan_semua_model()
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Pastikan koneksi internet tersedia untuk mengunduh data Yahoo Finance.")
            return

    # Pilih split untuk ditampilkan
    split_pilihan = st.radio(
        "Tampilkan metrik dari split:",
        ["train", "val", "test"],
        index=2,
        horizontal=True,
        format_func=lambda x: {
            "train": "🟢 TRAIN (70%)",
            "val"  : "🟡 VAL (15%)",
            "test" : "🔴 TEST (15%) — Evaluasi Akhir"
        }[x]
    )

    # ── Kartu Metrik Utama ────────────────────────────────────────
    st.markdown("#### 📊 Metrik Utama")
    cols = st.columns(3)

    for i, (nama, data) in enumerate(hasil.items()):
        m = data[split_pilihan]
        with cols[i]:
            warna = WARNA[nama].replace("#", "")
            st.markdown(f"""
            <div style='background:#111827;border:2px solid #{warna}44;
            border-radius:12px;padding:16px;'>
                <div style='color:#{warna};font-weight:700;font-size:0.95rem;
                margin-bottom:12px;'>{nama}</div>
                <div style='display:grid;grid-template-columns:1fr 1fr;gap:8px;'>
                    <div>
                        <div style='color:#64748b;font-size:0.7rem;'>SHARPE</div>
                        <div style='color:#f1f5f9;font-size:1.1rem;font-weight:700;
                        font-family:monospace;'>{m['sharpe_ratio']:.3f}</div>
                    </div>
                    <div>
                        <div style='color:#64748b;font-size:0.7rem;'>RETURN/THN</div>
                        <div style='color:#f1f5f9;font-size:1.1rem;font-weight:700;
                        font-family:monospace;'>{m['ann_return']*100:.1f}%</div>
                    </div>
                    <div>
                        <div style='color:#64748b;font-size:0.7rem;'>VOLATILITAS</div>
                        <div style='color:#f1f5f9;font-size:1.1rem;font-weight:700;
                        font-family:monospace;'>{m['ann_vol']*100:.1f}%</div>
                    </div>
                    <div>
                        <div style='color:#64748b;font-size:0.7rem;'>MAX DRAWDOWN</div>
                        <div style='color:#ef4444;font-size:1.1rem;font-weight:700;
                        font-family:monospace;'>{m['max_drawdown']*100:.1f}%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Grafik ────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Kinerja Kumulatif",
        "🎯 Radar Chart",
        "📊 Perbandingan Metrik",
        "⚖️ Bobot Saham",
    ])

    with tab1:
        fig_cum = plot_performa_kumulatif(hasil, price_data)
        st.plotly_chart(fig_cum, use_container_width=True)
        st.markdown("""
        <div style='background:rgba(0,212,170,0.06);border:1px solid rgba(0,212,170,0.2);
        border-radius:8px;padding:12px 16px;font-size:0.82rem;color:#6ee7d4;'>
        💡 Garis putus-putus vertikal memisahkan TRAIN / VAL / TEST.
        Performa di area TEST (paling kanan) adalah yang paling jujur
        karena model belum pernah melihat data tersebut.
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        fig_radar = plot_radar(hasil, split_pilihan)
        st.plotly_chart(fig_radar, use_container_width=True)
        st.markdown("""
        <div style='background:rgba(59,130,246,0.08);border:1px solid rgba(59,130,246,0.25);
        border-radius:8px;padding:12px 16px;font-size:0.82rem;color:#93c5fd;'>
        💡 Radar chart menunjukkan 5 dimensi sekaligus.
        Model yang lebih baik memiliki area yang lebih luas.
        "Rendah Volatilitas" sudah dibalik — makin tinggi makin bagus.
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        fig_metrik = plot_metrik_grouped(hasil, split_pilihan)
        st.plotly_chart(fig_metrik, use_container_width=True)

        # Tabel lengkap
        st.markdown("#### Tabel Metrik Lengkap")
        rows = []
        metrik_keys = [
            ("Return/Tahun (%)",     "ann_return",    100,  False),
            ("Volatilitas (%)",      "ann_vol",        100,  True),
            ("Sharpe Ratio",         "sharpe_ratio",     1,  False),
            ("Max Drawdown (%)",     "max_drawdown",   100,  True),
            ("Sortino Ratio",        "sortino_ratio",    1,  False),
            ("Calmar Ratio",         "calmar_ratio",     1,  False),
            ("Win Rate (%)",         "win_rate",        100, False),
            ("VaR 95% (%)",          "var_95",          100,  True),
            ("Total Return (%)",     "total_return",    100, False),
        ]

        for label, key, mult, lower_better in metrik_keys:
            row = {"Metrik": label}
            nilais = {}
            for nama in hasil:
                nilais[nama] = hasil[nama][split_pilihan][key] * mult

            if lower_better:
                best = min(nilais.values())
            else:
                best = max(nilais.values())

            for nama, nilai in nilais.items():
                is_best = abs(nilai - best) < 0.0001
                row[nama] = f"{'★ ' if is_best else ''}{nilai:.4f}"
            rows.append(row)

        df_tabel = pd.DataFrame(rows)
        st.dataframe(df_tabel, hide_index=True, use_container_width=True)

    with tab4:
        fig_bobot = plot_bobot_grouped(hasil)
        st.plotly_chart(fig_bobot, use_container_width=True)

        # Tabel bobot
        st.markdown("#### Tabel Bobot Detail")
        bobot_rows = []
        for ticker in TICKERS:
            row = {"Saham": ticker.replace(".JK", "")}
            for nama, data in hasil.items():
                row[nama] = f"{data['weights'].get(ticker, 0)*100:.2f}%"
            bobot_rows.append(row)
        df_bobot = pd.DataFrame(bobot_rows)
        st.dataframe(df_bobot, hide_index=True, use_container_width=True)

    # ── Kesimpulan Otomatis ───────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🏆 Kesimpulan Otomatis")

    bintang = {nama: 0 for nama in hasil}
    for label, key, mult, lower_better in metrik_keys:
        nilais = {
            nama: hasil[nama]["test"][key] * mult
            for nama in hasil
        }
        best = min(nilais.values()) if lower_better else max(nilais.values())
        for nama, nilai in nilais.items():
            if abs(nilai - best) < 0.0001:
                bintang[nama] += 1

    pemenang = max(bintang, key=bintang.get)
    col_a, col_b, col_c = st.columns(3)

    for col_ui, nama in zip([col_a, col_b, col_c], hasil.keys()):
        with col_ui:
            is_winner = nama == pemenang
            border    = "3px solid " + WARNA[nama] if is_winner else "1px solid #1e2d3d"
            st.markdown(f"""
            <div style='background:#111827;border:{border};border-radius:12px;
            padding:16px;text-align:center;'>
                <div style='font-size:1.5rem;'>
                    {"🥇" if is_winner else "🥈" if bintang[nama] == sorted(bintang.values())[-2] else "🥉"}
                </div>
                <div style='color:{WARNA[nama]};font-weight:700;margin:8px 0;'>
                    {nama}
                </div>
                <div style='color:#f1f5f9;font-size:1.8rem;font-weight:700;
                font-family:monospace;'>{bintang[nama]}</div>
                <div style='color:#64748b;font-size:0.8rem;'>bintang dari {len(metrik_keys)}</div>
                {"<div style='color:#00d4aa;font-size:0.8rem;margin-top:8px;font-weight:700;'>✓ MODEL TERBAIK</div>" if is_winner else ""}
            </div>
            """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style='background:rgba(0,212,170,0.06);border:1px solid rgba(0,212,170,0.2);
    border-radius:10px;padding:16px;margin-top:16px;font-size:0.85rem;color:#6ee7d4;'>
    <strong>📝 Interpretasi:</strong><br>
    Perbandingan di atas menggunakan data TEST (15% terakhir yang tidak pernah dilihat model).
    Model dengan bintang terbanyak menunjukkan performa paling konsisten di berbagai metrik.
    Tidak ada satu model yang selalu terbaik di semua kondisi pasar —
    gunakan hasil ini sebagai panduan, bukan keputusan final.
    </div>
    """, unsafe_allow_html=True)


# ===========================================================================
# JALANKAN MANDIRI
# ===========================================================================

if __name__ == "__main__":
    st.set_page_config(
        page_title="Model Comparison",
        page_icon="⚖️",
        layout="wide"
    )
    tampilkan_halaman_perbandingan()
