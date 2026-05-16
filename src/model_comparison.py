"""
Model Comparison Module
=======================
Membandingkan 3 model optimasi portofolio:
  1. Markowitz Efficient Frontier  — Modern Portfolio Theory (Max Sharpe)
  2. HRP            — Hierarchical Risk Parity
  3. Equal Weight   — Baseline sederhana (pembanding)

Menggunakan MLflow untuk mencatat semua hasil eksperimen
sehingga bisa dibandingkan secara objektif.

Jalankan:
    python src/model_comparison.py
"""
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import json
import warnings
from datetime import datetime
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt import objective_functions
import yfinance as yf

warnings.filterwarnings("ignore")


# ===========================================================================
# KONFIGURASI EKSPERIMEN
# ===========================================================================

TICKERS = [
    "BNGA.JK",   # Perbankan
    "ADRO.JK",   # Energi Batubara
    "ANTM.JK",   # Pertambangan
    "MDKA.JK",   # Pertambangan Emas
    "PGAS.JK",   # Energi Gas
]

START_DATE  = "2022-01-01"
END_DATE    = "2024-12-31"
RISK_FREE   = 0.0575       # BI Rate 5.75%
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# test_ratio otomatis = 0.15


# ===========================================================================
# FUNGSI BANTU
# ===========================================================================

def fetch_data(tickers: list, start: str, end: str) -> pd.DataFrame:
    """Unduh data harga saham dari Yahoo Finance."""
    print(f"\n📡 Mengunduh data {len(tickers)} saham...")
    all_data = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist  = stock.history(start=start, end=end, interval="1d")
            if not hist.empty and len(hist) >= 50:
                all_data[ticker] = hist["Close"]
                print(f"   ✓ {ticker}: {len(hist)} hari")
            else:
                print(f"   ✗ {ticker}: data tidak cukup")
        except Exception as e:
            print(f"   ✗ {ticker}: error — {e}")

    if not all_data:
        raise ValueError("Tidak ada data yang berhasil diunduh!")

    df = pd.DataFrame(all_data)
    df = df.ffill().bfill()
    print(f"   Total: {df.shape[0]} hari x {df.shape[1]} saham")
    return df


def split_data(price_data: pd.DataFrame,
               train_ratio: float = 0.70,
               val_ratio: float   = 0.15):
    """Split data secara kronologis 70/15/15."""
    n             = len(price_data)
    train_end     = int(n * train_ratio)
    val_end       = train_end + int(n * val_ratio)

    train = price_data.iloc[:train_end]
    val   = price_data.iloc[train_end:val_end]
    test  = price_data.iloc[val_end:]

    print(f"\n✂️  Split Data:")
    print(f"   TRAIN : {len(train)} hari [{train.index[0].date()} → {train.index[-1].date()}]")
    print(f"   VAL   : {len(val)}   hari [{val.index[0].date()} → {val.index[-1].date()}]")
    print(f"   TEST  : {len(test)}  hari [{test.index[0].date()} → {test.index[-1].date()}]")

    return train, val, test


def hitung_metrik(price_data: pd.DataFrame,
                  weights: dict,
                  risk_free: float = 0.0575) -> dict:
    """
    Hitung semua metrik performa portofolio.

    Parameters:
    -----------
    price_data : pd.DataFrame — data harga
    weights    : dict         — bobot per saham
    risk_free  : float        — BI Rate

    Returns:
    --------
    dict : semua metrik
    """
    returns = price_data.pct_change().dropna()

    # Return harian portofolio
    port_ret = sum(
        returns[col] * weights.get(col, 0)
        for col in returns.columns
        if col in weights
    )

    # Metrik utama
    ann_return  = (1 + port_ret.mean()) ** 252 - 1
    ann_vol     = port_ret.std() * np.sqrt(252)
    sharpe      = (ann_return - risk_free) / ann_vol if ann_vol > 0 else 0

    # Drawdown
    cumulative  = (1 + port_ret).cumprod()
    rolling_max = cumulative.cummax()
    drawdown    = cumulative / rolling_max - 1
    max_dd      = drawdown.min()

    # VaR & CVaR
    var_95      = np.percentile(port_ret, 5)
    cvar_95     = port_ret[port_ret <= var_95].mean()

    # Sortino
    downside    = port_ret[port_ret < 0].std() * np.sqrt(252)
    sortino     = (ann_return - risk_free) / downside if downside > 0 else 0

    # Calmar
    calmar      = ann_return / abs(max_dd) if max_dd != 0 else 0

    # Win Rate
    win_rate    = (port_ret > 0).mean()

    # Total Return
    total_ret   = cumulative.iloc[-1] - 1

    return {
        "ann_return"   : round(ann_return, 6),
        "ann_vol"      : round(ann_vol, 6),
        "sharpe_ratio" : round(sharpe, 6),
        "max_drawdown" : round(max_dd, 6),
        "var_95"       : round(var_95, 6),
        "cvar_95"      : round(cvar_95, 6),
        "sortino_ratio": round(sortino, 6),
        "calmar_ratio" : round(calmar, 6),
        "win_rate"     : round(win_rate, 6),
        "total_return" : round(total_ret, 6),
    }

# ===========================================================================
# MODEL 1: MARKOWITZ  (Max Sharpe)
# ===========================================================================

def model_markowitz(train_data: pd.DataFrame,
                    risk_free: float = 0.0575,
                    weight_bounds: tuple = (0.05, 0.40)) -> dict:
    """
    Optimasi portofolio menggunakan Markowitz Efficient Frontier.
    Mencari bobot yang memaksimalkan Sharpe Ratio.

    Cara kerja:
    - Hitung expected return dari rata-rata historis
    - Hitung matriks kovariansi dengan Ledoit-Wolf shrinkage
    - Cari bobot yang memaksimalkan (Return - Rf) / Volatilitas

    Parameters:
    -----------
    train_data    : pd.DataFrame — data training
    risk_free     : float        — BI Rate
    weight_bounds : tuple        — (min_bobot, max_bobot)

    Returns:
    --------
    dict : bobot optimal per saham
    """
    print("\n🔵 Menghitung Model Markowitz Efficient Frontier...")

    # Estimasi parameter
    mu = expected_returns.mean_historical_return(train_data)
    S  = risk_models.CovarianceShrinkage(train_data).ledoit_wolf()

    # Optimasi
    ef = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
    ef.add_objective(objective_functions.L2_reg, gamma=0.1)
    ef.max_sharpe(risk_free_rate=risk_free)

    weights = ef.clean_weights(cutoff=0.01)
    weights = {k: v for k, v in weights.items() if v > 0.001}

    print(f"   ✓ Bobot: {weights}")
    return dict(weights)


# ===========================================================================
# MODEL 2: HRP (Hierarchical Risk Parity)
# ===========================================================================

def model_hrp(train_data: pd.DataFrame) -> dict:
    """
    Optimasi portofolio menggunakan Hierarchical Risk Parity (HRP).

    Cara kerja HRP (berbeda dari Markowitz):
    1. Hitung matriks korelasi antar saham
    2. Ubah korelasi menjadi jarak: distance = sqrt(0.5 * (1 - korelasi))
    3. Lakukan hierarchical clustering — kelompokkan saham yang mirip
    4. Alokasikan bobot berdasarkan risiko tiap cluster
       (saham berisiko tinggi dapat bobot lebih kecil)

    Keunggulan HRP vs Markowitz:
    - Tidak butuh invers matriks kovariansi (lebih stabil)
    - Lebih tahan terhadap error estimasi
    - Diversifikasi lebih merata berdasarkan hierarki risiko

    Parameters:
    -----------
    train_data : pd.DataFrame — data training

    Returns:
    --------
    dict : bobot optimal per saham
    """
    print("\n🟢 Menghitung Model HRP (Hierarchical Risk Parity)...")

    returns = train_data.pct_change().dropna()
    tickers = list(returns.columns)
    n       = len(tickers)

    # Langkah 1: Matriks korelasi
    corr = returns.corr()

    # Langkah 2: Ubah korelasi → jarak
    # Rumus: distance = sqrt(0.5 * (1 - korelasi))
    # Jarak = 0 berarti identik, jarak = 1 berarti tidak berkorelasi
    dist_matrix = np.sqrt(0.5 * (1 - corr.values))
    np.fill_diagonal(dist_matrix, 0)

    # Langkah 3: Hierarchical clustering
    # Gunakan metode "single linkage" — standar untuk HRP
    dist_condensed = squareform(dist_matrix)
    linkage_matrix  = linkage(dist_condensed, method="single")

    # Langkah 4: Urutkan saham berdasarkan hierarki
    # Tujuan: saham yang mirip dikelompokkan berdekatan
    def get_quasi_diag(link):
        """Rekursif — susun ulang indeks berdasarkan dendrogram."""
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]
        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            df0 = sort_ix[sort_ix >= num_items]
            i   = df0.index
            j   = df0.values - num_items
            sort_ix[i] = link[j, 0]
            df0_new     = pd.Series(link[j, 1], index=i + 1)
            sort_ix     = pd.concat([sort_ix, df0_new])
            sort_ix     = sort_ix.sort_index()
            sort_ix.index = range(sort_ix.shape[0])
        return sort_ix.tolist()

    sorted_idx = get_quasi_diag(linkage_matrix)
    sorted_idx = [i for i in sorted_idx if i < n]
    sorted_tickers = [tickers[i] for i in sorted_idx]

    # Langkah 5: Alokasi bobot rekursif berdasarkan risiko
    # Prinsip: cluster yang lebih berisiko dapat bobot lebih kecil
    def get_cluster_var(cov, cluster_items):
        """Hitung varians portofolio untuk satu cluster."""
        cov_slice  = cov.loc[cluster_items, cluster_items]
        # Inverse variance weighting dalam cluster
        ivp        = 1.0 / np.diag(cov_slice.values)
        ivp       /= ivp.sum()
        w          = pd.Series(ivp, index=cluster_items)
        cluster_var = np.dot(w.values, np.dot(cov_slice.values, w.values))
        return cluster_var

    def hrp_recursive(cov, sorted_items):
        """Alokasi bobot HRP secara rekursif."""
        weights = pd.Series(1.0, index=sorted_items)
        clusters = [sorted_items]

        while clusters:
            clusters = [
                i[j:k]
                for i in clusters
                for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
                if len(i) > 1
            ]

            for i in range(0, len(clusters), 2):
                if i + 1 >= len(clusters):
                    break
                cluster0   = clusters[i]
                cluster1   = clusters[i + 1]
                var0       = get_cluster_var(cov, cluster0)
                var1       = get_cluster_var(cov, cluster1)
                alpha      = 1 - var0 / (var0 + var1)
                weights[cluster0] *= alpha
                weights[cluster1] *= (1 - alpha)

        return weights

    cov      = returns.cov() * 252  # annualized
    weights  = hrp_recursive(cov, sorted_tickers)
    weights  = weights / weights.sum()  # normalisasi

    result = {ticker: round(float(w), 6) for ticker, w in weights.items()}
    print(f"   ✓ Bobot: {result}")
    return result


# ===========================================================================
# MODEL 3: EQUAL WEIGHT (Baseline)
# ===========================================================================

def model_equal_weight(tickers: list) -> dict:
    """
    Portofolio dengan bobot sama rata — dipakai sebagai baseline.

    Cara kerja:
    - Bagi modal secara merata ke semua saham
    - Tidak ada optimasi matematika sama sekali
    - Dipakai sebagai pembanding: apakah Markowitz & HRP
      benar-benar lebih baik dari sekedar bagi rata?

    Parameters:
    -----------
    tickers : list — daftar kode saham

    Returns:
    --------
    dict : bobot sama rata per saham
    """
    print("\n⚪ Menghitung Model Equal Weight (Baseline)...")
    w      = 1.0 / len(tickers)
    result = {t: round(w, 6) for t in tickers}
    print(f"   ✓ Bobot: {result}")
    return result


# ===========================================================================
# EKSPERIMEN UTAMA DENGAN MLFLOW
# ===========================================================================

def jalankan_eksperimen():
    """
    Fungsi utama yang menjalankan semua eksperimen dan mencatat ke MLflow.

    Alur:
    1. Unduh data saham
    2. Split data 70/15/15
    3. Latih 3 model menggunakan data TRAIN
    4. Evaluasi di VAL dan TEST
    5. Log semua parameter dan metrik ke MLflow
    6. Tampilkan ringkasan perbandingan
    """

    print("=" * 65)
    print("   EKSPERIMEN PERBANDINGAN MODEL PORTOFOLIO")
    print("   Markowitz Efficient Frontier vs HRP vs Equal Weight")
    print("=" * 65)

    # ── 1. Unduh Data ─────────────────────────────────────────────
    price_data = fetch_data(TICKERS, START_DATE, END_DATE)

    # ── 2. Split Data ──────────────────────────────────────────────
    train, val, test = split_data(price_data, TRAIN_RATIO, VAL_RATIO)

    # ── 3. Definisi semua model ────────────────────────────────────
    models = {
        "Markowitz_EF": lambda: model_markowitz(
            train, risk_free=RISK_FREE, weight_bounds=(0.05, 0.40)
        ),
        "HRP": lambda: model_hrp(train),
        "Equal_Weight": lambda: model_equal_weight(list(train.columns)),
    }

    # ── 4. Setup MLflow ───────────────────────────────────────────
    mlflow.set_experiment("IDX_Portfolio_Model_Comparison")

    semua_hasil = {}

    for nama_model, fungsi_model in models.items():

        print(f"\n{'='*65}")
        print(f"  Model: {nama_model}")
        print(f"{'='*65}")

        with mlflow.start_run(run_name=f"{nama_model}_{datetime.now().strftime('%H%M%S')}"):

            # Log parameter umum
            mlflow.log_param("model", nama_model)
            mlflow.log_param("tickers", str(TICKERS))
            mlflow.log_param("n_saham", len(TICKERS))
            mlflow.log_param("start_date", START_DATE)
            mlfw_params = {
                "end_date"   : END_DATE,
                "risk_free"  : RISK_FREE,
                "train_ratio": TRAIN_RATIO,
                "val_ratio"  : VAL_RATIO,
                "train_days" : len(train),
                "val_days"   : len(val),
                "test_days"  : len(test),
            }
            for k, v in mlfw_params.items():
                mlflow.log_param(k, v)

            # Tambahan parameter khusus Markowitz
            if nama_model == "Markowitz_EF":
                mlflow.log_param("weight_bound_min", 0.05)
                mlflow.log_param("weight_bound_max", 0.40)
                mlflow.log_param("regularisasi", "L2_gamma=0.1")
                mlflow.log_param("solver", "CLARABEL")
                mlflow.log_param("kovariansi", "Ledoit-Wolf Shrinkage")

            elif nama_model == "HRP":
                mlflow.log_param("clustering_method", "single_linkage")
                mlflow.log_param("distance_formula", "sqrt(0.5*(1-corr))")
                mlflow.log_param("alokasi", "recursive_bisection")

            elif nama_model == "Equal_Weight":
                mlflow.log_param("bobot_per_saham", round(1/len(TICKERS), 4))

            # Hitung bobot model
            weights = fungsi_model()

            # Log bobot sebagai parameter
            for ticker, w in weights.items():
                mlflow.log_param(f"bobot_{ticker.replace('.JK','')}", round(w, 4))

            # Evaluasi di semua split
            hasil_split = {}
            for split_name, split_data_df in [
                ("train", train),
                ("val", val),
                ("test", test)
            ]:
                metrik = hitung_metrik(split_data_df, weights, RISK_FREE)
                hasil_split[split_name] = metrik

                # Log metrik ke MLflow dengan prefix split
                for nama_metrik, nilai in metrik.items():
                    mlflow.log_metric(f"{split_name}_{nama_metrik}", nilai)

                print(f"\n   📊 [{split_name.upper()}]")
                print(f"      Return/Tahun  : {metrik['ann_return']*100:.2f}%")
                print(f"      Volatilitas   : {metrik['ann_vol']*100:.2f}%")
                print(f"      Sharpe Ratio  : {metrik['sharpe_ratio']:.4f}")
                print(f"      Max Drawdown  : {metrik['max_drawdown']*100:.2f}%")
                print(f"      Sortino Ratio : {metrik['sortino_ratio']:.4f}")
                print(f"      Calmar Ratio  : {metrik['calmar_ratio']:.4f}")
                print(f"      Win Rate      : {metrik['win_rate']*100:.1f}%")
                print(f"      VaR 95%       : {metrik['var_95']*100:.2f}%")

            # Simpan bobot ke file JSON sebagai artifact
            bobot_file = f"weights_{nama_model}.json"
            with open(bobot_file, "w") as f:
                json.dump({
                    "model"  : nama_model,
                    "weights": weights,
                    "metrik" : hasil_split
                }, f, indent=2)
            mlflow.log_artifact(bobot_file)

            semua_hasil[nama_model] = {
                "weights": weights,
                "metrik" : hasil_split
            }

    # ── 5. Tampilkan Ringkasan Perbandingan ───────────────────────
    print("\n" + "=" * 65)
    print("   RINGKASAN PERBANDINGAN MODEL")
    print("=" * 65)

    # Tabel metrik TEST (evaluasi akhir yang jujur)
    header = f"{'Metrik':<22} {'Markowitz':>12} {'HRP':>12} {'EqWeight':>12}"
    print(f"\n📊 EVALUASI TEST SET (data yang tidak pernah dilihat model):")
    print("-" * 65)
    print(header)
    print("-" * 65)

    metrik_tampil = [
        ("Return/Tahun (%)",    "ann_return",    100),
        ("Volatilitas (%)",     "ann_vol",        100),
        ("Sharpe Ratio",        "sharpe_ratio",     1),
        ("Max Drawdown (%)",    "max_drawdown",   100),
        ("Sortino Ratio",       "sortino_ratio",    1),
        ("Calmar Ratio",        "calmar_ratio",     1),
        ("Win Rate (%)",        "win_rate",        100),
        ("VaR 95% (%)",         "var_95",          100),
        ("Total Return (%)",    "total_return",    100),
    ]

    for label, key, multiplier in metrik_tampil:
        mpt = semua_hasil["Markowitz_EF"]["metrik"]["test"][key] * multiplier
        hrp = semua_hasil["HRP"]["metrik"]["test"][key] * multiplier
        eqw = semua_hasil["Equal_Weight"]["metrik"]["test"][key] * multiplier

        # Tandai yang terbaik dengan ★
        if key in ["ann_return", "sharpe_ratio", "sortino_ratio",
                   "calmar_ratio", "win_rate", "total_return"]:
            # Makin tinggi makin baik
            best = max(mpt, hrp, eqw)
            mpt_str = f"{mpt:>10.4f} {'★' if abs(mpt-best)<0.0001 else ' '}"
            hrp_str = f"{hrp:>10.4f} {'★' if abs(hrp-best)<0.0001 else ' '}"
            eqw_str = f"{eqw:>10.4f} {'★' if abs(eqw-best)<0.0001 else ' '}"
        else:
            # Makin rendah makin baik (volatilitas, drawdown, VaR)
            best = min(mpt, hrp, eqw)
            mpt_str = f"{mpt:>10.4f} {'★' if abs(mpt-best)<0.0001 else ' '}"
            hrp_str = f"{hrp:>10.4f} {'★' if abs(hrp-best)<0.0001 else ' '}"
            eqw_str = f"{eqw:>10.4f} {'★' if abs(eqw-best)<0.0001 else ' '}"

        print(f"{label:<22} {mpt_str:>12} {hrp_str:>12} {eqw_str:>12}")

    print("-" * 65)
    print("★ = Terbaik di kategori tersebut")

    # Hitung total bintang tiap model
    bintang = {"Markowitz_EF": 0, "HRP": 0, "Equal_Weight": 0}
    for label, key, multiplier in metrik_tampil:
        mpt = semua_hasil["Markowitz_EF"]["metrik"]["test"][key]
        hrp = semua_hasil["HRP"]["metrik"]["test"][key]
        eqw = semua_hasil["Equal_Weight"]["metrik"]["test"][key]

        if key in ["ann_return", "sharpe_ratio", "sortino_ratio",
                   "calmar_ratio", "win_rate", "total_return"]:
            best = max(mpt, hrp, eqw)
        else:
            best = min(mpt, hrp, eqw)

        if abs(mpt - best) < 0.0001:
            bintang["Markowitz_EF"] += 1
        if abs(hrp - best) < 0.0001:
            bintang["HRP"] += 1
        if abs(eqw - best) < 0.0001:
            bintang["Equal_Weight"] += 1

    print(f"\n🏆 TOTAL BINTANG:")
    print(f"   Markowitz Efficient Frontier : {bintang['Markowitz_EF']} bintang")
    print(f"   HRP           : {bintang['HRP']} bintang")
    print(f"   Equal Weight  : {bintang['Equal_Weight']} bintang")

    pemenang = max(bintang, key=bintang.get)
    nama_pemenang = {
        "Markowitz_EF": "Markowitz Efficient Frontier",
        "HRP"          : "HRP (Hierarchical Risk Parity)",
        "Equal_Weight" : "Equal Weight"
    }

    print(f"\n🥇 MODEL TERBAIK: {nama_pemenang[pemenang]}")

    # Log hasil akhir ke MLflow
    with mlflow.start_run(run_name="RINGKASAN_PERBANDINGAN"):
        mlflow.log_param("model_terbaik", pemenang)
        for model, stars in bintang.items():
            mlflow.log_metric(f"bintang_{model}", stars)
        for model in ["Markowitz_EF", "HRP", "Equal_Weight"]:
            for split in ["train", "val", "test"]:
                sharpe = semua_hasil[model]["metrik"][split]["sharpe_ratio"]
                mlflow.log_metric(f"{model}_{split}_sharpe", sharpe)

    print(f"\n✅ Semua hasil tersimpan di MLflow.")
    print(f"   Jalankan: mlflow ui")
    print(f"   Buka    : http://localhost:5000")
    print("=" * 65)

    return semua_hasil


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    hasil = jalankan_eksperimen()
