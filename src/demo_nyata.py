"""
Demo Penggunaan Data Nyata
==========================
Script ini mendemonstrasikan cara pakai data_fetcher.py
dengan data simulasi yang strukturnya IDENTIK dengan
data nyata dari Yahoo Finance.

Jalankan di terminal:
    python demo_nyata.py

Catatan: Ganti bagian SIMULASI dengan pemanggilan
fetch_stock_data() asli saat dijalankan di luar sandbox.
"""

import pandas as pd
import numpy as np
from datetime import datetime

# ---------------------------------------------------------------------------
# IMPORT DARI FILE KITA
# ---------------------------------------------------------------------------
# from data_fetcher import fetch_stock_data, split_data, SplitResult
# Karena demo ini standalone, kita copy fungsinya langsung di bawah:

from dataclasses import dataclass

@dataclass
class SplitResult:
    train: pd.DataFrame
    val:   pd.DataFrame
    test:  pd.DataFrame
    train_start: datetime
    train_end:   datetime
    val_start:   datetime
    val_end:     datetime
    test_start:  datetime
    test_end:    datetime

    def summary(self) -> str:
        total = len(self.train) + len(self.val) + len(self.test)
        lines = [
            "=" * 58,
            "            RINGKASAN PEMBAGIAN DATA",
            "=" * 58,
            f"  Total data   : {total} hari perdagangan",
            f"  Jumlah saham : {len(self.train.columns)} saham",
            "-" * 58,
            f"  TRAIN  (70%) : {len(self.train):>4} hari  "
            f"[{self.train_start.date()} → {self.train_end.date()}]",
            f"  VAL    (15%) : {len(self.val):>4} hari  "
            f"[{self.val_start.date()} → {self.val_end.date()}]",
            f"  TEST   (15%) : {len(self.test):>4} hari  "
            f"[{self.test_start.date()} → {self.test_end.date()}]",
            "=" * 58,
        ]
        return "\n".join(lines)


def split_data(price_data, train_ratio=0.70, val_ratio=0.15):
    if not isinstance(price_data.index, pd.DatetimeIndex):
        price_data.index = pd.to_datetime(price_data.index)
    price_data = price_data.sort_index()

    n = len(price_data)
    train_end_idx = int(n * train_ratio)
    val_end_idx   = train_end_idx + int(n * val_ratio)

    train = price_data.iloc[:train_end_idx]
    val   = price_data.iloc[train_end_idx:val_end_idx]
    test  = price_data.iloc[val_end_idx:]

    return SplitResult(
        train=train, val=val, test=test,
        train_start=train.index[0].to_pydatetime(),
        train_end=train.index[-1].to_pydatetime(),
        val_start=val.index[0].to_pydatetime(),
        val_end=val.index[-1].to_pydatetime(),
        test_start=test.index[0].to_pydatetime(),
        test_end=test.index[-1].to_pydatetime(),
    )


# ===========================================================================
# KONFIGURASI
# ===========================================================================

TICKERS = [
    "BBCA.JK",   # Bank Central Asia
    "BBRI.JK",   # Bank Rakyat Indonesia
    "BMRI.JK",   # Bank Mandiri
    "TLKM.JK",   # Telkom Indonesia
    "ASII.JK",   # Astra International
    "UNVR.JK",   # Unilever Indonesia
    "GOTO.JK",   # GoTo Group
    "BYAN.JK",   # Bayan Resources
    "ICBP.JK",   # Indofood CBP
    "KLBF.JK",   # Kalbe Farma
]

START_DATE = datetime(2020, 1, 1)
END_DATE   = datetime(2024, 12, 31)


# ===========================================================================
# [OPSI A] CARA PAKAI DATA NYATA (aktifkan saat di luar sandbox)
# ===========================================================================
# import streamlit as st
# from data_fetcher import fetch_stock_data
#
# price_data, failed, split = fetch_stock_data(
#     tickers=TICKERS,
#     start_date=START_DATE,
#     end_date=END_DATE,
#     train_ratio=0.70,
#     val_ratio=0.15,
# )
# if failed:
#     print(f"Saham gagal diunduh: {failed}")


# ===========================================================================
# [OPSI B] SIMULASI DATA (struktur identik dengan Yahoo Finance)
# Harga saham IDX disimulasikan dengan harga awal, volatilitas, dan
# drift yang realistis per saham.
# ===========================================================================

def simulate_idx_prices(
    tickers: list[str],
    start_date: datetime,
    end_date: datetime,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Membuat data harga saham simulasi yang strukturnya identik
    dengan output yfinance (DatetimeIndex, kolom = ticker).

    Setiap saham memiliki harga awal dan volatilitas realistis.
    """
    np.random.seed(seed)

    # Parameter realistis tiap saham (harga awal IDR, volatilitas harian)
    params = {
        "BBCA.JK":  {"harga_awal": 31_000, "vol": 0.013, "drift": 0.0003},
        "BBRI.JK":  {"harga_awal": 4_200,  "vol": 0.016, "drift": 0.0002},
        "BMRI.JK":  {"harga_awal": 7_200,  "vol": 0.014, "drift": 0.0003},
        "TLKM.JK":  {"harga_awal": 3_800,  "vol": 0.012, "drift": 0.0001},
        "ASII.JK":  {"harga_awal": 6_300,  "vol": 0.015, "drift": 0.0001},
        "UNVR.JK":  {"harga_awal": 8_500,  "vol": 0.011, "drift": -0.0001},
        "GOTO.JK":  {"harga_awal": 380,    "vol": 0.028, "drift": -0.0002},
        "BYAN.JK":  {"harga_awal": 15_000, "vol": 0.022, "drift": 0.0005},
        "ICBP.JK":  {"harga_awal": 9_600,  "vol": 0.013, "drift": 0.0002},
        "KLBF.JK":  {"harga_awal": 1_450,  "vol": 0.014, "drift": 0.0001},
    }

    # Hanya hari bursa (Business Days)
    dates = pd.date_range(start=start_date, end=end_date, freq="B")

    data = {}
    for ticker in tickers:
        p = params.get(ticker, {"harga_awal": 5_000, "vol": 0.015, "drift": 0.0002})
        n = len(dates)

        # Geometric Brownian Motion (model harga saham standar)
        daily_returns = np.random.normal(
            loc=p["drift"],
            scale=p["vol"],
            size=n
        )

        # Simulasikan crash kecil di 2020 (COVID) untuk realisme
        crash_mask = (dates >= "2020-02-20") & (dates <= "2020-04-01")
        daily_returns[crash_mask] += np.random.normal(-0.02, 0.03, crash_mask.sum())

        prices = p["harga_awal"] * np.cumprod(1 + daily_returns)
        data[ticker] = prices

    df = pd.DataFrame(data, index=dates)
    df.index.name = "Date"
    return df


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":

    print("=" * 58)
    print("   DEMO DATA FETCHER + SPLIT (70 / 15 / 15)")
    print("=" * 58)

    # 1. Buat / unduh data
    print("\n[1] Membuat data simulasi IDX...")
    price_data = simulate_idx_prices(TICKERS, START_DATE, END_DATE)
    print(f"    ✓ Data berhasil: {price_data.shape[0]} baris x {price_data.shape[1]} saham")
    print(f"    Rentang: {price_data.index[0].date()} → {price_data.index[-1].date()}")

    # 2. Split data
    print("\n[2] Membagi data (70% Train | 15% Val | 15% Test)...")
    split = split_data(price_data, train_ratio=0.70, val_ratio=0.15)
    print(split.summary())

    # 3. Preview tiap split
    print("\n[3] Preview data TRAIN (5 baris pertama):")
    print(split.train.head().to_string())

    print("\n[4] Preview data VAL (5 baris pertama):")
    print(split.val.head().to_string())

    print("\n[5] Preview data TEST (5 baris pertama):")
    print(split.test.head().to_string())

    # 4. Statistik return per split
    print("\n[6] Statistik Return Harian per Split:")
    for nama, df in [("TRAIN", split.train), ("VAL", split.val), ("TEST", split.test)]:
        ret = df.pct_change().dropna()
        print(f"\n  {nama}:")
        print(f"    Mean return  : {ret.mean().mean()*100:.4f}%")
        print(f"    Std (vol)    : {ret.std().mean()*100:.4f}%")
        print(f"    Min          : {ret.min().min()*100:.4f}%")
        print(f"    Max          : {ret.max().max()*100:.4f}%")

    # 5. Contoh integrasi dengan PortfolioOptimizer
    print("\n[7] Contoh integrasi dengan PortfolioOptimizer:")
    print("""
    from portfolio_optimizer import PortfolioOptimizer

    # Optimasi HANYA menggunakan data TRAIN
    optimizer = PortfolioOptimizer(split.train, risk_free_rate=0.0575)
    weights, stats = optimizer.optimize("max_sharpe")

    print("Bobot Optimal:", weights)
    print("Sharpe Ratio :", stats['sharpe_ratio'])

    # Evaluasi di VAL
    opt_val = PortfolioOptimizer(split.val, risk_free_rate=0.0575)
    _, stats_val = opt_val.optimize("max_sharpe")
    print("Sharpe di Val:", stats_val['sharpe_ratio'])

    # Evaluasi akhir di TEST
    opt_test = PortfolioOptimizer(split.test, risk_free_rate=0.0575)
    _, stats_test = opt_test.optimize("max_sharpe")
    print("Sharpe di Test:", stats_test['sharpe_ratio'])
    """)

    print("\n✓ Demo selesai!")
    print("  Untuk data nyata, aktifkan [OPSI A] di atas dan jalankan")
    print("  di environment lokal yang memiliki akses internet.")
