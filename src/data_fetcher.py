"""
Data Fetcher Module
Mengunduh data historis saham IDX dari Yahoo Finance
+ Split data Train / Validation / Test (70% / 15% / 15%)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
import time
import streamlit as st


# ===========================================================================
# DATACLASS HASIL SPLIT
# ===========================================================================

@dataclass
class SplitResult:
    """
    Menyimpan hasil pembagian data beserta informasi tanggal dan ukurannya.

    Attributes:
    -----------
    train : pd.DataFrame  — Data latih (70%)
    val   : pd.DataFrame  — Data validasi (15%)
    test  : pd.DataFrame  — Data uji akhir (15%)
    """
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
        """Tampilkan ringkasan pembagian data."""
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


# ===========================================================================
# FUNGSI SPLIT DATA
# ===========================================================================

def split_data(
    price_data: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> SplitResult:
    """
    Membagi DataFrame harga saham secara kronologis menjadi
    Train / Validation / Test.

    Data TIDAK diacak agar urutan waktu tetap terjaga.

    Parameters:
    -----------
    price_data : pd.DataFrame
        DataFrame harga saham hasil fetch_stock_data()
    train_ratio : float
        Proporsi data train (default: 0.70 → 70%)
    val_ratio : float
        Proporsi data validasi (default: 0.15 → 15%)
        Sisa otomatis menjadi test (15%)

    Returns:
    --------
    SplitResult : Objek berisi train, val, test beserta info tanggalnya

    Raises:
    -------
    ValueError : Jika data terlalu sedikit atau rasio tidak valid
    """
    # Pastikan index bertipe datetime dan terurut
    if not isinstance(price_data.index, pd.DatetimeIndex):
        price_data.index = pd.to_datetime(price_data.index)
    price_data = price_data.sort_index()

    test_ratio = round(1.0 - train_ratio - val_ratio, 10)

    # Validasi rasio
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Total train_ratio + val_ratio harus kurang dari 1.0")
    if test_ratio <= 0:
        raise ValueError(
            "test_ratio bernilai 0 atau negatif. "
            "Kurangi train_ratio atau val_ratio."
        )

    n = len(price_data)
    if n < 100:
        raise ValueError(
            f"Data terlalu sedikit ({n} baris). "
            "Minimal 100 hari perdagangan diperlukan."
        )

    # Hitung indeks batas
    train_end_idx = int(n * train_ratio)
    val_end_idx   = train_end_idx + int(n * val_ratio)

    # Validasi tiap split minimal 30 baris
    for nama, ukuran in [
        ("Train", train_end_idx),
        ("Val",   val_end_idx - train_end_idx),
        ("Test",  n - val_end_idx),
    ]:
        if ukuran < 30:
            raise ValueError(
                f"Split '{nama}' terlalu kecil ({ukuran} baris). "
                "Tambahkan lebih banyak data historis."
            )

    train = price_data.iloc[:train_end_idx]
    val   = price_data.iloc[train_end_idx:val_end_idx]
    test  = price_data.iloc[val_end_idx:]

    return SplitResult(
        train=train,
        val=val,
        test=test,
        train_start=train.index[0].to_pydatetime(),
        train_end=train.index[-1].to_pydatetime(),
        val_start=val.index[0].to_pydatetime(),
        val_end=val.index[-1].to_pydatetime(),
        test_start=test.index[0].to_pydatetime(),
        test_end=test.index[-1].to_pydatetime(),
    )


# ===========================================================================
# FUNGSI FETCH DATA
# ===========================================================================

def fetch_stock_data(
    tickers: list[str],
    start_date: datetime,
    end_date: datetime,
    max_retries: int = 3,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame | None, list[str], SplitResult | None]:
    """
    Mengunduh data harga penutupan saham IDX dari Yahoo Finance
    sekaligus membaginya menjadi Train / Validation / Test.

    Parameters:
    -----------
    tickers : list[str]
        Daftar kode saham dengan suffix .JK (contoh: ['BBCA.JK', 'TLKM.JK'])
    start_date : datetime
        Tanggal mulai pengambilan data
    end_date : datetime
        Tanggal akhir pengambilan data
    max_retries : int
        Jumlah maksimal percobaan ulang jika gagal (default: 3)
    train_ratio : float
        Proporsi data train (default: 0.70)
    val_ratio : float
        Proporsi data validasi (default: 0.15)

    Returns:
    --------
    tuple :
        - pd.DataFrame        : Seluruh data harga (sebelum split)
        - list[str]           : Daftar saham yang gagal diunduh
        - SplitResult | None  : Hasil split train/val/test
                                (None jika data tidak cukup)
    """
    all_data = {}
    failed_stocks = []

    progress_bar = st.progress(0, text="Mengunduh data saham...")

    for i, ticker in enumerate(tickers):
        progress = (i + 1) / len(tickers)
        progress_bar.progress(progress, text=f"Mengunduh {ticker.replace('.JK', '')}...")

        for attempt in range(max_retries):
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(
                    start=start_date.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d"),
                    interval="1d"
                )

                if hist.empty or len(hist) < 50:
                    raise ValueError(
                        f"Data tidak cukup untuk {ticker}: hanya {len(hist)} baris"
                    )

                all_data[ticker] = hist["Close"]
                break  # Sukses, keluar dari loop retry

            except Exception:
                if attempt == max_retries - 1:
                    failed_stocks.append(ticker)
                else:
                    time.sleep(1)

    progress_bar.empty()

    if not all_data:
        return None, failed_stocks, None

    # Gabungkan & bersihkan data
    df = pd.DataFrame(all_data)
    # Forward fill untuk hari libur/tidak ada perdagangan
    df = df.ffill()
    # Drop baris dengan terlalu banyak NaN (misalnya saham baru)
    df = df.dropna(thresh=int(len(df.columns) * 0.8)) # Minimal 80% data tersedia
    # Drop kolom (saham) yang memiliki lebih dari 10% NaN
    df = df.loc[:, df.isnull().mean() < 0.1]
    # Isi sisa NaN dengan backward fill lalu forward fill
    df = df.bfill().ffill()

    # Split data
    try:
        split = split_data(df, train_ratio=train_ratio, val_ratio=val_ratio)
    except ValueError:
        split = None

    return df, failed_stocks, split


def get_stock_info(ticker: str) -> dict:
    """
    Mengambil informasi dasar sebuah saham IDX.

    Parameters:
    -----------
    ticker : str
        Kode saham (contoh: 'BBCA.JK')

    Returns:
    --------
    dict : Informasi saham (nama, sektor, market cap, dll.)
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        return {
            "nama": info.get("longName", ticker),
            "sektor": info.get("sector", "N/A"),
            "industri": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE", None),
            "dividen_yield": info.get("dividendYield", 0),
            "52w_high": info.get("fiftyTwoWeekHigh", None),
            "52w_low": info.get("fiftyTwoWeekLow", None),
            "harga_terakhir": info.get("currentPrice", None),
            "mata_uang": info.get("currency", "IDR"),
            "website": info.get("website", ""),
            "deskripsi": info.get("longBusinessSummary", "")[:300] + "..."
                         if info.get("longBusinessSummary") else ""
        }
    except Exception:
        return {"nama": ticker, "sektor": "N/A"}


def get_benchmark_data(
    start_date: datetime,
    end_date: datetime,
    benchmark: str = "^JKSE"
) -> pd.Series:
    """
    Mengambil data benchmark (default: IHSG).

    Parameters:
    -----------
    start_date, end_date : datetime
    benchmark : str
        Yahoo Finance ticker untuk benchmark ('^JKSE' = IHSG)

    Returns:
    --------
    pd.Series : Harga penutupan benchmark
    """
    try:
        data = yf.download(
            benchmark,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False
        )["Close"]

        if hasattr(data, 'squeeze'):
            data = data.squeeze()

        return data.ffill().dropna()
    except Exception:
        return pd.Series(dtype=float)


def validate_ticker(ticker: str) -> bool:
    """
    Memvalidasi apakah kode saham valid di Yahoo Finance.

    Parameters:
    -----------
    ticker : str
        Kode saham untuk divalidasi

    Returns:
    --------
    bool : True jika valid, False jika tidak
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")
        return not hist.empty
    except Exception:
        return False
