"""
Data Fetcher Module
Mengunduh data historis saham IDX dari Yahoo Finance
+ Split data Train / Validation / Test (70% / 15% / 15%)
"""

import os
import json
import urllib.request
import urllib.error
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
import time
import streamlit as st
from curl_cffi import requests as cffi_requests


# ===========================================================================
# KONFIGURASI PROXY & CACHE (biar resilien terhadap rate-limit Yahoo)
# ===========================================================================
# Kenapa ini ada: Yahoo Finance kadang blokir sementara (rate-limit) satu
# alamat IP yang terlalu sering request - ini yang bikin app jalan mulus
# pas di-deploy (IP server beda, belum kena flag) tapi gagal di lokal
# (IP kamu udah sering nembak Yahoo pas develop/testing).
#
# Fix-nya dua lapis:
#   1. Cache lokal (folder cache_saham/) - biar data yang sama nggak
#      diminta ulang ke Yahoo tiap kali app di-restart pas development.
#      Ini biasanya paling berdampak, karena akar masalahnya adalah
#      volume request, bukan cuma soal proxy.
#   2. Fallback proxy - kalau permintaan langsung gagal, coba proxy
#      lain bergantian, dipasangkan dengan browser-impersonation
#      (curl_cffi) karena Yahoo juga mendeteksi dari fingerprint
#      TLS/browser, bukan cuma dari IP-nya.
#
# GANTI daftar di bawah dengan proxy asli kamu (format:
# "http://user:pass@host:port" atau "http://host:port" tanpa auth).
# Proxy gratis/publik biasanya sudah diblacklist Yahoo duluan -
# proxy residential/rotating berbayar jauh lebih mungkin berhasil.
PROXIES: list[str | None] = [
    None,  # coba koneksi langsung dulu (siapa tau blokirnya udah lepas)
    "http://user:pass@proxy1.example.com:8000",
    "http://user:pass@proxy2.example.com:8000",
    "http://user:pass@proxy3.example.com:8000",
]

CACHE_DIR = "cache_saham"
os.makedirs(CACHE_DIR, exist_ok=True)

_WARNED_PLACEHOLDER = False

# Diisi tiap kali satu ticker gagal total, supaya app.py bisa nampilin
# rincian teknisnya di UI (expander) tanpa perlu buka terminal.
LAST_FETCH_ERRORS: dict[str, str] = {}


def _warn_if_placeholder_proxies() -> None:
    """Cetak peringatan sekali kalau PROXIES masih berisi URL contoh."""
    global _WARNED_PLACEHOLDER
    if _WARNED_PLACEHOLDER:
        return
    placeholder_hit = [p for p in PROXIES if p and "example.com" in p]
    if placeholder_hit:
        print(
            "[data_fetcher] PERINGATAN: PROXIES masih berisi URL contoh "
            f"({placeholder_hit}) - ini TIDAK akan pernah berhasil konek. "
            "Isi dengan proxy asli (yang kamu bayar/punya), atau kosongkan "
            "list-nya jadi cuma [None] kalau memang belum ada proxy - cache "
            "lokal tetap akan jalan tanpa proxy."
        )
    _WARNED_PLACEHOLDER = True


def _get_session(proxy: str | None) -> cffi_requests.Session:
    """Session yang menyamar sebagai browser Chrome asli, opsional lewat proxy."""
    return cffi_requests.Session(impersonate="chrome", proxy=proxy)


_CHROME_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def _method_requests_useragent(
    ticker: str, start_date: datetime, end_date: datetime, interval: str
) -> pd.DataFrame:
    """Method 1: requests.Session polos + header User-Agent Chrome."""
    import requests  # lib beda dari curl_cffi, cuma dipakai di method fallback ini

    session = requests.Session()
    session.headers.update({
        "User-Agent": _CHROME_UA,
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
    })
    return yf.Ticker(ticker, session=session).history(
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        interval=interval,
    )


def _method_yf_download(
    ticker: str, start_date: datetime, end_date: datetime, interval: str
) -> pd.DataFrame:
    """Method 2: yf.download() polos, tanpa session/proxy kustom."""
    hist = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)
    return hist


def _method_raw_urllib(
    ticker: str, start_date: datetime, end_date: datetime, interval: str
) -> pd.DataFrame:
    """Method 3: urllib langsung ke endpoint v8/finance/chart Yahoo."""
    period1 = int(start_date.timestamp())
    period2 = int(end_date.timestamp())
    url = (
        f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={period1}&period2={period2}&interval={interval}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": _CHROME_UA})
    with urllib.request.urlopen(req, timeout=10) as resp:
        payload = json.loads(resp.read().decode())

    result = payload["chart"]["result"][0]
    timestamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]
    hist = pd.DataFrame(
        {
            "Open": quote["open"],
            "High": quote["high"],
            "Low": quote["low"],
            "Close": quote["close"],
            "Volume": quote["volume"],
        },
        index=pd.to_datetime(timestamps, unit="s"),
    )
    return hist.dropna(how="all")


# Dicoba sebagai lapisan TERAKHIR, cuma sekali per method, kalau curl_cffi +
# semua proxy di atas sudah gagal total. Lihat catatan jujur di
# _fetch_history_resilient soal kenapa ini kemungkinan besar TETAP gagal
# untuk blokir yang berbasis fingerprint TLS (bukan cuma header).
_FALLBACK_METHODS = [
    ("requests + User-Agent Chrome", _method_requests_useragent),
    ("yf.download() polos", _method_yf_download),
    ("urllib langsung ke v8/finance/chart", _method_raw_urllib),
]


def _safe_name(ticker: str) -> str:
    return ticker.replace("/", "_").replace("^", "idx")


def _price_cache_path(ticker: str, start_date: datetime, end_date: datetime) -> str:
    return os.path.join(
        CACHE_DIR,
        f"{_safe_name(ticker)}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv",
    )


def _info_cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"{_safe_name(ticker)}_info.json")


def _fetch_history_resilient(
    ticker: str,
    start_date: datetime,
    end_date: datetime,
    max_retries: int = 3,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Ambil OHLCV historis 1 ticker dengan cache lokal + fallback proxy.

    Urutan: cek cache -> ulangi sampai max_retries putaran, tiap putaran
    coba semua proxy di PROXIES satu-satu -> raise Exception kalau semua
    gagal (supaya kode pemanggil tetap bisa nangkep & skip ticker itu).
    """
    cache_file = _price_cache_path(ticker, start_date, end_date)
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    _warn_if_placeholder_proxies()

    attempt_log = []
    for attempt in range(max_retries):
        for proxy in PROXIES:
            proxy_label = proxy if proxy else "langsung (tanpa proxy)"
            try:
                session = _get_session(proxy)
                hist = yf.Ticker(ticker, session=session).history(
                    start=start_date.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d"),
                    interval=interval,
                )
                if hist is not None and not hist.empty:
                    hist.to_csv(cache_file)
                    LAST_FETCH_ERRORS.pop(ticker, None)
                    return hist
                attempt_log.append(f"  - putaran {attempt + 1}, {proxy_label}: data kosong")
            except Exception as e:
                attempt_log.append(
                    f"  - putaran {attempt + 1}, {proxy_label}: {type(e).__name__}: {e}"
                )
        time.sleep(1)  # jeda sebelum ulangi seluruh daftar proxy

    # --- Lapisan fallback tambahan (atas permintaan): metode header-spoofing ---
    # CATATAN JUJUR: Yahoo sekarang mendeteksi bot terutama dari fingerprint
    # TLS/JA3 koneksinya (itu kenapa curl_cffi dipakai di atas), bukan cuma
    # dari header User-Agent. requests/urllib polos di bawah ini tetap
    # "ngomong" pakai TLS fingerprint Python standar walau header-nya
    # di-set persis kayak Chrome asli - jadi kemungkinan besar tetap kena
    # blokir yang sama kalau akar masalahnya memang di situ. Tetap dicoba
    # karena nggak ada ruginya, dan bisa membantu kalau blokirnya ternyata
    # cuma soal header/rate sederhana, bukan fingerprint.
    for name, method in _FALLBACK_METHODS:
        try:
            hist = method(ticker, start_date, end_date, interval)
            if hist is not None and not hist.empty:
                hist.to_csv(cache_file)
                LAST_FETCH_ERRORS.pop(ticker, None)
                return hist
            attempt_log.append(f"  - fallback [{name}]: data kosong")
        except Exception as e:
            attempt_log.append(f"  - fallback [{name}]: {type(e).__name__}: {e}")

    placeholder_note = ""
    if any(p and "example.com" in p for p in PROXIES):
        placeholder_note = (
            "CATATAN: PROXIES masih berisi URL contoh (proxy1/2/3.example.com) - "
            "itu TIDAK akan pernah konek. Ganti dulu dengan proxy asli kamu.\n\n"
        )
    detail = placeholder_note + "\n".join(attempt_log)
    LAST_FETCH_ERRORS[ticker] = detail
    print(f"[data_fetcher] GAGAL TOTAL ambil '{ticker}'. Rincian tiap percobaan:\n{detail}")
    raise RuntimeError(f"Gagal ambil data '{ticker}'. Lihat terminal atau expander 'Detail teknis' untuk rincian.")


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

        try:
            hist = _fetch_history_resilient(
                ticker, start_date, end_date, max_retries=max_retries
            )
            if hist.empty or len(hist) < 50:
                raise ValueError(
                    f"Data tidak cukup untuk {ticker}: hanya {len(hist)} baris"
                )
            all_data[ticker] = hist["Close"]

        except Exception:
            failed_stocks.append(ticker)

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
    cache_file = _info_cache_path(ticker)
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass  # cache korup/tidak kebaca, lanjut fetch ulang di bawah

    for proxy in PROXIES:
        try:
            stock = yf.Ticker(ticker, session=_get_session(proxy))
            info = stock.info

            result = {
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
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(result, f)
            return result
        except Exception:
            time.sleep(1)

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
        hist = _fetch_history_resilient(benchmark, start_date, end_date)
        data = hist["Close"]

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
    for proxy in PROXIES:
        try:
            stock = yf.Ticker(ticker, session=_get_session(proxy))
            hist = stock.history(period="5d")
            if not hist.empty:
                return True
        except Exception:
            continue
    return False
