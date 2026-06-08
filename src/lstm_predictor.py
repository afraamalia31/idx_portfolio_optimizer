"""
src/lstm_predictor.py
─────────────────────
Prediksi harga saham menggunakan LSTM + sinyal BUY/HOLD/SELL.
Terintegrasi penuh dengan MLflow untuk tracking experiment.
"""

import numpy as np
import pandas as pd
import mlflow
import mlflow.pyfunc
import tempfile
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler

# ── TensorFlow (lazy import agar tidak crash kalau belum install) ──────────────
def _get_keras():
    try:
        from tensorflow import keras
        return keras
    except ImportError:
        raise ImportError(
            "TensorFlow belum terinstall. Jalankan: pip install tensorflow"
        )


# =============================================================================
# KONSTANTA DEFAULT
# =============================================================================
LOOKBACK      = 30    # jumlah hari historis sebagai input sequence
FORECAST_DAYS = 5     # prediksi N hari ke depan
EPOCHS        = 50
BATCH_SIZE    = 32
LSTM_UNITS_1  = 64
LSTM_UNITS_2  = 32
DROPOUT       = 0.2
BUY_THRESHOLD = 0.02  # prediksi naik ≥ 2% → BUY
SELL_THRESHOLD = -0.02  # prediksi turun ≥ 2% → SELL


# =============================================================================
# HELPER: FITUR TEKNIKAL
# =============================================================================
def _add_technical_features(df: pd.Series) -> pd.DataFrame:
    """
    Tambahkan indikator teknikal sederhana ke series harga tunggal.
    Return DataFrame dengan kolom: close, ma5, ma20, rsi14, pct_change
    """
    out = pd.DataFrame({"close": df})

    # Moving Average
    out["ma5"]  = df.rolling(5).mean()
    out["ma20"] = df.rolling(20).mean()

    # RSI 14
    delta = df.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-8)
    out["rsi14"] = 100 - (100 / (1 + rs))

    # Perubahan harian (%)
    out["pct_change"] = df.pct_change()

    return out.dropna()


def _make_sequences(data: np.ndarray, lookback: int, forecast_days: int):
    """
    Buat pasangan (X, y) dari data time-series.
    X shape : (n_samples, lookback, n_features)
    y shape : (n_samples,)  — harga penutupan t+forecast_days
    """
    X, y = [], []
    for i in range(len(data) - lookback - forecast_days + 1):
        X.append(data[i : i + lookback])
        y.append(data[i + lookback + forecast_days - 1, 0])  # kolom 0 = close
    return np.array(X), np.array(y)


# =============================================================================
# MODEL WRAPPER (MLflow pyfunc)
# =============================================================================
class LSTMPortfolioModel(mlflow.pyfunc.PythonModel):
    """
    MLflow custom model untuk LSTM portfolio predictor.
    Disimpan bersama scaler dan metadata.
    """

    def __init__(self, model, scaler, feature_cols, lookback, forecast_days):
        self.model         = model
        self.scaler        = scaler
        self.feature_cols  = feature_cols
        self.lookback      = lookback
        self.forecast_days = forecast_days

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        """
        Input  : DataFrame dengan kolom harga (tanggal sebagai index)
        Output : DataFrame kolom ['stock', 'predicted_price',
                                   'current_price', 'pct_change', 'signal']
        """
        results = []
        for stock in model_input.columns:
            series = model_input[stock].dropna()
            if len(series) < self.lookback + 20:
                continue
            feat_df = _add_technical_features(series)
            scaled  = self.scaler.transform(feat_df[self.feature_cols].values)
            seq     = scaled[-self.lookback:].reshape(1, self.lookback, -1)
            pred_scaled = self.model.predict(seq, verbose=0)[0, 0]

            # inverse transform hanya kolom close (index 0)
            dummy = np.zeros((1, len(self.feature_cols)))
            dummy[0, 0] = pred_scaled
            pred_price   = self.scaler.inverse_transform(dummy)[0, 0]
            curr_price   = series.iloc[-1]
            pct          = (pred_price - curr_price) / curr_price

            if pct >= BUY_THRESHOLD:
                signal = "BUY"
            elif pct <= SELL_THRESHOLD:
                signal = "SELL"
            else:
                signal = "HOLD"

            results.append({
                "stock":           stock,
                "current_price":   round(curr_price, 2),
                "predicted_price": round(pred_price, 2),
                "pct_change":      round(pct * 100, 2),
                "signal":          signal,
            })
        return pd.DataFrame(results)


# =============================================================================
# TRAINING & MLFLOW LOGGING
# =============================================================================
def train_lstm_and_predict(
    price_data: pd.DataFrame,
    lookback:      int   = LOOKBACK,
    forecast_days: int   = FORECAST_DAYS,
    epochs:        int   = EPOCHS,
    batch_size:    int   = BATCH_SIZE,
    lstm_units_1:  int   = LSTM_UNITS_1,
    lstm_units_2:  int   = LSTM_UNITS_2,
    dropout:       float = DROPOUT,
    buy_threshold: float = BUY_THRESHOLD,
    sell_threshold:float = SELL_THRESHOLD,
    mlflow_run_id: str   = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Train satu model LSTM per saham, log semua ke MLflow (nested run),
    kembalikan DataFrame sinyal dan dict ringkasan.

    Parameters
    ----------
    price_data      : DataFrame harga penutupan (index=tanggal, kolom=ticker)
    mlflow_run_id   : ID parent run MLflow (dari blok Markowitz)

    Returns
    -------
    signals_df      : DataFrame sinyal per saham
    lstm_summary    : dict berisi avg_mae, avg_rmse, model_path, dll.
    """
    keras = _get_keras()
    feature_cols = ["close", "ma5", "ma20", "rsi14", "pct_change"]

    all_signals  = []
    all_maes     = []
    all_rmses    = []
    all_accs     = []

    # ── Nested run LSTM di dalam parent run Markowitz ─────────────────────────
    with mlflow.start_run(
        run_name="LSTM_Training",
        nested=True,
        parent_run_id=mlflow_run_id,
    ) as lstm_run:

        # ── LOG HYPERPARAMETER ────────────────────────────────────────────────
        mlflow.log_params({
            "lstm_lookback":      lookback,
            "lstm_forecast_days": forecast_days,
            "lstm_epochs":        epochs,
            "lstm_batch_size":    batch_size,
            "lstm_units_1":       lstm_units_1,
            "lstm_units_2":       lstm_units_2,
            "lstm_dropout":       dropout,
            "lstm_buy_threshold": buy_threshold,
            "lstm_sell_threshold":sell_threshold,
            "lstm_n_stocks":      price_data.shape[1],
            "lstm_feature_cols":  ",".join(feature_cols),
        })

        # ── TRAINING PER SAHAM ────────────────────────────────────────────────
        for stock in price_data.columns:
            series = price_data[stock].dropna()
            if len(series) < lookback + forecast_days + 30:
                continue

            # Fitur
            feat_df = _add_technical_features(series)
            feat_arr = feat_df[feature_cols].values

            # Normalisasi
            scaler  = MinMaxScaler()
            scaled  = scaler.fit_transform(feat_arr)

            # Sequences
            X, y = _make_sequences(scaled, lookback, forecast_days)
            if len(X) < 10:
                continue

            split  = int(len(X) * 0.8)
            X_tr, X_te = X[:split], X[split:]
            y_tr, y_te = y[:split], y[split:]

            # ── Build model ──────────────────────────────────────────────────
            model = keras.Sequential([
                keras.layers.LSTM(
                    lstm_units_1,
                    return_sequences=True,
                    input_shape=(lookback, len(feature_cols))
                ),
                keras.layers.Dropout(dropout),
                keras.layers.LSTM(lstm_units_2, return_sequences=False),
                keras.layers.Dropout(dropout),
                keras.layers.Dense(16, activation="relu"),
                keras.layers.Dense(1),
            ])
            model.compile(optimizer="adam", loss="mse",
                          metrics=["mae"])

            # ── Train ────────────────────────────────────────────────────────
            history = model.fit(
                X_tr, y_tr,
                validation_data=(X_te, y_te),
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
            )

            # ── Evaluasi ─────────────────────────────────────────────────────
            y_pred_scaled = model.predict(X_te, verbose=0).flatten()

            # Inverse transform
            dummy_pred = np.zeros((len(y_pred_scaled), len(feature_cols)))
            dummy_true = np.zeros((len(y_te),          len(feature_cols)))
            dummy_pred[:, 0] = y_pred_scaled
            dummy_true[:, 0] = y_te
            y_pred_price = scaler.inverse_transform(dummy_pred)[:, 0]
            y_true_price = scaler.inverse_transform(dummy_true)[:, 0]

            mae  = float(np.mean(np.abs(y_pred_price - y_true_price)))
            rmse = float(np.sqrt(np.mean((y_pred_price - y_true_price) ** 2)))

            # Akurasi arah (naik/turun)
            dir_true = np.sign(np.diff(y_true_price))
            dir_pred = np.sign(np.diff(y_pred_price))
            acc = float(np.mean(dir_true == dir_pred)) if len(dir_true) > 0 else 0.0

            all_maes.append(mae)
            all_rmses.append(rmse)
            all_accs.append(acc)

            # ── Log per-saham ke MLflow ───────────────────────────────────────
            mlflow.log_metrics({
                f"{stock}_mae":           round(mae,  4),
                f"{stock}_rmse":          round(rmse, 4),
                f"{stock}_dir_accuracy":  round(acc,  4),
                f"{stock}_train_loss":    round(history.history["loss"][-1],     4),
                f"{stock}_val_loss":      round(history.history["val_loss"][-1], 4),
            })

            # ── Prediksi harga ke depan ───────────────────────────────────────
            last_seq = scaled[-lookback:].reshape(1, lookback, -1)
            pred_scaled = float(model.predict(last_seq, verbose=0)[0, 0])
            dummy_f = np.zeros((1, len(feature_cols)))
            dummy_f[0, 0] = pred_scaled
            pred_price  = float(scaler.inverse_transform(dummy_f)[0, 0])
            curr_price  = float(series.iloc[-1])
            pct         = (pred_price - curr_price) / curr_price

            if pct >= buy_threshold:
                signal = "BUY"
            elif pct <= sell_threshold:
                signal = "SELL"
            else:
                signal = "HOLD"

            all_signals.append({
                "stock":           stock,
                "current_price":   round(curr_price,  2),
                "predicted_price": round(pred_price,   2),
                "pct_change":      round(pct * 100,    2),
                "signal":          signal,
            })

        # ── AGGREGATE METRICS ─────────────────────────────────────────────────
        avg_mae  = float(np.mean(all_maes))  if all_maes  else 0.0
        avg_rmse = float(np.mean(all_rmses)) if all_rmses else 0.0
        avg_acc  = float(np.mean(all_accs))  if all_accs  else 0.0

        signals_df   = pd.DataFrame(all_signals)
        buy_count    = int((signals_df["signal"] == "BUY").sum())  if len(signals_df) else 0
        hold_count   = int((signals_df["signal"] == "HOLD").sum()) if len(signals_df) else 0
        sell_count   = int((signals_df["signal"] == "SELL").sum()) if len(signals_df) else 0

        mlflow.log_metrics({
            "lstm_avg_mae":          round(avg_mae,  4),
            "lstm_avg_rmse":         round(avg_rmse, 4),
            "lstm_avg_dir_accuracy": round(avg_acc,  4),
            "lstm_signal_buy":       buy_count,
            "lstm_signal_hold":      hold_count,
            "lstm_signal_sell":      sell_count,
        })

        # ── LOG SIGNALS ARTIFACT ──────────────────────────────────────────────
        with tempfile.TemporaryDirectory() as tmp_dir:
            signals_path = os.path.join(tmp_dir, "lstm_signals.csv")
            signals_df.to_csv(signals_path, index=False)
            mlflow.log_artifact(signals_path, artifact_path="lstm")

        # ── REGISTER MODEL (train ulang satu model gabungan untuk registry) ───
        # Gabungkan semua data untuk satu model representatif (saham pertama)
        rep_stock = price_data.columns[0]
        rep_series = price_data[rep_stock].dropna()
        rep_feat   = _add_technical_features(rep_series)
        rep_scaler = MinMaxScaler()
        rep_scaled = rep_scaler.fit_transform(rep_feat[feature_cols].values)
        Xr, yr     = _make_sequences(rep_scaled, lookback, forecast_days)

        rep_model  = keras.Sequential([
            keras.layers.LSTM(lstm_units_1, return_sequences=True,
                              input_shape=(lookback, len(feature_cols))),
            keras.layers.Dropout(dropout),
            keras.layers.LSTM(lstm_units_2),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1),
        ])
        rep_model.compile(optimizer="adam", loss="mse")
        rep_model.fit(Xr, yr, epochs=min(epochs, 20),
                      batch_size=batch_size, verbose=0)

        lstm_pyfunc = LSTMPortfolioModel(
            model         = rep_model,
            scaler        = rep_scaler,
            feature_cols  = feature_cols,
            lookback      = lookback,
            forecast_days = forecast_days,
        )

        mlflow.pyfunc.log_model(
            artifact_path         = "lstm_model",
            python_model          = lstm_pyfunc,
            registered_model_name = "LSTM_IDX_Predictor",
            pip_requirements      = [
                "tensorflow>=2.13.0",
                "scikit-learn",
                "pandas",
                "numpy",
            ],
        )

        lstm_run_id = lstm_run.info.run_id

    lstm_summary = {
        "avg_mae":    avg_mae,
        "avg_rmse":   avg_rmse,
        "avg_acc":    avg_acc,
        "buy_count":  buy_count,
        "hold_count": hold_count,
        "sell_count": sell_count,
        "run_id":     lstm_run_id,
    }

    return signals_df, lstm_summary
