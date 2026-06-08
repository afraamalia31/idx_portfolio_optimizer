"""
src/lstm_predictor.py
─────────────────────
Prediksi harga saham menggunakan LSTM + sinyal BUY/HOLD/SELL.
Terintegrasi penuh dengan MLflow untuk tracking experiment.
"""

import os
import warnings
import tempfile

import numpy as np
import pandas as pd
import mlflow
import mlflow.pyfunc

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # suppress TF log spam

from sklearn.preprocessing import MinMaxScaler


# =============================================================================
# KERAS IMPORT — support TF <2.16 dan TF ≥2.16 (keras standalone)
# =============================================================================
def _get_keras():
    # Keras 3.x standalone (TF ≥ 2.16)
    try:
        import keras
        # pastikan bukan keras 2 yang old
        if hasattr(keras, "Sequential"):
            return keras
    except ImportError:
        pass
    # Fallback: keras bundled di dalam TF
    try:
        from tensorflow import keras
        return keras
    except ImportError:
        raise ImportError(
            "Install TensorFlow: pip install tensorflow"
        )


# =============================================================================
# KONSTANTA DEFAULT
# =============================================================================
LOOKBACK       = 30
FORECAST_DAYS  = 5
EPOCHS         = 50
BATCH_SIZE     = 32
LSTM_UNITS_1   = 64
LSTM_UNITS_2   = 32
DROPOUT        = 0.2
BUY_THRESHOLD  =  0.02   # naik ≥ 2%  → BUY
SELL_THRESHOLD = -0.02   # turun ≥ 2% → SELL


# =============================================================================
# HELPER: FITUR TEKNIKAL
# =============================================================================
def _add_technical_features(series: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame({"close": series})
    out["ma5"]       = series.rolling(5).mean()
    out["ma20"]      = series.rolling(20).mean()
    delta            = series.diff()
    gain             = delta.clip(lower=0).rolling(14).mean()
    loss             = (-delta.clip(upper=0)).rolling(14).mean()
    rs               = gain / (loss + 1e-8)
    out["rsi14"]     = 100 - (100 / (1 + rs))
    out["pct_change"]= series.pct_change()
    return out.dropna()


def _make_sequences(data: np.ndarray, lookback: int, forecast_days: int):
    """
    Return X (float32) dan y (float32) — TensorFlow wajib float32.
    """
    X, y = [], []
    for i in range(len(data) - lookback - forecast_days + 1):
        X.append(data[i : i + lookback])
        y.append(data[i + lookback + forecast_days - 1, 0])
    # ⬇ KUNCI FIX: paksa float32
    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32),
    )


def _build_model(keras, lookback: int, n_features: int,
                 units1: int, units2: int, dropout: float):
    """Bangun model LSTM yang sama di dua tempat."""
    model = keras.Sequential([
        keras.layers.Input(shape=(lookback, n_features)),   # explicit Input layer
        keras.layers.LSTM(units1, return_sequences=True),
        keras.layers.Dropout(dropout),
        keras.layers.LSTM(units2, return_sequences=False),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


# =============================================================================
# MLflow pyfunc WRAPPER
# =============================================================================
class LSTMPortfolioModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model, scaler, feature_cols, lookback, forecast_days):
        self.model         = model
        self.scaler        = scaler
        self.feature_cols  = feature_cols
        self.lookback      = lookback
        self.forecast_days = forecast_days

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        results = []
        for stock in model_input.columns:
            series = model_input[stock].dropna()
            if len(series) < self.lookback + 20:
                continue
            feat_df = _add_technical_features(series)
            scaled  = self.scaler.transform(
                feat_df[self.feature_cols].values
            ).astype(np.float32)                          # float32
            seq = scaled[-self.lookback:].reshape(1, self.lookback, -1)
            pred_s = float(self.model.predict(seq, verbose=0)[0, 0])

            dummy = np.zeros((1, len(self.feature_cols)), dtype=np.float32)
            dummy[0, 0] = pred_s
            pred_price = float(self.scaler.inverse_transform(dummy)[0, 0])
            curr_price = float(series.iloc[-1])
            pct        = (pred_price - curr_price) / curr_price

            signal = (
                "BUY"  if pct >= BUY_THRESHOLD  else
                "SELL" if pct <= SELL_THRESHOLD else
                "HOLD"
            )
            results.append({
                "stock":           stock,
                "current_price":   round(curr_price,  2),
                "predicted_price": round(pred_price,  2),
                "pct_change":      round(pct * 100,   2),
                "signal":          signal,
            })
        return pd.DataFrame(results)


# =============================================================================
# TRAINING + MLflow LOGGING
# =============================================================================
def train_lstm_and_predict(
    price_data:     pd.DataFrame,
    lookback:       int   = LOOKBACK,
    forecast_days:  int   = FORECAST_DAYS,
    epochs:         int   = EPOCHS,
    batch_size:     int   = BATCH_SIZE,
    lstm_units_1:   int   = LSTM_UNITS_1,
    lstm_units_2:   int   = LSTM_UNITS_2,
    dropout:        float = DROPOUT,
    buy_threshold:  float = BUY_THRESHOLD,
    sell_threshold: float = SELL_THRESHOLD,
    mlflow_run_id:  str   = None,
):
    keras        = _get_keras()
    feature_cols = ["close", "ma5", "ma20", "rsi14", "pct_change"]
    n_features   = len(feature_cols)

    all_signals, all_maes, all_rmses, all_accs = [], [], [], []

    with mlflow.start_run(
        run_name      = "LSTM_Training",
        nested        = True,
        parent_run_id = mlflow_run_id,
    ) as lstm_run:

        # ── HYPERPARAMETER ────────────────────────────────────────────────────
        mlflow.log_params({
            "lstm_lookback":       lookback,
            "lstm_forecast_days":  forecast_days,
            "lstm_epochs":         epochs,
            "lstm_batch_size":     batch_size,
            "lstm_units_1":        lstm_units_1,
            "lstm_units_2":        lstm_units_2,
            "lstm_dropout":        dropout,
            "lstm_buy_threshold":  buy_threshold,
            "lstm_sell_threshold": sell_threshold,
            "lstm_n_stocks":       price_data.shape[1],
        })

        # ── TRAINING PER SAHAM ────────────────────────────────────────────────
        for stock in price_data.columns:
            series = price_data[stock].dropna()
            if len(series) < lookback + forecast_days + 30:
                continue

            feat_arr = _add_technical_features(series)[feature_cols].values

            scaler = MinMaxScaler()
            # ⬇ float32 sejak awal
            scaled = scaler.fit_transform(feat_arr).astype(np.float32)

            X, y = _make_sequences(scaled, lookback, forecast_days)
            if len(X) < 10:
                continue

            sp   = int(len(X) * 0.8)
            X_tr, X_te = X[:sp], X[sp:]
            y_tr, y_te = y[:sp], y[sp:]

            model = _build_model(
                keras, lookback, n_features,
                lstm_units_1, lstm_units_2, dropout
            )

            history = model.fit(
                X_tr, y_tr,
                validation_data=(X_te, y_te),
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
            )

            # ── Evaluasi ─────────────────────────────────────────────────────
            y_pred_s = model.predict(X_te, verbose=0).flatten()

            dummy_p = np.zeros((len(y_pred_s), n_features), dtype=np.float32)
            dummy_t = np.zeros((len(y_te),     n_features), dtype=np.float32)
            dummy_p[:, 0] = y_pred_s
            dummy_t[:, 0] = y_te
            y_pred_price = scaler.inverse_transform(dummy_p)[:, 0]
            y_true_price = scaler.inverse_transform(dummy_t)[:, 0]

            mae  = float(np.mean(np.abs(y_pred_price - y_true_price)))
            rmse = float(np.sqrt(np.mean((y_pred_price - y_true_price) ** 2)))
            d_t  = np.sign(np.diff(y_true_price))
            d_p  = np.sign(np.diff(y_pred_price))
            acc  = float(np.mean(d_t == d_p)) if len(d_t) > 0 else 0.0

            all_maes.append(mae); all_rmses.append(rmse); all_accs.append(acc)

            mlflow.log_metrics({
                f"{stock}_mae":          round(mae,  4),
                f"{stock}_rmse":         round(rmse, 4),
                f"{stock}_dir_accuracy": round(acc,  4),
                f"{stock}_train_loss":   round(float(history.history["loss"][-1]),     4),
                f"{stock}_val_loss":     round(float(history.history["val_loss"][-1]), 4),
            })

            # ── Sinyal prediksi ───────────────────────────────────────────────
            last_seq  = scaled[-lookback:].reshape(1, lookback, -1)
            pred_s    = float(model.predict(last_seq, verbose=0)[0, 0])
            dummy_f   = np.zeros((1, n_features), dtype=np.float32)
            dummy_f[0, 0] = pred_s
            pred_price  = float(scaler.inverse_transform(dummy_f)[0, 0])
            curr_price  = float(series.iloc[-1])
            pct         = (pred_price - curr_price) / curr_price

            signal = (
                "BUY"  if pct >= buy_threshold  else
                "SELL" if pct <= sell_threshold else
                "HOLD"
            )
            all_signals.append({
                "stock":           stock,
                "current_price":   round(curr_price, 2),
                "predicted_price": round(pred_price,  2),
                "pct_change":      round(pct * 100,   2),
                "signal":          signal,
            })

        # ── AGGREGATE ─────────────────────────────────────────────────────────
        avg_mae  = float(np.mean(all_maes))  if all_maes  else 0.0
        avg_rmse = float(np.mean(all_rmses)) if all_rmses else 0.0
        avg_acc  = float(np.mean(all_accs))  if all_accs  else 0.0

        signals_df = pd.DataFrame(all_signals)
        buy_count  = int((signals_df["signal"] == "BUY").sum())  if len(signals_df) else 0
        hold_count = int((signals_df["signal"] == "HOLD").sum()) if len(signals_df) else 0
        sell_count = int((signals_df["signal"] == "SELL").sum()) if len(signals_df) else 0

        mlflow.log_metrics({
            "lstm_avg_mae":          round(avg_mae,  4),
            "lstm_avg_rmse":         round(avg_rmse, 4),
            "lstm_avg_dir_accuracy": round(avg_acc,  4),
            "lstm_signal_buy":       buy_count,
            "lstm_signal_hold":      hold_count,
            "lstm_signal_sell":      sell_count,
        })

        # ── ARTIFACT: signals CSV ─────────────────────────────────────────────
        with tempfile.TemporaryDirectory() as tmp_dir:
            sig_path = os.path.join(tmp_dir, "lstm_signals.csv")
            signals_df.to_csv(sig_path, index=False)
            mlflow.log_artifact(sig_path, artifact_path="lstm")

        # ── REGISTER MODEL ────────────────────────────────────────────────────
        rep_stock  = price_data.columns[0]
        rep_series = price_data[rep_stock].dropna()
        rep_feat   = _add_technical_features(rep_series)[feature_cols].values
        rep_scaler = MinMaxScaler()
        rep_scaled = rep_scaler.fit_transform(rep_feat).astype(np.float32)
        Xr, yr     = _make_sequences(rep_scaled, lookback, forecast_days)

        rep_model  = _build_model(
            keras, lookback, n_features,
            lstm_units_1, lstm_units_2, dropout
        )
        rep_model.fit(
            Xr, yr,
            epochs=min(epochs, 20),
            batch_size=batch_size,
            verbose=0,
        )

        mlflow.pyfunc.log_model(
            artifact_path         = "lstm_model",
            python_model          = LSTMPortfolioModel(
                model         = rep_model,
                scaler        = rep_scaler,
                feature_cols  = feature_cols,
                lookback      = lookback,
                forecast_days = forecast_days,
            ),
            registered_model_name = "LSTM_IDX_Predictor",
            pip_requirements      = [
                "tensorflow>=2.13.0",
                "scikit-learn",
                "pandas",
                "numpy",
            ],
        )

        lstm_run_id = lstm_run.info.run_id

    return signals_df, {
        "avg_mae":    avg_mae,
        "avg_rmse":   avg_rmse,
        "avg_acc":    avg_acc,
        "buy_count":  buy_count,
        "hold_count": hold_count,
        "sell_count": sell_count,
        "run_id":     lstm_run_id,
    }
