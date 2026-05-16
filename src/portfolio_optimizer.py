"""
Portfolio Optimizer Module
Implementasi Markowitz Efficient Frontier menggunakan PyPortfolioOpt
"""

import numpy as np
import pandas as pd
from pypfopt import (
    EfficientFrontier,
    risk_models,
    expected_returns,
    plotting,
    objective_functions
)
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
import warnings
warnings.filterwarnings('ignore')
import mlflow
import mlflow.sklearn

def optimize(self, method, weight_bounds, 
             target_return=None, target_volatility=None):
    
    # Mulai log eksperimen
    with mlflow.start_run():
        
        # Log parameter
        mlflow.log_param("method", method)
        mlflow.log_param("weight_bounds_min", weight_bounds[0])
        mlflow.log_param("weight_bounds_max", weight_bounds[1])
        mlflow.log_param("risk_free_rate", self.risk_free_rate)
        mlflow.log_param("n_assets", len(self.tickers))
        mlflow.log_param("tickers", str(self.tickers))
        
        if target_return:
            mlflow.log_param("target_return", target_return)
        if target_volatility:
            mlflow.log_param("target_volatility", target_volatility)
        
        # --- proses optimasi seperti biasa ---
        weights, stats = ... # kode optimasi yang sudah ada
        
        # Log hasil/metrik
        mlflow.log_metric("sharpe_ratio", stats["sharpe_ratio"])
        mlflow.log_metric("expected_return", stats["expected_return"])
        mlflow.log_metric("volatility", stats["volatility"])
        mlflow.log_metric("max_drawdown", stats["max_drawdown"])
        mlflow.log_metric("sortino_ratio", stats["sortino_ratio"])
        mlflow.log_metric("calmar_ratio", stats["calmar_ratio"])
        mlflow.log_metric("var_95", stats["var_95"])
        mlflow.log_metric("n_assets_active", stats["n_assets"])
        
        # Log bobot optimal sebagai artifact
        import json
        with open("weights.json", "w") as f:
            json.dump(weights, f)
        mlflow.log_artifact("weights.json")
    
    return weights, stats
    
class PortfolioOptimizer:
    """
    Kelas utama untuk optimasi portofolio menggunakan Markowitz MPT.
    
    Attributes:
    -----------
    price_data : pd.DataFrame
        Data harga historis saham
    risk_free_rate : float
        Tingkat bunga bebas risiko (annualized)
    returns : pd.DataFrame
        Return harian dari price_data
    mu : pd.Series
        Expected returns yang diestimasikan
    S : pd.DataFrame
        Matriks kovariansi yang diestimasikan
    """
    
    def __init__(self, price_data: pd.DataFrame, risk_free_rate: float = 0.0575):
        """
        Inisialisasi PortfolioOptimizer.
        
        Parameters:
        -----------
        price_data : pd.DataFrame
            DataFrame dengan kolom = ticker saham, baris = tanggal
        risk_free_rate : float
            BI Rate atau tingkat bunga bebas risiko (default: 5.75%)
        """
        self.price_data = price_data.copy()
        self.risk_free_rate = risk_free_rate
        self.tickers = list(price_data.columns)
        
        # Hitung returns
        self.returns = price_data.pct_change().dropna()
        
        # Estimasi expected returns (mean historical return, annualized)
        self.mu = expected_returns.mean_historical_return(price_data)
        
        # Estimasi matriks kovariansi (Ledoit-Wolf shrinkage untuk stabilitas)
        self.S = risk_models.CovarianceShrinkage(price_data).ledoit_wolf()
        
        # Simpan hasil optimasi
        self._optimal_weights = None
        self._optimal_stats = None
    
    def optimize(
        self,
        method: str = "max_sharpe",
        weight_bounds: tuple = (0.0, 0.1),
        target_return: float | None = None,
        target_volatility: float | None = None
    ) -> tuple[dict, dict]:
        """
        Jalankan optimasi portofolio dengan berbagai metode.
        
        Parameters:
        -----------
        method : str
            Metode optimasi:
            - 'max_sharpe': Maksimalkan Sharpe Ratio
            - 'min_volatility': Minimasi volatilitas
            - 'efficient_return': Cari bobot untuk target return tertentu
            - 'efficient_risk': Cari bobot untuk target volatilitas tertentu
        weight_bounds : tuple
            (min_weight, max_weight) untuk setiap aset
        target_return : float, optional
            Target return tahunan untuk metode 'efficient_return'
        target_volatility : float, optional
            Target volatilitas tahunan untuk metode 'efficient_risk'
        
        Returns:
        --------
        tuple : (weights_dict, stats_dict)
            - weights_dict: Bobot optimal per saham
            - stats_dict: Statistik portofolio (return, volatilitas, sharpe)
        """
        
        # Buat objek EfficientFrontier
        ef = EfficientFrontier(
            self.mu,
            self.S,
            weight_bounds=weight_bounds,
           
        )
        
        # Tambahkan regularisasi L2 untuk diversifikasi
        ef.add_objective(objective_functions.L2_reg, gamma=0.1)
        
        # Validasi parameter wajib — dipisah dari try/except optimasi
        # agar ValueError tidak tertangkap dan di-fallback diam-diam
        if method == "efficient_return" and target_return is None:
            raise ValueError(
                "target_return harus diisi untuk metode 'efficient_return'. "
                "Gunakan slider 'Target Return' di sidebar."
            )
        if method == "efficient_risk" and target_volatility is None:
            raise ValueError(
                "target_volatility harus diisi untuk metode 'efficient_risk'. "
                "Gunakan slider 'Target Risiko' di sidebar."
            )
        if method not in ("max_sharpe", "min_volatility", "efficient_return", "efficient_risk"):
            raise ValueError(f"Metode '{method}' tidak dikenal.")

        # Pilih metode optimasi — hanya tangkap error numerik/solver di sini
        try:
            if method == "max_sharpe":
                ef.max_sharpe(risk_free_rate=self.risk_free_rate)
            
            elif method == "min_volatility":
                ef.min_volatility()
            
            elif method == "efficient_return":
                ef.efficient_return(target_return=target_return)
            
            elif method == "efficient_risk":
                ef.efficient_risk(target_volatility=target_volatility)

        except Exception as e:
            # Hanya fallback untuk error solver/numerik (bukan validasi),
            # dan beri peringatan eksplisit agar tidak senyap
            import warnings
            warnings.warn(
                f"Optimasi dengan metode '{method}' gagal ({e}). "
                f"Fallback ke min_volatility.",
                RuntimeWarning,
                stacklevel=2
            )
            ef = EfficientFrontier(self.mu, self.S, weight_bounds=weight_bounds)
            ef.add_objective(objective_functions.L2_reg, gamma=0.1)
            ef.min_volatility()
        
        # Bersihkan bobot (buang yang < 1%)
        weights = ef.clean_weights(cutoff=0.01)
        
        # Hitung statistik portofolio
        perf = ef.portfolio_performance(verbose=False, risk_free_rate=self.risk_free_rate)
        expected_ret, volatility, sharpe = perf
        
        # Hitung max drawdown
        port_returns = sum(
            self.returns[col] * weights.get(col, 0) 
            for col in self.tickers 
            if col in weights
        )
        cumulative = (1 + port_returns).cumprod()
        max_drawdown = (cumulative / cumulative.cummax() - 1).min()
        
        # Hitung Value at Risk (95% confidence)
        var_95 = np.percentile(port_returns, 5)
        
        # Hitung Sortino Ratio
        downside_returns = port_returns[port_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino = (expected_ret - self.risk_free_rate) / downside_std if downside_std > 0 else 0
        
        # Hitung Calmar Ratio
        calmar = expected_ret / abs(max_drawdown) if max_drawdown != 0 else 0
        
        stats = {
            "expected_return": expected_ret,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "var_95": var_95,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "n_assets": len([v for v in weights.values() if v > 0.01])
        }
        
        self._optimal_weights = dict(weights)
        self._optimal_stats = stats
        
        return dict(weights), stats
    
    def monte_carlo_simulation(
        self,
        n_portfolios: int = 5000
    ) -> pd.DataFrame:
        """
        Simulasi Monte Carlo untuk menghasilkan portofolio acak.
        Digunakan untuk memvisualisasikan Efficient Frontier.
        
        Parameters:
        -----------
        n_portfolios : int
            Jumlah portofolio yang disimulasikan
        
        Returns:
        --------
        pd.DataFrame
            DataFrame dengan kolom: weights, return, volatility, sharpe
        """
        n_assets = len(self.tickers)
        results = []
        
        np.random.seed(42)  # Reproducibility
        
        for _ in range(n_portfolios):
            # Generate bobot acak
            weights = np.random.dirichlet(np.ones(n_assets))
            
            # Hitung return dan risiko portofolio
            port_return = np.dot(weights, self.mu)
            port_vol = np.sqrt(
                np.dot(weights.T, np.dot(self.S.values, weights))
            )
            
            # Hitung Sharpe Ratio
            sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0
            
            results.append({
                "return": port_return,
                "volatility": port_vol,
                "sharpe": sharpe,
                **{f"w_{ticker}": w for ticker, w in zip(self.tickers, weights)}
            })
        
        return pd.DataFrame(results)
    
    def get_efficient_frontier_points(
        self,
        n_points: int = 100,
        weight_bounds: tuple = (0.0, 1.0)
    ) -> pd.DataFrame:
        """
        Hitung titik-titik di sepanjang Efficient Frontier.
        
        Parameters:
        -----------
        n_points : int
            Jumlah titik pada frontier
        weight_bounds : tuple
            Batas bobot minimum dan maksimum
        
        Returns:
        --------
        pd.DataFrame : Titik-titik frontier dengan return dan volatilitas
        """
        frontier_points = []
        
        # Cari range return yang valid
        min_ret = float(self.mu.min())
        max_ret = float(self.mu.max()) * 0.95  # Tidak perlu capai return tertinggi
        
        target_returns = np.linspace(min_ret, max_ret, n_points)
        
        for target in target_returns:
            try:
                ef = EfficientFrontier(
                    self.mu,
                    self.S,
                    weight_bounds=weight_bounds
                )
                ef.efficient_return(target_return=target)
                perf = ef.portfolio_performance(
                    verbose=False,
                    risk_free_rate=self.risk_free_rate
                )
                frontier_points.append({
                    "return": perf[0],
                    "volatility": perf[1],
                    "sharpe": perf[2]
                })
            except Exception:
                continue
        
        return pd.DataFrame(frontier_points)
    
    def get_discrete_allocation(
        self,
        weights: dict,
        total_portfolio_value: float = 10_000_000
    ) -> tuple[dict, float]:
        """
        Konversi bobot optimal ke jumlah lot saham yang harus dibeli.
        
        Parameters:
        -----------
        weights : dict
            Bobot optimal per saham
        total_portfolio_value : float
            Total nilai investasi dalam Rupiah
        
        Returns:
        --------
        tuple : (allocation_dict, leftover)
            - allocation_dict: Jumlah lot per saham
            - leftover: Sisa uang yang tidak teralokasikan
        """
        try:
            latest_prices = get_latest_prices(self.price_data)
            
            da = DiscreteAllocation(
                weights,
                latest_prices,
                total_portfolio_value=total_portfolio_value,
                short_ratio=None
            )
            
            allocation, leftover = da.lp_portfolio()
            return allocation, leftover
        
        except Exception as e:
            # Fallback ke greedy allocation
            allocation = {}
            for ticker, weight in weights.items():
                if weight > 0.01 and ticker in self.price_data.columns:
                    price = self.price_data[ticker].iloc[-1]
                    value = weight * total_portfolio_value
                    # Di BEI, 1 lot = 100 lembar
                    n_lots = int(value / (price * 100))
                    if n_lots > 0:
                        allocation[ticker] = n_lots
            return allocation, 0
    
    def get_covariance_matrix(self) -> pd.DataFrame:
        """Kembalikan matriks kovariansi yang telah diestimasikan."""
        return self.S
    
    def get_expected_returns(self) -> pd.Series:
        """Kembalikan expected returns per saham."""
        return self.mu
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        """Hitung dan kembalikan matriks korelasi."""
        return self.returns.corr()
    
    def rolling_optimization(
        self,
        window: int = 252,
        rebalance_freq: int = 63,  # Triwulanan
        method: str = "max_sharpe"
    ) -> pd.DataFrame:
        """
        Lakukan rolling optimization untuk backtest strategi rebalancing.
        
        Parameters:
        -----------
        window : int
            Window estimasi parameter (hari)
        rebalance_freq : int
            Frekuensi rebalancing (hari)
        method : str
            Metode optimasi
        
        Returns:
        --------
        pd.DataFrame : Bobot portofolio sepanjang waktu
        """
        all_weights = []
        dates = self.price_data.index[window:]
        
        for i, date in enumerate(dates):
            if i % rebalance_freq == 0:
                # Window data untuk estimasi
                window_data = self.price_data.iloc[
                    max(0, self.price_data.index.get_loc(date) - window):
                    self.price_data.index.get_loc(date)
                ]
                
                if len(window_data) < 50:
                    continue
                
                try:
                    mu_roll = expected_returns.mean_historical_return(window_data)
                    S_roll = risk_models.CovarianceShrinkage(window_data).ledoit_wolf()
                    ef = EfficientFrontier(mu_roll, S_roll, weight_bounds=(0.0, 0.5))
                    
                    if method == "max_sharpe":
                        ef.max_sharpe(risk_free_rate=self.risk_free_rate)
                    else:
                        ef.min_volatility()
                    
                    w = ef.clean_weights()
                    w['date'] = date
                    all_weights.append(w)
                except Exception:
                    continue
        
        if not all_weights:
            return pd.DataFrame()
        
        weights_df = pd.DataFrame(all_weights).set_index('date')
        return weights_df.fillna(0)
