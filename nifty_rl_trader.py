import os, math, warnings, time
import numpy as np
import pandas as pd
# Optional plotting (headless). If matplotlib is unavailable, we will skip charts.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False
    plt = None  # type: ignore[assignment]

# Optional gymnasium. If unavailable, provide minimal fallbacks for spaces and Env.
try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYM = True
except Exception:
    from types import SimpleNamespace
    HAS_GYM = False

    class _Env:
        def __init__(self):
            pass
        def reset(self, seed=None, options=None):
            return

    class _Box:
        def __init__(self, low, high, shape, dtype):
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype
        def sample(self):
            return np.zeros(self.shape, dtype=self.dtype)

    class _Discrete:
        def __init__(self, n):
            self.n = int(n)
        def sample(self):
            return int(np.random.randint(self.n))
        def contains(self, x):
            try:
                xi = int(x)
            except Exception:
                return False
            return 0 <= xi < self.n

    spaces = SimpleNamespace(Box=_Box, Discrete=_Discrete)
    gym = SimpleNamespace(Env=_Env)

# Try RL libs; fall back if not available (e.g., Python 3.13 without torch wheels)
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    HAS_SB3 = True
except Exception:
    PPO = None  # type: ignore[assignment]
    DummyVecEnv = None  # type: ignore[assignment]
    HAS_SB3 = False

warnings.filterwarnings("ignore")

# ---------- Helper: Features ----------
def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    series = series.squeeze()
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1/n, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1/n, adjust=False).mean()
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))

def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def zscore(series: pd.Series, win: int) -> pd.Series:
    mean = series.rolling(win).mean()
    std = series.rolling(win).std(ddof=0)
    return (series - mean) / (std + 1e-12)

# ---------- Data helpers ----------
def _parse_period_to_days(period: str) -> int:
    period = str(period).lower()
    if period.endswith("d"):
        return int(period[:-1])
    if period.endswith("mo"):
        return int(period[:-2]) * 30
    if period.endswith("y"):
        return int(period[:-1]) * 365
    # default fallback
    return 60

def _generate_synthetic_ohlcv(period: str = "60d", interval: str = "15m") -> pd.DataFrame:
    rng = np.random.default_rng(42)
    days = _parse_period_to_days(period)

    if interval.endswith("m"):
        # Intraday session ~09:15–15:30 => 25 bars per day at 15m resolution
        end_day = pd.Timestamp.now().normalize()
        bdays = pd.bdate_range(end=end_day, periods=max(days, 5))
        timestamps = []
        for d in bdays:
            date_str = d.strftime("%Y-%m-%d")
            intraday = pd.date_range(f"{date_str} 09:15", f"{date_str} 15:30", freq="15min")
            timestamps.append(intraday)
        index = pd.DatetimeIndex(np.concatenate([np.array(t.values, dtype="datetime64[ns]") for t in timestamps]))
    else:
        # Daily bars
        index = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=max(days, 60))

    n = len(index)
    start_price = 20000.0
    vol = 0.003 if interval.endswith("m") else 0.015

    # Random walk for close
    rets = rng.normal(0.0, vol, size=n)
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[start_price], close[:-1]])
    spread = np.maximum(0.0005 * close, rng.normal(0.0, vol, size=n) * close * 0.25)
    high = np.maximum(open_, close) * (1 + np.abs(spread) / close)
    low = np.minimum(open_, close) * (1 - np.abs(spread) / close)
    volume = rng.integers(1_000_000, 10_000_000, size=n)

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume.astype(float),
    }, index=pd.DatetimeIndex(index))
    return df

# ---------- Data: Download or load CSV ----------
def fetch_nifty(interval: str = "15m", period: str = "60d") -> pd.DataFrame:
    """
    Downloads NIFTY index (^NSEI) candles from Yahoo Finance.
    Fallback to local CSV or synthetic data if download fails/offline.
    """
    print(f"Downloading NIFTY data: period={period}, interval={interval} ...")

    # Optional local CSV override
    local_csv = os.environ.get("NIFTY_CSV_PATH")
    if local_csv and os.path.exists(local_csv):
        print(f"Loading local CSV: {local_csv}")
        df = pd.read_csv(local_csv)
        # Try to standardize
        if "datetime" in {c.lower() for c in df.columns}:
            df["datetime"] = pd.to_datetime(df["datetime"])  # type: ignore[index]
            df = df.set_index("datetime")
        df = df.rename(columns=str.lower)
        df = df[["open","high","low","close","volume"]].dropna()
        df.index = pd.to_datetime(df.index)
        if interval.endswith("m"):
            try:
                df = df.between_time("09:15", "15:30")
            except Exception:
                pass
        return df

    # Try Yahoo Finance
    try:
        import yfinance as yf
        df = yf.download("^NSEI", period=period, interval=interval, auto_adjust=False, progress=False)
        if df is None or df.empty:
            raise RuntimeError("Empty dataframe from yfinance")
        df = df.rename(columns=str.lower)
        df = df[["open","high","low","close","volume"]]
        df = df.dropna()
        df.index = pd.to_datetime(df.index)
        if interval.endswith("m"):
            df = df.between_time("09:15", "15:30")
        return df
    except Exception as e:
        print(f"Warning: yfinance download failed ({e}). Falling back to synthetic data.")
        return _generate_synthetic_ohlcv(period=period, interval=interval)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Drop duplicate-named columns, keep first occurrence
    if out.columns.duplicated().any():
        out = out.loc[:, ~out.columns.duplicated()].copy()

    def _ensure_series(s: pd.Series | pd.DataFrame | np.ndarray) -> pd.Series:
        if isinstance(s, pd.DataFrame):
            # take first column if multiple
            if s.shape[1] == 0:
                return pd.Series(index=out.index, dtype=float)
            return pd.to_numeric(s.iloc[:, 0], errors="coerce")
        if isinstance(s, np.ndarray):
            return pd.Series(s.reshape(-1), index=out.index)
        return pd.to_numeric(s.squeeze(), errors="coerce")

    # Ensure standard columns exist and are Series
    for col in ["open","high","low","close","volume"]:
        if col not in out.columns:
            raise KeyError(f"Missing required column: {col}")
        out[col] = _ensure_series(out[col])

    close_s = _ensure_series(out["close"])  # ensure Series
    out["ret_1"] = close_s.pct_change().fillna(0.0)
    out["rsi_14"] = _ensure_series(rsi(close_s, 14))
    out["atr_14"] = _ensure_series(atr(_ensure_series(out["high"]), _ensure_series(out["low"]), close_s, 14))

    ma_fast_series = close_s.rolling(window=5).mean()
    ma_slow_series = close_s.rolling(window=20).mean()
    ma_fast = ma_fast_series.to_numpy().reshape(-1)
    ma_slow = ma_slow_series.to_numpy().reshape(-1)
    close_v = close_s.to_numpy().reshape(-1)

    out["ma_fast"] = ma_fast
    out["ma_slow"] = ma_slow
    ma_diff = (ma_fast - ma_slow) / (close_v + 1e-9)
    out["ma_diff"] = ma_diff

    # Rolling z-scores to normalize (prevent look-ahead)
    out["zret_20"] = _ensure_series(zscore(out["ret_1"], 20))
    out["zclose_50"] = _ensure_series(zscore(np.log(close_s.clip(lower=1)), 50))

    # Normalize RSI to ~[-1,1]
    out["rsi_14_n"] = (out["rsi_14"] - 50.0) / 50.0
    # Relative ATR (volatility) scaled by price
    out["atr_rel"] = out["atr_14"] / (close_s + 1e-9)

    out = out.dropna().copy()
    return out

# ---------- Gym Environment ----------
class NiftyEnv(gym.Env):
    """
    Discrete actions: 0=Short, 1=Flat, 2=Long
    Reward per step in basis points (bps) of return minus costs when we change position.
    Observation: flattened vector of last K bars of selected features.
    """
    metadata = {"render_modes": []}

    def __init__(self, df: pd.DataFrame, features: list[str], k: int = 20,
                 trade_cost_bps: float = 4.0, slippage_bps: float = 2.0):
        super().__init__()
        if len(df) <= k + 1:
            raise ValueError(f"Not enough data for k={k}. Need > k+1 rows, got {len(df)}")
        self.df = df.reset_index(drop=False)  # keep timestamp
        self.features = features
        self.k = k

        self.trade_cost_bps = float(trade_cost_bps)  # per position change (enter/exit)
        self.slippage_bps = float(slippage_bps)      # per change, crude model

        self.n_feat = len(features)
        obs_size = self.k * self.n_feat

        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(obs_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        # Precompute close series for speed
        self.close = self.df["close"].values.astype(np.float64)

        # Prebuild state matrix (features only)
        self.X = self.df[self.features].values.astype(np.float32)

        self.reset(seed=None)

    def _get_obs(self):
        start = self.i - self.k
        end = self.i
        window = self.X[start:end, :]
        return window.reshape(-1).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Start at K so we can grab a full window
        self.i = self.k
        self.pos = 0  # -1, 0, +1
        self.equity_bps = 0.0
        self.trades = 0
        obs = self._get_obs()
        return obs, {}

    def step(self, action: int):
        assert self.action_space.contains(action)
        target_pos = int(action) - 1  # map 0,1,2 -> -1,0,1

        # Return from holding previous step: (P_t - P_{t-1}) / P_{t-1} in bps
        p_t = self.close[self.i]
        p_tm1 = self.close[self.i - 1]
        ret_bps = 10000.0 * ((p_t - p_tm1) / max(p_tm1, 1e-9))
        pnl_bps = self.pos * ret_bps

        # Transaction costs + slippage if we change position
        cost_bps = 0.0
        if target_pos != self.pos:
            cost_bps = self.trade_cost_bps + self.slippage_bps
            self.trades += 1

        reward = pnl_bps - cost_bps
        self.equity_bps += reward
        self.pos = target_pos

        self.i += 1
        terminated = (self.i >= len(self.df))
        truncated = False
        info = {"equity_bps": float(self.equity_bps), "pos": int(self.pos), "index": int(self.i)}
        obs = self._get_obs() if not terminated else np.zeros_like(self.observation_space.sample())
        return obs, float(reward), terminated, truncated, info

# ---------- Fallback model when SB3/Torch unavailable ----------
class RuleBasedModel:
    def __init__(self, features: list[str], k: int):
        self.features = features
        self.k = k

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        # obs is flattened window of shape (k * n_feat,)
        n_feat = len(self.features)
        if obs is None or len(obs) < n_feat:
            return 1, None  # Flat
        last_row = np.array(obs[-n_feat:])
        feature_to_index = {name: idx for idx, name in enumerate(self.features)}
        zret = last_row[feature_to_index.get("zret_20", -1)] if feature_to_index.get("zret_20", -1) >= 0 else 0.0
        rsi_n = last_row[feature_to_index.get("rsi_14_n", -1)] if feature_to_index.get("rsi_14_n", -1) >= 0 else 0.0
        ma_diff = last_row[feature_to_index.get("ma_diff", -1)] if feature_to_index.get("ma_diff", -1) >= 0 else 0.0
        # Simple heuristic
        if (rsi_n > 0.2 and ma_diff > 0) or (zret > 0.5 and ma_diff > 0):
            return 2, None  # Long
        if (rsi_n < -0.2 and ma_diff < 0) or (zret < -0.5 and ma_diff < 0):
            return 0, None  # Short
        return 1, None  # Flat

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".zip", "w") as f:
            f.write("RuleBasedModel placeholder")

# ---------- Train / Evaluate ----------
def split_train_test(df: pd.DataFrame, test_frac: float = 0.2):
    n = len(df)
    cut = int(n * (1 - test_frac))
    train, test = df.iloc[:cut].copy(), df.iloc[cut:].copy()
    return train, test

def train_agent(train_df, features, k=20, steps=200_000):
    if HAS_SB3:
        def make_env():
            return NiftyEnv(train_df, features=features, k=k,
                        trade_cost_bps=4.0, slippage_bps=2.0)
        env = DummyVecEnv([make_env])
        model = PPO("MlpPolicy", env,
                    verbose=1,
                    n_steps=2048,
                    batch_size=256,
                    learning_rate=3e-4,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    ent_coef=0.0,
                    vf_coef=0.5,
                    tensorboard_log=None)
        print(f"Training PPO for {steps:,} timesteps ...")
        model.learn(total_timesteps=int(steps))
        return model
    else:
        print("stable-baselines3/torch not available. Using a simple rule-based model as fallback.")
        return RuleBasedModel(features=features, k=k)

def evaluate_agent(model, test_df, features, k=20):
    env = NiftyEnv(test_df, features=features, k=k,
                   trade_cost_bps=4.0, slippage_bps=2.0)
    obs, _ = env.reset()
    done = False
    # Store signals for plotting/report
    rows = []
    # We also want to know when position changes to mark entries/exits
    prev_pos = 0
    last_info = {"equity_bps": 0.0}
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(int(action))
        last_info = info
        idx = int(info["index"]) - 1  # step consumed
        # Guard against any index drift near the end
        idx = min(max(0, idx), len(test_df) - 1)
        ts = test_df.index[idx]
        price = float(test_df["close"].iloc[idx])
        pos = info["pos"]
        if pos != prev_pos:
            rows.append({
                "timestamp": ts,
                "price": price,
                "pos": pos,                     # -1,0,+1
                "signal": "ENTER_LONG" if pos == 1 else ("ENTER_SHORT" if pos == -1 else "EXIT"),
                "equity_bps": float(info["equity_bps"])
            })
        prev_pos = pos
    # Final equity snapshot
    rows.append({
        "timestamp": test_df.index[-1],
        "price": float(test_df["close"].iloc[-1]),
        "pos": prev_pos,
        "signal": "END",
        "equity_bps": float(last_info.get("equity_bps", 0.0))
    })
    sigs = pd.DataFrame(rows)
    return sigs

# ---------- Reporting ----------
def plot_results(train_df, test_df, sigs, out_prefix="nifty_rl"):
    os.makedirs("outputs", exist_ok=True)

    # Equity curve (bps -> %)
    eq_pct = sigs["equity_bps"] / 100.0
    # Align timestamps
    eq_index = sigs["timestamp"]

    if not HAS_MPL:
        print("matplotlib not available; skipping charts.")
        return

    # --- Plot equity curve
    plt.figure(figsize=(10, 4))
    plt.plot(eq_index, eq_pct)
    plt.title("RL Strategy Equity Curve (Test)")
    plt.xlabel("Time")
    plt.ylabel("Return (%)")
    plt.tight_layout()
    eq_path = os.path.join("outputs", f"{out_prefix}_equity_curve.png")
    plt.savefig(eq_path, dpi=150)
    plt.close()
    print(f"Saved: {eq_path}")

    # --- Plot price with markers
    test = test_df.copy()
    plt.figure(figsize=(12, 5))
    plt.plot(test.index, test["close"], linewidth=1.2)
    # Add markers for entries/exits
    for _, r in sigs.iterrows():
        if r["signal"] == "ENTER_LONG":
            plt.scatter(r["timestamp"], r["price"], marker="^", s=60)
        elif r["signal"] == "ENTER_SHORT":
            plt.scatter(r["timestamp"], r["price"], marker="v", s=60)
        elif r["signal"] == "EXIT":
            plt.scatter(r["timestamp"], r["price"], marker="o", s=40)
    plt.title("NIFTY Close with RL Signals (Test)")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.tight_layout()
    price_path = os.path.join("outputs", f"{out_prefix}_price_signals.png")
    plt.savefig(price_path, dpi=150)
    plt.close()
    print(f"Saved: {price_path}")

def basic_metrics(sigs: pd.DataFrame):
    # Approximate step-to-step returns from equity_bps deltas
    if len(sigs) < 2:
        return {"total_return_%": float(sigs["equity_bps"].iloc[-1] / 100.0)}
    rets_bps = sigs["equity_bps"].diff().dropna()
    mean = rets_bps.mean()
    std = rets_bps.std(ddof=0) + 1e-9
    steps_per_day = 25  # ~15-minute bars during 09:15–15:30
    ann_factor = math.sqrt(252 * steps_per_day)
    sharpe = (mean / std) * ann_factor
    return {
        "total_return_%": float(sigs["equity_bps"].iloc[-1] / 100.0),
        "sharpe_approx": float(sharpe),
        "num_signals": int((sigs["signal"] != "END").sum())
    }

# ---------- Main ----------
def main():
    # 1) Get data
    # For more history (less resolution): use interval="1d", period="10y"
    interval = os.environ.get("INTERVAL", "15m")
    period = os.environ.get("PERIOD", "60d")
    df = fetch_nifty(interval=interval, period=period)
    df_feat = build_features(df)

    # Choose features (keep it small & stable)
    FEATURES = ["zret_20", "zclose_50", "rsi_14_n", "atr_rel", "ma_diff"]

    # 2) Split
    train_df, test_df = split_train_test(df_feat, test_frac=0.2)
    # Ensure enough data for chosen window size
    k = int(os.environ.get("K", 20))
    if len(train_df) <= k + 1 or len(test_df) <= k + 1:
        # shrink k if needed
        k = max(5, min(20, max(2, min(len(train_df) // 3, len(test_df) // 3))))
        print(f"Adjusted k to {k} due to limited data.")
    print(f"Train bars: {len(train_df):,} | Test bars: {len(test_df):,} | k={k}")

    # 3) Train RL (PPO or fallback)
    steps = int(os.environ.get("SB3_TIMESTEPS", 200_000))
    model = train_agent(train_df, features=FEATURES, k=k, steps=steps)

    # 4) Evaluate on unseen test data
    sigs = evaluate_agent(model, test_df, features=FEATURES, k=k)

    # 5) Save artifacts
    os.makedirs("outputs", exist_ok=True)
    # Ensure save exists even for fallback model
    model.save(os.path.join("outputs", "ppo_nifty_model"))
    sigs.to_csv(os.path.join("outputs", "signals_test.csv"), index=False)
    print("Saved: outputs/ppo_nifty_model.zip")
    print("Saved: outputs/signals_test.csv")

    # 6) Charts + Metrics
    plot_results(train_df, test_df, sigs)
    m = basic_metrics(sigs)
    print("\n=== Test Metrics (very rough) ===")
    for k_, v in m.items():
        print(f"{k_}: {v}")

    print("\nDone. Open the PNGs in the outputs/ folder to see results.")

if __name__ == "__main__":
    main()