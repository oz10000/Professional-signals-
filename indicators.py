"""
Indicadores técnicos con fórmulas Wilder correctas y manejo defensivo.
"""

import numpy as np
import pandas as pd
from typing import Optional


# ============================================================
# HELPERS INTERNOS
# ============================================================

def _safe_series(df: pd.DataFrame) -> bool:
    """Verifica que df tenga las columnas necesarias."""
    if df is None or df.empty:
        return False
    required = ['high', 'low', 'close']
    return all(c in df.columns for c in required)


# ============================================================
# TRUE RANGE Y ATR
# ============================================================

def true_range(df: pd.DataFrame) -> pd.Series:
    """True Range de Wilder con manejo de errores."""
    if not _safe_series(df):
        return pd.Series(dtype=float)
    
    try:
        high = df['high']
        low = df['low']
        prev_close = df['close'].shift(1)
        
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        
        return tr.fillna(high - low)
    except Exception:
        return pd.Series(dtype=float)


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR con suavizado Wilder (RMA)."""
    if not _safe_series(df) or len(df) < period:
        return pd.Series(0.0, index=df.index if df is not None else None)
    
    try:
        tr = true_range(df)
        atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
        return atr.fillna(0).replace([np.inf, -np.inf], 0)
    except Exception:
        return pd.Series(0.0, index=df.index)


# ============================================================
# ADX WILDER
# ============================================================

def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ADX Wilder correcto con defensa contra errores."""
    if not _safe_series(df) or len(df) < period * 2:
        return pd.Series(0.0, index=df.index)
    
    try:
        high, low = df['high'], df['low']
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
        
        alpha = 1.0 / period
        tr = true_range(df)
        atr_s = tr.ewm(alpha=alpha, adjust=False).mean().replace(0, np.nan)
        
        plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_s
        minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_s
        
        di_sum = (plus_di + minus_di).replace(0, np.nan)
        dx = (plus_di - minus_di).abs() / di_sum * 100
        dx = dx.fillna(0).replace([np.inf, -np.inf], 0)
        adx = dx.ewm(alpha=alpha, adjust=False).mean()
        
        return adx.fillna(0).replace([np.inf, -np.inf], 0)
    except Exception:
        return pd.Series(0.0, index=df.index)


# ============================================================
# RSI WILDER
# ============================================================

def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """RSI de Wilder."""
    if not _safe_series(df) or len(df) < period:
        return pd.Series(50.0, index=df.index)
    
    try:
        close = df['close']
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        alpha = 1.0 / period
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50).replace([np.inf, -np.inf], 50).clip(0, 100)
    except Exception:
        return pd.Series(50.0, index=df.index)


# ============================================================
# MACD
# ============================================================

def compute_macd(df: pd.DataFrame, fast: int = 12,
                 slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD con histograma."""
    if not _safe_series(df) or len(df) < slow:
        return pd.DataFrame({
            'macd': 0.0, 'signal': 0.0, 'histogram': 0.0
        }, index=df.index if df is not None else None)
    
    try:
        close = df['close']
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        
        return pd.DataFrame({
            'macd': macd.fillna(0),
            'signal': signal_line.fillna(0),
            'histogram': histogram.fillna(0),
        }, index=df.index)
    except Exception:
        return pd.DataFrame({
            'macd': 0.0, 'signal': 0.0, 'histogram': 0.0
        }, index=df.index)


# ============================================================
# ROC
# ============================================================

def compute_roc(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """Rate of Change."""
    if not _safe_series(df) or len(df) < period:
        return pd.Series(0.0, index=df.index)
    
    try:
        roc = (df['close'] / df['close'].shift(period) - 1) * 100
        return roc.fillna(0).replace([np.inf, -np.inf], 0).clip(-100, 100)
    except Exception:
        return pd.Series(0.0, index=df.index)


# ============================================================
# BOLLINGER
# ============================================================

def compute_bollinger(df: pd.DataFrame, period: int = 20,
                      std_mult: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands."""
    if not _safe_series(df) or len(df) < period:
        return pd.DataFrame({
            'bb_upper': 0.0, 'bb_middle': 0.0,
            'bb_lower': 0.0, 'bb_width': 0.0
        }, index=df.index if df is not None else None)
    
    try:
        close = df['close']
        sma = close.rolling(period).mean()
        std = close.rolling(period).std()
        
        upper = sma + std_mult * std
        lower = sma - std_mult * std
        width = (upper - lower) / sma.replace(0, np.nan)
        
        return pd.DataFrame({
            'bb_upper': upper.fillna(close),
            'bb_middle': sma.fillna(close),
            'bb_lower': lower.fillna(close),
            'bb_width': width.fillna(0).replace([np.inf, -np.inf], 0),
        }, index=df.index)
    except Exception:
        return pd.DataFrame({
            'bb_upper': 0.0, 'bb_middle': 0.0,
            'bb_lower': 0.0, 'bb_width': 0.0
        }, index=df.index)


# ============================================================
# EMA / SMA
# ============================================================

def compute_ema(df: pd.DataFrame, period: int) -> pd.Series:
    """EMA estándar."""
    if not _safe_series(df) or len(df) < 1:
        return pd.Series(0.0, index=df.index if df is not None else None)
    try:
        return df['close'].ewm(span=period, adjust=False).mean().fillna(0)
    except Exception:
        return pd.Series(0.0, index=df.index)


def compute_sma(df: pd.DataFrame, period: int) -> pd.Series:
    """SMA estándar."""
    if not _safe_series(df) or len(df) < period:
        return pd.Series(0.0, index=df.index if df is not None else None)
    try:
        return df['close'].rolling(period).mean().fillna(0)
    except Exception:
        return pd.Series(0.0, index=df.index)


# ============================================================
# VOLUMEN INTELIGENTE
# ============================================================

def compute_volume_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Volume Ratio = Vol actual / SMA(Vol, period)."""
    if df is None or df.empty or 'volume' not in df.columns:
        return pd.Series(1.0, index=df.index if df is not None else None)
    
    try:
        avg = df['volume'].rolling(period).mean()
        vr = df['volume'] / avg.replace(0, np.nan)
        return vr.fillna(1.0).replace([np.inf, -np.inf], 1.0).clip(0, 20)
    except Exception:
        return pd.Series(1.0, index=df.index)


def compute_delta_volume(df: pd.DataFrame) -> pd.Series:
    """Delta Volume aproximado."""
    if df is None or df.empty:
        return pd.Series(0.0, index=df.index if df is not None else None)
    
    try:
        range_size = (df['high'] - df['low']).replace(0, np.nan)
        close_pos = (df['close'] - df['low']) / range_size
        delta = df['volume'] * (2 * close_pos - 1)
        return delta.fillna(0).replace([np.inf, -np.inf], 0)
    except Exception:
        return pd.Series(0.0, index=df.index)


def compute_cvd(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Cumulative Volume Delta."""
    if df is None or df.empty:
        return pd.Series(0.0, index=df.index if df is not None else None)
    
    try:
        delta = compute_delta_volume(df)
        cvd = delta.rolling(period).sum()
        return cvd.fillna(0).replace([np.inf, -np.inf], 0)
    except Exception:
        return pd.Series(0.0, index=df.index)


def compute_tfi(df: pd.DataFrame, period: int = 5) -> pd.Series:
    """
    Trade Flow Imbalance (TFI) aproximado.
    TFI = (Buy_Vol - Sell_Vol) / Total_Vol, normalizado a [-1, 1].
    """
    if df is None or df.empty:
        return pd.Series(0.0, index=df.index if df is not None else None)
    
    try:
        range_size = (df['high'] - df['low']).replace(0, np.nan)
        buy_ratio = (df['close'] - df['low']) / range_size
        buy_ratio = buy_ratio.clip(0, 1).fillna(0.5)
        
        buy_vol = df['volume'] * buy_ratio
        sell_vol = df['volume'] * (1 - buy_ratio)
        
        tfi = (buy_vol - sell_vol).rolling(period).sum()
        total = df['volume'].rolling(period).sum().replace(0, np.nan)
        tfi_norm = tfi / total
        
        return tfi_norm.fillna(0).replace([np.inf, -np.inf], 0).clip(-1, 1)
    except Exception:
        return pd.Series(0.0, index=df.index)


def compute_ofi(df: pd.DataFrame, period: int = 5) -> pd.Series:
    """Order Flow Imbalance (OFI) aproximado."""
    if df is None or df.empty:
        return pd.Series(0.0, index=df.index if df is not None else None)
    
    try:
        price_change = df['close'].diff().fillna(0)
        vol_weighted = price_change * df['volume']
        
        ofi = vol_weighted.rolling(period).sum()
        total = (df['close'].abs().rolling(period).sum() * df['volume'].rolling(period).sum()).replace(0, np.nan)
        ofi_norm = ofi / total
        
        return ofi_norm.fillna(0).replace([np.inf, -np.inf], 0).clip(-1, 1)
    except Exception:
        return pd.Series(0.0, index=df.index)


# ============================================================
# KAUFMAN EFFICIENCY RATIO
# ============================================================

def compute_ker(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """Kaufman Efficiency Ratio."""
    if df is None or df.empty or len(df) < period:
        return pd.Series(0.0, index=df.index if df is not None else None)
    
    try:
        close = df['close']
        change = close.diff(period).abs()
        volatility = close.diff().abs().rolling(period).sum()
        ker = change / (volatility + 1e-9)
        return ker.fillna(0).replace([np.inf, -np.inf], 0).clip(0, 1)
    except Exception:
        return pd.Series(0.0, index=df.index)


# ============================================================
# RÉGIMEN
# ============================================================

def compute_regime(df: pd.DataFrame, adx_val: float,
                   atr_pct: float) -> str:
    """Clasifica régimen de mercado."""
    try:
        if df is None or df.empty or len(df) < 30:
            return 'Chop'
        if not isinstance(adx_val, (int, float)) or adx_val != adx_val:
            return 'Chop'
        if not isinstance(atr_pct, (int, float)) or atr_pct != atr_pct:
            return 'Chop'
        
        if adx_val > 40 and atr_pct > 0.02:
            return 'Expansion'
        if adx_val > 30:
            return 'Trend_Strong'
        if adx_val > 20:
            return 'Trend_Weak'
        return 'Chop'
    except Exception:
        return 'Chop'
