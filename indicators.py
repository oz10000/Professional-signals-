"""
Indicadores técnicos con fórmulas Wilder correctas.
"""

import numpy as np
import pandas as pd
from typing import Optional


# ============================================================
# TRUE RANGE Y ATR (WILDER)
# ============================================================

def true_range(df: pd.DataFrame) -> pd.Series:
    """True Range de Wilder."""
    high = df['high']
    low = df['low']
    prev_close = df['close'].shift(1)
    
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    
    return tr


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR con suavizado Wilder (RMA)."""
    if df.empty or len(df) < period:
        return pd.Series(0.0, index=df.index)
    tr = true_range(df)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    return atr.fillna(0).replace([np.inf, -np.inf], 0)


# ============================================================
# ADX WILDER
# ============================================================

def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ADX Wilder correcto."""
    if df.empty or len(df) < period:
        return pd.Series(0.0, index=df.index)
    
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


# ============================================================
# RSI WILDER
# ============================================================

def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """RSI de Wilder."""
    if df.empty or len(df) < period:
        return pd.Series(50.0, index=df.index)
    
    close = df['close']
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.fillna(50).replace([np.inf, -np.inf], 50)


# ============================================================
# MACD
# ============================================================

def compute_macd(df: pd.DataFrame, fast: int = 12,
                 slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD con histograma."""
    close = df['close']
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    
    return pd.DataFrame({
        'macd': macd,
        'signal': signal_line,
        'histogram': histogram,
    }, index=df.index)


# ============================================================
# ROC (Rate of Change)
# ============================================================

def compute_roc(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """Rate of Change."""
    if len(df) < period:
        return pd.Series(0.0, index=df.index)
    return (df['close'] / df['close'].shift(period) - 1) * 100


# ============================================================
# BOLLINGER BANDS
# ============================================================

def compute_bollinger(df: pd.DataFrame, period: int = 20,
                      std_mult: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands."""
    close = df['close']
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    width = (upper - lower) / sma.replace(0, np.nan)
    
    return pd.DataFrame({
        'bb_upper': upper,
        'bb_middle': sma,
        'bb_lower': lower,
        'bb_width': width,
    }, index=df.index)


# ============================================================
# EMA / SMA
# ============================================================

def compute_ema(df: pd.DataFrame, period: int) -> pd.Series:
    """EMA estándar."""
    return df['close'].ewm(span=period, adjust=False).mean()


def compute_sma(df: pd.DataFrame, period: int) -> pd.Series:
    """SMA estándar."""
    return df['close'].rolling(period).mean()


# ============================================================
# VOLUMEN INTELIGENTE
# ============================================================

def compute_volume_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Volume Ratio = Vol actual / SMA(Vol, period)."""
    avg = df['volume'].rolling(period).mean()
    return df['volume'] / avg.replace(0, np.nan)


def compute_delta_volume(df: pd.DataFrame) -> pd.Series:
    """
    Delta Volume aproximado.
    Delta = Volume × (2 × (Close - Low) / (High - Low) - 1)
    """
    range_size = (df['high'] - df['low']).replace(0, np.nan)
    close_pos = (df['close'] - df['low']) / range_size
    delta = df['volume'] * (2 * close_pos - 1)
    return delta.fillna(0)


def compute_cvd(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Cumulative Volume Delta."""
    delta = compute_delta_volume(df)
    return delta.rolling(period).sum()


def compute_tfi(df: pd.DataFrame, period: int = 5) -> pd.Series:
    """
    Trade Flow Imbalance (TFI) simplificado.
    
    TFI = (Volume_Buy - Volume_Sell) / Volume_Total
    Aproximado usando posición del cierre en el rango de la vela.
    """
    range_size = (df['high'] - df['low']).replace(0, np.nan)
    buy_ratio = (df['close'] - df['low']) / range_size
    
    buy_vol = df['volume'] * buy_ratio
    sell_vol = df['volume'] * (1 - buy_ratio)
    
    tfi = (buy_vol - sell_vol).rolling(period).sum()
    tfi_norm = tfi / df['volume'].rolling(period).sum().replace(0, np.nan)
    
    return tfi_norm.fillna(0).clip(-1, 1)


def compute_ofi(df: pd.DataFrame, period: int = 5) -> pd.Series:
    """
    Order Flow Imbalance (OFI) simplificado.
    
    Aproximado usando cambios de precio ponderados por volumen.
    """
    price_change = df['close'].diff()
    vol_weighted = price_change * df['volume']
    
    ofi = vol_weighted.rolling(period).sum()
    ofi_norm = ofi / df['volume'].rolling(period).sum().replace(0, np.nan)
    
    return ofi_norm.fillna(0).clip(-1, 1)


# ============================================================
# KAUFMAN EFFICIENCY RATIO
# ============================================================

def compute_ker(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """Kaufman Efficiency Ratio."""
    if df.empty or len(df) < period:
        return pd.Series(0.0, index=df.index)
    close = df['close']
    change = close.diff(period).abs()
    volatility = close.diff().abs().rolling(period).sum()
    ker = change / (volatility + 1e-9)
    return ker.fillna(0).clip(0, 1)


# ============================================================
# RÉGIMEN
# ============================================================

def compute_regime(df: pd.DataFrame, adx_val: float,
                   atr_pct: float) -> str:
    """Clasifica régimen de mercado."""
    if df.empty or len(df) < 30:
        return 'Chop'
    if adx_val > 40 and atr_pct > 0.02:
        return 'Expansion'
    if adx_val > 30:
        return 'Trend_Strong'
    if adx_val > 20:
        return 'Trend_Weak'
    return 'Chop'
