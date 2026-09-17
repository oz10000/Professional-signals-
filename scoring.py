"""
Score Compuesto Normalizado 0-100.

Combina 6 componentes ponderados con normalización robusta.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from indicators import (
    compute_atr, compute_adx, compute_rsi, compute_macd,
    compute_roc, compute_bollinger, compute_ema, compute_sma,
    compute_volume_ratio, compute_delta_volume, compute_cvd,
    compute_tfi, compute_ofi, compute_ker, compute_regime,
)


@dataclass
class ScoreResult:
    """Resultado del scoring para un activo."""
    symbol: str
    total_score: float
    direction: str
    prob_long: float
    prob_short: float
    trend_score: float
    momentum_score: float
    volume_score: float
    volatility_score: float
    liquidity_score: float
    regime_score: float
    tier: str
    adx: float
    atr_pct: float
    rvol: float
    tfi: float
    ofi: float
    rsi: float
    regime_label: str
    conditions: List[str] = field(default_factory=list)
    is_valid: bool = True
    reason: str = "OK"


class CompositeScorer:
    """
    Score Compuesto Normalizado 0-100.
    
    Componentes:
    - Tendencia (18%)
    - Momentum (16%)
    - Volumen Inteligente (44%)
    - Volatilidad (12%)
    - Liquidez (10%)
    - Régimen (8%)
    """
    
    def __init__(self, config):
        self.config = config
        self.weights = config.weights
        self.volume_sub_weights = config.raw['scoring']['sub_weights']['volume']
        self.tiers = config.tiers
        self.thresholds = config.thresholds
    
    def compute(self, symbol: str, data: Dict[str, pd.DataFrame]) -> ScoreResult:
        """
        Calcula Score Compuesto para un activo.
        
        Args:
            symbol: Símbolo del activo.
            data: Diccionario con DataFrames por timeframe.
        
        Returns:
            ScoreResult con score y componentes.
        """
        df_5m = data.get('5m')
        df_15m = data.get('15m')
        df_1h = data.get('1h')
        
        if df_5m is None or df_5m.empty or len(df_5m) < 200:
            return self._empty_result(symbol, "Datos insuficientes 5m")
        
        # Sin look-ahead: descartar última vela
        df_5m = df_5m.iloc[:-1]
        if df_15m is not None and not df_15m.empty:
            df_15m = df_15m.iloc[:-1]
        if df_1h is not None and not df_1h.empty:
            df_1h = df_1h.iloc[:-1]
        
        # Calcular features
        features = self._compute_features(df_5m, df_15m, df_1h)
        
        # Normalizar
        norm = self._normalize(features)
        
        # Score ponderado
        total_score = (
            self.weights['trend'] * norm['trend'] +
            self.weights['momentum'] * norm['momentum'] +
            self.weights['volume_intelligence'] * norm['volume'] +
            self.weights['volatility'] * norm['volatility'] +
            self.weights['liquidity'] * norm['liquidity'] +
            self.weights['regime'] * norm['regime']
        ) * 100
        
        total_score = float(np.clip(total_score, 0, 100))
        tier = self._classify_tier(total_score)
        
        # Dirección
        prob_long = self._compute_prob_long(features)
        prob_short = 1.0 - prob_long
        
        if prob_long > self.thresholds['min_prob_directional']:
            direction = 'LONG'
        elif prob_short > self.thresholds['min_prob_directional']:
            direction = 'SHORT'
        else:
            direction = 'NEUTRAL'
        
        # Validación
        is_valid, reason = self._validate(features, total_score, direction)
        
        # Condiciones
        conditions = self._collect_conditions(features, direction)
        
        return ScoreResult(
            symbol=symbol,
            total_score=round(total_score, 2),
            direction=direction,
            prob_long=round(prob_long, 4),
            prob_short=round(prob_short, 4),
            trend_score=round(norm['trend'] * 100, 2),
            momentum_score=round(norm['momentum'] * 100, 2),
            volume_score=round(norm['volume'] * 100, 2),
            volatility_score=round(norm['volatility'] * 100, 2),
            liquidity_score=round(norm['liquidity'] * 100, 2),
            regime_score=round(norm['regime'] * 100, 2),
            tier=tier,
            adx=round(features['adx'], 2),
            atr_pct=round(features['atr_pct'], 5),
            rvol=round(features['rvol'], 2),
            tfi=round(features['tfi'], 4),
            ofi=round(features['ofi'], 4),
            rsi=round(features['rsi'], 2),
            regime_label=features['regime_label'],
            conditions=conditions,
            is_valid=is_valid,
            reason=reason,
        )
    
    def _compute_features(self, df_5m: pd.DataFrame,
                          df_15m: Optional[pd.DataFrame],
                          df_1h: Optional[pd.DataFrame]) -> Dict:
        """Calcula todas las features necesarias."""
        close = df_5m['close']
        
        # Tendencia (5m)
        ema_9 = compute_ema(df_5m, 9).iloc[-1]
        ema_21 = compute_ema(df_5m, 21).iloc[-1]
        ema_50 = compute_ema(df_5m, 50).iloc[-1]
        ema_200 = compute_ema(df_5m, 200).iloc[-1]
        adx = compute_adx(df_5m, 14).iloc[-1]
        
        # Momentum
        rsi = compute_rsi(df_5m, 14).iloc[-1]
        macd_df = compute_macd(df_5m)
        macd_hist = macd_df['histogram'].iloc[-1]
        roc = compute_roc(df_5m, 10).iloc[-1]
        
        # Volumen
        rvol = compute_volume_ratio(df_5m, 20).iloc[-1]
        tfi = compute_tfi(df_5m, 5).iloc[-1]
        ofi = compute_ofi(df_5m, 5).iloc[-1]
        cvd = compute_cvd(df_5m, 20).iloc[-1]
        delta_vol = compute_delta_volume(df_5m).iloc[-1]
        
        # Volatilidad
        atr = compute_atr(df_5m, 14).iloc[-1]
        atr_pct = atr / close.iloc[-1] if close.iloc[-1] > 0 else 0
        bb = compute_bollinger(df_5m)
        bb_width = bb['bb_width'].iloc[-1]
        
        # KER
        ker = compute_ker(df_5m, 10).iloc[-1]
        
        # Régimen
        regime_label = compute_regime(df_5m, adx, atr_pct)
        
        # Multi-TF alignment
        mtf_aligned = False
        if df_15m is not None and not df_15m.empty:
            ema_50_15m = compute_ema(df_15m, 50).iloc[-1]
            ema_200_15m = compute_ema(df_15m, 200).iloc[-1] if len(df_15m) >= 200 else ema_50_15m
            mtf_aligned = bool(ema_50_15m > ema_200_15m)
        
        # Liquidez
        avg_volume_usd = (
            df_5m['volume'].tail(100).mean() *
            df_5m['close'].tail(100).mean()
        )
        
        return {
            'ema_9': ema_9, 'ema_21': ema_21, 'ema_50': ema_50,
            'ema_200': ema_200, 'adx': adx, 'rsi': rsi,
            'macd_hist': macd_hist, 'roc': roc,
            'rvol': rvol, 'tfi': tfi, 'ofi': ofi,
            'cvd': cvd, 'delta_vol': delta_vol,
            'atr_pct': atr_pct, 'bb_width': bb_width,
            'ker': ker, 'regime_label': regime_label,
            'mtf_aligned': mtf_aligned,
            'avg_volume_usd': avg_volume_usd,
        }
    
    def _normalize(self, f: Dict) -> Dict[str, float]:
        """Normaliza features a [0, 1]."""
        norm = {}
        
        # ---- TENDENCIA ----
        ema_score = 0.0
        if f['ema_9'] > f['ema_21']: ema_score += 0.33
        if f['ema_21'] > f['ema_50']: ema_score += 0.33
        if f['ema_50'] > f['ema_200']: ema_score += 0.34
        
        adx_norm = min(f['adx'] / 40.0, 1.0)
        ker_norm = min(f['ker'], 1.0)
        norm['trend'] = 0.40 * ema_score + 0.35 * adx_norm + 0.25 * ker_norm
        
        # ---- MOMENTUM ----
        rsi_norm = 1.0 - abs(f['rsi'] - 50.0) / 50.0
        macd_norm = min(max(f['macd_hist'] * 100 + 0.5, 0), 1)
        roc_norm = min(max(f['roc'] / 5.0 + 0.5, 0), 1)
        norm['momentum'] = 0.40 * rsi_norm + 0.30 * macd_norm + 0.30 * roc_norm
        
        # ---- VOLUMEN INTELIGENTE ----
        rvol_norm = min(f['rvol'] / 3.0, 1.0)
        tfi_norm = min(abs(f['tfi']) * 2.5, 1.0)
        ofi_norm = min(abs(f['ofi']) * 2.5, 1.0)
        cvd_norm = min(abs(f['cvd']) * 0.5, 1.0)
        
        sw = self.volume_sub_weights
        norm['volume'] = (
            sw['tfi'] * tfi_norm +
            sw['ofi'] * ofi_norm +
            sw['rvol'] * rvol_norm +
            sw['cvd'] * cvd_norm
        )
        
        # ---- VOLATILIDAD ----
        atr_pct = f['atr_pct']
        if 0.005 <= atr_pct <= 0.015:
            vol_score = 1.0
        elif 0.003 <= atr_pct <= 0.025:
            vol_score = 0.7
        else:
            vol_score = 0.3
        
        bb_norm = 1.0 - abs(f['bb_width'] - 0.02) / 0.03 if f['bb_width'] else 0.5
        bb_norm = max(0, min(bb_norm, 1))
        norm['volatility'] = 0.6 * vol_score + 0.4 * bb_norm
        
        # ---- LIQUIDEZ ----
        v = f['avg_volume_usd']
        if v > 1e9: liq = 1.0
        elif v > 1e8: liq = 0.85
        elif v > 1e7: liq = 0.65
        elif v > 1e6: liq = 0.40
        else: liq = 0.20
        norm['liquidity'] = liq
        
        # ---- RÉGIMEN ----
        regime_map = {
            'Expansion': 1.0,
            'Trend_Strong': 0.9,
            'Trend_Weak': 0.6,
            'Chop': 0.2,
        }
        norm['regime'] = regime_map.get(f['regime_label'], 0.3)
        
        return norm
    
    def _compute_prob_long(self, f: Dict) -> float:
        """Calcula probabilidad LONG (0-1)."""
        score = 0.0
        
        # TFI (peso alto)
        score += 0.35 * (f['tfi'] * 2.5 + 0.5)
        
        # OFI
        score += 0.25 * (f['ofi'] * 2.5 + 0.5)
        
        # EMA alignment
        ema_bull = 1.0 if f['ema_9'] > f['ema_21'] else 0.0
        score += 0.20 * ema_bull
        
        # RSI
        rsi_bull = 1.0 if f['rsi'] > 50 else 0.0
        score += 0.10 * rsi_bull
        
        # MTF
        score += 0.10 * (1.0 if f['mtf_aligned'] else 0.0)
        
        return float(np.clip(score, 0.0, 1.0))
    
    def _classify_tier(self, score: float) -> str:
        """Clasifica score en tier."""
        for tier, threshold in self.tiers.items():
            if score >= threshold:
                return tier
        return 'NO_TRADE'
    
    def _validate(self, f: Dict, score: float, direction: str) -> tuple:
        """Valida si la señal es operable."""
        if score < self.thresholds['min_score']:
            return False, f"Score {score:.1f} < {self.thresholds['min_score']}"
        if direction == 'NEUTRAL':
            return False, "Dirección neutral"
        if f['rvol'] < self.thresholds['min_volume_ratio']:
            return False, f"RVOL {f['rvol']:.2f} < {self.thresholds['min_volume_ratio']}"
        if f['adx'] < self.thresholds['min_adx']:
            return False, f"ADX {f['adx']:.1f} < {self.thresholds['min_adx']}"
        if not (self.thresholds['min_atr_pct'] <= f['atr_pct'] <= self.thresholds['max_atr_pct']):
            return False, f"ATR% {f['atr_pct']:.4f} fuera de rango"
        if f['regime_label'] == 'Chop':
            return False, "Régimen Chop"
        return True, "OK"
    
    def _collect_conditions(self, f: Dict, direction: str) -> List[str]:
        """Recolecta condiciones favorables."""
        c = []
        if f['adx'] >= 30:
            c.append(f"ADX≥30 ({f['adx']:.0f})")
        if f['ker'] >= 0.55:
            c.append(f"KER≥0.55 ({f['ker']:.2f})")
        if f['rvol'] >= 2.0:
            c.append(f"RVOL≥2.0 ({f['rvol']:.2f})")
        if abs(f['tfi']) >= 0.2:
            c.append(f"TFI {'+' if f['tfi']>0 else ''}{f['tfi']:.2f}")
        if abs(f['ofi']) >= 0.2:
            c.append(f"OFI {'+' if f['ofi']>0 else ''}{f['ofi']:.2f}")
        if f['mtf_aligned']:
            c.append("MTF✓")
        if f['regime_label'] in ('Expansion', 'Trend_Strong'):
            c.append(f"Régimen:{f['regime_label']}")
        return c
    
    def _empty_result(self, symbol: str, reason: str) -> ScoreResult:
        """Resultado vacío."""
        return ScoreResult(
            symbol=symbol, total_score=0.0, direction='NEUTRAL',
            prob_long=0.5, prob_short=0.5, trend_score=0.0,
            momentum_score=0.0, volume_score=0.0, volatility_score=0.0,
            liquidity_score=0.0, regime_score=0.0, tier='NO_TRADE',
            adx=0.0, atr_pct=0.0, rvol=0.0, tfi=0.0, ofi=0.0, rsi=50.0,
            regime_label='Chop', is_valid=False, reason=reason,
        )
