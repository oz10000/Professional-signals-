"""
Score Compuesto Normalizado 0-100 con calibración y umbrales dinámicos.
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from indicators import (
    compute_atr, compute_adx, compute_rsi, compute_macd,
    compute_roc, compute_bollinger, compute_ema,
    compute_volume_ratio, compute_delta_volume, compute_cvd,
    compute_tfi, compute_ofi, compute_ker, compute_regime,
)

logger = logging.getLogger(__name__)


@dataclass
class ScoreResult:
    symbol: str
    total_score: float = 0.0
    direction: str = 'NEUTRAL'
    prob_long: float = 0.5
    prob_short: float = 0.5
    trend_score: float = 0.0
    momentum_score: float = 0.0
    volume_score: float = 0.0
    volatility_score: float = 0.0
    liquidity_score: float = 0.0
    regime_score: float = 0.0
    tier: str = 'NO_TRADE'
    adx: float = 0.0
    atr_pct: float = 0.0
    rvol: float = 0.0
    tfi: float = 0.0
    ofi: float = 0.0
    rsi: float = 50.0
    regime_label: str = 'Chop'
    conditions: List[str] = field(default_factory=list)
    is_valid: bool = False
    reason: str = "Sin datos"


class CompositeScorer:
    """Score Compuesto Normalizado con calibración y umbrales por régimen."""
    
    def __init__(self, config, calibrator=None):
        self.config = config
        self.weights = config.weights
        self.volume_sub_weights = config.volume_sub_weights
        self.tiers = config.tiers
        self.thresholds = config.thresholds
        self.regime_thresholds = config.regime_thresholds
        self.min_bars = config.timeframes.get('min_bars_required', 200)
        self.calibrator = calibrator
    
    def compute(self, symbol: str, data: Dict[str, pd.DataFrame]) -> ScoreResult:
        try:
            df_5m = data.get('5m')
            df_15m = data.get('15m')
            df_1h = data.get('1h')
            
            if df_5m is None or df_5m.empty or len(df_5m) < self.min_bars:
                return self._empty_result(symbol, f"Datos insuficientes 5m")
            
            df_5m = df_5m.iloc[:-1]
            if df_15m is not None and not df_15m.empty:
                df_15m = df_15m.iloc[:-1]
            if df_1h is not None and not df_1h.empty:
                df_1h = df_1h.iloc[:-1]
            
            features = self._compute_features(df_5m, df_15m, df_1h)
            norm = self._normalize(features)
            total_score = self._weighted_score(norm)
            total_score = float(np.clip(total_score, 0, 100))
            
            if total_score != total_score:
                return self._empty_result(symbol, "Score NaN")
            
            tier = self._classify_tier(total_score)
            
            raw_prob = self._compute_raw_prob_long(features)
            if self.calibrator and self.calibrator.is_fitted:
                prob_long = self.calibrator.calibrate(raw_prob)
            else:
                prob_long = raw_prob
            
            prob_short = 1.0 - prob_long
            
            min_prob = self.thresholds.get('min_prob_directional', 0.60)
            if prob_long > min_prob:
                direction = 'LONG'
            elif prob_short > min_prob:
                direction = 'SHORT'
            else:
                direction = 'NEUTRAL'
            
            is_valid, reason = self._validate_dynamic(
                features, total_score, direction
            )
            conditions = self._collect_conditions(features, direction)
            
            return ScoreResult(
                symbol=symbol,
                total_score=round(total_score, 2),
                direction=direction,
                prob_long=round(prob_long, 4),
                prob_short=round(prob_short, 4),
                trend_score=round(norm.get('trend', 0) * 100, 2),
                momentum_score=round(norm.get('momentum', 0) * 100, 2),
                volume_score=round(norm.get('volume', 0) * 100, 2),
                volatility_score=round(norm.get('volatility', 0) * 100, 2),
                liquidity_score=round(norm.get('liquidity', 0) * 100, 2),
                regime_score=round(norm.get('regime', 0) * 100, 2),
                tier=tier,
                adx=round(features.get('adx', 0), 2),
                atr_pct=round(features.get('atr_pct', 0), 5),
                rvol=round(features.get('rvol', 0), 2),
                tfi=round(features.get('tfi', 0), 4),
                ofi=round(features.get('ofi', 0), 4),
                rsi=round(features.get('rsi', 50), 2),
                regime_label=features.get('regime_label', 'Chop'),
                conditions=conditions,
                is_valid=is_valid,
                reason=reason,
            )
        except Exception as e:
            logger.error(f"Error calculando score para {symbol}: {e}", exc_info=True)
            return self._empty_result(symbol, f"Error: {str(e)[:100]}")
    
    def _compute_features(self, df_5m, df_15m, df_1h):
        f = {}
        try:
            close = df_5m['close']
            f['ema_9'] = float(compute_ema(df_5m, 9).iloc[-1]) if len(df_5m) >= 9 else float(close.iloc[-1])
            f['ema_21'] = float(compute_ema(df_5m, 21).iloc[-1]) if len(df_5m) >= 21 else float(close.iloc[-1])
            f['ema_50'] = float(compute_ema(df_5m, 50).iloc[-1]) if len(df_5m) >= 50 else float(close.iloc[-1])
            f['ema_200'] = float(compute_ema(df_5m, 200).iloc[-1]) if len(df_5m) >= 200 else float(close.iloc[-1])
            f['adx'] = float(compute_adx(df_5m, 14).iloc[-1])
            f['rsi'] = float(compute_rsi(df_5m, 14).iloc[-1])
            macd_df = compute_macd(df_5m)
            f['macd_hist'] = float(macd_df['histogram'].iloc[-1])
            f['roc'] = float(compute_roc(df_5m, 10).iloc[-1])
            f['rvol'] = float(compute_volume_ratio(df_5m, 20).iloc[-1])
            f['tfi'] = float(compute_tfi(df_5m, 5).iloc[-1])
            f['ofi'] = float(compute_ofi(df_5m, 5).iloc[-1])
            f['cvd'] = float(compute_cvd(df_5m, 20).iloc[-1])
            f['delta_vol'] = float(compute_delta_volume(df_5m).iloc[-1])
            atr = float(compute_atr(df_5m, 14).iloc[-1])
            price = float(close.iloc[-1]) if len(close) > 0 else 1.0
            f['atr_pct'] = atr / price if price > 0 else 0.0
            bb = compute_bollinger(df_5m)
            f['bb_width'] = float(bb['bb_width'].iloc[-1])
            f['ker'] = float(compute_ker(df_5m, 10).iloc[-1])
            f['regime_label'] = compute_regime(df_5m, f['adx'], f['atr_pct'])
            f['mtf_aligned'] = False
            if df_15m is not None and not df_15m.empty and len(df_15m) >= 50:
                try:
                    ema_50_15m = float(compute_ema(df_15m, 50).iloc[-1])
                    ema_200_15m = float(compute_ema(df_15m, 200).iloc[-1]) if len(df_15m) >= 200 else ema_50_15m
                    f['mtf_aligned'] = bool(ema_50_15m > ema_200_15m)
                except Exception:
                    f['mtf_aligned'] = False
            try:
                avg_vol = float(df_5m['volume'].tail(100).mean())
                avg_price = float(close.tail(100).mean())
                f['avg_volume_usd'] = avg_vol * avg_price
            except Exception:
                f['avg_volume_usd'] = 0.0
            
            for k, v in f.items():
                if isinstance(v, float) and (v != v or v in (float('inf'), float('-inf'))):
                    f[k] = 0.0
        except Exception as e:
            logger.error(f"Error en _compute_features: {e}")
        return f
    
    def _normalize(self, f):
        n = {}
        try:
            ema_score = 0.0
            if f.get('ema_9', 0) > f.get('ema_21', 0): ema_score += 0.33
            if f.get('ema_21', 0) > f.get('ema_50', 0): ema_score += 0.33
            if f.get('ema_50', 0) > f.get('ema_200', 0): ema_score += 0.34
            adx_norm = min(max(f.get('adx', 0) / 40.0, 0), 1.0)
            ker_norm = min(max(f.get('ker', 0), 0), 1.0)
            n['trend'] = 0.40 * ema_score + 0.35 * adx_norm + 0.25 * ker_norm
            
            rsi = f.get('rsi', 50)
            rsi_norm = 1.0 - abs(rsi - 50.0) / 50.0
            macd_norm = min(max(f.get('macd_hist', 0) * 100 + 0.5, 0), 1)
            roc_norm = min(max(f.get('roc', 0) / 5.0 + 0.5, 0), 1)
            n['momentum'] = 0.40 * rsi_norm + 0.30 * macd_norm + 0.30 * roc_norm
            
            rvol_norm = min(f.get('rvol', 1.0) / 3.0, 1.0)
            tfi_norm = min(abs(f.get('tfi', 0)) * 2.5, 1.0)
            ofi_norm = min(abs(f.get('ofi', 0)) * 2.5, 1.0)
            cvd_norm = min(abs(f.get('cvd', 0)) * 0.5, 1.0)
            sw = self.volume_sub_weights
            n['volume'] = (
                sw.get('tfi', 0.35) * tfi_norm +
                sw.get('ofi', 0.25) * ofi_norm +
                sw.get('rvol', 0.25) * rvol_norm +
                sw.get('cvd', 0.15) * cvd_norm
            )
            
            atr_pct = f.get('atr_pct', 0)
            if 0.005 <= atr_pct <= 0.015:
                vol_score = 1.0
            elif 0.003 <= atr_pct <= 0.025:
                vol_score = 0.7
            else:
                vol_score = 0.3
            bb_w = f.get('bb_width', 0.02)
            bb_norm = 1.0 - abs(bb_w - 0.02) / 0.03 if bb_w else 0.5
            bb_norm = max(0, min(bb_norm, 1))
            n['volatility'] = 0.6 * vol_score + 0.4 * bb_norm
            
            v = f.get('avg_volume_usd', 0)
            if v > 1e9: liq = 1.0
            elif v > 1e8: liq = 0.85
            elif v > 1e7: liq = 0.65
            elif v > 1e6: liq = 0.40
            else: liq = 0.20
            n['liquidity'] = liq
            
            regime_map = {
                'Expansion': 1.0, 'Trend_Strong': 0.9,
                'Trend_Weak': 0.6, 'Chop': 0.2,
            }
            n['regime'] = regime_map.get(f.get('regime_label', 'Chop'), 0.3)
            
            for k, v in n.items():
                if not isinstance(v, (int, float)) or v != v:
                    n[k] = 0.0
                n[k] = max(0.0, min(1.0, float(v)))
        except Exception as e:
            logger.error(f"Error en _normalize: {e}")
            n = {k: 0.0 for k in ['trend', 'momentum', 'volume', 'volatility', 'liquidity', 'regime']}
        return n
    
    def _weighted_score(self, norm):
        try:
            w = self.weights
            return (
                w.get('trend', 0.17) * norm.get('trend', 0) +
                w.get('momentum', 0.15) * norm.get('momentum', 0) +
                w.get('volume_intelligence', 0.40) * norm.get('volume', 0) +
                w.get('volatility', 0.11) * norm.get('volatility', 0) +
                w.get('liquidity', 0.10) * norm.get('liquidity', 0) +
                w.get('regime', 0.07) * norm.get('regime', 0)
            ) * 100
        except Exception:
            return 0.0
    
    def _compute_raw_prob_long(self, f):
        try:
            score = 0.0
            score += 0.35 * (f.get('tfi', 0) * 2.5 + 0.5)
            score += 0.25 * (f.get('ofi', 0) * 2.5 + 0.5)
            score += 0.20 * (1.0 if f.get('ema_9', 0) > f.get('ema_21', 0) else 0.0)
            score += 0.10 * (1.0 if f.get('rsi', 50) > 50 else 0.0)
            score += 0.10 * (1.0 if f.get('mtf_aligned', False) else 0.0)
            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.5
    
    def _classify_tier(self, score):
        try:
            sorted_tiers = sorted(self.tiers.items(), key=lambda x: -x[1])
            for tier, threshold in sorted_tiers:
                if score >= threshold:
                    return tier
        except Exception:
            pass
        return 'NO_TRADE'
    
    def _validate_dynamic(self, f, score, direction):
        """Validación con umbrales específicos del régimen."""
        try:
            regime = f.get('regime_label', 'Chop')
            thresholds = self.regime_thresholds.get(regime,
                          self.regime_thresholds.get('Chop',
                          {'min_score': 65, 'min_adx': 15, 'min_rvol': 2.0}))
            
            if score < thresholds['min_score']:
                return False, f"Score {score:.1f} < {thresholds['min_score']}"
            if direction == 'NEUTRAL':
                return False, "Dirección neutral"
            if f.get('rvol', 0) < thresholds['min_rvol']:
                return False, f"RVOL {f.get('rvol', 0):.2f} < {thresholds['min_rvol']}"
            if f.get('adx', 0) < thresholds['min_adx']:
                return False, f"ADX {f.get('adx', 0):.1f} < {thresholds['min_adx']}"
            
            # Cost-aware filter
            expected_move = f.get('atr_pct', 0) * 1.5
            min_move = self.thresholds.get('min_expected_move_pct', 0.0032)
            if expected_move < min_move:
                return False, f"Movimiento esperado {expected_move:.4f} < {min_move}"
            
            return True, "OK"
        except Exception as e:
            return False, f"Error validación: {str(e)[:50]}"
    
    def _collect_conditions(self, f, direction):
        c = []
        try:
            if f.get('adx', 0) >= 30:
                c.append(f"ADX≥30 ({f['adx']:.0f})")
            if f.get('ker', 0) >= 0.55:
                c.append(f"KER≥0.55 ({f['ker']:.2f})")
            if f.get('rvol', 0) >= 2.0:
                c.append(f"RVOL≥2.0 ({f['rvol']:.2f})")
            if abs(f.get('tfi', 0)) >= 0.2:
                c.append(f"TFI {'+' if f['tfi']>0 else ''}{f['tfi']:.2f}")
            if abs(f.get('ofi', 0)) >= 0.2:
                c.append(f"OFI {'+' if f['ofi']>0 else ''}{f['ofi']:.2f}")
            if f.get('mtf_aligned', False):
                c.append("MTF✓")
            if f.get('regime_label') in ('Expansion', 'Trend_Strong'):
                c.append(f"Régimen:{f['regime_label']}")
        except Exception:
            pass
        return c
    
    def _empty_result(self, symbol, reason):
        return ScoreResult(
            symbol=symbol, total_score=0.0, direction='NEUTRAL',
            prob_long=0.5, prob_short=0.5, tier='NO_TRADE',
            is_valid=False, reason=reason,
        )
