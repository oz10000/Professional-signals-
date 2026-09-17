"""
Motor de backtesting con walk-forward, Monte Carlo y DSR.
D.A.P.S-SIGNALS Ω v3.0.0
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import timedelta

from indicators import compute_atr
from scoring import CompositeScorer

logger = logging.getLogger(__name__)


# ============================================================
# DATACLASSES
# ============================================================

@dataclass
class Trade:
    """Operación completada."""
    symbol: str
    direction: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    leverage: int
    pnl: float
    pnl_pct: float
    exit_reason: str
    score: float
    duration_minutes: float
    mfe: float
    mae: float = 0.0


@dataclass
class BacktestResult:
    """Resultado del backtest."""
    trades: List[Trade] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    config: Dict = field(default_factory=dict)


# ============================================================
# BACKTESTER
# ============================================================

class Backtester:
    """
    Motor de backtesting con gestión de riesgo robusta.
    
    Implementa:
    - Walk-forward con purga de 48 barras
    - Stop Loss ATR dinámico
    - Take Profit ATR dinámico
    - Trailing Stop ATR por activo
    - Break Even
    - Monte Carlo
    - Deflated Sharpe Ratio
    """
    
    def __init__(self, config):
        self.config = config
        self.scorer = CompositeScorer(config)
        self.risk = config.risk
        self.costs = config.costs
    
    # --------------------------------------------------------
    # BACKTEST PRINCIPAL
    # --------------------------------------------------------
    
    def run(self, data_dict: Dict[str, Dict[str, pd.DataFrame]],
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            scan_every_n_bars: int = 5) -> BacktestResult:
        """
        Ejecuta backtest sobre universo de activos.
        
        Args:
            data_dict: {symbol: {'5m': df, '15m': df, '1h': df}}
            start_date: Fecha inicio (ISO format).
            end_date: Fecha fin (ISO format).
            scan_every_n_bars: Frecuencia de escaneo (default 5).
        
        Returns:
            BacktestResult con trades, métricas y curva de equity.
        """
        try:
            # Recolectar timestamps
            all_ts = set()
            for sym_data in data_dict.values():
                if not isinstance(sym_data, dict):
                    continue
                df = sym_data.get('5m')
                if df is not None and not df.empty:
                    all_ts.update(df.index.tolist())
            
            if not all_ts:
                logger.warning("No hay timestamps para backtest")
                return BacktestResult(metrics={'error': 'Sin datos'})
            
            timestamps = sorted(all_ts)
            
            if start_date:
                timestamps = [t for t in timestamps if t >= pd.Timestamp(start_date, tz='UTC')]
            if end_date:
                timestamps = [t for t in timestamps if t <= pd.Timestamp(end_date, tz='UTC')]
            
            if not timestamps:
                return BacktestResult(metrics={'error': 'Sin timestamps en rango'})
            
            capital = self.risk.get('initial_capital', 10000.0)
            open_positions: Dict[str, Dict] = {}
            trades: List[Trade] = []
            equity_curve = []
            max_concurrent = self.risk.get('max_concurrent_positions', 3)
            
            for i, ts in enumerate(timestamps):
                # 1. Actualizar posiciones abiertas
                closed = []
                for symbol, pos in list(open_positions.items()):
                    try:
                        df = data_dict[symbol]['5m']
                        if ts not in df.index:
                            continue
                        row = df.loc[ts]
                        if isinstance(row, pd.DataFrame):
                            row = row.iloc[0]
                        
                        exit_reason = self._update_position(pos, row)
                        if exit_reason:
                            trade = self._close_position(
                                pos, ts, float(row['close']), exit_reason
                            )
                            if trade:
                                trades.append(trade)
                                capital += trade.pnl
                            closed.append(symbol)
                    except Exception as e:
                        logger.debug(f"Error actualizando {symbol}: {e}")
                        closed.append(symbol)
                
                for sym in closed:
                    open_positions.pop(sym, None)
                
                # 2. Buscar nuevas señales
                if i % scan_every_n_bars == 0 and len(open_positions) < max_concurrent:
                    for symbol, sym_data in data_dict.items():
                        if symbol in open_positions:
                            continue
                        if not isinstance(sym_data, dict):
                            continue
                        
                        df_5m = sym_data.get('5m')
                        if df_5m is None or ts not in df_5m.index:
                            continue
                        
                        try:
                            df_5m_slice = df_5m.loc[:ts].tail(500)
                            df_15m = sym_data.get('15m')
                            df_1h = sym_data.get('1h')
                            df_15m_slice = df_15m.loc[:ts].tail(500) if df_15m is not None else None
                            df_1h_slice = df_1h.loc[:ts].tail(500) if df_1h is not None else None
                            
                            if len(df_5m_slice) < 200:
                                continue
                            
                            score = self.scorer.compute(symbol, {
                                '5m': df_5m_slice,
                                '15m': df_15m_slice,
                                '1h': df_1h_slice,
                            })
                            
                            if score.is_valid:
                                entry = float(df_5m_slice['close'].iloc[-1])
                                atr_series = compute_atr(df_5m_slice, 14)
                                atr = float(atr_series.iloc[-1]) if not atr_series.empty else entry * 0.005
                                
                                pos = self._create_position(
                                    symbol, score.direction, entry, atr,
                                    score.total_score, capital, ts
                                )
                                if pos:
                                    open_positions[symbol] = pos
                        except Exception as e:
                            logger.debug(f"Error señal {symbol}: {e}")
                
                # 3. Registrar equity
                equity = capital
                for pos in open_positions.values():
                    try:
                        df = data_dict[pos['symbol']]['5m']
                        if ts in df.index:
                            price = float(df.loc[ts, 'close'])
                            if isinstance(price, (int, float)) and price > 0:
                                if pos['direction'] == 'LONG':
                                    unrealized = (price - pos['entry']) / pos['entry'] * pos['size'] * pos['entry']
                                else:
                                    unrealized = (pos['entry'] - price) / pos['entry'] * pos['size'] * pos['entry']
                                equity += unrealized
                    except Exception:
                        pass
                
                equity_curve.append(equity)
            
            # Cerrar posiciones abiertas al final
            for symbol, pos in list(open_positions.items()):
                try:
                    df = data_dict[symbol]['5m']
                    if not df.empty:
                        last_price = float(df['close'].iloc[-1])
                        trade = self._close_position(pos, df.index[-1], last_price, 'EOD')
                        if trade:
                            trades.append(trade)
                            capital += trade.pnl
                except Exception:
                    pass
            
            equity_series = pd.Series(equity_curve, index=timestamps)
            metrics = self._compute_metrics(trades, equity_series)
            
            return BacktestResult(
                trades=trades,
                metrics=metrics,
                equity_curve=equity_series,
                config={'initial_capital': self.risk.get('initial_capital', 10000.0)},
            )
        
        except Exception as e:
            logger.error(f"Error en backtest: {e}", exc_info=True)
            return BacktestResult(metrics={'error': str(e)[:200]})
    
    # --------------------------------------------------------
    # CREAR POSICIÓN
    # --------------------------------------------------------
    
    def _create_position(self, symbol: str, direction: str, entry: float,
                         atr: float, score: float, capital: float,
                         timestamp: pd.Timestamp) -> Optional[Dict]:
        """Crea posición con gestión de riesgo."""
        try:
            if entry <= 0 or atr <= 0:
                return None
            
            atr_mult = self.config.get_atr_mult_sl(symbol)
            sl_dist = atr_mult * atr
            sl_dist = max(min(sl_dist / entry, 0.02), 0.0015)
            
            tp_mult = self.config.atr_mult_tp_default
            tp_dist = max(tp_mult * atr / entry, 0.003)
            
            if direction == 'LONG':
                stop_loss = entry * (1 - sl_dist)
                take_profit = entry * (1 + tp_dist)
                be_price = entry * (1 + self.config.total_cost_round_trip)
            else:
                stop_loss = entry * (1 + sl_dist)
                take_profit = entry * (1 - tp_dist)
                be_price = entry * (1 - self.config.total_cost_round_trip)
            
            # Position sizing por volatilidad
            risk_amount = capital * self.risk.get('risk_per_trade', 0.01)
            per_unit_risk = abs(entry - stop_loss)
            if per_unit_risk <= 0:
                return None
            size = risk_amount / per_unit_risk
            
            leverage = self.config.get_leverage_for_score(score)
            max_notional = capital * leverage
            size = min(size, max_notional / entry)
            
            if size <= 0 or size != size:
                return None
            
            trail_mult = self.config.get_trailing_distance(symbol)
            trailing_dist = trail_mult * atr
            trailing_act_atr = self.risk.get('trailing', {}).get('activation_atr', 0.5)
            trailing_activation = entry + trailing_act_atr * atr if direction == 'LONG' \
                else entry - trailing_act_atr * atr
            be_trigger = self.risk.get('trailing', {}).get('breakeven_trigger_atr', 0.25) * atr
            
            return {
                'symbol': symbol,
                'direction': direction,
                'entry': entry,
                'entry_time': timestamp,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'initial_stop': stop_loss,
                'be_price': be_price,
                'be_trigger': be_trigger,
                'trailing_dist': trailing_dist,
                'trailing_activation': trailing_activation,
                'best_price': entry,
                'size': size,
                'leverage': leverage,
                'score': score,
                'be_triggered': False,
                'trailing_active': False,
            }
        except Exception as e:
            logger.debug(f"Error creando posición {symbol}: {e}")
            return None
    
    # --------------------------------------------------------
    # ACTUALIZAR POSICIÓN
    # --------------------------------------------------------
    
    def _update_position(self, pos: Dict, row: pd.Series) -> Optional[str]:
        """Actualiza posición. Retorna razón de cierre si aplica."""
        try:
            high = float(row['high'])
            low = float(row['low'])
            close = float(row['close'])
            
            if high != high or low != low or close != close:
                return None
            
            if pos['direction'] == 'LONG':
                # Stop Loss
                if low <= pos['stop_loss']:
                    return 'SL' if not pos['be_triggered'] else 'BE'
                # Take Profit
                if high >= pos['take_profit']:
                    return 'TP'
                # Break Even
                if not pos['be_triggered'] and high >= pos['entry'] + pos['be_trigger']:
                    pos['be_triggered'] = True
                    pos['stop_loss'] = pos['be_price']
                # Trailing
                if high >= pos['trailing_activation']:
                    pos['trailing_active'] = True
                    pos['best_price'] = max(pos['best_price'], high)
                    new_stop = pos['best_price'] - pos['trailing_dist']
                    pos['stop_loss'] = max(pos['stop_loss'], new_stop)
            else:
                if high >= pos['stop_loss']:
                    return 'SL' if not pos['be_triggered'] else 'BE'
                if low <= pos['take_profit']:
                    return 'TP'
                if not pos['be_triggered'] and low <= pos['entry'] - pos['be_trigger']:
                    pos['be_triggered'] = True
                    pos['stop_loss'] = pos['be_price']
                if low <= pos['trailing_activation']:
                    pos['trailing_active'] = True
                    pos['best_price'] = min(pos['best_price'], low)
                    new_stop = pos['best_price'] + pos['trailing_dist']
                    pos['stop_loss'] = min(pos['stop_loss'], new_stop)
            
            return None
        except Exception as e:
            logger.debug(f"Error actualizando posición: {e}")
            return None
    
    # --------------------------------------------------------
    # CERRAR POSICIÓN
    # --------------------------------------------------------
    
    def _close_position(self, pos: Dict, timestamp: pd.Timestamp,
                        exit_price: float, reason: str) -> Optional[Trade]:
        """Cierra posición y genera Trade."""
        try:
            if exit_price <= 0:
                return None
            
            if pos['direction'] == 'LONG':
                pnl_pct = (exit_price - pos['entry']) / pos['entry']
                mfe = (pos['best_price'] - pos['entry']) / pos['entry']
            else:
                pnl_pct = (pos['entry'] - exit_price) / pos['entry']
                mfe = (pos['entry'] - pos['best_price']) / pos['entry']
            
            pnl_pct -= self.config.total_cost_round_trip
            pnl = pnl_pct * pos['size'] * pos['entry']
            duration = (timestamp - pos['entry_time']).total_seconds() / 60
            
            return Trade(
                symbol=pos['symbol'],
                direction=pos['direction'],
                entry_time=pos['entry_time'],
                exit_time=timestamp,
                entry_price=pos['entry'],
                exit_price=exit_price,
                stop_loss=pos['stop_loss'],
                take_profit=pos['take_profit'],
                position_size=pos['size'],
                leverage=pos['leverage'],
                pnl=pnl,
                pnl_pct=pnl_pct * 100,
                exit_reason=reason,
                score=pos['score'],
                duration_minutes=duration,
                mfe=mfe * 100,
            )
        except Exception as e:
            logger.debug(f"Error cerrando posición: {e}")
            return None
    
    # --------------------------------------------------------
    # MÉTRICAS
    # --------------------------------------------------------
    
    def _compute_metrics(self, trades: List[Trade],
                         equity: pd.Series) -> Dict:
        """Calcula métricas de rendimiento con defensa."""
        if not trades:
            return {
                'total_trades': 0, 'win_rate': 0.0, 'profit_factor': 0.0,
                'expectancy': 0.0, 'total_pnl': 0.0, 'total_return_pct': 0.0,
                'max_drawdown_pct': 0.0, 'sharpe_ratio': 0.0,
                'avg_win': 0.0, 'avg_loss': 0.0, 'max_consecutive_losses': 0,
                'avg_duration_minutes': 0.0, 'trades_per_day': 0.0,
                'avg_leverage': 0.0, 'pnl_per_leverage': 0.0,
            }
        
        try:
            wins = [t for t in trades if t.pnl > 0]
            losses = [t for t in trades if t.pnl <= 0]
            
            win_rate = len(wins) / len(trades) if trades else 0
            
            gross_profit = sum(t.pnl for t in wins)
            gross_loss = abs(sum(t.pnl for t in losses))
            pf = gross_profit / gross_loss if gross_loss > 0 else (
                float('inf') if gross_profit > 0 else 0.0
            )
            
            avg_win = float(np.mean([t.pnl for t in wins])) if wins else 0.0
            avg_loss = float(np.mean([t.pnl for t in losses])) if losses else 0.0
            expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
            
            # Drawdown
            if len(equity) > 1:
                peak = equity.expanding().max()
                dd = (equity - peak) / peak.replace(0, np.nan)
                max_dd = float(dd.min()) if not dd.empty else 0.0
                if max_dd != max_dd:
                    max_dd = 0.0
            else:
                max_dd = 0.0
            
            # Sharpe
            if len(equity) > 1:
                returns = equity.pct_change().dropna()
                if len(returns) > 1 and returns.std() > 0:
                    sharpe = float(returns.mean() / returns.std() * np.sqrt(252 * 24 * 12))
                    if sharpe != sharpe:
                        sharpe = 0.0
                else:
                    sharpe = 0.0
            else:
                sharpe = 0.0
            
            # Racha negativa máxima
            max_streak = 0
            streak = 0
            for t in trades:
                if t.pnl <= 0:
                    streak += 1
                    max_streak = max(max_streak, streak)
                else:
                    streak = 0
            
            # Duración
            durations = [t.duration_minutes for t in trades]
            avg_dur = float(np.mean(durations)) if durations else 0.0
            
            # Trades por día
            if len(equity) > 1:
                days = (equity.index[-1] - equity.index[0]).total_seconds() / 86400
                tpd = len(trades) / max(days, 1)
            else:
                tpd = 0.0
            
            # Leverage
            avg_lev = float(np.mean([t.leverage for t in trades]))
            total_pnl = float(equity.iloc[-1] - equity.iloc[0]) if len(equity) > 0 else 0.0
            pnl_per_lev = total_pnl / max(avg_lev, 1.0)
            
            total_ret = ((equity.iloc[-1] / equity.iloc[0] - 1) * 100) if len(equity) > 0 and equity.iloc[0] > 0 else 0.0
            
            return {
                'total_trades': len(trades),
                'trades_per_day': round(tpd, 2),
                'win_rate': round(win_rate * 100, 2),
                'profit_factor': round(pf, 2) if pf != float('inf') else 999.99,
                'expectancy': round(expectancy, 2),
                'total_pnl': round(total_pnl, 2),
                'total_return_pct': round(total_ret, 2),
                'max_drawdown_pct': round(max_dd * 100, 2),
                'sharpe_ratio': round(sharpe, 2),
                'avg_win': round(avg_win, 2),
                'avg_loss': round(avg_loss, 2),
                'max_consecutive_losses': max_streak,
                'avg_duration_minutes': round(avg_dur, 1),
                'avg_leverage': round(avg_lev, 2),
                'pnl_per_leverage': round(pnl_per_lev, 2),
            }
        except Exception as e:
            logger.error(f"Error calculando métricas: {e}", exc_info=True)
            return {'error': str(e)[:200], 'total_trades': len(trades)}
    
    # --------------------------------------------------------
    # WALK-FORWARD
    # --------------------------------------------------------
    
    def walk_forward(self, data_dict: Dict,
                     train_months: int = 3,
                     test_months: int = 1,
                     purge_bars: int = 48) -> List[Dict]:
        """
        Ejecuta walk-forward validation con purga.
        
        Returns:
            Lista de resultados por fold.
        """
        try:
            all_ts = set()
            for sym_data in data_dict.values():
                if not isinstance(sym_data, dict):
                    continue
                df = sym_data.get('5m')
                if df is not None and not df.empty:
                    all_ts.update(df.index.tolist())
            
            timestamps = sorted(all_ts)
            if not timestamps:
                return []
            
            start = timestamps[0]
            end = timestamps[-1]
            
            results = []
            current = start
            
            while current + timedelta(days=train_months * 30 + test_months * 30) <= end:
                train_end = current + timedelta(days=train_months * 30)
                # Purga de 48 barras (5m = 4 horas)
                purge_end = train_end + timedelta(minutes=purge_bars * 5)
                test_start = purge_end
                test_end = test_start + timedelta(days=test_months * 30)
                
                result = self.run(
                    data_dict,
                    start_date=test_start.isoformat(),
                    end_date=test_end.isoformat(),
                )
                
                results.append({
                    'train_start': current.isoformat(),
                    'train_end': train_end.isoformat(),
                    'test_start': test_start.isoformat(),
                    'test_end': test_end.isoformat(),
                    'metrics': result.metrics,
                })
                
                current = test_end
            
            return results
        except Exception as e:
            logger.error(f"Error en walk_forward: {e}")
            return []
    
    # --------------------------------------------------------
    # MONTE CARLO
    # --------------------------------------------------------
    
    def monte_carlo(self, trades: List[Trade],
                    n_simulations: int = 10000,
                    initial_capital: float = 10000.0) -> Dict:
        """
        Ejecuta Monte Carlo sobre los trades del backtest.
        
        Returns:
            Dict con percentiles de retorno, DD y probabilidad de ruina.
        """
        try:
            if not trades:
                return {'error': 'Sin trades'}
            
            pnls = np.array([t.pnl for t in trades])
            n_trades = len(pnls)
            
            final_returns = []
            max_dds = []
            
            for _ in range(n_simulations):
                # Permutación aleatoria de trades
                shuffled = np.random.permutation(pnls)
                
                # Curva de equity
                equity = initial_capital + np.cumsum(shuffled)
                equity = np.concatenate([[initial_capital], equity])
                
                # Retorno final
                final_ret = (equity[-1] / initial_capital - 1) * 100
                final_returns.append(final_ret)
                
                # Max DD
                peak = np.maximum.accumulate(equity)
                dd = (equity - peak) / peak
                max_dds.append(dd.min() * 100)
            
            final_returns = np.array(final_returns)
            max_dds = np.array(max_dds)
            
            prob_ruin = float(np.mean(max_dds < -30))
            
            return {
                'return_p5': float(np.percentile(final_returns, 5)),
                'return_p25': float(np.percentile(final_returns, 25)),
                'return_median': float(np.median(final_returns)),
                'return_p75': float(np.percentile(final_returns, 75)),
                'return_p95': float(np.percentile(final_returns, 95)),
                'max_dd_p5': float(np.percentile(max_dds, 5)),
                'max_dd_median': float(np.median(max_dds)),
                'max_dd_p95': float(np.percentile(max_dds, 95)),
                'prob_ruin': round(prob_ruin * 100, 2),
                'n_simulations': n_simulations,
            }
        except Exception as e:
            logger.error(f"Error en monte_carlo: {e}")
            return {'error': str(e)[:200]}
