"""
Motor de backtesting con walk-forward validation.
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
    mae: float


@dataclass
class BacktestResult:
    """Resultado del backtest."""
    trades: List[Trade]
    metrics: Dict
    equity_curve: pd.Series
    config: Dict = field(default_factory=dict)


class Backtester:
    """
    Motor de backtesting con gestión de riesgo.
    
    Implementa:
    - Stop Loss ATR dinámico
    - Take Profit ATR dinámico
    - Trailing Stop
    - Break Even
    """
    
    def __init__(self, config):
        self.config = config
        self.scorer = CompositeScorer(config)
        self.risk = config.risk
        self.costs = config.costs
    
    def run(self, data_dict: Dict[str, Dict[str, pd.DataFrame]],
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            scan_every_n_bars: int = 5) -> BacktestResult:
        """
        Ejecuta backtest sobre universo de activos.
        
        Args:
            data_dict: {symbol: {'5m': df, '15m': df, '1h': df}}
            start_date: Fecha inicio.
            end_date: Fecha fin.
            scan_every_n_bars: Frecuencia de escaneo.
        
        Returns:
            BacktestResult
        """
        # Recolectar todos los timestamps
        all_ts = set()
        for sym_data in data_dict.values():
            df = sym_data.get('5m')
            if df is not None and not df.empty:
                all_ts.update(df.index.tolist())
        
        timestamps = sorted(all_ts)
        if start_date:
            timestamps = [t for t in timestamps if t >= pd.Timestamp(start_date, tz='UTC')]
        if end_date:
            timestamps = [t for t in timestamps if t <= pd.Timestamp(end_date, tz='UTC')]
        
        capital = self.risk['initial_capital']
        open_positions: Dict[str, Dict] = {}
        trades: List[Trade] = []
        equity_curve = []
        
        for i, ts in enumerate(timestamps):
            # 1. Actualizar posiciones abiertas
            closed = []
            for symbol, pos in list(open_positions.items()):
                df = data_dict[symbol]['5m']
                if ts not in df.index:
                    continue
                
                row = df.loc[ts]
                exit_reason = self._update_position(pos, row)
                
                if exit_reason:
                    trade = self._close_position(pos, ts, row['close'], exit_reason)
                    trades.append(trade)
                    capital += trade.pnl
                    closed.append(symbol)
            
            for sym in closed:
                del open_positions[sym]
            
            # 2. Buscar nuevas señales
            if i % scan_every_n_bars == 0 and len(open_positions) < self.risk['max_concurrent_positions']:
                for symbol, sym_data in data_dict.items():
                    if symbol in open_positions:
                        continue
                    
                    df_5m = sym_data['5m']
                    if ts not in df_5m.index:
                        continue
                    
                    # Datos hasta ts
                    df_5m_slice = df_5m.loc[:ts].tail(500)
                    df_15m_slice = sym_data['15m'].loc[:ts].tail(500) if sym_data['15m'] is not None else None
                    df_1h_slice = sym_data['1h'].loc[:ts].tail(500) if sym_data['1h'] is not None else None
                    
                    if len(df_5m_slice) < 200:
                        continue
                    
                    try:
                        score = self.scorer.compute(symbol, {
                            '5m': df_5m_slice,
                            '15m': df_15m_slice,
                            '1h': df_1h_slice,
                        })
                        
                        if score.is_valid:
                            entry = df_5m_slice['close'].iloc[-1]
                            atr = compute_atr(df_5m_slice, 14).iloc[-1]
                            
                            pos = self._create_position(
                                symbol=symbol,
                                direction=score.direction,
                                entry=entry,
                                atr=atr,
                                score=score.total_score,
                                capital=capital,
                                timestamp=ts,
                            )
                            open_positions[symbol] = pos
                    except Exception as e:
                        logger.debug(f"Error señal {symbol}: {e}")
            
            # 3. Registrar equity
            equity = capital
            for pos in open_positions.values():
                df = data_dict[pos['symbol']]['5m']
                if ts in df.index:
                    price = df.loc[ts, 'close']
                    if pos['direction'] == 'LONG':
                        unrealized = (price - pos['entry']) / pos['entry'] * pos['size'] * pos['entry']
                    else:
                        unrealized = (pos['entry'] - price) / pos['entry'] * pos['size'] * pos['entry']
                    equity += unrealized
            
            equity_curve.append(equity)
        
        # Cerrar posiciones abiertas al final
        for symbol, pos in open_positions.items():
            df = data_dict[symbol]['5m']
            last_price = df['close'].iloc[-1]
            trade = self._close_position(pos, df.index[-1], last_price, 'EOD')
            trades.append(trade)
            capital += trade.pnl
        
        equity_series = pd.Series(equity_curve, index=timestamps)
        metrics = self._compute_metrics(trades, equity_series)
        
        return BacktestResult(
            trades=trades,
            metrics=metrics,
            equity_curve=equity_series,
            config={'initial_capital': self.risk['initial_capital']},
        )
    
    def _create_position(self, symbol: str, direction: str, entry: float,
                         atr: float, score: float, capital: float,
                         timestamp: pd.Timestamp) -> Dict:
        """Crea posición con gestión de riesgo."""
        atr_mults = self.config.atr_mult_sl
        atr_mult = atr_mults.get(symbol, self.config.atr_mult_sl_default)
        
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
        
        # Position sizing
        risk_amount = capital * self.risk['risk_per_trade']
        per_unit_risk = abs(entry - stop_loss)
        size = risk_amount / per_unit_risk if per_unit_risk > 0 else 0
        
        leverage = self.config.get_leverage_for_score(score)
        max_notional = capital * leverage
        size = min(size, max_notional / entry)
        
        # Trailing
        trail_mults = self.config.trailing_distance_by_symbol
        trail_mult = trail_mults.get(symbol, trail_mults.get('default', 2.0))
        trailing_dist = trail_mult * atr
        trailing_activation = entry + self.risk['trailing']['activation_atr'] * atr \
            if direction == 'LONG' else entry - self.risk['trailing']['activation_atr'] * atr
        be_trigger = self.risk['trailing']['breakeven_trigger_atr'] * atr
        
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
    
    def _update_position(self, pos: Dict, row: pd.Series) -> Optional[str]:
        """Actualiza posición. Retorna razón de cierre si aplica."""
        high, low, close = row['high'], row['low'], row['close']
        
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
    
    def _close_position(self, pos: Dict, timestamp: pd.Timestamp,
                        exit_price: float, reason: str) -> Trade:
        """Cierra posición y genera Trade."""
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
            mae=0.0,
        )
    
    def _compute_metrics(self, trades: List[Trade],
                         equity: pd.Series) -> Dict:
        """Calcula métricas de rendimiento."""
        if not trades:
            return {'error': 'No trades', 'total_trades': 0}
        
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        
        win_rate = len(wins) / len(trades)
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        avg_loss = np.mean([t.pnl for t in losses]) if losses else 0
        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
        
        # Drawdown
        peak = equity.expanding().max()
        dd = (equity - peak) / peak
        max_dd = dd.min()
        
        # Sharpe (aproximado)
        returns = equity.pct_change().dropna()
        sharpe = (returns.mean() / returns.std() * np.sqrt(252 * 24 * 12)
                  if len(returns) > 1 and returns.std() > 0 else 0)
        
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
        
        # Trades por día (aprox)
        if len(equity) > 1:
            days = (equity.index[-1] - equity.index[0]).total_seconds() / 86400
            trades_per_day = len(trades) / max(days, 1)
        else:
            trades_per_day = 0
        
        # PnL por leverage
        avg_leverage = np.mean([t.leverage for t in trades])
        pnl_per_leverage = equity.iloc[-1] / max(avg_leverage, 1) if len(equity) > 0 else 0
        
        return {
            'total_trades': len(trades),
            'trades_per_day': round(trades_per_day, 2),
            'win_rate': round(win_rate * 100, 2),
            'profit_factor': round(pf, 2),
            'expectancy': round(expectancy, 2),
            'total_pnl': round(equity.iloc[-1] - equity.iloc[0], 2) if len(equity) > 0 else 0,
            'total_return_pct': round((equity.iloc[-1] / equity.iloc[0] - 1) * 100, 2) if len(equity) > 0 else 0,
            'max_drawdown_pct': round(max_dd * 100, 2),
            'sharpe_ratio': round(sharpe, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'max_consecutive_losses': max_streak,
            'avg_duration_minutes': round(np.mean(durations), 1) if durations else 0,
            'avg_leverage': round(avg_leverage, 2),
            'pnl_per_leverage': round(pnl_per_leverage, 2),
        }
    
    def walk_forward(self, data_dict: Dict, train_months: int = 3,
                     test_months: int = 1, purge_bars: int = 48) -> List[Dict]:
        """Ejecuta walk-forward validation."""
        # Recolectar timestamps
        all_ts = set()
        for sym_data in data_dict.values():
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
            test_start = train_end + timedelta(bars=purge_bars)
            test_end = test_start + timedelta(days=test_months * 30)
            
            # Backtest en OOS
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
