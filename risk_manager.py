"""
Gestión de riesgo con position sizing ajustado por volatilidad
y trailing stop ATR por activo.
"""

import logging
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Position:
    symbol: str
    direction: str
    entry_price: float
    entry_time: object
    stop_loss: float
    take_profit: float
    position_size: float
    leverage: int
    initial_stop: float
    break_even_price: float
    trailing_activation: float
    trailing_distance: float
    be_triggered: bool = False
    trailing_active: bool = False
    best_price: float = 0.0


class RiskManager:
    """Gestiona el riesgo con position sizing por volatilidad."""
    
    def __init__(self, config):
        self.config = config
        self.risk = config.risk
        self.costs = config.costs
    
    def create_position(self, symbol, direction, entry_price, atr, score, capital, timestamp):
        """
        Crea posición con sizing ajustado por volatilidad.
        
        Formula: Size = (Capital × Risk%) / (ATR × Multiplier)
        """
        try:
            if entry_price <= 0 or atr <= 0:
                return None
            
            atr_mult = self.config.get_atr_mult_sl(symbol)
            sl_dist = atr_mult * atr
            sl_dist = max(min(sl_dist / entry_price, 0.02), 0.0015)
            
            tp_mult = self.config.atr_mult_tp_default
            tp_dist = max(tp_mult * atr / entry_price, 0.003)
            
            if direction == 'LONG':
                stop_loss = entry_price * (1 - sl_dist)
                take_profit = entry_price * (1 + tp_dist)
                be_price = entry_price * (1 + self.config.total_cost_round_trip)
            else:
                stop_loss = entry_price * (1 + sl_dist)
                take_profit = entry_price * (1 - tp_dist)
                be_price = entry_price * (1 - self.config.total_cost_round_trip)
            
            # Position sizing por volatilidad
            if self.risk.get('use_volatility_sizing', True):
                risk_amount = capital * self.risk.get('risk_per_trade', 0.01)
                per_unit_risk = abs(entry_price - stop_loss)
                if per_unit_risk <= 0:
                    return None
                size = risk_amount / per_unit_risk
            else:
                size = capital * self.risk.get('risk_per_trade', 0.01) / entry_price
            
            leverage = self.config.get_leverage_for_score(score)
            max_notional = capital * leverage
            size = min(size, max_notional / entry_price)
            
            if size <= 0 or size != size:
                return None
            
            trail_mult = self.config.get_trailing_distance(symbol)
            trailing_dist = trail_mult * atr
            trailing_act_atr = self.risk.get('trailing', {}).get('activation_atr', 0.5)
            trailing_activation = entry_price + trailing_act_atr * atr if direction == 'LONG' \
                else entry_price - trailing_act_atr * atr
            be_trigger = self.risk.get('trailing', {}).get('breakeven_trigger_atr', 0.25) * atr
            
            return Position(
                symbol=symbol, direction=direction,
                entry_price=entry_price, entry_time=timestamp,
                stop_loss=stop_loss, take_profit=take_profit,
                position_size=size, leverage=leverage,
                initial_stop=stop_loss, break_even_price=be_price,
                trailing_activation=trailing_activation,
                trailing_distance=trailing_dist,
            )
        except Exception as e:
            logger.error(f"Error creando posición {symbol}: {e}")
            return None
