"""
Cargador de configuración YAML con validación.
"""

import os
import yaml
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_yaml(path: str) -> dict:
    """Carga archivo YAML con manejo de errores."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuración no encontrada: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


@dataclass
class ScannerConfig:
    """Configuración del scanner."""
    
    raw: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, path: str = "config.yaml") -> "ScannerConfig":
        """Carga configuración desde YAML."""
        raw = _load_yaml(path)
        instance = cls(raw=raw)
        instance._validate()
        instance._ensure_dirs()
        return instance
    
    def _validate(self) -> None:
        """Valida coherencia de la configuración."""
        # Pesos deben sumar 1.0
        weights = self.raw['scoring']['weights']
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Pesos del scoring deben sumar 1.0, suma actual: {total}")
        
        # Sub-pesos de volumen deben sumar 1.0
        vol_weights = self.raw['scoring']['sub_weights']['volume']
        total_vol = sum(vol_weights.values())
        if abs(total_vol - 1.0) > 1e-6:
            raise ValueError(f"Sub-pesos de volumen deben sumar 1.0, suma actual: {total_vol}")
        
        # Al menos un exchange
        if not self.raw['exchanges']['priority']:
            raise ValueError("Debe haber al menos un exchange en priority")
    
    def _ensure_dirs(self) -> None:
        """Crea directorios necesarios."""
        for key, path in self.raw['directories'].items():
            os.makedirs(path, exist_ok=True)
    
    # --------------------------------------------------------
    # PROPIEDADES DE ACCESO RÁPIDO
    # --------------------------------------------------------
    @property
    def symbols(self) -> List[str]:
        """Lista plana de todos los símbolos."""
        syms = []
        for category in self.raw['symbols'].values():
            syms.extend(category)
        return syms
    
    @property
    def weights(self) -> Dict[str, float]:
        return self.raw['scoring']['weights']
    
    @property
    def thresholds(self) -> Dict[str, float]:
        return self.raw['thresholds']
    
    @property
    def risk(self) -> Dict[str, Any]:
        return self.raw['risk']
    
    @property
    def costs(self) -> Dict[str, float]:
        return self.raw['costs']
    
    @property
    def timeframes(self) -> Dict[str, Any]:
        return self.raw['timeframes']
    
    @property
    def exchanges(self) -> Dict[str, Any]:
        return self.raw['exchanges']
    
    @property
    def cache_dir(self) -> str:
        return self.raw['directories']['cache']
    
    @property
    def reports_dir(self) -> str:
        return self.raw['directories']['reports']
    
    @property
    def tiers(self) -> Dict[str, float]:
        return self.raw['tiers']
    
    @property
    def atr_mult_sl(self) -> Dict[str, float]:
        return self.raw['risk']['stop_loss']['atr_mult_by_symbol']
    
    @property
    def atr_mult_sl_default(self) -> float:
        return self.raw['risk']['stop_loss']['atr_mult_default']
    
    @property
    def atr_mult_tp_default(self) -> float:
        return self.raw['risk']['take_profit']['atr_mult_default']
    
    @property
    def trailing_distance_by_symbol(self) -> Dict[str, float]:
        return self.raw['risk']['trailing']['distance_atr_by_symbol']
    
    @property
    def total_cost_round_trip(self) -> float:
        c = self.costs
        return c['fee_per_side'] * 2 + c['slippage'] + c['spread'] + c['funding_rate_8h']
    
    def get_leverage_for_score(self, score: float) -> int:
        """Retorna leverage máximo según score."""
        for range_str, lev in self.raw['risk']['leverage_by_score'].items():
            low, high = range_str.split('-')
            if float(low) <= score <= float(high):
                return min(lev, self.raw['risk']['max_leverage'])
        return 1
