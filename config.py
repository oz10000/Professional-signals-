"""
Configuración central del scanner con auto-normalización robusta.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import yaml

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Error de configuración."""
    pass


def _load_yaml(path: str) -> dict:
    """Carga archivo YAML con manejo robusto de errores."""
    if not os.path.exists(path):
        raise ConfigError(f"Configuración no encontrada: {path}")
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"YAML inválido en {path}: {e}")
    except Exception as e:
        raise ConfigError(f"Error leyendo {path}: {e}")
    
    if not isinstance(data, dict):
        raise ConfigError(f"YAML debe ser un diccionario")
    
    return data


def _normalize_weights(weights: Dict[str, float], name: str = "weights") -> Dict[str, float]:
    """Normaliza un diccionario de pesos a suma 1.0."""
    if not weights:
        logger.warning(f"⚠️ {name} vacío, usando pesos iguales")
        return {}
    
    total = sum(weights.values())
    
    if total <= 0:
        logger.warning(f"⚠️ {name} suma <= 0, usando pesos iguales")
        n = len(weights)
        return {k: 1.0 / n for k in weights}
    
    if abs(total - 1.0) > 1e-6:
        logger.warning(f"⚠️ {name} suma {total:.4f}, no 1.0. Auto-normalizando...")
        return {k: v / total for k, v in weights.items()}
    
    return weights


@dataclass
class ScannerConfig:
    """Configuración del scanner con acceso por propiedades."""
    
    raw: Dict[str, Any] = field(default_factory=dict)
    _validated: bool = False
    
    @classmethod
    def from_yaml(cls, path: str = "config.yaml") -> "ScannerConfig":
        """Carga configuración desde YAML."""
        raw = _load_yaml(path)
        instance = cls(raw=raw)
        instance._validate()
        instance._ensure_dirs()
        instance._validated = True
        logger.info(f"✅ Configuración cargada desde {path}")
        return instance
    
    def _validate(self) -> None:
        """Validación defensiva. Solo crashea en errores graves."""
        required_keys = ['project', 'symbols', 'exchanges', 'scoring', 'risk']
        for key in required_keys:
            if key not in self.raw:
                raise ConfigError(f"Falta clave requerida: '{key}'")
        
        exchanges = self.raw.get('exchanges', {})
        if not exchanges.get('priority'):
            raise ConfigError("Debe haber al menos un exchange en 'priority'")
        
        symbols = self.raw.get('symbols', {})
        total_symbols = sum(len(v) for v in symbols.values() if isinstance(v, list))
        if total_symbols == 0:
            raise ConfigError("No hay símbolos definidos en 'symbols'")
        logger.info(f"✅ {total_symbols} símbolos configurados")
        
        try:
            weights = self.raw['scoring']['weights']
            self.raw['scoring']['weights'] = _normalize_weights(weights, "pesos del scoring")
        except KeyError:
            raise ConfigError("Falta 'scoring.weights' en configuración")
        
        try:
            vol_weights = self.raw['scoring']['sub_weights']['volume']
            self.raw['scoring']['sub_weights']['volume'] = _normalize_weights(
                vol_weights, "sub-pesos de volumen"
            )
        except KeyError:
            logger.warning("⚠️ No hay sub-pesos de volumen, usando defaults")
            self.raw['scoring'].setdefault('sub_weights', {})
            self.raw['scoring']['sub_weights']['volume'] = {
                'tfi': 0.35, 'ofi': 0.25, 'rvol': 0.25, 'cvd': 0.15
            }
    
    def _ensure_dirs(self) -> None:
        """Crea directorios necesarios."""
        dirs = self.raw.get('directories', {})
        for key, path in dirs.items():
            try:
                os.makedirs(path, exist_ok=True)
            except Exception as e:
                logger.warning(f"No se pudo crear directorio {path}: {e}")
    
    @property
    def project(self) -> Dict:
        return self.raw.get('project', {})
    
    @property
    def symbols(self) -> List[str]:
        syms = []
        for category in self.raw.get('symbols', {}).values():
            if isinstance(category, list):
                syms.extend(category)
        seen = set()
        return [s for s in syms if not (s in seen or seen.add(s))]
    
    @property
    def weights(self) -> Dict[str, float]:
        weights = self.raw.get('scoring', {}).get('weights', {})
        return _normalize_weights(weights, "pesos del scoring")
    
    @property
    def volume_sub_weights(self) -> Dict[str, float]:
        weights = self.raw.get('scoring', {}).get('sub_weights', {}).get('volume', {})
        return _normalize_weights(weights, "sub-pesos de volumen")
    
    @property
    def thresholds(self) -> Dict[str, float]:
        return self.raw.get('thresholds', {})
    
    @property
    def regime_thresholds(self) -> Dict:
        return self.raw.get('thresholds', {}).get('regime_thresholds', {})
    
    @property
    def risk(self) -> Dict[str, Any]:
        return self.raw.get('risk', {})
    
    @property
    def costs(self) -> Dict[str, float]:
        return self.raw.get('costs', {})
    
    @property
    def timeframes(self) -> Dict[str, Any]:
        return self.raw.get('timeframes', {})
    
    @property
    def exchanges(self) -> Dict[str, Any]:
        return self.raw.get('exchanges', {})
    
    @property
    def cache_dir(self) -> str:
        return self.raw.get('directories', {}).get('cache', './cache')
    
    @property
    def reports_dir(self) -> str:
        return self.raw.get('directories', {}).get('reports', './reports')
    
    @property
    def tiers(self) -> Dict[str, float]:
        return self.raw.get('tiers', {})
    
    @property
    def atr_mult_sl(self) -> Dict[str, float]:
        return self.raw.get('risk', {}).get('stop_loss', {}).get('atr_mult_by_symbol', {})
    
    @property
    def atr_mult_sl_default(self) -> float:
        return self.raw.get('risk', {}).get('stop_loss', {}).get('atr_mult_default', 1.8)
    
    @property
    def atr_mult_tp_default(self) -> float:
        return self.raw.get('risk', {}).get('take_profit', {}).get('atr_mult_default', 2.5)
    
    @property
    def trailing_distance_by_symbol(self) -> Dict[str, float]:
        return self.raw.get('risk', {}).get('trailing', {}).get('atr_mult_by_symbol', {'default': 2.0})
    
    @property
    def total_cost_round_trip(self) -> float:
        c = self.costs
        fee = c.get('fee_per_side', 0.0005) * 2
        slip = c.get('slippage', 0.0003)
        spread = c.get('spread', 0.0002)
        funding = c.get('funding_rate_8h', 0.0001)
        return fee + slip + spread + funding
    
    def get_leverage_for_score(self, score: float) -> int:
        if not isinstance(score, (int, float)) or score != score:
            return 1
        leverage_map = self.raw.get('risk', {}).get('leverage_by_score', {})
        max_lev = self.raw.get('risk', {}).get('max_leverage', 10)
        for range_str, lev in leverage_map.items():
            try:
                low, high = range_str.split('-')
                if float(low) <= score <= float(high):
                    return min(int(lev), max_lev)
            except (ValueError, AttributeError):
                continue
        return 1
    
    def get_atr_mult_sl(self, symbol: str) -> float:
        return self.atr_mult_sl.get(symbol, self.atr_mult_sl_default)
    
    def get_trailing_distance(self, symbol: str) -> float:
        td = self.trailing_distance_by_symbol
        return td.get(symbol, td.get('default', 2.0))
