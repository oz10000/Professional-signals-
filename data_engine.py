"""
Motor de datos con fallback multi-exchange y caché robusta.
"""

import os
import time
import logging
from typing import Optional, Dict, List
from datetime import datetime
import pandas as pd
import numpy as np

try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    logging.warning("CCXT no disponible. Instalar: pip install ccxt")

logger = logging.getLogger(__name__)


class DataEngine:
    """
    Motor de datos robusto con fallback multi-exchange.
    
    Características:
    - Prioridad de exchanges configurable
    - Fallback automático en caso de error
    - Caché local en Parquet
    - Validación defensiva de datos
    - Timeouts configurables
    """
    
    CACHE_TTL = 3600
    MIN_BARS = 50
    
    def __init__(self, config):
        self.config = config
        self.cache_dir = config.cache_dir
        self.exchanges: Dict[str, Any] = {}
        self._available = False
        self._connect()
    
    def _connect(self) -> None:
        """Conecta a exchanges en orden de prioridad con fallback."""
        if not CCXT_AVAILABLE:
            logger.error("❌ CCXT no disponible. Solo caché funcionará.")
            return
        
        priority = self.config.exchanges.get('priority', [])
        blocked = set(self.config.exchanges.get('blocked', []))
        max_ex = self.config.exchanges.get('max_exchanges', 3)
        timeout_ms = self.config.exchanges.get('timeout_ms', 30000)
        
        for ex_id in priority:
            if ex_id in blocked:
                logger.debug(f"⏭️ Saltando {ex_id} (bloqueado)")
                continue
            if len(self.exchanges) >= max_ex:
                break
            
            try:
                exchange_class = getattr(ccxt, ex_id, None)
                if exchange_class is None:
                    logger.warning(f"⚠️ Exchange {ex_id} no existe en CCXT")
                    continue
                
                exchange = exchange_class({
                    'enableRateLimit': True,
                    'options': {'defaultType': 'spot'},
                    'timeout': timeout_ms,
                })
                exchange.load_markets()
                self.exchanges[ex_id] = exchange
                logger.info(f"✅ Conectado a {ex_id} ({len(exchange.markets)} mercados)")
                
            except Exception as e:
                logger.warning(f"⚠️ No se pudo conectar a {ex_id}: {e}")
                continue
        
        self._available = len(self.exchanges) > 0
        if self._available:
            logger.info(f"✅ {len(self.exchanges)} exchanges disponibles")
        else:
            logger.error("❌ Ningún exchange disponible. Solo caché.")
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = '5m',
                    limit: int = 2000, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Obtiene datos OHLCV con fallback robusto.
        
        Args:
            symbol: Par de trading (ej: 'BTC/USDT').
            timeframe: Temporalidad ('5m', '15m', '1h').
            limit: Número de velas.
            use_cache: Usar caché si está disponible.
        
        Returns:
            DataFrame con columnas OHLCV, o None si falla.
        """
        if not symbol or not isinstance(symbol, str):
            logger.warning(f"Símbolo inválido: {symbol}")
            return None
        
        try:
            cache_file = os.path.join(
                self.cache_dir,
                f"{symbol.replace('/', '_')}_{timeframe}_{limit}.parquet"
            )
        except Exception as e:
            logger.warning(f"Error construyendo path de caché: {e}")
            cache_file = None
        
        # 1. Intentar caché fresca
        if use_cache and cache_file and os.path.exists(cache_file):
            try:
                df = pd.read_parquet(cache_file)
                if self._validate(df) and self._cache_fresh(df):
                    logger.debug(f"✅ {symbol} desde caché ({len(df)} velas)")
                    return df
            except Exception as e:
                logger.debug(f"Caché corrupta {cache_file}: {e}")
        
        # 2. Descargar de exchanges
        for ex_id, exchange in self.exchanges.items():
            df = self._fetch_from_exchange(exchange, ex_id, symbol, timeframe, limit)
            if df is not None:
                # Guardar caché
                if use_cache and cache_file:
                    try:
                        df.to_parquet(cache_file)
                    except Exception as e:
                        logger.debug(f"Error guardando caché: {e}")
                return df
        
        # 3. Fallback: caché obsoleta
        if cache_file and os.path.exists(cache_file):
            try:
                df = pd.read_parquet(cache_file)
                if not df.empty:
                    logger.warning(f"⚠️ Usando caché obsoleta para {symbol}")
                    return df
            except Exception:
                pass
        
        logger.warning(f"❌ No se pudieron obtener datos para {symbol}")
        return None
    
    def _fetch_from_exchange(self, exchange, ex_id: str,
                              symbol: str, timeframe: str,
                              limit: int) -> Optional[pd.DataFrame]:
        """Intenta descargar de un exchange específico con retries."""
        max_retries = self.config.exchanges.get('max_retries', 3)
        
        for attempt in range(max_retries):
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                if not ohlcv:
                    continue
                
                df = pd.DataFrame(
                    ohlcv,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                df = df.set_index('timestamp').sort_index()
                df = df[~df.index.duplicated(keep='last')]
                
                # Limpiar NaN
                df = df.dropna(subset=['open', 'high', 'low', 'close'])
                
                if not self._validate(df):
                    continue
                
                logger.debug(f"✅ {symbol}@{ex_id} ({len(df)} velas)")
                return df
                
            except Exception as e:
                logger.debug(f"Intento {attempt+1}/{max_retries} {symbol}@{ex_id}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1.0 + attempt * 0.5)
        
        return None
    
    def fetch_multi_timeframe(self, symbol: str) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Obtiene múltiples timeframes para un símbolo.
        
        Returns:
            Dict con keys '5m', '15m', '1h'. Valores pueden ser None.
        """
        tfs = self.config.timeframes
        limit = tfs.get('history_limit', 2000)
        
        result = {}
        for tf_key, tf_val in [('5m', tfs.get('entry', '5m')),
                                ('15m', tfs.get('confirm', '15m')),
                                ('1h', tfs.get('trend', '1h'))]:
            try:
                result[tf_key] = self.fetch_ohlcv(symbol, tf_val, limit)
            except Exception as e:
                logger.warning(f"Error fetching {symbol} {tf_val}: {e}")
                result[tf_key] = None
        
        return result
    
    def _validate(self, df: Optional[pd.DataFrame]) -> bool:
        """Validación defensiva de integridad de datos."""
        try:
            if df is None or df.empty:
                return False
            if len(df) < self.MIN_BARS:
                return False
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(c in df.columns for c in required_cols):
                return False
            if df[required_cols].isna().any().any():
                return False
            # High debe ser >= Low
            if (df['high'] < df['low']).any():
                return False
            # Precios positivos
            if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
                return False
            # Volumen no negativo
            if (df['volume'] < 0).any():
                return False
            return True
        except Exception as e:
            logger.debug(f"Error validando datos: {e}")
            return False
    
    def _cache_fresh(self, df: pd.DataFrame) -> bool:
        """Verifica si la caché está fresca."""
        try:
            if df.empty:
                return False
            last = df.index[-1]
            if last.tzinfo is None:
                last = last.tz_localize('UTC')
            age = (pd.Timestamp.now(tz='UTC') - last).total_seconds()
            return age < self.CACHE_TTL
        except Exception:
            return False
    
    @property
    def is_available(self) -> bool:
        """Retorna True si al menos un exchange está conectado."""
        return self._available
    
    def get_status(self) -> Dict:
        """Retorna estado del data engine."""
        return {
            'available': self._available,
            'exchanges': list(self.exchanges.keys()),
            'cache_dir': self.cache_dir,
        }
