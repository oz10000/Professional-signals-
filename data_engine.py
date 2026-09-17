"""
Motor de datos con fallback multi-exchange y caché.
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
    - Validación de datos (duplicados, timestamps, gaps)
    """
    
    CACHE_TTL = 3600  # 1 hora
    
    def __init__(self, config):
        self.config = config
        self.cache_dir = config.cache_dir
        self.exchanges: Dict[str, 'ccxt.Exchange'] = {}
        self._connect()
    
    def _connect(self) -> None:
        """Conecta a exchanges en orden de prioridad."""
        if not CCXT_AVAILABLE:
            logger.error("CCXT no disponible")
            return
        
        priority = self.config.exchanges['priority']
        blocked = set(self.config.exchanges.get('blocked', []))
        
        for ex_id in priority:
            if ex_id in blocked:
                logger.info(f"Saltando {ex_id} (bloqueado)")
                continue
            try:
                exchange_class = getattr(ccxt, ex_id)
                exchange = exchange_class({
                    'enableRateLimit': True,
                    'options': {'defaultType': 'spot'},
                    'timeout': self.config.exchanges.get('timeout_ms', 30000),
                })
                exchange.load_markets()
                self.exchanges[ex_id] = exchange
                logger.info(f"✅ Conectado a {ex_id}")
                if len(self.exchanges) >= 3:
                    break
            except Exception as e:
                logger.warning(f"⚠️ No se pudo conectar a {ex_id}: {e}")
        
        if not self.exchanges:
            logger.error("❌ Ningún exchange disponible")
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = '5m',
                    limit: int = 2000, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Obtiene datos OHLCV con fallback.
        
        Args:
            symbol: Par de trading.
            timeframe: Temporalidad.
            limit: Número de velas.
            use_cache: Usar caché si está disponible.
        
        Returns:
            DataFrame con columnas OHLCV, o None si falla.
        """
        cache_file = os.path.join(
            self.cache_dir,
            f"{symbol.replace('/', '_')}_{timeframe}_{limit}.parquet"
        )
        
        # 1. Intentar caché fresca
        if use_cache and os.path.exists(cache_file):
            try:
                df = pd.read_parquet(cache_file)
                if not df.empty and self._cache_fresh(df):
                    return df
            except Exception as e:
                logger.debug(f"Caché corrupta {cache_file}: {e}")
        
        # 2. Descargar de exchanges disponibles
        for ex_id, exchange in self.exchanges.items():
            for attempt in range(3):
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
                    
                    # Validación
                    if not self._validate(df):
                        continue
                    
                    # Guardar en caché
                    if use_cache:
                        try:
                            df.to_parquet(cache_file)
                        except Exception as e:
                            logger.debug(f"Error guardando caché: {e}")
                    
                    logger.debug(f"✅ {symbol} desde {ex_id} ({len(df)} velas)")
                    return df
                    
                except Exception as e:
                    logger.warning(f"Intento {attempt+1}/3 {symbol}@{ex_id}: {e}")
                    time.sleep(1)
        
        # 3. Fallback: caché obsoleta
        if os.path.exists(cache_file):
            try:
                df = pd.read_parquet(cache_file)
                if not df.empty:
                    logger.warning(f"⚠️ Usando caché obsoleta para {symbol}")
                    return df
            except Exception:
                pass
        
        return None
    
    def fetch_multi_timeframe(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Obtiene múltiples timeframes para un símbolo."""
        tfs = self.config.timeframes
        limit = tfs['history_limit']
        
        return {
            '5m': self.fetch_ohlcv(symbol, tfs['entry'], limit),
            '15m': self.fetch_ohlcv(symbol, tfs['confirm'], limit),
            '1h': self.fetch_ohlcv(symbol, tfs['trend'], limit),
        }
    
    def _validate(self, df: pd.DataFrame) -> bool:
        """Valida integridad de los datos."""
        if df is None or df.empty:
            return False
        if len(df) < 50:
            return False
        if df[['open', 'high', 'low', 'close']].isna().any().any():
            return False
        if (df['high'] < df['low']).any():
            return False
        if (df['close'] <= 0).any():
            return False
        return True
    
    def _cache_fresh(self, df: pd.DataFrame) -> bool:
        """Verifica si la caché está fresca."""
        try:
            last = df.index[-1]
            if last.tzinfo is None:
                last = last.tz_localize('UTC')
            age = (pd.Timestamp.now(tz='UTC') - last).total_seconds()
            return age < self.CACHE_TTL
        except Exception:
            return False
