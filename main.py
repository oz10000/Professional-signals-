"""
Orquestador CLI del scanner con manejo de errores.
"""

import logging
import sys
import traceback

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def main():
    """Ejecuta el pipeline completo."""
    try:
        from config import ScannerConfig, ConfigError
        from data_engine import DataEngine
        from scoring import CompositeScorer
        from report import to_ranking_dataframe, to_text_alert
        from indicators import compute_atr
    except ImportError as e:
        logger.error(f"Error importando módulos: {e}")
        sys.exit(1)
    
    try:
        config = ScannerConfig.from_yaml("config.yaml")
    except ConfigError as e:
        logger.error(f"Error de configuración: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error inesperado: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    data_engine = DataEngine(config)
    if not data_engine.is_available:
        logger.error("No hay exchanges disponibles")
        sys.exit(1)
    
    scorer = CompositeScorer(config)
    
    logger.info(f"Escaneando {len(config.symbols)} activos...")
    
    results = []
    for symbol in config.symbols:
        try:
            data = data_engine.fetch_multi_timeframe(symbol)
            if data.get('5m') is not None and not data['5m'].empty:
                score = scorer.compute(symbol, data)
                results.append(score)
                logger.info(f"  {symbol}: Score {score.total_score:.1f} [{score.tier}] {score.direction}")
        except Exception as e:
            logger.warning(f"Error {symbol}: {e}")
    
    if not results:
        logger.error("No se obtuvieron resultados")
        sys.exit(1)
    
    df = to_ranking_dataframe(results)
    print("\n" + "="*100)
    print("  RANKING COMPLETO")
    print("="*100)
    print(df.to_string(index=False))
    
    valid = [r for r in results if r.is_valid]
    if valid:
        best = max(valid, key=lambda x: x.total_score)
        print("\n" + "="*60)
        print(f"  MEJOR OPORTUNIDAD: {best.symbol} ({best.direction})")
        print("="*60)
        
        data = data_engine.fetch_multi_timeframe(best.symbol)
        entry = float(data['5m']['close'].iloc[-1])
        atr_series = compute_atr(data['5m'], 14)
        atr = float(atr_series.iloc[-1]) if not atr_series.empty else entry * 0.005
        
        sl_mult = config.get_atr_mult_sl(best.symbol)
        tp_mult = config.atr_mult_tp_default
        
        if best.direction == 'LONG':
            sl = entry - sl_mult * atr
            tp = entry + tp_mult * atr
        else:
            sl = entry + sl_mult * atr
            tp = entry - tp_mult * atr
        
        trail_mult = config.get_trailing_distance(best.symbol)
        trailing = trail_mult * atr / entry
        lev = config.get_leverage_for_score(best.total_score)
        
        print(to_text_alert(best, entry, sl, tp, trailing, lev))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrumpido por el usuario.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error fatal: {e}")
        traceback.print_exc()
        sys.exit(1)
