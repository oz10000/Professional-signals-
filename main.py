"""
Orquestador CLI del scanner.
"""

import logging
import sys
from config import ScannerConfig
from data_engine import DataEngine
from scoring import CompositeScorer
from backtester import Backtester
from report import to_ranking_dataframe, to_text_alert
from indicators import compute_atr

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def main():
    """Ejecuta el pipeline completo."""
    config = ScannerConfig.from_yaml("config.yaml")
    data_engine = DataEngine(config)
    scorer = CompositeScorer(config)
    
    logger.info(f"Escaneando {len(config.symbols)} activos...")
    
    results = []
    for symbol in config.symbols:
        try:
            data = data_engine.fetch_multi_timeframe(symbol)
            if data.get('5m') is not None and not data['5m'].empty:
                score = scorer.compute(symbol, data)
                results.append(score)
                logger.info(f"  {symbol}: Score {score.total_score:.1f} [{score.tier}]")
        except Exception as e:
            logger.warning(f"Error {symbol}: {e}")
    
    # Mostrar ranking
    df = to_ranking_dataframe(results)
    print("\n" + "="*100)
    print("  RANKING COMPLETO")
    print("="*100)
    print(df.to_string(index=False))
    
    # Mostrar mejores oportunidades
    valid = [r for r in results if r.is_valid]
    if valid:
        best = max(valid, key=lambda x: x.total_score)
        print("\n" + "="*60)
        print(f"  MEJOR OPORTUNIDAD: {best.symbol} ({best.direction})")
        print("="*60)
        
        data = data_engine.fetch_multi_timeframe(best.symbol)
        entry = data['5m']['close'].iloc[-1]
        atr = compute_atr(data['5m'], 14).iloc[-1]
        sl_mult = config.atr_mult_sl.get(best.symbol, config.atr_mult_sl_default)
        sl = entry - sl_mult * atr if best.direction == 'LONG' else entry + sl_mult * atr
        tp = entry + config.atr_mult_tp_default * atr if best.direction == 'LONG' else entry - config.atr_mult_tp_default * atr
        lev = config.get_leverage_for_score(best.total_score)
        
        print(to_text_alert(best, entry, sl, tp, 0.017, lev))


if __name__ == "__main__":
    main()
