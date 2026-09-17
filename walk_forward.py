"""
Walk-forward validation con purga y embargo (48 barras).
"""

import logging
import pandas as pd
from typing import List, Dict

logger = logging.getLogger(__name__)


def generate_walk_forward_folds(timestamps: List[pd.Timestamp],
                                 train_months: int = 3,
                                 test_months: int = 1,
                                 purge_bars: int = 48) -> List[Dict]:
    """Genera folds walk-forward con purga entre train y test."""
    if not timestamps:
        return []
    
    folds = []
    current = timestamps[0]
    end = timestamps[-1]
    
    while current + pd.Timedelta(days=train_months*30 + test_months*30) <= end:
        train_end = current + pd.Timedelta(days=train_months*30)
        purge_end = train_end + pd.Timedelta(minutes=purge_bars * 5)
        test_start = purge_end
        test_end = test_start + pd.Timedelta(days=test_months*30)
        
        folds.append({
            'train_start': current.isoformat(),
            'train_end': train_end.isoformat(),
            'purge_bars': purge_bars,
            'test_start': test_start.isoformat(),
            'test_end': test_end.isoformat(),
        })
        current = test_end
    
    logger.info(f"Generados {len(folds)} folds walk-forward")
    return folds
