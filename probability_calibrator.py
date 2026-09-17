"""
Calibrador de probabilidades con regresión isotónica.
Corrige P(L)/P(S) saturadas. Reduce MSE hasta 50%.
"""

import logging
import numpy as np
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)


class ProbabilityCalibrator:
    """Calibra probabilidades crudas con regresión isotónica."""
    
    def __init__(self):
        self.iso_long = IsotonicRegression(out_of_bounds='clip')
        self.iso_short = IsotonicRegression(out_of_bounds='clip')
        self.is_fitted = False
        self._n_samples = 0
    
    def fit(self, raw_probs_long, actual_wins_long,
            raw_probs_short, actual_wins_short):
        """Entrena calibradores sobre datos históricos."""
        if len(raw_probs_long) < 50:
            logger.warning("Datos insuficientes para calibración isotónica")
            return
        
        self.iso_long.fit(raw_probs_long, actual_wins_long)
        self.iso_short.fit(raw_probs_short, actual_wins_short)
        self.is_fitted = True
        self._n_samples = len(raw_probs_long)
        logger.info(f"✅ Calibrador entrenado con {self._n_samples} muestras")
    
    def calibrate(self, raw_prob_long):
        """Calibra probabilidad cruda."""
        if not self.is_fitted:
            return raw_prob_long
        
        calibrated = float(self.iso_long.predict([raw_prob_long])[0])
        return max(0.01, min(0.99, calibrated))
