"""
Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).
Filtra estrategias sobreajustadas.
"""

import logging
import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)


def deflated_sharpe_ratio(observed_sr: float,
                          num_trials: int,
                          num_obs: int,
                          skewness: float = 0.0,
                          kurtosis: float = 3.0) -> tuple:
    """
    Calcula el Deflated Sharpe Ratio.
    
    DSR > 0 y p < 0.05: Estrategia con alpha real.
    DSR < 0: Sharpe esperado por azar supera al observado.
    """
    if num_trials <= 1 or num_obs < 2:
        return observed_sr, 0.5
    
    euler_mascheroni = 0.5772156649
    e_max = ((1 - euler_mascheroni) * norm.ppf(1 - 1/num_trials) +
             euler_mascheroni * norm.ppf(1 - 1/(num_trials * np.e)))
    
    sr_std = np.sqrt((1 + 0.5 * observed_sr**2 - skewness * observed_sr +
                      (kurtosis - 3) / 4 * observed_sr**2) / (num_obs - 1))
    
    dsr = (observed_sr - e_max * sr_std) / sr_std if sr_std > 0 else 0
    p_value = 1 - norm.cdf(dsr)
    
    return float(dsr), float(p_value)
