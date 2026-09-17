"""
Optimización Bayesiana (TPE) de pesos con fallback a random search.
"""

import logging
import numpy as np
from typing import Dict, Callable, List
from dataclasses import dataclass

try:
    from skopt import gp_minimize
    from skopt.space import Real
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    logging.warning("skopt no disponible, usando random search")

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    best_weights: Dict[str, float]
    best_score: float
    convergence: List[float]
    n_iterations: int
    method: str = "unknown"


class BayesianWeightOptimizer:
    """Optimizador Bayesiano con fallback a random search."""
    
    PARAM_NAMES = [
        'trend', 'momentum', 'volume_intelligence',
        'volatility', 'liquidity', 'regime'
    ]
    
    def optimize(self, objective_fn: Callable[[Dict], float],
                 n_calls: int = 50,
                 n_initial_points: int = 10) -> OptimizationResult:
        if SKOPT_AVAILABLE:
            try:
                return self._optimize_bayesian(objective_fn, n_calls, n_initial_points)
            except Exception as e:
                logger.warning(f"Bayesian optimization falló: {e}. Usando random search.")
        return self._fallback_random(objective_fn, n_calls)
    
    def _optimize_bayesian(self, objective_fn, n_calls, n_initial_points):
        dimensions = [Real(0.0, 1.0, name=n) for n in self.PARAM_NAMES]
        
        @use_named_args(dimensions)
        def objective(**params):
            total = sum(params.values())
            if total <= 0:
                return 1e6
            normalized = {k: v / total for k, v in params.items()}
            try:
                score = objective_fn(normalized)
                if score != score:
                    return 1e6
                return -score
            except Exception:
                return 1e6
        
        result = gp_minimize(
            objective, dimensions,
            n_calls=n_calls,
            n_initial_points=n_initial_points,
            random_state=42,
        )
        
        best_params = dict(zip(self.PARAM_NAMES, result.x))
        total = sum(best_params.values())
        best_weights = {k: v / total for k, v in best_params.items()} if total > 0 else {}
        
        return OptimizationResult(
            best_weights=best_weights,
            best_score=-result.fun,
            convergence=[-v for v in result.func_vals],
            n_iterations=len(result.func_vals),
            method="bayesian",
        )
    
    def _fallback_random(self, objective_fn, n_calls):
        best_weights = None
        best_score = -float('inf')
        convergence = []
        for _ in range(n_calls):
            raw = np.random.dirichlet(np.ones(len(self.PARAM_NAMES)))
            weights = dict(zip(self.PARAM_NAMES, raw))
            try:
                score = objective_fn(weights)
                if score != score:
                    score = -1e6
            except Exception:
                score = -1e6
            convergence.append(score)
            if score > best_score:
                best_score = score
                best_weights = weights
        
        if best_weights is None:
            n = len(self.PARAM_NAMES)
            best_weights = {k: 1.0 / n for k in self.PARAM_NAMES}
        
        return OptimizationResult(
            best_weights=best_weights,
            best_score=best_score,
            convergence=convergence,
            n_iterations=n_calls,
            method="random",
        )
