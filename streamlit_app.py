"""
D.A.P.S-SIGNALS Ω — Scanner Cuantitativo (Streamlit)
Versión robusta con manejo completo de errores.
"""

import logging
import traceback
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np

# ---- Configurar logging ----
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# ---- Configuración de página ----
st.set_page_config(
    page_title="D.A.P.S-SIGNALS Ω Scanner",
    page_icon="Ω",
    layout="wide",
)


# ============================================================
# CARGA DE MÓDULOS CON MANEJO DE ERRORES
# ============================================================

def _safe_import():
    """Importa módulos con manejo de errores."""
    modules = {}
    errors = {}
    
    try:
        from config import ScannerConfig, ConfigError
        modules['ScannerConfig'] = ScannerConfig
        modules['ConfigError'] = ConfigError
    except Exception as e:
        errors['config'] = str(e)
    
    try:
        from data_engine import DataEngine
        modules['DataEngine'] = DataEngine
    except Exception as e:
        errors['data_engine'] = str(e)
    
    try:
        from scoring import CompositeScorer, ScoreResult
        modules['CompositeScorer'] = CompositeScorer
        modules['ScoreResult'] = ScoreResult
    except Exception as e:
        errors['scoring'] = str(e)
    
    try:
        from backtester import Backtester
        modules['Backtester'] = Backtester
    except Exception as e:
        errors['backtester'] = str(e)
    
    try:
        from optimizer import BayesianWeightOptimizer
        modules['BayesianWeightOptimizer'] = BayesianWeightOptimizer
    except Exception as e:
        errors['optimizer'] = str(e)
    
    try:
        from report import (
            to_ranking_dataframe, to_text_alert,
            comparative_table_traditional_vs_scanner,
            comparative_table_leverage, comparative_table_by_asset,
            validation_report,
        )
        modules['to_ranking_dataframe'] = to_ranking_dataframe
        modules['to_text_alert'] = to_text_alert
        modules['comparative_table_traditional_vs_scanner'] = comparative_table_traditional_vs_scanner
        modules['comparative_table_leverage'] = comparative_table_leverage
        modules['comparative_table_by_asset'] = comparative_table_by_asset
        modules['validation_report'] = validation_report
    except Exception as e:
        errors['report'] = str(e)
    
    try:
        from indicators import compute_atr
        modules['compute_atr'] = compute_atr
    except Exception as e:
        errors['indicators'] = str(e)
    
    return modules, errors


_modules, _errors = _safe_import()

if _errors:
    st.error("❌ Errores al importar módulos:")
    for mod, err in _errors.items():
        st.warning(f"**{mod}**: {err}")
    st.stop()


# ============================================================
# CARGA DE CONFIGURACIÓN
# ============================================================

@st.cache_resource
def load_config():
    """Carga configuración con caché y manejo de errores."""
    try:
        return _modules['ScannerConfig'].from_yaml("config.yaml")
    except Exception as e:
        st.error(f"❌ Error cargando config.yaml: {e}")
        st.code(traceback.format_exc())
        st.stop()


@st.cache_resource
def load_data_engine():
    """Carga data engine con caché."""
    try:
        return _modules['DataEngine'](load_config())
    except Exception as e:
        st.warning(f"⚠️ Error cargando DataEngine: {e}")
        return None


# ---- Cargar ----
try:
    config = load_config()
    data_engine = load_data_engine()
    scorer = _modules['CompositeScorer'](config)
    backtester = _modules['Backtester'](config)
except Exception as e:
    st.error(f"❌ Error inicializando sistema: {e}")
    st.code(traceback.format_exc())
    st.stop()


# ============================================================
# HEADER
# ============================================================
st.title("Ω D.A.P.S-SIGNALS — Scanner Cuantitativo")
st.caption("Score Compuesto Normalizado · Multi-Timeframe · Walk-Forward Validado")


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("⚙️ Configuración")
    
    try:
        st.caption(f"Modo: **{config.project.get('mode', 'manual')}**")
        st.caption(f"Versión: **{config.project.get('version', 'N/A')}**")
        st.caption(f"Activos: **{len(config.symbols)}**")
        
        tfs = config.timeframes
        st.caption(f"TF: {tfs.get('entry', '5m')} / {tfs.get('confirm', '15m')} / {tfs.get('trend', '1h')}")
        
        st.markdown("---")
        st.markdown("**Pesos del Scoring**")
        weights = config.weights
        for k, v in weights.items():
            st.caption(f"  {k}: {v:.3f}")
        st.caption(f"  **Suma: {sum(weights.values()):.3f}**")
        
        st.markdown("---")
        
        # Estado del data engine
        if data_engine is not None:
            status = data_engine.get_status()
            if status['available']:
                st.success(f"✅ {len(status['exchanges'])} exchanges")
                for ex in status['exchanges']:
                    st.caption(f"  • {ex}")
            else:
                st.warning("⚠️ Sin exchanges. Solo caché.")
        else:
            st.error("❌ DataEngine no disponible")
        
        st.markdown("---")
        scan_btn = st.button("🔄 ESCANEAR", type="primary", use_container_width=True)
        backtest_btn = st.button("📊 BACKTEST 6M", use_container_width=True)
    
    except Exception as e:
        st.error(f"Error en sidebar: {e}")


# ============================================================
# TABS
# ============================================================
try:
    tab_scanner, tab_ranking, tab_backtest, tab_comparative, tab_validation = st.tabs([
        "📡 Scanner", "🏆 Ranking", "📊 Backtest", "📈 Comparativas", "✅ Validación"
    ])
except Exception as e:
    st.error(f"Error creando tabs: {e}")
    st.stop()


# ============================================================
# TAB 1: SCANNER
# ============================================================
with tab_scanner:
    st.subheader("📡 Scanner de Oportunidades")
    
    if scan_btn or 'scan_results' not in st.session_state:
        if data_engine is None:
            st.error("❌ DataEngine no disponible. No se puede escanear.")
        else:
            with st.spinner("Escaneando activos..."):
                results = []
                symbols = config.symbols
                progress = st.progress(0.0)
                status_text = st.empty()
                
                for i, symbol in enumerate(symbols):
                    try:
                        status_text.caption(f"Escaneando {symbol} ({i+1}/{len(symbols)})")
                        data = data_engine.fetch_multi_timeframe(symbol)
                        
                        if data.get('5m') is not None and not data['5m'].empty:
                            score = scorer.compute(symbol, data)
                            results.append(score)
                    except Exception as e:
                        logger.warning(f"Error escaneando {symbol}: {e}")
                    
                    progress.progress((i + 1) / len(symbols))
                
                progress.empty()
                status_text.empty()
                st.session_state.scan_results = results
                st.session_state.last_scan = datetime.now().strftime("%H:%M:%S")
                st.session_state.scan_time = datetime.now()
    
    results = st.session_state.get('scan_results', [])
    
    if not results:
        st.info("Presioná **ESCANEAR** en la sidebar.")
    else:
        try:
            valid = [r for r in results if r.is_valid]
            long_valid = [r for r in valid if r.direction == 'LONG']
            short_valid = [r for r in valid if r.direction == 'SHORT']
            
            st.success(
                f"✅ {len(results)} activos escaneados · "
                f"Último: {st.session_state.get('last_scan', 'N/A')}"
            )
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total", len(results))
            c2.metric("Válidas", len(valid))
            c3.metric("LONG", len(long_valid))
            c4.metric("SHORT", len(short_valid))
            
            st.markdown("---")
            
            if long_valid:
                best_long = max(long_valid, key=lambda x: x.total_score)
                st.markdown("### 🌟 MEJOR LONG")
                
                data = data_engine.fetch_multi_timeframe(best_long.symbol)
                if data.get('5m') is not None and not data['5m'].empty:
                    entry = float(data['5m']['close'].iloc[-1])
                    atr_series = _modules['compute_atr'](data['5m'], 14)
                    atr = float(atr_series.iloc[-1]) if not atr_series.empty else entry * 0.005
                    
                    if atr > 0:
                        sl_mult = config.get_atr_mult_sl(best_long.symbol)
                        tp_mult = config.atr_mult_tp_default
                        sl = entry - sl_mult * atr
                        tp = entry + tp_mult * atr
                        trail_mult = config.get_trailing_distance(best_long.symbol)
                        trailing = trail_mult * atr / entry
                        lev = config.get_leverage_for_score(best_long.total_score)
                        
                        alert = _modules['to_text_alert'](
                            best_long, entry, sl, tp, trailing, lev
                        )
                        st.code(alert)
            
            if short_valid:
                best_short = max(short_valid, key=lambda x: x.total_score)
                st.markdown("### 🌟 MEJOR SHORT")
                
                data = data_engine.fetch_multi_timeframe(best_short.symbol)
                if data.get('5m') is not None and not data['5m'].empty:
                    entry = float(data['5m']['close'].iloc[-1])
                    atr_series = _modules['compute_atr'](data['5m'], 14)
                    atr = float(atr_series.iloc[-1]) if not atr_series.empty else entry * 0.005
                    
                    if atr > 0:
                        sl_mult = config.get_atr_mult_sl(best_short.symbol)
                        tp_mult = config.atr_mult_tp_default
                        sl = entry + sl_mult * atr
                        tp = entry - tp_mult * atr
                        trail_mult = config.get_trailing_distance(best_short.symbol)
                        trailing = trail_mult * atr / entry
                        lev = config.get_leverage_for_score(best_short.total_score)
                        
                        alert = _modules['to_text_alert'](
                            best_short, entry, sl, tp, trailing, lev
                        )
                        st.code(alert)
        except Exception as e:
            st.error(f"Error mostrando resultados: {e}")
            st.code(traceback.format_exc())


# ============================================================
# TAB 2: RANKING
# ============================================================
with tab_ranking:
    st.subheader("🏆 Ranking Completo")
    
    results = st.session_state.get('scan_results', [])
    if not results:
        st.info("Primero escaneá activos en la pestaña Scanner.")
    else:
        try:
            df = _modules['to_ranking_dataframe'](results)
            if df.empty:
                st.warning("No se pudo generar el ranking.")
            else:
                st.dataframe(df, use_container_width=True, height=600)
                
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Descargar CSV",
                    csv,
                    f"ranking_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    "text/csv",
                )
        except Exception as e:
            st.error(f"Error generando ranking: {e}")


# ============================================================
# TAB 3: BACKTEST
# ============================================================
with tab_backtest:
    st.subheader("📊 Backtest y Walk-Forward")
    
    if backtest_btn or 'backtest_result' not in st.session_state:
        if data_engine is None:
            st.error("❌ DataEngine no disponible.")
        else:
            with st.spinner("Ejecutando backtest..."):
                try:
                    data_dict = {}
                    symbols = config.symbols[:config.raw.get('backtest', {}).get('max_symbols_backtest', 10)]
                    progress = st.progress(0.0)
                    
                    for i, symbol in enumerate(symbols):
                        sym_data = data_engine.fetch_multi_timeframe(symbol)
                        if sym_data.get('5m') is not None and not sym_data['5m'].empty:
                            data_dict[symbol] = sym_data
                        progress.progress((i + 1) / len(symbols))
                    
                    progress.empty()
                    
                    if data_dict:
                        result = backtester.run(data_dict)
                        st.session_state.backtest_result = result
                    else:
                        st.error("No se pudieron cargar datos para backtest.")
                except Exception as e:
                    st.error(f"Error en backtest: {e}")
                    st.code(traceback.format_exc())
    
    result = st.session_state.get('backtest_result')
    if result and result.metrics and 'error' not in result.metrics:
        metrics = result.metrics
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Trades", metrics.get('total_trades', 0))
        c2.metric("Win Rate", f"{metrics.get('win_rate', 0):.1f}%")
        c3.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")
        c4.metric("Sharpe", f"{metrics.get('sharpe_ratio', 0):.2f}")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Max DD", f"{metrics.get('max_drawdown_pct', 0):.1f}%")
        c2.metric("Expectativa", f"{metrics.get('expectancy', 0):.2f}")
        c3.metric("Trades/Día", f"{metrics.get('trades_per_day', 0):.1f}")
        c4.metric("Leverage Prom.", f"{metrics.get('avg_leverage', 0):.1f}×")
        
        if not result.equity_curve.empty:
            st.markdown("### Curva de Equity")
            st.line_chart(result.equity_curve)
        
        st.markdown("### Métricas Completas")
        st.json(metrics)


# ============================================================
# TAB 4: COMPARATIVAS
# ============================================================
with tab_comparative:
    st.subheader("📈 Tablas Comparativas")
    
    result = st.session_state.get('backtest_result')
    metrics = result.metrics if result else {}
    
    st.markdown("### Modelos Tradicionales vs Scanner")
    try:
        df_trad = _modules['comparative_table_traditional_vs_scanner'](metrics)
        st.dataframe(df_trad, use_container_width=True)
    except Exception as e:
        st.warning(f"Error: {e}")
    
    st.markdown("### PnL por Nivel de Apalancamiento")
    try:
        df_lev = _modules['comparative_table_leverage'](
            metrics, config.risk.get('initial_capital', 10000.0)
        )
        st.dataframe(df_lev, use_container_width=True)
    except Exception as e:
        st.warning(f"Error: {e}")
    
    st.markdown("### Rendimiento por Activo")
    if result and result.trades:
        try:
            df_asset = _modules['comparative_table_by_asset'](result.trades)
            st.dataframe(df_asset, use_container_width=True)
        except Exception as e:
            st.warning(f"Error: {e}")
    else:
        st.info("Ejecutá un backtest primero.")


# ============================================================
# TAB 5: VALIDACIÓN
# ============================================================
with tab_validation:
    st.subheader("✅ Informe de Validación")
    try:
        st.code(_modules['validation_report']())
    except Exception as e:
        st.error(f"Error: {e}")


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption(
    f"D.A.P.S-SIGNALS Ω v{config.project.get('version', 'N/A')} · "
    f"Modo: {config.project.get('mode', 'manual')} · "
    f"Última actualización: {st.session_state.get('last_scan', 'Nunca')}"
)
