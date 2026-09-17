"""
D.A.P.S-SIGNALS Ω — Scanner Cuantitativo (Streamlit)
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List

from config import ScannerConfig
from data_engine import DataEngine
from scoring import CompositeScorer, ScoreResult
from backtester import Backtester
from optimizer import BayesianWeightOptimizer
from report import (
    to_ranking_dataframe,
    to_text_alert,
    comparative_table_traditional_vs_scanner,
    comparative_table_leverage,
    comparative_table_by_asset,
    validation_report,
)
from indicators import compute_atr


# ============================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================
st.set_page_config(
    page_title="D.A.P.S-SIGNALS Ω Scanner",
    page_icon="Ω",
    layout="wide",
)

st.title("Ω D.A.P.S-SIGNALS — Scanner Cuantitativo")
st.caption("Score Compuesto Normalizado · Multi-Timeframe · Walk-Forward Validado")


# ============================================================
# CARGA DE CONFIGURACIÓN
# ============================================================
@st.cache_resource
def load_config():
    return ScannerConfig.from_yaml("config.yaml")


@st.cache_resource
def load_data_engine():
    return DataEngine(load_config())


config = load_config()
data_engine = load_data_engine()
scorer = CompositeScorer(config)
backtester = Backtester(config)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("⚙️ Configuración")
    st.caption(f"Modo: **{config.raw['project']['mode']}**")
    st.caption(f"Activos: **{len(config.symbols)}**")
    st.caption(f"TF: {config.timeframes['entry']} / {config.timeframes['confirm']} / {config.timeframes['trend']}")
    
    st.markdown("---")
    st.markdown("**Pesos del Scoring**")
    for k, v in config.weights.items():
        st.caption(f"  {k}: {v:.2f}")
    
    st.markdown("---")
    scan_btn = st.button("🔄 ESCANEAR", type="primary", use_container_width=True)
    backtest_btn = st.button("📊 BACKTEST 6M", use_container_width=True)
    optimize_btn = st.button("🧠 OPTIMIZAR PESOS", use_container_width=True)


# ============================================================
# TABS
# ============================================================
tab_scanner, tab_ranking, tab_backtest, tab_comparative, tab_validation = st.tabs([
    "📡 Scanner",
    "🏆 Ranking",
    "📊 Backtest",
    "📈 Comparativas",
    "✅ Validación",
])


# ============================================================
# TAB 1: SCANNER
# ============================================================
with tab_scanner:
    st.subheader("📡 Scanner de Oportunidades")
    
    if scan_btn or 'scan_results' not in st.session_state:
        with st.spinner("Escaneando activos..."):
            results: List[ScoreResult] = []
            progress = st.progress(0)
            
            for i, symbol in enumerate(config.symbols):
                try:
                    data = data_engine.fetch_multi_timeframe(symbol)
                    if data.get('5m') is not None and not data['5m'].empty:
                        score = scorer.compute(symbol, data)
                        results.append(score)
                except Exception as e:
                    st.warning(f"Error {symbol}: {e}")
                
                progress.progress((i + 1) / len(config.symbols))
            
            st.session_state.scan_results = results
            st.session_state.last_scan = datetime.now().strftime("%H:%M:%S")
    
    results = st.session_state.get('scan_results', [])
    
    if not results:
        st.info("Presioná **ESCANEAR** en la sidebar.")
    else:
        st.success(f"✅ {len(results)} activos escaneados · Último: {st.session_state.get('last_scan', 'N/A')}")
        
        valid = [r for r in results if r.is_valid]
        long_valid = [r for r in valid if r.direction == 'LONG']
        short_valid = [r for r in valid if r.direction == 'SHORT']
        
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
            if data.get('5m') is not None:
                entry = data['5m']['close'].iloc[-1]
                atr = compute_atr(data['5m'], 14).iloc[-1]
                sl_mult = config.atr_mult_sl.get(best_long.symbol, config.atr_mult_sl_default)
                sl = entry - sl_mult * atr
                tp = entry + config.atr_mult_tp_default * atr
                trailing = config.trailing_distance_by_symbol.get(
                    best_long.symbol,
                    config.trailing_distance_by_symbol.get('default', 2.0)
                ) * atr / entry
                lev = config.get_leverage_for_score(best_long.total_score)
                
                st.code(to_text_alert(best_long, entry, sl, tp, trailing, lev))
        
        if short_valid:
            best_short = max(short_valid, key=lambda x: x.total_score)
            st.markdown("### 🌟 MEJOR SHORT")
            
            data = data_engine.fetch_multi_timeframe(best_short.symbol)
            if data.get('5m') is not None:
                entry = data['5m']['close'].iloc[-1]
                atr = compute_atr(data['5m'], 14).iloc[-1]
                sl_mult = config.atr_mult_sl.get(best_short.symbol, config.atr_mult_sl_default)
                sl = entry + sl_mult * atr
                tp = entry - config.atr_mult_tp_default * atr
                trailing = config.trailing_distance_by_symbol.get(
                    best_short.symbol,
                    config.trailing_distance_by_symbol.get('default', 2.0)
                ) * atr / entry
                lev = config.get_leverage_for_score(best_short.total_score)
                
                st.code(to_text_alert(best_short, entry, sl, tp, trailing, lev))


# ============================================================
# TAB 2: RANKING
# ============================================================
with tab_ranking:
    st.subheader("🏆 Ranking Completo")
    
    results = st.session_state.get('scan_results', [])
    if not results:
        st.info("Primero escaneá activos en la pestaña Scanner.")
    else:
        df = to_ranking_dataframe(results)
        st.dataframe(df, use_container_width=True, height=600)
        
        # Descargar CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Descargar CSV",
            csv,
            "ranking.csv",
            "text/csv",
        )


# ============================================================
# TAB 3: BACKTEST
# ============================================================
with tab_backtest:
    st.subheader("📊 Backtest y Walk-Forward")
    
    if backtest_btn or 'backtest_result' not in st.session_state:
        with st.spinner("Ejecutando backtest (6 meses)..."):
            try:
                # Cargar datos históricos
                data_dict = {}
                for symbol in config.symbols[:10]:  # Limitar a 10 para velocidad
                    sym_data = data_engine.fetch_multi_timeframe(symbol)
                    if sym_data.get('5m') is not None and not sym_data['5m'].empty:
                        data_dict[symbol] = sym_data
                
                if data_dict:
                    result = backtester.run(data_dict)
                    st.session_state.backtest_result = result
                else:
                    st.error("No se pudieron cargar datos")
            except Exception as e:
                st.error(f"Error en backtest: {e}")
    
    result = st.session_state.get('backtest_result')
    if result:
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
    df_trad = comparative_table_traditional_vs_scanner(metrics)
    st.dataframe(df_trad, use_container_width=True)
    
    st.markdown("### PnL por Nivel de Apalancamiento")
    df_lev = comparative_table_leverage(metrics, config.risk['initial_capital'])
    st.dataframe(df_lev, use_container_width=True)
    
    st.markdown("### Rendimiento por Activo")
    if result and result.trades:
        df_asset = comparative_table_by_asset(result.trades)
        st.dataframe(df_asset, use_container_width=True)
    else:
        st.info("Ejecutá un backtest primero.")


# ============================================================
# TAB 5: VALIDACIÓN
# ============================================================
with tab_validation:
    st.subheader("✅ Informe de Validación")
    st.code(validation_report())


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption(
    f"D.A.P.S-SIGNALS Ω v{config.raw['project']['version']} · "
    f"Modo: {config.raw['project']['mode']} · "
    f"Última actualización: {st.session_state.get('last_scan', 'Nunca')}"
)
