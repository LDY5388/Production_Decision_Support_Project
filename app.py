"""
app.py - 제조업 수요 예측 및 안전재고 시뮬레이션 대시보드
=======================================================
실행: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Demand Forecast & Safety Stock",
    page_icon="📦",
    layout="wide",
)

COLORS = {
    'primary': '#2563eb',
    'red': '#dc2626',
    'green': '#16a34a',
    'orange': '#d97706',
    'purple': '#7c3aed',
    'gray': '#6b7280',
}

WH_COLORS = {
    'Whse_A': COLORS['primary'],
    'Whse_C': COLORS['red'],
    'Whse_J': COLORS['green'],
    'Whse_S': COLORS['orange'],
}


# ─────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────
@st.cache_data
def load_data():
    monthly = pd.read_csv('data/monthly_total.csv', parse_dates=['YearMonth'])
    monthly_wh = pd.read_csv('data/monthly_by_warehouse.csv', parse_dates=['YearMonth'])
    monthly_cat = pd.read_csv('data/monthly_by_category.csv', parse_dates=['YearMonth'])

    forecast = pd.read_csv('data/forecast_results.csv', index_col=0, parse_dates=True)
    wh_forecast = pd.read_csv('data/warehouse_forecast_results.csv', parse_dates=['YearMonth'])
    metrics = pd.read_csv('data/model_metrics.csv')
    ss_levels = pd.read_csv('data/safety_stock_by_service_level.csv')
    wh_ss = pd.read_csv('data/warehouse_safety_stock.csv')
    cost = pd.read_csv('data/cost_tradeoff.csv')

    return monthly, monthly_wh, monthly_cat, forecast, wh_forecast, metrics, ss_levels, wh_ss, cost


monthly, monthly_wh, monthly_cat, forecast_df, wh_forecast_df, metrics_df, ss_df, wh_ss_df, cost_df = load_data()

# 분석 범위 필터링
monthly_filtered = monthly[
    (monthly['YearMonth'] >= '2012-01-01') & (monthly['YearMonth'] <= '2016-12-01')
]


# ─────────────────────────────────────────
# 사이드바
# ─────────────────────────────────────────
with st.sidebar:
    st.title("📦 설정")
    st.divider()

    selected_warehouse = st.selectbox(
        "창고 선택", ['전체'] + sorted(wh_forecast_df['Warehouse'].unique().tolist())
    )

    service_level = st.slider(
        "서비스 수준 (%)", min_value=50, max_value=99, value=95, step=1
    )

    cost_ratio = st.slider(
        "품절/과잉 비용 비율", min_value=1, max_value=20, value=5, step=1,
        help="품절 단위비용 / 과잉 생산 단위비용"
    )

    st.divider()
    st.caption("데이터: Kaggle - Product Demand Forecasting")
    st.caption("분석 기간: 2012-01 ~ 2016-12")


# ─────────────────────────────────────────
# 메인
# ─────────────────────────────────────────
st.title("제조업 수요 예측 및 안전재고 시뮬레이션")
st.caption("수요 예측 → 안전재고 산출 → 생산 의사결정 지원")

# KPI 카드
total_demand = monthly_filtered['total_demand'].sum()
avg_monthly = monthly_filtered['total_demand'].mean()
best_mape = metrics_df['MAPE (%)'].min()
best_model = metrics_df.loc[metrics_df['MAPE (%)'].idxmin(), 'Model']

c1, c2, c3, c4 = st.columns(4)
c1.metric("분석 기간 총 수요", f"{total_demand/1e9:.1f}B")
c2.metric("월평균 수요", f"{avg_monthly/1e6:.1f}M")
c3.metric("최적 모델", best_model)
c4.metric("최적 MAPE", f"{best_mape:.1f}%")

st.divider()

# ─────────────────────────────────────────
# 탭
# ─────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "수요 추이",
    "예측 모델 비교",
    "안전재고 시뮬레이션",
    "창고별 분석",
])


# ═══════════════════════════════════════
# 탭 1: 수요 추이
# ═══════════════════════════════════════
with tab1:
    st.subheader("월별 수요 추이")

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=monthly_filtered['YearMonth'],
        y=monthly_filtered['total_demand'],
        mode='lines+markers',
        line=dict(color=COLORS['primary'], width=2),
        marker=dict(size=4),
        name='Total Demand',
    ))
    fig_trend.update_layout(
        height=400,
        xaxis_title='Date', yaxis_title='Demand',
        hovermode='x unified',
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # 창고별 비교
    col_a, col_b = st.columns([2, 1])

    with col_a:
        st.subheader("창고별 수요 추이")
        wh_filtered = monthly_wh[
            (monthly_wh['YearMonth'] >= '2012-01-01') & (monthly_wh['YearMonth'] <= '2016-12-01')
        ]
        fig_wh = go.Figure()
        for wh in sorted(wh_filtered['Warehouse'].unique()):
            wh_data = wh_filtered[wh_filtered['Warehouse'] == wh]
            fig_wh.add_trace(go.Scatter(
                x=wh_data['YearMonth'], y=wh_data['total_demand'],
                name=wh, line=dict(color=WH_COLORS.get(wh, COLORS['gray']), width=1.5),
            ))
        fig_wh.update_layout(height=350, hovermode='x unified')
        st.plotly_chart(fig_wh, use_container_width=True)

    with col_b:
        st.subheader("창고별 비중")
        wh_total = wh_filtered.groupby('Warehouse')['total_demand'].sum().reset_index()
        fig_pie = px.pie(
            wh_total, values='total_demand', names='Warehouse',
            color='Warehouse',
            color_discrete_map=WH_COLORS,
        )
        fig_pie.update_layout(height=350)
        st.plotly_chart(fig_pie, use_container_width=True)

    # 계절성 패턴
    st.subheader("월별 계절성 패턴")
    monthly_filtered_cp = monthly_filtered.copy()
    monthly_filtered_cp['Month'] = monthly_filtered_cp['YearMonth'].dt.month
    monthly_filtered_cp['Year'] = monthly_filtered_cp['YearMonth'].dt.year

    fig_season = go.Figure()
    for year in sorted(monthly_filtered_cp['Year'].unique()):
        yr_data = monthly_filtered_cp[monthly_filtered_cp['Year'] == year]
        fig_season.add_trace(go.Scatter(
            x=yr_data['Month'], y=yr_data['total_demand'],
            name=str(year), mode='lines+markers',
            marker=dict(size=4), line=dict(width=1.5),
        ))
    fig_season.update_layout(
        height=350,
        xaxis=dict(
            tickmode='array', tickvals=list(range(1, 13)),
            ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ),
        hovermode='x unified',
    )
    st.plotly_chart(fig_season, use_container_width=True)


# ═══════════════════════════════════════
# 탭 2: 예측 모델 비교
# ═══════════════════════════════════════
with tab2:
    st.subheader("모델 성능 비교")

    col1, col2, col3 = st.columns(3)
    for i, (_, row) in enumerate(metrics_df.iterrows()):
        col = [col1, col2, col3][i]
        with col:
            st.metric(row['Model'], f"MAPE {row['MAPE (%)']:.1f}%")
            st.caption(f"RMSE: {row['RMSE']:,.0f}")
            st.caption(f"95% Coverage: {row['95% CI Coverage (%)']:.0f}%")

    st.divider()

    # 예측 결과 그래프
    st.subheader("예측 결과 비교")

    train_data = monthly_filtered[monthly_filtered['YearMonth'] < forecast_df.index[0]]

    fig_fc = go.Figure()

    # 학습 데이터
    fig_fc.add_trace(go.Scatter(
        x=train_data['YearMonth'], y=train_data['total_demand'],
        mode='lines', line=dict(color=COLORS['gray'], width=1),
        name='Train', opacity=0.6,
    ))

    # 실제값
    fig_fc.add_trace(go.Scatter(
        x=forecast_df.index, y=forecast_df['actual'],
        mode='lines+markers', line=dict(color='black', width=2.5),
        marker=dict(size=6), name='Actual',
    ))

    # 모델별 예측
    model_configs = [
        ('SARIMA', COLORS['primary']),
        ('Holt-Winters', COLORS['red']),
        ('Prophet', COLORS['green']),
    ]
    for model_name, color in model_configs:
        pred_col = f'{model_name}_pred'
        lower_col = f'{model_name}_lower'
        upper_col = f'{model_name}_upper'

        if pred_col in forecast_df.columns:
            fig_fc.add_trace(go.Scatter(
                x=forecast_df.index, y=forecast_df[pred_col],
                mode='lines+markers', line=dict(color=color, width=1.5, dash='dash'),
                marker=dict(size=5), name=f'{model_name}',
            ))
            fig_fc.add_trace(go.Scatter(
                x=list(forecast_df.index) + list(forecast_df.index[::-1]),
                y=list(forecast_df[upper_col]) + list(forecast_df[lower_col][::-1]),
                fill='toself', fillcolor=color, opacity=0.1,
                line=dict(width=0), name=f'{model_name} 95% PI',
                showlegend=False,
            ))

    fig_fc.add_vline(x=forecast_df.index[0], line_dash='dot', line_color=COLORS['gray'])
    fig_fc.update_layout(height=450, hovermode='x unified')
    st.plotly_chart(fig_fc, use_container_width=True)

    # 예측 오차 비교
    st.subheader("예측 오차 (Actual - Predicted)")
    fig_err = go.Figure()
    months = list(range(1, len(forecast_df) + 1))
    for model_name, color in model_configs:
        pred_col = f'{model_name}_pred'
        if pred_col in forecast_df.columns:
            errors = forecast_df['actual'] - forecast_df[pred_col]
            fig_err.add_trace(go.Bar(
                x=[f'Month {m}' for m in months], y=errors,
                name=model_name, marker_color=color, opacity=0.7,
            ))
    fig_err.add_hline(y=0, line_color='black', line_width=1)
    fig_err.update_layout(height=350, barmode='group')
    st.plotly_chart(fig_err, use_container_width=True)


# ═══════════════════════════════════════
# 탭 3: 안전재고 시뮬레이션
# ═══════════════════════════════════════
with tab3:
    st.subheader("몬테카를로 시뮬레이션")

    # 실시간 시뮬레이션 (사이드바 서비스 수준 반영)
    hw_train = monthly_filtered.set_index('YearMonth')['total_demand'][:-6]
    hw_model = ExponentialSmoothing(
        hw_train, trend='add', seasonal='add', seasonal_periods=12
    ).fit(optimized=True)
    resid_std = np.std(hw_model.resid.dropna())
    base_pred = hw_model.forecast(6).iloc[-1]

    np.random.seed(42)
    N_SIM = 10000
    sim_demand = np.random.normal(base_pred, resid_std, N_SIM)

    z_score = stats.norm.ppf(service_level / 100)
    safety_stock = z_score * resid_std
    production_target = base_pred + safety_stock
    actual_coverage = np.mean(sim_demand <= production_target) * 100

    # KPI
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("예측값 (기본 생산량)", f"{base_pred/1e6:.1f}M")
    col_s2.metric("안전재고", f"{safety_stock/1e6:.1f}M")
    col_s3.metric(f"생산 목표 (SL {service_level}%)", f"{production_target/1e6:.1f}M")
    col_s4.metric("시뮬레이션 커버리지", f"{actual_coverage:.1f}%")

    # 시뮬레이션 분포
    fig_sim = go.Figure()
    fig_sim.add_trace(go.Histogram(
        x=sim_demand, nbinsx=80, name='Simulated Demand',
        marker_color=COLORS['primary'], opacity=0.5,
    ))
    fig_sim.add_vline(x=base_pred, line_color='black', line_width=2,
                      annotation_text=f"Forecast: {base_pred/1e6:.1f}M")
    fig_sim.add_vline(x=production_target, line_color=COLORS['red'], line_width=2,
                      line_dash='dash',
                      annotation_text=f"Target (SL {service_level}%): {production_target/1e6:.1f}M")
    fig_sim.update_layout(height=400, xaxis_title='Demand', yaxis_title='Frequency')
    st.plotly_chart(fig_sim, use_container_width=True)

    st.divider()

    # 비용 트레이드오프 (실시간 비용 비율 반영)
    st.subheader("비용 트레이드오프 분석")
    st.caption(f"현재 설정: 품절 비용 = 과잉 생산 비용의 {cost_ratio}배")

    prod_range = np.linspace(base_pred - 2 * resid_std, base_pred + 3 * resid_std, 200)
    total_costs, over_costs, stock_costs = [], [], []

    for prod in prod_range:
        overprod = np.maximum(prod - sim_demand, 0)
        stockout = np.maximum(sim_demand - prod, 0)
        oc = 1 * np.mean(overprod)
        sc = cost_ratio * np.mean(stockout)
        over_costs.append(oc)
        stock_costs.append(sc)
        total_costs.append(oc + sc)

    optimal_idx = np.argmin(total_costs)
    optimal_prod = prod_range[optimal_idx]
    optimal_sl = np.mean(sim_demand <= optimal_prod) * 100

    fig_cost = go.Figure()
    fig_cost.add_trace(go.Scatter(
        x=prod_range, y=over_costs, name='Overproduction Cost',
        line=dict(color=COLORS['orange'], width=1.5),
    ))
    fig_cost.add_trace(go.Scatter(
        x=prod_range, y=stock_costs, name='Stockout Cost',
        line=dict(color=COLORS['red'], width=1.5),
    ))
    fig_cost.add_trace(go.Scatter(
        x=prod_range, y=total_costs, name='Total Cost',
        line=dict(color=COLORS['primary'], width=2.5),
    ))
    fig_cost.add_vline(x=optimal_prod, line_color=COLORS['green'], line_dash='dash')
    fig_cost.add_annotation(
        x=optimal_prod, y=total_costs[optimal_idx],
        text=f"Optimal: {optimal_prod/1e6:.1f}M (SL: {optimal_sl:.0f}%)",
        showarrow=True, arrowhead=2, arrowcolor=COLORS['green'],
    )
    fig_cost.update_layout(height=400, xaxis_title='Production Target', yaxis_title='Expected Cost')
    st.plotly_chart(fig_cost, use_container_width=True)

    col_o1, col_o2 = st.columns(2)
    col_o1.metric("비용 최적 생산량", f"{optimal_prod/1e6:.1f}M")
    col_o2.metric("비용 최적 서비스 수준", f"{optimal_sl:.1f}%")

    # 서비스 수준 곡선
    st.subheader("서비스 수준 vs 안전재고")
    sl_range = np.linspace(0.50, 0.999, 200)
    ss_range = stats.norm.ppf(sl_range) * resid_std

    fig_sl = go.Figure()
    fig_sl.add_trace(go.Scatter(
        x=sl_range * 100, y=ss_range,
        mode='lines', line=dict(color=COLORS['primary'], width=2),
        name='Safety Stock Curve',
    ))
    fig_sl.add_trace(go.Scatter(
        x=[service_level], y=[safety_stock],
        mode='markers', marker=dict(size=12, color=COLORS['red']),
        name=f'Current: SL {service_level}%',
    ))
    fig_sl.update_layout(
        height=350,
        xaxis_title='Service Level (%)', yaxis_title='Safety Stock',
        xaxis=dict(range=[50, 100]),
    )
    st.plotly_chart(fig_sl, use_container_width=True)


# ═══════════════════════════════════════
# 탭 4: 창고별 분석
# ═══════════════════════════════════════
with tab4:
    st.subheader("창고별 예측 및 안전재고")

    if selected_warehouse == '전체':
        # 전체 요약
        col_w1, col_w2 = st.columns(2)

        with col_w1:
            st.subheader("창고별 변동계수 (CV)")
            fig_cv = go.Figure(go.Bar(
                x=wh_ss_df['Warehouse'], y=wh_ss_df['CV (%)'],
                marker_color=[WH_COLORS.get(w, COLORS['gray']) for w in wh_ss_df['Warehouse']],
                text=[f"{v:.1f}%" for v in wh_ss_df['CV (%)']],
                textposition='outside',
            ))
            fig_cv.update_layout(height=350, yaxis_title='CV (%)')
            st.plotly_chart(fig_cv, use_container_width=True)
            st.caption("CV가 높을수록 수요 변동이 크고, 더 많은 안전재고가 필요합니다.")

        with col_w2:
            st.subheader("창고별 생산 목표 (95% SL)")
            fig_target = go.Figure()
            fig_target.add_trace(go.Bar(
                x=wh_ss_df['Warehouse'], y=wh_ss_df['avg_predicted'],
                name='Predicted Demand',
                marker_color=[WH_COLORS.get(w, COLORS['gray']) for w in wh_ss_df['Warehouse']],
                opacity=0.7,
            ))
            fig_target.add_trace(go.Bar(
                x=wh_ss_df['Warehouse'], y=wh_ss_df['safety_stock_95'],
                name='Safety Stock',
                marker_color=[WH_COLORS.get(w, COLORS['gray']) for w in wh_ss_df['Warehouse']],
                opacity=0.3,
            ))
            fig_target.update_layout(height=350, barmode='stack', yaxis_title='Units')
            st.plotly_chart(fig_target, use_container_width=True)

        # 요약 테이블
        st.subheader("창고별 요약")
        display_df = wh_ss_df.copy()
        display_df['avg_predicted'] = display_df['avg_predicted'].apply(lambda x: f"{x:,.0f}")
        display_df['forecast_std'] = display_df['forecast_std'].apply(lambda x: f"{x:,.0f}")
        display_df['safety_stock_95'] = display_df['safety_stock_95'].apply(lambda x: f"{x:,.0f}")
        display_df['production_target_95'] = display_df['production_target_95'].apply(lambda x: f"{x:,.0f}")
        display_df['CV (%)'] = display_df['CV (%)'].apply(lambda x: f"{x:.1f}%")
        display_df.columns = ['창고', '월평균 예측수요', '예측 표준편차', 'CV', '안전재고 (95%)', '생산목표 (95%)']
        st.dataframe(display_df, hide_index=True, use_container_width=True)

    else:
        # 개별 창고 상세
        wh = selected_warehouse
        wh_data = monthly_wh[
            (monthly_wh['Warehouse'] == wh) &
            (monthly_wh['YearMonth'] >= '2012-01-01') &
            (monthly_wh['YearMonth'] <= '2016-12-01')
        ]

        wh_fc = wh_forecast_df[wh_forecast_df['Warehouse'] == wh]
        wh_info = wh_ss_df[wh_ss_df['Warehouse'] == wh].iloc[0]

        col_i1, col_i2, col_i3, col_i4 = st.columns(4)
        col_i1.metric("월평균 예측수요", f"{wh_info['avg_predicted']/1e6:.1f}M")
        col_i2.metric("예측 표준편차", f"{wh_info['forecast_std']/1e6:.1f}M")
        col_i3.metric("CV", f"{wh_info['CV (%)']:.1f}%")
        col_i4.metric("안전재고 (95%)", f"{wh_info['safety_stock_95']/1e6:.1f}M")

        # 수요 추이 + 예측
        fig_wh_detail = go.Figure()
        color = WH_COLORS.get(wh, COLORS['gray'])

        fig_wh_detail.add_trace(go.Scatter(
            x=wh_data['YearMonth'], y=wh_data['total_demand'],
            mode='lines+markers', line=dict(color=color, width=1.5),
            marker=dict(size=3), name='Actual',
        ))

        if len(wh_fc) > 0:
            fig_wh_detail.add_trace(go.Scatter(
                x=wh_fc['YearMonth'], y=wh_fc['predicted'],
                mode='lines+markers', line=dict(color=COLORS['red'], width=2, dash='dash'),
                marker=dict(size=5), name='Forecast',
            ))
            fig_wh_detail.add_trace(go.Scatter(
                x=list(wh_fc['YearMonth']) + list(wh_fc['YearMonth'][::-1]),
                y=list(wh_fc['upper_95']) + list(wh_fc['lower_95'][::-1]),
                fill='toself', fillcolor=COLORS['red'], opacity=0.1,
                line=dict(width=0), name='95% PI', showlegend=False,
            ))

        fig_wh_detail.update_layout(height=400, xaxis_title='Date', yaxis_title='Demand')
        st.plotly_chart(fig_wh_detail, use_container_width=True)

        # 해당 창고 시뮬레이션
        wh_std = wh_info['forecast_std']
        wh_pred = wh_info['avg_predicted']
        wh_z = stats.norm.ppf(service_level / 100)
        wh_ss = wh_z * wh_std
        wh_target = wh_pred + wh_ss

        wh_sim = np.random.normal(wh_pred, wh_std, N_SIM)
        wh_coverage = np.mean(wh_sim <= wh_target) * 100

        st.subheader(f"{wh} 안전재고 시뮬레이션 (SL {service_level}%)")

        fig_wh_sim = go.Figure()
        fig_wh_sim.add_trace(go.Histogram(
            x=wh_sim, nbinsx=60, marker_color=color, opacity=0.5,
        ))
        fig_wh_sim.add_vline(x=wh_pred, line_color='black', line_width=2,
                             annotation_text=f"Forecast: {wh_pred/1e6:.1f}M")
        fig_wh_sim.add_vline(x=wh_target, line_color=COLORS['red'], line_width=2,
                             line_dash='dash',
                             annotation_text=f"Target: {wh_target/1e6:.1f}M")
        fig_wh_sim.update_layout(height=350, xaxis_title='Demand', yaxis_title='Frequency')
        st.plotly_chart(fig_wh_sim, use_container_width=True)

        col_ws1, col_ws2, col_ws3 = st.columns(3)
        col_ws1.metric("생산 목표", f"{wh_target/1e6:.1f}M")
        col_ws2.metric("안전재고", f"{wh_ss/1e6:.1f}M")
        col_ws3.metric("커버리지", f"{wh_coverage:.1f}%")
