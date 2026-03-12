"""
03_safety_stock_simulation.py
=============================
안전재고 시뮬레이션

목적:
  - 예측 오차 분포를 기반으로 안전재고 수준 산출
  - 몬테카를로 시뮬레이션으로 서비스 수준별 최적 재고량 도출
  - 과잉 생산 비용 vs 품절 비용 트레이드오프 분석
  - 창고별 안전재고 추천
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

OUTPUT_DIR = "outputs/"
DATA_DIR = "data/"
np.random.seed(42)


# ============================================================
# 1. 데이터 로드
# ============================================================
print("=" * 60)
print("  1. 데이터 로드")
print("=" * 60)

forecast_df = pd.read_csv(f"{DATA_DIR}forecast_results.csv", index_col=0, parse_dates=True)
wh_forecast_df = pd.read_csv(f"{DATA_DIR}warehouse_forecast_results.csv", parse_dates=['YearMonth'])

# Holt-Winters 기준 (최적 모델)
actual = forecast_df['actual'].values
predicted = forecast_df['Holt-Winters_pred'].values
forecast_errors = actual - predicted

print(f"예측 오차 통계:")
print(f"  평균: {np.mean(forecast_errors):,.0f}")
print(f"  표준편차: {np.std(forecast_errors):,.0f}")
print(f"  최소: {np.min(forecast_errors):,.0f}")
print(f"  최대: {np.max(forecast_errors):,.0f}")


# ============================================================
# 2. 예측 오차 분포 분석
# ============================================================
print("\n" + "=" * 60)
print("  2. 예측 오차 분포 분석")
print("=" * 60)

# 정규성 검정 (Shapiro-Wilk) - 표본이 작으므로 참고용
if len(forecast_errors) >= 3:
    stat, p_val = stats.shapiro(forecast_errors)
    print(f"Shapiro-Wilk 검정: statistic={stat:.4f}, p-value={p_val:.4f}")
    if p_val > 0.05:
        print("  -> 정규 분포를 따른다는 귀무가설 기각 불가")
    else:
        print("  -> 정규 분포를 따르지 않을 수 있음 (표본 작아 참고만)")

# 전체 학습 기간 잔차 기반 표준편차 사용
# (테스트셋 6개월은 표본이 너무 작으므로, 학습 잔차 std 활용)
monthly_df = pd.read_csv(f"{DATA_DIR}monthly_total.csv", parse_dates=['YearMonth'])
monthly_df = monthly_df.set_index('YearMonth')['total_demand']['2012-01':'2016-12']

from statsmodels.tsa.holtwinters import ExponentialSmoothing
hw_full = ExponentialSmoothing(
    monthly_df[:-6], trend='add', seasonal='add', seasonal_periods=12
).fit(optimized=True)

resid_std = np.std(hw_full.resid.dropna())
demand_mean = monthly_df[:-6].mean()
print(f"\n학습 잔차 표준편차: {resid_std:,.0f}")
print(f"학습 기간 평균 수요: {demand_mean:,.0f}")
print(f"변동계수 (CV): {resid_std / demand_mean * 100:.1f}%")


# ============================================================
# 3. 몬테카를로 시뮬레이션
# ============================================================
print("\n" + "=" * 60)
print("  3. 몬테카를로 시뮬레이션")
print("=" * 60)

N_SIM = 10000
service_levels = [0.90, 0.95, 0.99]

# 기본 생산 목표 = 예측값 (다음 달 예측 기준, 마지막 예측값 사용)
base_production = predicted[-1]  # 가장 최근 예측값
print(f"기본 생산 목표 (예측값): {base_production:,.0f}")

# 수요를 정규분포로 시뮬레이션: N(예측값, 잔차_std)
simulated_demand = np.random.normal(base_production, resid_std, N_SIM)

# 각 서비스 수준별 안전재고
print(f"\n서비스 수준별 안전재고:")
safety_stock_results = []
for sl in service_levels:
    z_score = stats.norm.ppf(sl)
    safety_stock = z_score * resid_std
    production_target = base_production + safety_stock

    # 시뮬레이션에서 실제 서비스 수준 확인
    actual_sl = np.mean(simulated_demand <= production_target) * 100

    safety_stock_results.append({
        'service_level': sl * 100,
        'z_score': z_score,
        'safety_stock': safety_stock,
        'production_target': production_target,
        'actual_coverage': actual_sl,
    })
    print(f"  {sl*100:.0f}%: 안전재고 = {safety_stock:,.0f}, "
          f"생산 목표 = {production_target:,.0f}, "
          f"시뮬레이션 커버리지 = {actual_sl:.1f}%")

ss_df = pd.DataFrame(safety_stock_results)


# ============================================================
# 4. 비용 트레이드오프 분석
# ============================================================
print("\n" + "=" * 60)
print("  4. 비용 트레이드오프 분석")
print("=" * 60)

# 비용 가정 (상대적 비율)
# 과잉 생산 단위 비용: 1 (재고 보관비, 감가상각 등)
# 품절 단위 비용: 5 (매출 손실, 고객 이탈 등 - 일반적으로 품절이 더 비쌈)
OVERPRODUCTION_COST = 1
STOCKOUT_COST = 5

production_range = np.linspace(
    base_production - 2 * resid_std,
    base_production + 3 * resid_std,
    200
)

cost_results = []
for prod in production_range:
    # 과잉: 생산량 > 수요인 경우
    overprod = np.maximum(prod - simulated_demand, 0)
    # 품절: 수요 > 생산량인 경우
    stockout = np.maximum(simulated_demand - prod, 0)

    total_cost = (
        OVERPRODUCTION_COST * np.mean(overprod) +
        STOCKOUT_COST * np.mean(stockout)
    )
    overprod_cost = OVERPRODUCTION_COST * np.mean(overprod)
    stockout_cost = STOCKOUT_COST * np.mean(stockout)
    service_rate = np.mean(simulated_demand <= prod) * 100

    cost_results.append({
        'production': prod,
        'total_cost': total_cost,
        'overproduction_cost': overprod_cost,
        'stockout_cost': stockout_cost,
        'service_rate': service_rate,
    })

cost_df = pd.DataFrame(cost_results)
optimal_idx = cost_df['total_cost'].idxmin()
optimal_prod = cost_df.loc[optimal_idx, 'production']
optimal_cost = cost_df.loc[optimal_idx, 'total_cost']
optimal_sl = cost_df.loc[optimal_idx, 'service_rate']

print(f"비용 가정: 과잉 생산 단위비용 = {OVERPRODUCTION_COST}, 품절 단위비용 = {STOCKOUT_COST}")
print(f"최적 생산량: {optimal_prod:,.0f}")
print(f"최적 생산량의 서비스 수준: {optimal_sl:.1f}%")
print(f"최적 안전재고: {optimal_prod - base_production:,.0f}")
print(f"최소 총비용: {optimal_cost:,.0f}")


# ============================================================
# 5. 시각화
# ============================================================
print("\n" + "=" * 60)
print("  5. 시각화")
print("=" * 60)

# --- 5-1. 시뮬레이션 수요 분포 + 안전재고 수준 ---
fig, ax = plt.subplots(figsize=(14, 6))

ax.hist(simulated_demand, bins=80, color='#2563eb', alpha=0.5, density=True,
        edgecolor='white', label='Simulated Demand Distribution')

# 정규분포 곡선 오버레이
x_range = np.linspace(simulated_demand.min(), simulated_demand.max(), 200)
pdf = stats.norm.pdf(x_range, base_production, resid_std)
ax.plot(x_range, pdf, color='#2563eb', linewidth=2, label='Normal PDF')

# 서비스 수준별 라인
sl_colors = {90: '#d97706', 95: '#dc2626', 99: '#7c3aed'}
for row in safety_stock_results:
    sl = row['service_level']
    prod = row['production_target']
    color = sl_colors[int(sl)]
    ax.axvline(x=prod, color=color, linewidth=1.5, linestyle='--',
               label=f"SL {sl:.0f}%: {prod:,.0f}")

# 기본 예측값
ax.axvline(x=base_production, color='black', linewidth=2, linestyle='-',
           label=f'Forecast: {base_production:,.0f}')

ax.set_title('Monte Carlo Simulation: Demand Distribution & Safety Stock Levels', fontsize=13, pad=12)
ax.set_xlabel('Demand / Production Target')
ax.set_ylabel('Density')
ax.legend(loc='upper right', fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}15_simulation_distribution.png', dpi=150)
plt.close()
print("저장: 15_simulation_distribution.png")


# --- 5-2. 비용 트레이드오프 곡선 ---
fig, ax1 = plt.subplots(figsize=(14, 6))

ax1.plot(cost_df['production'], cost_df['overproduction_cost'],
         color='#d97706', linewidth=1.5, label='Overproduction Cost')
ax1.plot(cost_df['production'], cost_df['stockout_cost'],
         color='#dc2626', linewidth=1.5, label='Stockout Cost')
ax1.plot(cost_df['production'], cost_df['total_cost'],
         color='#2563eb', linewidth=2.5, label='Total Cost')

ax1.axvline(x=optimal_prod, color='#16a34a', linewidth=1.5, linestyle='--')
ax1.plot(optimal_prod, optimal_cost, 'o', color='#16a34a', markersize=10, zorder=5)
ax1.annotate(f'Optimal: {optimal_prod:,.0f}\n(SL: {optimal_sl:.1f}%)',
             xy=(optimal_prod, optimal_cost),
             xytext=(optimal_prod + resid_std * 0.5, optimal_cost * 1.3),
             fontsize=10,
             arrowprops=dict(arrowstyle='->', color='#16a34a'),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0fdf4', alpha=0.9))

ax1.axvline(x=base_production, color='black', linewidth=1, linestyle=':',
            alpha=0.5, label=f'Forecast: {base_production:,.0f}')

ax1.set_title('Cost Trade-off Analysis: Overproduction vs Stockout', fontsize=13, pad=12)
ax1.set_xlabel('Production Target')
ax1.set_ylabel('Expected Cost')
ax1.legend(loc='upper center', fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}16_cost_tradeoff.png', dpi=150)
plt.close()
print("저장: 16_cost_tradeoff.png")


# --- 5-3. 서비스 수준 vs 안전재고 곡선 ---
sl_range = np.linspace(0.50, 0.999, 200)
ss_range = stats.norm.ppf(sl_range) * resid_std

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(sl_range * 100, ss_range, color='#2563eb', linewidth=2)

for row in safety_stock_results:
    sl = row['service_level']
    ss = row['safety_stock']
    color = sl_colors[int(sl)]
    ax.plot(sl, ss, 'o', color=color, markersize=8, zorder=5)
    ax.annotate(f'{sl:.0f}%: {ss:,.0f}', xy=(sl, ss),
                xytext=(sl - 8, ss + resid_std * 0.3), fontsize=9, color=color)

ax.set_title('Service Level vs Safety Stock', fontsize=13, pad=12)
ax.set_xlabel('Service Level (%)')
ax.set_ylabel('Safety Stock (units)')
ax.set_xlim(50, 100)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}17_service_level_curve.png', dpi=150)
plt.close()
print("저장: 17_service_level_curve.png")


# --- 5-4. 창고별 안전재고 추천 ---
print("\n창고별 안전재고 (95% 서비스 수준):")

wh_ss_data = []
warehouses = wh_forecast_df['Warehouse'].unique()
for wh in sorted(warehouses):
    wh_data = wh_forecast_df[wh_forecast_df['Warehouse'] == wh]
    wh_std = wh_data['forecast_std'].iloc[0]
    wh_pred_mean = wh_data['predicted'].mean()

    z_95 = stats.norm.ppf(0.95)
    wh_safety_stock = z_95 * wh_std
    wh_cv = wh_std / wh_pred_mean * 100

    wh_ss_data.append({
        'Warehouse': wh,
        'avg_predicted': wh_pred_mean,
        'forecast_std': wh_std,
        'CV (%)': wh_cv,
        'safety_stock_95': wh_safety_stock,
        'production_target_95': wh_pred_mean + wh_safety_stock,
    })
    print(f"  {wh}: 예측평균={wh_pred_mean:,.0f}, "
          f"안전재고={wh_safety_stock:,.0f}, "
          f"생산목표={wh_pred_mean + wh_safety_stock:,.0f}, "
          f"CV={wh_cv:.1f}%")

wh_ss_df = pd.DataFrame(wh_ss_data)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
wh_colors_list = ['#2563eb', '#dc2626', '#16a34a', '#d97706']

# 예측 평균 vs 안전재고
x = range(len(wh_ss_df))
width = 0.35
axes[0].bar([xi - width/2 for xi in x], wh_ss_df['avg_predicted'],
            width, color=wh_colors_list, alpha=0.7, label='Predicted Demand')
axes[0].bar([xi + width/2 for xi in x], wh_ss_df['safety_stock_95'],
            width, color=wh_colors_list, alpha=0.4, label='Safety Stock (95%)',
            hatch='//')
axes[0].set_xticks(x)
axes[0].set_xticklabels(wh_ss_df['Warehouse'], fontsize=9)
axes[0].set_title('Predicted Demand vs Safety Stock', fontsize=11)
axes[0].legend(fontsize=8)

# 변동계수 비교
axes[1].bar(x, wh_ss_df['CV (%)'], color=wh_colors_list, alpha=0.7)
axes[1].set_xticks(x)
axes[1].set_xticklabels(wh_ss_df['Warehouse'], fontsize=9)
axes[1].set_title('Coefficient of Variation (%)', fontsize=11)
axes[1].set_ylabel('CV (%)')
for j, val in enumerate(wh_ss_df['CV (%)']):
    axes[1].text(j, val, f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

# 생산 목표
axes[2].bar(x, wh_ss_df['production_target_95'], color=wh_colors_list, alpha=0.7)
axes[2].set_xticks(x)
axes[2].set_xticklabels(wh_ss_df['Warehouse'], fontsize=9)
axes[2].set_title('Recommended Production Target (95% SL)', fontsize=11)
for j, val in enumerate(wh_ss_df['production_target_95']):
    axes[2].text(j, val, f'{val:,.0f}', ha='center', va='bottom', fontsize=8)

plt.suptitle('Warehouse Safety Stock Analysis', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}18_warehouse_safety_stock.png', dpi=150)
plt.close()
print("저장: 18_warehouse_safety_stock.png")


# ============================================================
# 6. 결과 저장
# ============================================================
print("\n" + "=" * 60)
print("  6. 결과 저장")
print("=" * 60)

ss_df.to_csv(f'{DATA_DIR}safety_stock_by_service_level.csv', index=False)
wh_ss_df.to_csv(f'{DATA_DIR}warehouse_safety_stock.csv', index=False)
cost_df.to_csv(f'{DATA_DIR}cost_tradeoff.csv', index=False)

print(f"저장: safety_stock_by_service_level.csv")
print(f"저장: warehouse_safety_stock.csv")
print(f"저장: cost_tradeoff.csv")

print("\n" + "=" * 60)
print("  안전재고 시뮬레이션 완료")
print("=" * 60)
