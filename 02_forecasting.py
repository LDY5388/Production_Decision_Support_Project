"""
02_forecasting.py
=================
수요 예측 모델링

목적:
  - 전체 월별 수요에 대해 ARIMA, Holt-Winters, Prophet 모델 비교
  - 창고별 수요 예측
  - 예측 구간(prediction interval) 산출
  - 모델 성능 비교 (RMSE, MAPE, 커버리지)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings("ignore")

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

OUTPUT_DIR = "outputs/"
DATA_DIR = "data/"


# ============================================================
# 1. 데이터 준비
# ============================================================
print("=" * 60)
print("  1. 데이터 준비")
print("=" * 60)

df = pd.read_csv(f"{DATA_DIR}monthly_total.csv", parse_dates=['YearMonth'])
df = df.set_index('YearMonth')

# 2012-01 ~ 2016-12만 사용 (2011년은 데이터 부족, 2017년은 1월만 존재)
df = df['2012-01':'2016-12']
print(f"분석 기간: {df.index.min().date()} ~ {df.index.max().date()}")
print(f"총 {len(df)}개월")

ts = df['total_demand']

# train/test 분할: 마지막 6개월을 테스트셋
train = ts[:-6]
test = ts[-6:]
print(f"학습셋: {len(train)}개월 ({train.index[0].date()} ~ {train.index[-1].date()})")
print(f"테스트셋: {len(test)}개월 ({test.index[0].date()} ~ {test.index[-1].date()})")

h = len(test)  # forecast horizon


# ============================================================
# 2. 모델 학습 및 예측
# ============================================================
print("\n" + "=" * 60)
print("  2. 모델 학습 및 예측")
print("=" * 60)

results = {}

# --- 2-1. SARIMA ---
print("\n[SARIMA]")
# (p,d,q) x (P,D,Q,s) - ADF 검정에서 d=1 확인, 12개월 계절성
try:
    sarima = ARIMA(train, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))
    sarima_fit = sarima.fit()
    sarima_pred = sarima_fit.forecast(steps=h)
    sarima_ci = sarima_fit.get_forecast(steps=h).conf_int(alpha=0.05)

    results['SARIMA'] = {
        'pred': sarima_pred,
        'lower': sarima_ci.iloc[:, 0],
        'upper': sarima_ci.iloc[:, 1],
    }
    print(f"  AIC: {sarima_fit.aic:.1f}")
    print(f"  BIC: {sarima_fit.bic:.1f}")
except Exception as e:
    print(f"  SARIMA 실패: {e}")


# --- 2-2. Holt-Winters ---
print("\n[Holt-Winters]")
try:
    hw = ExponentialSmoothing(
        train, trend='add', seasonal='add', seasonal_periods=12
    ).fit(optimized=True)
    hw_pred = hw.forecast(steps=h)

    # Holt-Winters 예측 구간: 잔차 기반 계산
    hw_resid_std = np.std(hw.resid)
    hw_lower = hw_pred - 1.96 * hw_resid_std
    hw_upper = hw_pred + 1.96 * hw_resid_std

    results['Holt-Winters'] = {
        'pred': hw_pred,
        'lower': hw_lower,
        'upper': hw_upper,
    }
    print(f"  smoothing_level (alpha): {hw.params['smoothing_level']:.4f}")
    print(f"  smoothing_trend (beta): {hw.params['smoothing_trend']:.4f}")
    print(f"  smoothing_seasonal (gamma): {hw.params['smoothing_seasonal']:.4f}")
except Exception as e:
    print(f"  Holt-Winters 실패: {e}")


# --- 2-3. Prophet ---
print("\n[Prophet]")
try:
    prophet_df = train.reset_index()
    prophet_df.columns = ['ds', 'y']

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.95,
    )
    m.fit(prophet_df)

    future = m.make_future_dataframe(periods=h, freq='MS')
    forecast = m.predict(future)
    prophet_pred = forecast.tail(h).set_index('ds')

    results['Prophet'] = {
        'pred': pd.Series(prophet_pred['yhat'].values, index=test.index),
        'lower': pd.Series(prophet_pred['yhat_lower'].values, index=test.index),
        'upper': pd.Series(prophet_pred['yhat_upper'].values, index=test.index),
    }
    print("  학습 완료")
except Exception as e:
    print(f"  Prophet 실패: {e}")


# ============================================================
# 3. 성능 평가
# ============================================================
print("\n" + "=" * 60)
print("  3. 성능 평가")
print("=" * 60)

metrics_list = []
for name, res in results.items():
    pred = res['pred'].values
    actual = test.values
    lower = res['lower'].values
    upper = res['upper'].values

    rmse = np.sqrt(mean_squared_error(actual, pred))
    mape = mean_absolute_percentage_error(actual, pred) * 100
    coverage = np.mean((actual >= lower) & (actual <= upper)) * 100

    metrics_list.append({
        'Model': name,
        'RMSE': rmse,
        'MAPE (%)': mape,
        '95% CI Coverage (%)': coverage,
    })
    print(f"\n  {name}:")
    print(f"    RMSE: {rmse:,.0f}")
    print(f"    MAPE: {mape:.2f}%")
    print(f"    95% 예측 구간 커버리지: {coverage:.1f}%")

metrics_df = pd.DataFrame(metrics_list)
metrics_df.to_csv(f'{DATA_DIR}model_metrics.csv', index=False)
print(f"\n성능 지표 저장: {DATA_DIR}model_metrics.csv")


# ============================================================
# 4. 시각화
# ============================================================
print("\n" + "=" * 60)
print("  4. 시각화")
print("=" * 60)

model_colors = {
    'SARIMA': '#2563eb',
    'Holt-Winters': '#dc2626',
    'Prophet': '#16a34a',
}

# --- 4-1. 모델별 예측 결과 비교 (한 그래프) ---
fig, ax = plt.subplots(figsize=(14, 6))

# 학습 데이터
ax.plot(train.index, train.values, color='#374151', linewidth=1.2, label='Train')
# 실제값
ax.plot(test.index, test.values, color='black', linewidth=2, marker='o',
        markersize=5, label='Actual', zorder=5)

for name, res in results.items():
    color = model_colors[name]
    ax.plot(test.index, res['pred'].values, color=color, linewidth=1.5,
            linestyle='--', marker='s', markersize=4, label=f'{name} Forecast')
    ax.fill_between(test.index, res['lower'].values, res['upper'].values,
                    color=color, alpha=0.1)

ax.axvline(x=test.index[0], color='#6b7280', linestyle=':', linewidth=1, alpha=0.7)
ax.text(test.index[0], ax.get_ylim()[1] * 0.95, ' Test Period',
        fontsize=9, color='#6b7280')

ax.set_title('Demand Forecast Comparison (Last 6 Months)', fontsize=13, pad=12)
ax.set_xlabel('Date')
ax.set_ylabel('Total Demand')
ax.legend(loc='lower left', fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}10_forecast_comparison.png', dpi=150)
plt.close()
print("저장: 10_forecast_comparison.png")


# --- 4-2. 모델별 개별 예측 (서브플롯) ---
fig, axes = plt.subplots(len(results), 1, figsize=(14, 4 * len(results)), sharex=True)
if len(results) == 1:
    axes = [axes]

for i, (name, res) in enumerate(results.items()):
    ax = axes[i]
    color = model_colors[name]

    ax.plot(train.index, train.values, color='#374151', linewidth=1, alpha=0.7)
    ax.plot(test.index, test.values, color='black', linewidth=2, marker='o',
            markersize=5, label='Actual')
    ax.plot(test.index, res['pred'].values, color=color, linewidth=1.5,
            linestyle='--', marker='s', markersize=4, label=f'{name}')
    ax.fill_between(test.index, res['lower'].values, res['upper'].values,
                    color=color, alpha=0.15, label='95% PI')
    ax.axvline(x=test.index[0], color='#6b7280', linestyle=':', linewidth=1, alpha=0.5)

    # RMSE, MAPE 표시
    rmse = np.sqrt(mean_squared_error(test.values, res['pred'].values))
    mape = mean_absolute_percentage_error(test.values, res['pred'].values) * 100
    ax.text(0.02, 0.95, f'RMSE: {rmse:,.0f}  |  MAPE: {mape:.1f}%',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.set_title(f'{name}', fontsize=11, loc='left')
    ax.set_ylabel('Demand')
    ax.legend(loc='lower left', fontsize=8)

plt.suptitle('Individual Model Forecasts with 95% Prediction Interval', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}11_forecast_individual.png', dpi=150)
plt.close()
print("저장: 11_forecast_individual.png")


# --- 4-3. 예측 오차 분포 ---
fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
if len(results) == 1:
    axes = [axes]

for i, (name, res) in enumerate(results.items()):
    errors = test.values - res['pred'].values
    color = model_colors[name]

    axes[i].bar(range(len(errors)), errors, color=color, alpha=0.7)
    axes[i].axhline(y=0, color='black', linewidth=0.8)
    axes[i].set_title(f'{name} - Forecast Error', fontsize=10)
    axes[i].set_xlabel('Month (test period)')
    axes[i].set_ylabel('Actual - Predicted')

    # 평균 오차 표시
    mean_err = np.mean(errors)
    axes[i].axhline(y=mean_err, color=color, linestyle='--', linewidth=1, alpha=0.7)
    axes[i].text(0.5, mean_err, f'Mean: {mean_err:,.0f}', fontsize=8, color=color)

plt.suptitle('Forecast Error by Model', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}12_forecast_errors.png', dpi=150)
plt.close()
print("저장: 12_forecast_errors.png")


# --- 4-4. 성능 비교 바 차트 ---
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

metric_names = ['RMSE', 'MAPE (%)', '95% CI Coverage (%)']
bar_colors = [model_colors[m] for m in metrics_df['Model']]

for i, metric in enumerate(metric_names):
    axes[i].bar(metrics_df['Model'], metrics_df[metric], color=bar_colors, alpha=0.85)
    axes[i].set_title(metric, fontsize=11)
    for j, val in enumerate(metrics_df[metric]):
        fmt = f'{val:,.0f}' if metric == 'RMSE' else f'{val:.1f}'
        axes[i].text(j, val, fmt, ha='center', va='bottom', fontsize=9)

plt.suptitle('Model Performance Comparison', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}13_model_comparison.png', dpi=150)
plt.close()
print("저장: 13_model_comparison.png")


# ============================================================
# 5. 창고별 예측 (최적 모델 적용)
# ============================================================
print("\n" + "=" * 60)
print("  5. 창고별 예측")
print("=" * 60)

wh_df = pd.read_csv(f"{DATA_DIR}monthly_by_warehouse.csv", parse_dates=['YearMonth'])

warehouses = sorted(wh_df['Warehouse'].unique())
wh_colors = {'Whse_A': '#2563eb', 'Whse_C': '#dc2626',
             'Whse_J': '#16a34a', 'Whse_S': '#d97706'}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

wh_forecasts = {}

for i, wh in enumerate(warehouses):
    wh_data = wh_df[wh_df['Warehouse'] == wh].set_index('YearMonth')['total_demand']
    wh_data = wh_data['2012-01':'2016-12']

    wh_train = wh_data[:-6]
    wh_test = wh_data[-6:]

    # Holt-Winters 적용 (일반적으로 안정적)
    try:
        hw_wh = ExponentialSmoothing(
            wh_train, trend='add', seasonal='add', seasonal_periods=12
        ).fit(optimized=True)
        wh_pred = hw_wh.forecast(steps=6)
        wh_resid_std = np.std(hw_wh.resid)
        wh_lower = wh_pred - 1.96 * wh_resid_std
        wh_upper = wh_pred + 1.96 * wh_resid_std

        rmse = np.sqrt(mean_squared_error(wh_test.values, wh_pred.values))
        mape = mean_absolute_percentage_error(wh_test.values, wh_pred.values) * 100

        wh_forecasts[wh] = {
            'pred': wh_pred,
            'lower': wh_lower,
            'upper': wh_upper,
            'rmse': rmse,
            'mape': mape,
            'resid_std': wh_resid_std,
        }

        color = wh_colors[wh]
        ax = axes[i]
        ax.plot(wh_train.index, wh_train.values, color='#374151', linewidth=1, alpha=0.7)
        ax.plot(wh_test.index, wh_test.values, color='black', linewidth=2,
                marker='o', markersize=4, label='Actual')
        ax.plot(wh_test.index, wh_pred.values, color=color, linewidth=1.5,
                linestyle='--', marker='s', markersize=4, label='Forecast')
        ax.fill_between(wh_test.index, wh_lower.values, wh_upper.values,
                        color=color, alpha=0.15)
        ax.axvline(x=wh_test.index[0], color='#6b7280', linestyle=':', linewidth=1, alpha=0.5)

        ax.set_title(f'{wh}  (MAPE: {mape:.1f}%)', fontsize=11)
        ax.set_ylabel('Demand')
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        print(f"  {wh}: RMSE={rmse:,.0f}, MAPE={mape:.1f}%")

    except Exception as e:
        print(f"  {wh} 실패: {e}")

plt.suptitle('Warehouse-level Demand Forecast (Holt-Winters)', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}14_warehouse_forecasts.png', dpi=150)
plt.close()
print("저장: 14_warehouse_forecasts.png")


# ============================================================
# 6. 예측 결과 저장 (시뮬레이션용)
# ============================================================
print("\n" + "=" * 60)
print("  6. 예측 결과 저장")
print("=" * 60)

# 전체 모델 예측값 저장
forecast_results = pd.DataFrame({'actual': test})
for name, res in results.items():
    forecast_results[f'{name}_pred'] = res['pred'].values
    forecast_results[f'{name}_lower'] = res['lower'].values
    forecast_results[f'{name}_upper'] = res['upper'].values
forecast_results.to_csv(f'{DATA_DIR}forecast_results.csv')
print(f"저장: forecast_results.csv")

# 창고별 예측 결과 + 잔차 표준편차 (안전재고 계산에 사용)
wh_forecast_list = []
for wh, res in wh_forecasts.items():
    for date, pred, lower, upper in zip(
        test.index, res['pred'].values, res['lower'].values, res['upper'].values
    ):
        wh_forecast_list.append({
            'Warehouse': wh,
            'YearMonth': date,
            'predicted': pred,
            'lower_95': lower,
            'upper_95': upper,
            'forecast_std': res['resid_std'],
        })
wh_forecast_df = pd.DataFrame(wh_forecast_list)
wh_forecast_df.to_csv(f'{DATA_DIR}warehouse_forecast_results.csv', index=False)
print(f"저장: warehouse_forecast_results.csv")

print("\n" + "=" * 60)
print("  수요 예측 모델링 완료")
print("=" * 60)
