"""
01_preprocessing_eda.py
=======================
전처리 및 탐색적 데이터 분석 (EDA)

목적:
  - 원본 데이터 정제 (타입 변환, 결측치 처리, 음수 주문 처리)
  - 월별 집계 데이터 생성
  - 창고별/카테고리별/제품별 수요 패턴 탐색
  - 시계열 특성 분석 (추세, 계절성, 정상성 검정)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
# 1. 데이터 로드 및 전처리
# ============================================================
print("=" * 60)
print("  1. 데이터 로드 및 전처리")
print("=" * 60)

df = pd.read_csv(f"{DATA_DIR}raw_demand.csv")
print(f"원본 데이터: {df.shape[0]:,}행 x {df.shape[1]}열")
print(f"컬럼: {df.columns.tolist()}\n")

# 1-1. Order_Demand 숫자 변환
# 괄호로 감싼 값은 음수 (반품/취소)
df['Order_Demand'] = (
    df['Order_Demand']
    .str.strip()
    .str.replace('(', '-', regex=False)
    .str.replace(')', '', regex=False)
)
df['Order_Demand'] = pd.to_numeric(df['Order_Demand'], errors='coerce')

# 1-2. 날짜 변환
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# 1-3. 결측치 현황
print("결측치 현황:")
null_counts = df.isnull().sum()
for col, cnt in null_counts.items():
    if cnt > 0:
        print(f"  {col}: {cnt:,}건 ({cnt/len(df)*100:.2f}%)")

# 1-4. 결측치 처리
# 날짜 결측 행 제거 (날짜 없이는 시계열 분석 불가)
before = len(df)
df = df.dropna(subset=['Date'])
print(f"\n날짜 결측 제거: {before:,} -> {len(df):,}행 ({before - len(df):,}건 제거)")

# 음수 주문 현황 확인 후 제거 (반품/취소는 수요 예측 대상이 아님)
neg_count = (df['Order_Demand'] < 0).sum()
zero_count = (df['Order_Demand'] == 0).sum()
print(f"음수 주문: {neg_count:,}건, 0 주문: {zero_count:,}건")

df = df[df['Order_Demand'] > 0].copy()
print(f"양수 주문만 필터링: {len(df):,}행\n")

# 1-5. 연/월 컬럼 추가
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['YearMonth'] = df['Date'].dt.to_period('M')

# 1-6. 전처리 완료 데이터 저장
df.to_csv(f"{DATA_DIR}cleaned_demand.csv", index=False)
print(f"전처리 완료 데이터 저장: {DATA_DIR}cleaned_demand.csv")
print(f"최종 데이터: {len(df):,}행, 기간: {df['Date'].min().date()} ~ {df['Date'].max().date()}")


# ============================================================
# 2. 기초 통계
# ============================================================
print("\n" + "=" * 60)
print("  2. 기초 통계")
print("=" * 60)

print(f"\n제품 수: {df['Product_Code'].nunique():,}")
print(f"창고 수: {df['Warehouse'].nunique()}")
print(f"카테고리 수: {df['Product_Category'].nunique()}")
print(f"주문 건수: {len(df):,}")
print(f"총 주문량: {df['Order_Demand'].sum():,.0f}")

print("\n주문량 분포:")
print(df['Order_Demand'].describe().to_string())

print("\n창고별 주문 비중:")
wh_summary = (
    df.groupby('Warehouse')['Order_Demand']
    .agg(['count', 'sum', 'mean', 'median'])
    .sort_values('sum', ascending=False)
)
wh_summary.columns = ['주문건수', '총주문량', '평균주문량', '중앙값']
wh_summary['비중(%)'] = (wh_summary['총주문량'] / wh_summary['총주문량'].sum() * 100).round(1)
print(wh_summary.to_string())


# ============================================================
# 3. 시각화
# ============================================================
print("\n" + "=" * 60)
print("  3. 시각화")
print("=" * 60)

# --- 3-1. 전체 월별 수요 추이 ---
monthly_total = (
    df.groupby('YearMonth')['Order_Demand']
    .sum()
    .reset_index()
)
monthly_total['YearMonth'] = monthly_total['YearMonth'].dt.to_timestamp()

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(monthly_total['YearMonth'], monthly_total['Order_Demand'],
        color='#2563eb', linewidth=1.5)
ax.fill_between(monthly_total['YearMonth'], monthly_total['Order_Demand'],
                alpha=0.1, color='#2563eb')
ax.set_title('Monthly Total Order Demand (All Products)', fontsize=13, pad=12)
ax.set_xlabel('Date')
ax.set_ylabel('Order Demand')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}01_monthly_total_demand.png', dpi=150)
plt.close()
print("저장: 01_monthly_total_demand.png")


# --- 3-2. 창고별 월별 수요 추이 ---
monthly_wh = (
    df.groupby(['YearMonth', 'Warehouse'])['Order_Demand']
    .sum()
    .reset_index()
)
monthly_wh['YearMonth'] = monthly_wh['YearMonth'].dt.to_timestamp()

warehouses = sorted(df['Warehouse'].unique())
colors_wh = {'Whse_A': '#2563eb', 'Whse_C': '#dc2626',
             'Whse_J': '#16a34a', 'Whse_S': '#d97706'}

fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
axes = axes.flatten()

for i, wh in enumerate(warehouses):
    wh_data = monthly_wh[monthly_wh['Warehouse'] == wh]
    axes[i].plot(wh_data['YearMonth'], wh_data['Order_Demand'],
                 color=colors_wh[wh], linewidth=1.2)
    axes[i].fill_between(wh_data['YearMonth'], wh_data['Order_Demand'],
                         alpha=0.1, color=colors_wh[wh])
    axes[i].set_title(f'{wh}', fontsize=11)
    axes[i].set_ylabel('Demand')
    axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[i].xaxis.set_major_locator(mdates.YearLocator())

plt.suptitle('Monthly Demand by Warehouse', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}02_monthly_demand_by_warehouse.png', dpi=150)
plt.close()
print("저장: 02_monthly_demand_by_warehouse.png")


# --- 3-3. 상위 카테고리별 수요 추이 ---
top_categories = (
    df.groupby('Product_Category')['Order_Demand']
    .sum()
    .nlargest(6)
    .index.tolist()
)

monthly_cat = (
    df[df['Product_Category'].isin(top_categories)]
    .groupby(['YearMonth', 'Product_Category'])['Order_Demand']
    .sum()
    .reset_index()
)
monthly_cat['YearMonth'] = monthly_cat['YearMonth'].dt.to_timestamp()

fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
axes = axes.flatten()
cat_colors = ['#2563eb', '#dc2626', '#16a34a', '#d97706', '#7c3aed', '#0891b2']

for i, cat in enumerate(top_categories):
    cat_data = monthly_cat[monthly_cat['Product_Category'] == cat]
    axes[i].plot(cat_data['YearMonth'], cat_data['Order_Demand'],
                 color=cat_colors[i], linewidth=1.2)
    axes[i].fill_between(cat_data['YearMonth'], cat_data['Order_Demand'],
                         alpha=0.1, color=cat_colors[i])
    axes[i].set_title(f'{cat}', fontsize=10)
    axes[i].set_ylabel('Demand')
    axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.suptitle('Monthly Demand - Top 6 Categories', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}03_monthly_demand_top_categories.png', dpi=150)
plt.close()
print("저장: 03_monthly_demand_top_categories.png")


# --- 3-4. 주문량 분포 (로그 스케일) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df['Order_Demand'], bins=100, color='#2563eb', alpha=0.7, edgecolor='white')
axes[0].set_title('Order Demand Distribution', fontsize=11)
axes[0].set_xlabel('Order Demand')
axes[0].set_ylabel('Frequency')
axes[0].set_xlim(0, df['Order_Demand'].quantile(0.99))

axes[1].hist(np.log1p(df['Order_Demand']), bins=100, color='#16a34a', alpha=0.7, edgecolor='white')
axes[1].set_title('Order Demand Distribution (log scale)', fontsize=11)
axes[1].set_xlabel('log(1 + Order Demand)')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}04_demand_distribution.png', dpi=150)
plt.close()
print("저장: 04_demand_distribution.png")


# --- 3-5. 월별 계절성 패턴 ---
monthly_season = (
    df.groupby(['Year', 'Month'])['Order_Demand']
    .sum()
    .reset_index()
)

fig, ax = plt.subplots(figsize=(14, 5))
years = sorted(monthly_season['Year'].unique())
year_colors = plt.cm.Blues(np.linspace(0.3, 1.0, len(years)))

for i, year in enumerate(years):
    yr_data = monthly_season[monthly_season['Year'] == year]
    ax.plot(yr_data['Month'], yr_data['Order_Demand'],
            color=year_colors[i], linewidth=1.2, marker='o', markersize=3,
            label=str(year), alpha=0.8)

ax.set_title('Seasonal Pattern by Year (Monthly Total Demand)', fontsize=13, pad=12)
ax.set_xlabel('Month')
ax.set_ylabel('Order Demand')
ax.set_xticks(range(1, 13))
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
ax.legend(loc='upper right', fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}05_seasonal_pattern.png', dpi=150)
plt.close()
print("저장: 05_seasonal_pattern.png")


# --- 3-6. 창고별 주문량 비중 ---
wh_total = df.groupby('Warehouse')['Order_Demand'].sum().sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

wh_colors = [colors_wh[w] for w in wh_total.index]
axes[0].bar(wh_total.index, wh_total.values, color=wh_colors, alpha=0.85)
axes[0].set_title('Total Demand by Warehouse', fontsize=11)
axes[0].set_ylabel('Total Order Demand')
for j, (idx, val) in enumerate(zip(wh_total.index, wh_total.values)):
    axes[0].text(j, val, f'{val/1e6:.1f}M', ha='center', va='bottom', fontsize=9)

axes[1].pie(wh_total.values, labels=wh_total.index, colors=wh_colors,
            autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
axes[1].set_title('Demand Share by Warehouse', fontsize=11)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}06_warehouse_comparison.png', dpi=150)
plt.close()
print("저장: 06_warehouse_comparison.png")


# --- 3-7. 상위 20개 제품 수요량 ---
top_products = (
    df.groupby('Product_Code')['Order_Demand']
    .sum()
    .nlargest(20)
    .sort_values()
)

fig, ax = plt.subplots(figsize=(10, 7))
colors_bar = plt.cm.Blues(np.linspace(0.4, 1.0, len(top_products)))
ax.barh(top_products.index, top_products.values, color=colors_bar)
ax.set_title('Top 20 Products by Total Demand', fontsize=13, pad=12)
ax.set_xlabel('Total Order Demand')
for i, (idx, val) in enumerate(zip(top_products.index, top_products.values)):
    ax.text(val, i, f' {val:,.0f}', va='center', fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}07_top20_products.png', dpi=150)
plt.close()
print("저장: 07_top20_products.png")


# ============================================================
# 4. 시계열 특성 분석
# ============================================================
print("\n" + "=" * 60)
print("  4. 시계열 특성 분석")
print("=" * 60)

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# 전체 월별 수요로 분해
ts_monthly = (
    df.groupby('YearMonth')['Order_Demand']
    .sum()
)
ts_monthly.index = ts_monthly.index.to_timestamp()

# 시계열 분해 (additive)
decomp = seasonal_decompose(ts_monthly, model='additive', period=12)

fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
components = [
    ('Observed', decomp.observed, '#2563eb'),
    ('Trend', decomp.trend, '#dc2626'),
    ('Seasonal', decomp.seasonal, '#16a34a'),
    ('Residual', decomp.resid, '#6b7280'),
]
for i, (title, data, color) in enumerate(components):
    axes[i].plot(data, color=color, linewidth=1.2)
    axes[i].set_title(title, fontsize=10, loc='left')
    axes[i].set_ylabel('')

plt.suptitle('Time Series Decomposition (Additive, period=12)', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}08_ts_decomposition.png', dpi=150)
plt.close()
print("저장: 08_ts_decomposition.png")

# ADF 정상성 검정
adf_result = adfuller(ts_monthly.dropna(), autolag='AIC')
print(f"\nADF 검정 (전체 월별 수요):")
print(f"  검정통계량: {adf_result[0]:.4f}")
print(f"  p-value: {adf_result[1]:.4f}")
print(f"  사용 lag: {adf_result[2]}")
for key, val in adf_result[4].items():
    print(f"  임계값 ({key}): {val:.4f}")

if adf_result[1] < 0.05:
    print("  -> 귀무가설 기각: 정상 시계열")
else:
    print("  -> 귀무가설 채택 실패: 비정상 시계열 (차분 필요)")

# 1차 차분 후 검정
ts_diff = ts_monthly.diff().dropna()
adf_diff = adfuller(ts_diff, autolag='AIC')
print(f"\nADF 검정 (1차 차분 후):")
print(f"  검정통계량: {adf_diff[0]:.4f}")
print(f"  p-value: {adf_diff[1]:.4f}")
if adf_diff[1] < 0.05:
    print("  -> 1차 차분으로 정상성 확보")
else:
    print("  -> 추가 차분 필요")


# --- ACF / PACF ---
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

plot_acf(ts_monthly.dropna(), lags=30, ax=axes[0, 0], color='#2563eb')
axes[0, 0].set_title('ACF - Original', fontsize=10)

plot_pacf(ts_monthly.dropna(), lags=30, ax=axes[0, 1], color='#2563eb')
axes[0, 1].set_title('PACF - Original', fontsize=10)

plot_acf(ts_diff.dropna(), lags=30, ax=axes[1, 0], color='#dc2626')
axes[1, 0].set_title('ACF - 1st Differenced', fontsize=10)

plot_pacf(ts_diff.dropna(), lags=30, ax=axes[1, 1], color='#dc2626')
axes[1, 1].set_title('PACF - 1st Differenced', fontsize=10)

plt.suptitle('ACF / PACF Analysis', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}09_acf_pacf.png', dpi=150)
plt.close()
print("저장: 09_acf_pacf.png")


# ============================================================
# 5. 월별 집계 데이터 저장 (모델링용)
# ============================================================
print("\n" + "=" * 60)
print("  5. 모델링용 데이터 저장")
print("=" * 60)

# 전체 월별
monthly_agg = (
    df.groupby('YearMonth')
    .agg(
        total_demand=('Order_Demand', 'sum'),
        order_count=('Order_Demand', 'count'),
        avg_demand=('Order_Demand', 'mean'),
        median_demand=('Order_Demand', 'median'),
        std_demand=('Order_Demand', 'std'),
        unique_products=('Product_Code', 'nunique'),
    )
    .reset_index()
)
monthly_agg['YearMonth'] = monthly_agg['YearMonth'].dt.to_timestamp()
monthly_agg.to_csv(f'{DATA_DIR}monthly_total.csv', index=False)
print(f"저장: monthly_total.csv ({len(monthly_agg)}행)")

# 창고 x 월별
monthly_wh_agg = (
    df.groupby(['YearMonth', 'Warehouse'])
    .agg(
        total_demand=('Order_Demand', 'sum'),
        order_count=('Order_Demand', 'count'),
        unique_products=('Product_Code', 'nunique'),
    )
    .reset_index()
)
monthly_wh_agg['YearMonth'] = monthly_wh_agg['YearMonth'].dt.to_timestamp()
monthly_wh_agg.to_csv(f'{DATA_DIR}monthly_by_warehouse.csv', index=False)
print(f"저장: monthly_by_warehouse.csv ({len(monthly_wh_agg)}행)")

# 카테고리 x 월별
monthly_cat_agg = (
    df.groupby(['YearMonth', 'Product_Category'])
    .agg(
        total_demand=('Order_Demand', 'sum'),
        order_count=('Order_Demand', 'count'),
    )
    .reset_index()
)
monthly_cat_agg['YearMonth'] = monthly_cat_agg['YearMonth'].dt.to_timestamp()
monthly_cat_agg.to_csv(f'{DATA_DIR}monthly_by_category.csv', index=False)
print(f"저장: monthly_by_category.csv ({len(monthly_cat_agg)}행)")

print("\n" + "=" * 60)
print("  전처리 및 EDA 완료")
print("=" * 60)
