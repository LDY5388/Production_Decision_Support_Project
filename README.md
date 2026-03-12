# 제조업 수요 예측 및 안전재고 시뮬레이션

## 프로젝트 개요

제조업 환경에서 제품별/창고별 수요를 예측하고, 예측 불확실성을 반영한 안전재고 수준을 시뮬레이션하여 생산 의사결정을 지원하는 프로젝트.

양산 현장에서 생산 계획을 수립할 때, "다음 달에 얼마나 만들어야 하는가?"에 대한 통계적 근거를 제공하는 것이 목표다.


## 프로젝트 구조

```
demand-forecast-project/
├── data/                  # 원본 데이터 및 전처리 데이터
├── notebooks/             # 분석 노트북 (EDA, 모델링 등)
├── src/                   # 재사용 가능한 함수 모듈
├── outputs/               # 분석 결과물 (그래프, 리포트 등)
├── dashboard/             # Streamlit 대시보드
├── README.md
├── requirements.txt
└── .gitignore
```


## 데이터

- 출처: Kaggle - Forecasts for Product Demand (https://www.kaggle.com/datasets/felixzhao/productdemandforecasting)
- 기간: 2011-01 ~ 2017-01 (약 6년)
- 규모: 1,048,575행 x 5열
- 컬럼: Product_Code, Warehouse, Product_Category, Date, Order_Demand
- 제품 수: 2,160개 / 창고: 4개 / 카테고리: 33개


## 분석 흐름

1. 전처리 및 EDA
   - 결측치 처리 (날짜 11,239건, 음수 주문 10,469건)
   - 월별 집계 및 시계열 구조 변환
   - 창고별/카테고리별 수요 패턴 탐색
   - 계절성, 추세, 정상성 검정

2. 수요 예측 모델링
   - ARIMA / SARIMA
   - 지수평활법 (Holt-Winters)
   - Prophet
   - 모델별 성능 비교 (RMSE, MAPE, 예측 구간 커버리지)

3. 안전재고 시뮬레이션
   - 예측 오차 분포 기반 안전재고 산출
   - 몬테카를로 시뮬레이션으로 서비스 수준별 재고 수준 도출
   - 과잉 생산 비용 vs 품절 비용 트레이드오프 분석

4. 대시보드
   - Streamlit 기반 생산 의사결정 지원 대시보드
   - 제품별 수요 예측 결과 및 신뢰구간 시각화
   - 안전재고 추천 및 시나리오 비교


## EDA 주요 발견

전처리 결과:
- 원본 1,048,575행에서 날짜 결측(11,239건), 음수 주문(5,899건), 0 주문(28,672건) 제거
- 최종 분석 대상: 1,002,765행 (2011-01 ~ 2017-01)

데이터 특성:
- 2011년은 데이터가 거의 없고, 2017년은 1월만 존재. 모델링 시 2012~2016년 사용 권장.
- Whse_J가 전체 수요의 65.5%를 차지. 반도체 양산 관점에서 메인 FAB에 해당.
- 주문량 분포가 극심하게 right-skewed (평균 5,105 vs 중앙값 300). 로그 변환 고려.
- Whse_C는 뚜렷한 상승 추세, Whse_S는 2016년부터 하락 추세.

시계열 특성:
- ADF 검정 결과 원본 시계열은 비정상(p=0.0599), 1차 차분 후 정상성 확보(p<0.001).
- 12개월 주기 계절성 존재하나 강하지 않음. 3월, 10월에 약한 피크.
- ARIMA 기반 모델링 시 d=1 사용, 계절 차분도 검토 필요.
- ACF가 서서히 감소하는 패턴 -> AR 성분 존재.


## 진행 기록

| 날짜 | 작업 내용 | 비고 |
|------|----------|------|
| 2026-03-12 | 프로젝트 구조 설정, README 작성 | 초기 세팅 |
| 2026-03-12 | 데이터 전처리 및 EDA 완료 | 시각화 9개, 집계 데이터 3개 생성 |


## 기술 스택

- Python (pandas, numpy, scipy, statsmodels, prophet, scikit-learn)
- 시각화: plotly, matplotlib
- 대시보드: Streamlit
- 데이터 처리: SQL 스타일 쿼리 (pandas)


## 실행 방법

```bash
pip install -r requirements.txt

# EDA 노트북 실행
jupyter notebook notebooks/01_eda.ipynb

# 대시보드 실행
streamlit run dashboard/app.py
```
