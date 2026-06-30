!pip install sdv pandas numpy

import numpy as np
import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

# ==========================================
# [Step 1] 통계연보 기반 가상 로우 데이터 추출 (Seed Data)
# ==========================================
# 2020~2024년 통계연보의 평균적 분포 비율을 기반으로 가상의 마이크로데이터 추출
np.random.seed(42)
num_samples_cyber = 5000

# 2. **실제 데이터 적용 시:**
#   * 위 코드는 통계연보의 비율을 반영한 가상 로우 데이터를 자동 생성(`Step 1`)하도록 설계되어 있습니다.
#   * 만약 통계청이나 경찰청에서 확보한 **실제 행(Row) 단위 마이크로데이터**가 있다면, `Step 1` 부분을 생성과정 없이 `df_cyber = pd.read_csv("실제데이터.csv")` 형태로 바로 로드하여 사용하시면 됩니다.
# 3. **학습 속도 및 고도화:**
#   * 데이터의 차원이 복잡하고 범주형 변수의 종류가 많을수록 `epochs` 숫자를 300 이상으로 늘려야 원본 통계와 재현데이터 간의 오차율(K-S 검정 점수 등)이 낮아집니다.
#   * GPU 환경(CUDA)이 지원된다면 CTGAN은 자동으로 GPU를 사용하여 연산 속도가 빨라집니다.

# 범주형 변수 정의
crime_types = ['정보통신망 침해범죄(해킹/디도스)', '정보통신망 이용범죄(사이버사기/피싱)', '불법콘텐츠범죄(사이버음란물/도박)']
age_groups = ['10대 이하', '20대', '30대', '40대', '50대 이상']

# 통계연보 기준 가중치 확률 (사이버 사기/이용 범죄가 압도적으로 높고, 10~30대 집중 분포)
crime_probs = [0.10, 0.75, 0.15]
age_probs = [0.25, 0.40, 0.20, 0.10, 0.05]

# 가중치를 적용한 마이크로데이터 추출(생성)
raw_cyber_data = {
    'Year': np.random.choice([2020, 2021, 2022, 2023, 2024], size=num_samples_cyber),
    'Crime_Type': np.random.choice(crime_types, size=num_samples_cyber, p=crime_probs),
    'Age_Group': np.random.choice(age_groups, size=num_samples_cyber, p=age_probs),
    'Damage_Amount_KB': np.random.exponential(scale=5000, size=num_samples_cyber).astype(int) # 피해액(연속형 변수 예시)
}

df_cyber = pd.DataFrame(raw_cyber_data)
print("--- 추출된 사이버 범죄 원본 데이터셋 샘플 ---")
print(df_cyber.head())

# ==========================================
# [Step 2] SDV 메타데이터 생성 및 CTGAN 설정
# ==========================================
metadata_cyber = SingleTableMetadata()
metadata_cyber.detect_from_dataframe(data=df_cyber)

# 변수 속성 미세 조정 (Year는 숫자가 아닌 범주형/유형으로 취급되도록 지정 가능)
metadata_cyber.update_column(column_name='Year', sdtype='categorical')

# CTGAN 모델 선언 및 학습
# epochs는 실증 연구 시 300~500회 이상을 추천합니다.
synthesizer_cyber = CTGANSynthesizer(metadata_cyber, epochs=200, batch_size=500)
print("\n[사이버 범죄] CTGAN 모델 학습 시작...")
synthesizer_cyber.fit(df_cyber)
print("[사이버 범죄] CTGAN 모델 학습 완료.")

# ==========================================
# [Step 3] 재현데이터 생성 및 저장
# ==========================================
synthetic_cyber_data = synthesizer_cyber.sample(num_rows=5000)
print("\n--- CTGAN으로 생성된 사이버 범죄 재현데이터 샘플 ---")
print(synthetic_cyber_data.head())

# CSV 저장 (향후 통계 유사성 평가용)
synthetic_cyber_data.to_csv("synthetic_cyber_crime_2020_2024.csv", index=False, encoding='utf-8-sig')
