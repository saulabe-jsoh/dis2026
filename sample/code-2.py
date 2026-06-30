#!pip install sdv pandas numpy

import numpy as np
import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

# ==========================================
# [Step 1] 통계연보 기반 가상 로우 데이터 추출 (Seed Data)
# ==========================================
np.random.seed(777)
num_samples_domestic = 5000

# 2. **실제 데이터 적용 시:**
#   * 위 코드는 통계연보의 비율을 반영한 가상 로우 데이터를 자동 생성(`Step 1`)하도록 설계되어 있습니다.
#   * 만약 통계청이나 경찰청에서 확보한 **실제 행(Row) 단위 마이크로데이터**가 있다면, `Step 1` 부분을 생성과정 없이 `df_cyber = pd.read_csv("실제데이터.csv")` 형태로 바로 로드하여 사용하시면 됩니다.
# 3. **학습 속도 및 고도화:**
#   * 데이터의 차원이 복잡하고 범주형 변수의 종류가 많을수록 `epochs` 숫자를 300 이상으로 늘려야 원본 통계와 재현데이터 간의 오차율(K-S 검정 점수 등)이 낮아집니다.
#   * GPU 환경(CUDA)이 지원된다면 CTGAN은 자동으로 GPU를 사용하여 연산 속도가 빨라집니다.

# 범주형 변수 정의
fv_types = ['폭행·상해', '체포·감금·협박', '재물손괴', '기타(주거침입 등)']
relationship_types = ['배우자(사실혼 포함)', '직계존비속', '기타 친족']
reoffense_yn = ['초범', '재범']

# 통계연보 기준 가중치 확률 (배우자 간 폭행·상해 비율이 지배적임)
fv_probs = [0.70, 0.15, 0.10, 0.05]
rel_probs = [0.82, 0.15, 0.03]
re_probs = [0.85, 0.15]

# 가중치를 적용한 마이크로데이터 추출(생성)
raw_domestic_data = {
    'Year': np.random.choice([2020, 2021, 2022, 2023, 2024], size=num_samples_domestic),
    'Violence_Type': np.random.choice(fv_types, size=num_samples_domestic, p=fv_probs),
    'Relationship': np.random.choice(relationship_types, size=num_samples_domestic, p=rel_probs),
    'Reoffense': np.random.choice(reoffense_yn, size=num_samples_domestic, p=re_probs),
    'Police_Dispatch_Time_Min': np.random.normal(loc=15, scale=5, size=num_samples_domestic).clip(3, 60).astype(int) # 출동시간
}

df_domestic = pd.DataFrame(raw_domestic_data)
print("--- 추출된 가정폭력 범죄 원본 데이터셋 샘플 ---")
print(df_domestic.head())

# ==========================================
# [Step 2] SDV 메타데이터 생성 및 CTGAN 설정
# ==========================================
metadata_domestic = SingleTableMetadata()
metadata_domestic.detect_from_dataframe(data=df_domestic)

# 변수 속성 지정
metadata_domestic.update_column(column_name='Year', sdtype='categorical')

# CTGAN 모델 선언 및 학습
synthesizer_domestic = CTGANSynthesizer(metadata_domestic, epochs=200, batch_size=500)
print("\n[가정폭력 범죄] CTGAN 모델 학습 시작...")
synthesizer_domestic.fit(df_domestic)
print("[가정폭력 범죄] CTGAN 모델 학습 완료.")

# ==========================================
# [Step 3] 재현데이터 생성 및 저장
# ==========================================
synthetic_domestic_data = synthesizer_domestic.sample(num_rows=5000)
print("\n--- CTGAN으로 생성된 가정폭력 범죄 재현데이터 샘플 ---")
print(synthetic_domestic_data.head())

# CSV 저장
synthetic_domestic_data.to_csv("synthetic_domestic_violence_2020_2024.csv", index=False, encoding='utf-8-sig')
