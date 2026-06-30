import numpy as np
import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

# ==========================================
# [Step 1] 통계연보 기반 92,464명 가상 원본(Seed Data) 구축
# ==========================================
np.random.seed(42)
target_n = 92464

# 20개 세부 죄종 및 통계 기반 발생 비중 정의
crime_profiles = {
    '해킹': ['정보통신망 침해', 0.00888, 0],
    '악성프로그램': ['정보통신망 침해', 0.00115, 0],
    '서비스거부공격(DDoS)': ['정보통신망 침해', 0.00012, 0],
    '기타 침해': ['정보통신망 침해', 0.00424, 0],
    '직거래 사기': ['정보통신망 이용', 0.52875, 2260874],
    '게임 사기': ['정보통신망 이용', 0.08788, 4093440],
    '기타 사이버사기': ['정보통신망 이용', 0.07686, 361283],
    '사이버저작권침해': ['정보통신망 이용', 0.11572, 0],
    '메신저피싱': ['정보통신망 이용', 0.02606, 46822738],
    '스미싱': ['정보통신망 이용', 0.01339, 12425386],
    '이메일 무역사기': ['정보통신망 이용', 0.02891, 10548859],
    '기타 정보탈취': ['정보통신망 이용', 0.00407, 5926811],
    '몸캠피싱': ['정보통신망 이용', 0.00096, 22129337],
    '개인위치정보침해': ['정보통신망 이용', 0.00120, 0],
    '쇼핑몰 사기': ['정보통신망 이용', 0.00072, 14249782178],
    '기타 이용': ['정보통신망 이용', 0.01195, 0],
    '사이버명예훼손·모욕': ['불법콘텐츠', 0.05734, 0],
    '사이버도박': ['불법콘텐츠', 0.01403, 0],
    '사이버성폭력': ['불법콘텐츠', 0.01138, 0],
    '기타 불법콘텐츠': ['불법콘텐츠', 0.00322, 0]
}

sub_categories = list(crime_profiles.keys())
probabilities = [profile[1] for profile in crime_profiles.values()]
probabilities = np.array(probabilities) / np.sum(probabilities) # 확률 정규화

# 1. 죄종 분배 생성
generated_subs = np.random.choice(sub_categories, size=target_n, p=probabilities)

df_seed = pd.DataFrame({
    'Suspect_ID': [f'S2024-{i+1:05d}' for i in range(target_n)],
    'Sub_Category': generated_subs
})
df_seed['Main_Category'] = df_seed['Sub_Category'].map(lambda x: crime_profiles[x][0])

# 2. 연보 332페이지 기반 인원수 성별/연령대 조건부 생성 규칙 적용
def 피의자_성별_매핑(sub):
    # 일반적인 사이버 성별 분포 적용 (남성 79.2%, 여성 20.8%)
    # 성별 도메인 지식 고도화: 도박/해킹은 남성 비중을 상향 조정, 명예훼손은 여성 비중 상향
    if sub in ['사이버도박', '해킹', '게임 사기']:
        return np.random.choice(['남성', '여성'], p=[0.88, 0.12])
    elif sub in ['사이버명예훼손·모욕', '사이버성폭력']:
        return np.random.choice(['남성', '여성'], p=[0.65, 0.35])
    return np.random.choice(['남성', '여성'], p=[0.792, 0.208])

def 피의자_연령_매핑(sub):
    ages = ['14세 미만', '14~19세', '20대', '30대', '40대', '50대', '60세 이상']
    # 332페이지 전국 평균 비율 기초선
    p_ages = [0.023, 0.145, 0.338, 0.261, 0.129, 0.071, 0.033]

    # 죄종별 연령 편향 도메인 지식 주입 (예: 게임사기는 10대 이하 증가, 이메일무역사기는 40대 이상 증가)
    if sub == '게임 사기':
        p_ages = [0.100, 0.450, 0.300, 0.100, 0.030, 0.015, 0.005]
    elif sub == '이메일 무역사기':
        p_ages = [0.000, 0.020, 0.150, 0.350, 0.300, 0.130, 0.050]

    return np.random.choice(ages, p=p_ages)

df_seed['Suspect_Gender'] = df_seed['Sub_Category'].apply(피의자_성별_매핑)
df_seed['Suspect_Age'] = df_seed['Sub_Category'].apply(피의자_연령_매핑)

# 3. 피해 금액 및 포렌식 증거 연계 생성
def 피해금액_매핑(sub):
    mean_val = crime_profiles[sub][2]
    if mean_val == 0: return 0
    sigma = 0.75
    mu = np.log(mean_val) - (sigma ** 2) / 2
    return int(np.round(np.random.lognormal(mean=mu, sigma=sigma), -3))

def 증거유형_매핑(sub):
    if sub in ['메신저피싱', '스미싱', '몸캠피싱', '직거래 사기']:
        return np.random.choice(['모바일 기기', '컴퓨터 기기', '디지털 기기', '파일/기타'], p=[0.92, 0.05, 0.02, 0.01])
    elif sub in ['해킹', '서비스거부공격(DDoS)', '악성프로그램']:
        return np.random.choice(['모바일 기기', '컴퓨터 기기', '디지털 기기', '파일/기타'], p=[0.15, 0.65, 0.05, 0.15])
    return np.random.choice(['모바일 기기', '컴퓨터 기기', '디지털 기기', '파일/기타'], p=[0.793, 0.171, 0.022, 0.014])

df_seed['Damage_Amount'] = df_seed['Sub_Category'].apply(피해금액_매핑)
df_seed['Evidence_Type'] = df_seed['Sub_Category'].apply(증거유형_매핑)

print(f"✅ Step 1: 가상 원본 데이터셋 구축 완료 (크기: {df_seed.shape})")

# ==========================================
# [Step 2] SDV 차세대 패키지 기반 CTGAN 인공지능 학습
# ==========================================
# 메타데이터 자동 감지 및 범주형 선언
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=df_seed)
metadata.update_column(column_name='Suspect_ID', sdtype='id')

# CTGAN 신경망 선언 (9만 건 학습용 파라미터 튜닝)
# 데이터 사이즈가 크므로 대형 배치(batch_size=1000) 및 효율적 에포크(epochs=150) 설정
synthesizer = CTGANSynthesizer(metadata, epochs=150, batch_size=1000, verbose=True)

print("\n🚀 Step 2: 92,464명 데이터셋 기반 CTGAN 모델 딥러닝 학습 시작...")
synthesizer.fit(df_seed)
print(" Ganz 학습 완료.")

# ==========================================
# [Step 3] 최종 92,464명의 재현 데이터 생성 및 파일 저장
# ==========================================
print("\n🔮 Step 3: 최종 개인정보 전처리 및 재현 데이터 생성 중...")
df_synthetic = synthesizer.sample(num_rows=target_n)

# 결과물 확인 및 저장
print("\n📊 [최종 결과] 생성된 92,464명 재현 데이터 상위 5개 행 명세:")
print(df_synthetic.head())

df_synthetic.to_csv('cyber_crime_92464_synthetic.csv', index=False, encoding='utf-8-sig')
print("\n💾 파일 저장 완료: 'cyber_crime_92464_synthetic.csv'")
