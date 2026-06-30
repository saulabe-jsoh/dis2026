import numpy as np
import pandas as pd

# 1. 시드 고정 (재현성 보장)
np.random.seed(42)

# 2. 시뮬레이션할 총 샘플 데이터 수 설정 (예: 50,000건)
num_samples = 50000

# 3. 데이터 통계 파라미터 정의
# 세부 죄종별 [대분류, 발생 비중, 검거 확률, 건당 평균 피해액(원)]
crime_profiles = {
    '해킹': ['정보통신망 침해', 0.00888, 0.1886, 0],
    '악성프로그램': ['정보통신망 침해', 0.00115, 0.1025, 0],
    '서비스거부공격(DDoS)': ['정보통신망 침해', 0.00012, 0.1026, 0],
    '기타 침해': ['정보통신망 침해', 0.00424, 0.3131, 0],
    '사이버사기': ['정보통신망 이용', 0.72630, 0.5170, 0],  # 세부 사기 유형으로 파생 예정
    '사이버저작권침해': ['정보통신망 이용', 0.11572, 0.6711, 0],
    '정보탈취연계형범죄': ['정보통신망 이용', 0.04447, 0.1999, 0],  # 세부 피싱 유형으로 파생 예정
    '개인위치정보침해': ['정보통신망 이용', 0.00120, 0.3910, 0],
    '기타 이용': ['정보통신망 이용', 0.01195, 0.4800, 0],
    '사이버명예훼손·모욕': ['불법콘텐츠', 0.05734, 0.8059, 0],
    '사이버도박': ['불법콘텐츠', 0.01403, 0.9304, 0],
    '사이버성폭력': ['불법콘텐츠', 0.01138, 0.7136, 0],
    '기타 불법콘텐츠': ['불법콘텐츠', 0.00322, 0.6677, 0]
}

# 금융 피해 데이터가 존재하는 세부 죄종별 평균 피해액 (단위: 원)
damage_profiles = {
    '메신저피싱': 46822738,
    '스미싱': 12425386,
    '몸캠피싱': 22129337,
    '기타 정보탈취': 5926811,
    '직거래 사기': 2260874,
    '쇼핑몰 사기': 14249782178,
    '게임 사기': 4093440,
    '이메일 무역사기': 10548859,
    '기타 사이버사기': 361283
}

# 4. 1단계: 세부 죄종(Sub_Category) 확률적 생성
sub_categories = list(crime_profiles.keys())
probabilities = [profile[1] for profile in crime_profiles.values()]
# 확률 합계를 1로 맞추기 위한 정규화
probabilities = np.array(probabilities) / np.sum(probabilities)

generated_subs = np.random.choice(sub_categories, size=num_samples, p=probabilities)

# 5. 데이터 프레임 초기 생성
df = pd.DataFrame({
    'Crime_ID': [f'C2024-{i+1:05d}' for i in range(num_samples)],
    'Sub_Category': generated_subs
})

# 대분류(Main_Category) 매핑
df['Main_Category'] = df['Sub_Category'].map(lambda x: crime_profiles[x][0])

# 6. 2단계: 피싱 및 사기 범죄 세부 분기 (조건부 생성)
def 세부분기_적용(row):
    sub = row['Sub_Category']
    if sub == '정보탈취연계형범죄':
        # 메신저피싱, 스미싱, 몸캠피싱, 기타정보탈취 비중 분할 (8558:4396:317:1339)
        return np.random.choice(['메신저피싱', '스미싱', '몸캠피싱', '기타 정보탈취'], p=[0.586, 0.301, 0.022, 0.091])
    elif sub == '사이버사기':
        # 직거래, 쇼핑몰, 게임, 이메일, 기타 비중 분할 (147738:202:24561:9077:21479)
        return np.random.choice(['직거래 사기', '쇼핑몰 사기', '게임 사기', '이메일 무역사기', '기타 사이버사기'],
                                p=[0.728, 0.001, 0.121, 0.045, 0.105])
    return sub

df['Sub_Category'] = df.apply(세부분기_적용, axis=1)

# 7. 3단계: 죄종별 검거 여부(Arrest_Status) 생성 (이항분포 적용)
def 검거여부_생성(sub_cat):
    # 세부 분기된 죄종의 원본 대분류 키를 찾아서 검거 확률 적용
    origin_key = sub_cat
    if sub_cat in ['메신저피싱', '스미싱', '몸캠피싱', '기타 정보탈취']:
        origin_key = '정보탈취연계형범죄'
    elif sub_cat in ['직거래 사기', '쇼핑몰 사기', '게임 사기', '이메일 무역사기', '기타 사이버사기']:
        origin_key = '사이버사기'

    p_arrest = crime_profiles[origin_key][2]
    return np.random.binomial(1, p_arrest)

df['Arrest_Status'] = df['Sub_Category'].apply(검거여부_생성)

# 8. 4단계: 피해 금액(Damage_Amount) 생성 (로그정규분포 적용)
def 피해금액_생성(sub_cat):
    if sub_cat in damage_profiles:
        mean_val = damage_profiles[sub_cat]
        if mean_val == 0:
            return 0
        # 로그정규분포 파라미터 (mu, sigma) 계산 -> 편차는 평균의 0.8배로 가정하여 비대칭 꼬리 구현
        sigma = 0.8
        mu = np.log(mean_val) - (sigma ** 2) / 2
        val = np.random.lognormal(mean = mu, sigma = sigma)
        return int(np.round(val, -3)) # 천원 단위 절사
    return 0

df['Damage_Amount'] = df['Sub_Category'].apply(피해금액_생성)

# 9. 5단계: 디지털 증거 유형(Evidence_Type) 조건부 생성
def 증거유형_생성(sub_cat):
    # 디바이스별 기본 확률: 모바일(79.33%), 컴퓨터(17.11%), 디지털기기(2.14%), 파일/기타(1.41%)
    if sub_cat in ['메신저피싱', '스미싱', '몸캠피싱', '직거래 사기']:
        # 모바일 비중이 압도적으로 높은 죄종 조정
        return np.random.choice(['모바일 기기', '컴퓨터 기기', '디지털 기기', '파일/기타'], p=[0.92, 0.05, 0.02, 0.01])
    elif sub_cat in ['해킹', '서비스거부공격(DDoS)', '악성프로그램', '이메일 무역사기']:
        # PC 및 시스템 파일 분석 비중이 높은 죄종 조정
        return np.random.choice(['모바일 기기', '컴퓨터 기기', '디지털 기기', '파일/기타'], p=[0.20, 0.60, 0.05, 0.15])
    else:
        # 일반 기본 통계 분포 적용
        return np.random.choice(['모바일 기기', '컴퓨터 기기', '디지털 기기', '파일/기타'], p=[0.7933, 0.1711, 0.0214, 0.0142])

df['Evidence_Type'] = df['Sub_Category'].apply(증거유형_생성)

# 10. 생성된 재현 데이터 결과 확인
print("--- 생성된 재현 데이터 요약 (상위 5개 행) ---")
print(df.head())

print("\n--- 죄종별 데이터 통계 검증 ---")
summary = df.groupby('Sub_Category').agg(
    사건수=('Crime_ID', 'count'),
    검거율=('Arrest_Status', 'mean'),
    평균피해액_원=('Damage_Amount', 'mean')
).round(3)
print(summary)

# CSV 파일로 저장하고 싶을 경우 주석 해제
# df.to_csv('cyber_crime_synthetic_data.csv', index=False, encoding='utf-8-sig')
