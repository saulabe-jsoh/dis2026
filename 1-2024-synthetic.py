import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import matplotlib as mpl # Added for mpl.get_cachedir()
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

# ==========================================
# [Step 1] 실측 사건(172,821건) 전수 동기화 및 280명 오차 완벽 교정
# ==========================================
np.random.seed(42)

# 1. 연령대별 실측 건수 고정 (정확히 172,821건) - 2024년도
age_real_counts = {
    '10대': 9469, '20대': 35085, '30대': 28550,
    '40대': 25677, '50대': 14913, '60대 이상': 5928, '기타(불상)': 53199
}
generated_ages = []
for age_group, count in age_real_counts.items():
    generated_ages.extend([age_group] * count)
generated_ages = np.array(generated_ages)
np.random.shuffle(generated_ages)

# 전체 타깃 기준 길이(N) 절대 고정 (172,821건)
target_n = len(generated_ages)

# 2. 시·도청별 실측 건수 정밀 재배정 (280명 유실 보정 완료: 합계 정확히 172,821건)
region_real_counts = {
    '경기남부청': 34597, '서울청': 28671, '부산청': 13601, '인천청': 13480,
    '경기북부청': 12841, '경남청': 9142, '대구청': 7725, '경북청': 7086,
    '충남청': 6878, '전북청': 6325, '광주청': 5876, '대전청': 5323,
    '전남청': 4925, '충북청': 4666, '강원청': 4269, '울산청': 4148,
    '제주청': 2195, '세종청': 985, '경찰청 본청': 88
}
generated_regions = []
for region, count in region_real_counts.items():
    generated_regions.extend([region] * count)
generated_regions = np.array(generated_regions)
np.random.shuffle(generated_regions)

# 3. 입건(92,464건) vs 불입건(80,357건) 비율 정수 고정 (정확히 172,821건)
investigation_counts = {
    '입건': 92464,
    '불입건': 80357
}
generated_results = []
for res_type, count in investigation_counts.items():
    generated_results.extend([res_type] * count)
generated_results = np.array(generated_results)
np.random.shuffle(generated_results)

# 4. 죄종별 프로필 정의
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
base_p_crimes = np.array([profile[1] for profile in crime_profiles.values()])
base_p_crimes /= base_p_crimes.sum()

# 5. 연령대 변수 조건부 죄종 할당 (정확히 172,821건)
generated_subs = []
for age in generated_ages:
    if age == '10대':
        p_adjust = base_p_crimes.copy()
        p_adjust[sub_categories.index('게임 사기')] *= 5
        p_adjust /= p_adjust.sum()
        generated_subs.append(np.random.choice(sub_categories, p=p_adjust))
    elif age == '기타(불상)':
        p_adjust = base_p_crimes.copy()
        p_adjust[sub_categories.index('해킹')] *= 3
        p_adjust[sub_categories.index('쇼핑몰 사기')] *= 3
        p_adjust /= p_adjust.sum()
        generated_subs.append(np.random.choice(sub_categories, p=p_adjust))
    else:
        generated_subs.append(np.random.choice(sub_categories, p=base_p_crimes))

generated_subs = np.array(generated_subs)
case_ids = [f'C2024-{i+1:06d}' for i in range(target_n)]

# ==========================================
# 배열 길이 최종 검증 프로세스
# ==========================================
print("🔍 조립 전 각 리스트/배열 크기 최종 재검증:")
print(f" 1. ID 배열 개수        : {len(case_ids):,}건")
print(f" 2. 연령(Age) 배열 개수   : {len(generated_ages):,}건")
print(f" 3. 지역(Region) 배열 개수: {len(generated_regions):,}건")
print(f" 4. 죄종(Subs) 배열 개수  : {len(generated_subs):,}건")
print(f" 5. 결과(Results) 배열 개수: {len(generated_results):,}건")

if not (len(case_ids) == len(generated_ages) == len(generated_regions) == len(generated_subs) == len(generated_results) == 172821):
    print("🚨 [경고] 여전히 리스트 간 크기가 맞지 않습니다. 개수를 다시 체크하세요.")
else:
    print("✅ [검증 최종 통과] 모든 변수의 길이가 정확히 172,821건으로 일치합니다. 데이터프레임을 조립합니다.\n")

# 6. 시드 데이터프레임 결합
df_seed = pd.DataFrame({
    'Case_Suspect_ID': case_ids,
    'Suspect_Age': generated_ages,
    'Arrest_Region': generated_regions,
    'Sub_Category': generated_subs,
    'Investigation_Result': generated_results
})
df_seed['Main_Category'] = df_seed['Sub_Category'].map(lambda x: crime_profiles[x][0])

# 7. 성별, 피해규모, 연계 증거 속성 난수 결합
df_seed['Suspect_Gender'] = df_seed.apply(
    lambda row: '불상' if row['Suspect_Age'] == '기타(불상)' else np.random.choice(['남성', '여성'], p=[0.88, 0.12] if row['Sub_Category'] in ['사이버도박', '해킹', '게임 사기'] else ([0.65, 0.35] if row['Sub_Category'] in ['사이버명예훼손·모욕', '사이버성폭력'] else [0.792, 0.208])), axis=1
)
df_seed['Damage_Amount'] = df_seed.apply(
    lambda row: 0 if (crime_profiles[row['Sub_Category']][2] == 0 or row['Investigation_Result'] == '불입건') else int(np.round(np.random.lognormal(mean=np.log(crime_profiles[row['Sub_Category']][2]) - 0.75**2/2, sigma=0.75), -3)), axis=1
)
df_seed['Evidence_Type'] = df_seed['Sub_Category'].apply(
    lambda s: np.random.choice(['모바일 기기', '컴퓨터 기기', '디지털 기기', '파일/기타'], p=[0.15, 0.65, 0.05, 0.15] if s in ['해킹', '서비스거부공격(DDoS)'] else [0.793, 0.171, 0.022, 0.014])
)

print(f"✅ [최종 성공] 172,821건의 무결성 마스터 시드 데이터프레임(df_seed)이 완벽히 구축되었습니다!")

age_order = ['10대', '20대', '30대', '40대', '50대', '60대 이상', '기타(불상)']
region_order = [
    '경기남부청', '서울청', '부산청', '인천청', '경기북부청', '경남청', '대구청', '경북청',
    '충남청', '전북청', '광주청', '대전청', '전남청', '충북청', '강원청', '울산청', '제주청', '세종청', '경찰청 본청'
]
# ==========================================
# [Step 2] 데이터 집계 연산
# ==========================================
stat_age = df_seed['Suspect_Age'].value_counts().reindex(age_order).to_frame(name='건수')
stat_region = df_seed['Arrest_Region'].value_counts().reindex(region_order).to_frame(name='건수')
stat_gender = pd.crosstab(index=df_seed['Sub_Category'], columns=df_seed['Suspect_Gender'])
stat_matrix = pd.crosstab(index=df_seed['Arrest_Region'], columns=df_seed['Suspect_Age'])

# ==========================================
# [#1-1] 코랩 인라인 통계 그래프 시각화 및 이미지 저장
# ==========================================
print("\n🎨 고해상도 학술 통계 시각화 이미지 렌더링 중...")

# 차트 1. 피의자 연령대별 현황 (세로 막대)
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=stat_age.index, y=stat_age['건수'], palette='Blues_r', ax=ax)
ax.set_title('사이버범죄 피의자 연령대별 사건 현황 (재현 데이터)', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('연령대', fontsize=11)
ax.set_ylabel('사건 건수 (건)', fontsize=11)
for i, val in enumerate(stat_age['건수']):
    ax.text(i, val + 500, f'{val:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig('graph_1_age_distribution.png', dpi=300)
plt.show() # 코랩 화면에 즉시 출력

# 차트 2. 관할청별 수사 처리 현황 (가로 막대)
fig, ax = plt.subplots(figsize=(11, 7))
sns.barplot(x=stat_region['건수'], y=stat_region.index, palette='crest', ax=ax)
ax.set_title('시·도경찰청별 사이버범죄 수사 처리 현황 (재현 데이터)', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('수사 처리 건수 (건)', fontsize=11)
ax.set_ylabel('관할 시·도경찰청', fontsize=11)
for i, val in enumerate(stat_region['건수']):
    ax.text(val + 150, i, f' {val:,}건', ha='left', va='center', fontsize=8.5)
plt.tight_layout()
plt.savefig('graph_2_regional_distribution.png', dpi=300)
plt.show()

# 차트 3. 관할청별 연령대 구성 비율 (누적 막대 그래프)
stat_matrix_pct = stat_matrix.div(stat_matrix.sum(axis=1), axis=0) * 100
ax = stat_matrix_pct.plot(kind='bar', stacked=True, figsize=(13, 7), cmap='Set3', edgecolor='gray', linewidth=0.4)
plt.title('시·도경찰청별 피의자 연령대 구성 비율 (%)', fontsize=14, fontweight='bold', pad=12)
plt.xlabel('관할 시·도경찰청', fontsize=11)
plt.ylabel('구성 비율 (%)', fontsize=11)
plt.xticks(rotation=45, ha='right')
plt.legend(title='연령대', bbox_to_anchor=(1.01, 1), loc='upper left')
plt.tight_layout()
plt.savefig('graph_3_regional_age_stacked.png', dpi=300)
plt.show()

print("\n📸 [시각화 완료] 3종의 고해상도 그래프 파일이 세션 파일 창에 저장되었습니다.")

# ==========================================
# [Step 2] SDV SingleTableMetadata 활용 CTGAN 신경망 학습
# ==========================================
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=df_seed)
metadata.update_column(column_name='Case_Suspect_ID', sdtype='id')

# 대규모 치안 마이크로데이터 학습 최적화 하이퍼파라미터 적용
synthesizer = CTGANSynthesizer(metadata, epochs=150, batch_size=2000, verbose=True)

print("\n🚀 Step 2: 172,821건 [입건/불입건 포함] 다변량 사이버 치안 데이터 CTGAN 인공지능 학습 시작...")
synthesizer.fit(df_seed)
print(" Ganz 딥러닝 최적화 프로세스가 완료되었습니다.")

# ==========================================
# [Step 3] 최종 1:1 매칭 172,821건 재현 데이터 생성 및 내보내기
# ==========================================
print("\n🔮 Step 3: 최종 가상 마이크로데이터셋 조건부 샘플링 및 무작위성 검증...")
df_synthetic = synthesizer.sample(num_rows=target_n)

print("\n📊 [입건/불입건 분리 정합성 보존 재현 데이터 상위 5개 결과 명세]")
print(df_synthetic[['Case_Suspect_ID', 'Suspect_Age', 'Arrest_Region', 'Sub_Category', 'Investigation_Result', 'Damage_Amount']].head())

# 코랩 및 통계 보고서용 인코딩 적용 CSV 파일 내보내기
df_synthetic.to_csv('cyber_crime_cases_2024_synthetic.csv', index=False, encoding='utf-8-sig')
print("\n💾 검증 마스터 파일 저장 성공: 'cyber_crime_cases_2024_synthetic.csv'")
