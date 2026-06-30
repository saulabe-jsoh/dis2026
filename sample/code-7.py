import os
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import matplotlib as mpl # Added for mpl.get_cachedir()

# ==========================================
# [구글 코랩 전용] 리눅스 서버 한글 폰트 강제 설치 및 자동 설정
# ==========================================
def setup_colab_korean():
    print("⏳ [코랩 환경 폰트 점검] 나눔 폰트 설치 및 환경 구성을 시작합니다...")

    # 1. 나눔 폰트 설치 확인 및 실행
    if not os.path.exists('/usr/share/fonts/truetype/nanum/NanumGothic.ttf'):
        print("📦 시스템에 한글 폰트가 없습니다. 나눔 폰트(Nanum) 설치 중...")
        # apt-get 명령어를 파이썬 내부에서 실행
        subprocess.run(["sudo", "apt-get", "update", "-qq"])
        subprocess.run(["sudo", "apt-get", "install", "-y", "-qq", "fonts-nanum"])

        # matplotlib 캐시 디렉터리 삭제 (폰트 재인식을 위함)
        import shutil
        # cache_dir = plt.rcParams['text.latex.preview'] # 임시 참조용이 아닌 캐시 경로 강제 타깃팅
        matplotlib_cache = mpl.get_cachedir() # Use mpl.get_cachedir() for robustness
        # matplotlib_cache = os.path.expanduser('~/.cache/matplotlib') # Fallback if get_cachedir fails, but usually not needed
        if os.path.exists(matplotlib_cache):
            shutil.rmtree(matplotlib_cache)
            print("🧹 Matplotlib 이전 폰트 캐시를 성공적으로 제거했습니다.")

    # 2. 설치된 나눔고딕 폰트를 matplotlib 시스템에 등록
    font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    font_prop = fm.FontProperties(fname=font_path)

    # 3. 전역 파라미터(RcParams) 변경
    plt.rcParams['font.family'] = 'NanumGothic'
    plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지

    # 4. Seaborn 테마에 나눔고딕 강제 바인딩
    sns.set_theme(style="whitegrid", font='NanumGothic', palette="muted")

    # 잘 매핑되었는지 폰트 매니저 리프레시
    fm._rebuild() if hasattr(fm, '_rebuild') else print("🔄 폰트 매니저 동기화 완료.")
    print("✅ 한글 깨짐 방지 패치가 성공적으로 적용되었습니다. (NanumGothic)\n")

# 코랩 폰트 환경 구축 실행
setup_colab_korean()


# ==========================================
# [Step 1] 재현 데이터셋 로드 (코랩 파일 시스템 기준)
# ==========================================
# ※ 주의: 좌측 파일 메뉴에 'cyber_crime_cases_172821_synthetic.csv' 파일이 업로드되어 있어야 합니다.
file_path = 'cyber_crime_cases_172821_synthetic.csv'

if not os.path.exists(file_path):
    print(f"❌ [오류] 코랩 세션에 '{file_path}' 파일이 존재하지 않습니다.")
    print("💡 해결방법: 코랩 왼쪽 [폴더 아이콘(파일)] 클릭 -> [업로드] 버튼을 통해 파일을 먼저 올려주세요.")
else:
    df_synthetic = pd.read_csv(file_path, encoding='utf-8-sig')
    print(f"✅ 재현 데이터 로드 완료! (총 {df_synthetic.shape[0]:,}행 구성)")

    # 경찰통계연보 정렬 순서 고정
    age_order = ['10대', '20대', '30대', '40대', '50대', '60대 이상', '기타(불상)']
    region_order = [
        '경기남부청', '서울청', '부산청', '인천청', '경기북부청', '경남청', '대구청', '경북청',
        '충남청', '전북청', '광주청', '대전청', '전남청', '충북청', '강원청', '울산청', '제주청', '세종청', '경찰청 본청'
    ]
    df_synthetic['Suspect_Age'] = pd.Categorical(df_synthetic['Suspect_Age'], categories=age_order, ordered=True)
    df_synthetic['Arrest_Region'] = pd.Categorical(df_synthetic['Arrest_Region'], categories=region_order, ordered=True)

    # ==========================================
    # [Step 2] 데이터 집계 연산
    # ==========================================
    stat_age = df_synthetic['Suspect_Age'].value_counts().reindex(age_order).to_frame(name='건수')
    stat_region = df_synthetic['Arrest_Region'].value_counts().reindex(region_order).to_frame(name='건수')
    stat_gender = pd.crosstab(index=df_synthetic['Main_Category'], columns=df_synthetic['Suspect_Gender'])
    stat_matrix = pd.crosstab(index=df_synthetic['Arrest_Region'], columns=df_synthetic['Suspect_Age'])

    # ==========================================
    # [Step 3] 코랩 인라인 통계 그래프 시각화 및 이미지 저장
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
    # [Step 4] Excel 보고서 생성
    # ==========================================
    excel_filename = 'cyber_crime_synthetic_summary_report.xlsx'
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        stat_age.to_excel(writer, sheet_name='1_연령대별_현황')
        stat_region.to_excel(writer, sheet_name='2_시도청별_현황')
        stat_gender.to_excel(writer, sheet_name='3_죄종별_성별_현황')
        stat_matrix.to_excel(writer, sheet_name='4_관할청_연령대_매트릭스')

    print(f"💾 [엑셀 출력 완료] 논문 교차표 작성용 통합 문서 저장 완료: '{excel_filename}'")
