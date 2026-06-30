# =========================================================================
# [코랩 필수] 리눅스 환경 나눔 한글 폰트 설치
# =========================================================================
#!apt-get install -y fonts-nanum

import json
import re
import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# SDV 라이브러리 임포트
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer

# =========================================================================
# [환경설정] 구글 코랩(Colab) 전용 한글 폰트 및 테마 설정
# =========================================================================
colab_font_path = "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf"
fm.fontManager.addfont(colab_font_path)

plt.rc("font", family="NanumBarunGothic")
plt.rc("axes", unicode_minus=False)
sns.set_theme(style="whitegrid", font="NanumBarunGothic")


# ==========================================
# 1단계: 5개 연도별 통계 파일 로드 및 통합
# ==========================================
def load_and_combine_stats(file_paths):
    combined_stats = {}
    for path in file_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            content_clean = re.sub(r"#.*", "", content)
            content_clean = content_clean.replace("(", "").replace(")", "")

            file_data = json.loads(content_clean)
            combined_stats.update(file_data)
            print(f"📖 파일 로드 완료: {path}")
        except Exception as e:
            print(f"⚠️ 파일 로드 실패 ({path}): {e}")
    return combined_stats


# ==========================================
# 2단계: 역사적 베이스라인 데이터 복원 (실제 통계 비율 반영)
# ==========================================
def generate_historical_baseline(stats_data, scale_factor=0.02):
    np.random.seed(42)
    all_year_dfs = []

    for year, year_data in stats_data.items():
        if "전국" not in year_data:
            continue

        actual_total = year_data["전국"]["발생건수"]
        total_rows = int(actual_total * scale_factor)

        # 시도청별 가중치
        regions = list(year_data["시도청별"].keys())
        region_counts = [
            year_data["시도청별"][r]["발생건수"] for r in regions
        ]
        region_probs = np.array(region_counts) / sum(region_counts)

        # 범죄 유형 가중치
        crime_main_list, crime_counts = [], []
        for main_cat, main_data in year_data["전국"]["범죄유형"].items():
            crime_main_list.append(main_cat)
            crime_counts.append(main_data["발생건수"])
        crime_probs = np.array(crime_counts) / sum(crime_counts)

        # 피의자 연령대 가중치
        ages = list(year_data["피의자연령"].keys())
        age_counts = [year_data["피의자연령"][a] for a in ages]
        age_probs = np.array(age_counts) / sum(age_counts)

        # 무작위 비례 추출
        sim_regions = np.random.choice(
            regions, size=total_rows, p=region_probs
        )
        crime_indices = np.random.choice(
            len(crime_counts), size=total_rows, p=crime_probs
        )
        sim_main_crimes = np.array(crime_main_list)[crime_indices]
        sim_ages = np.random.choice(ages, size=total_rows, p=age_probs)

        year_df = pd.DataFrame(
            {
                "연도": [str(year)] * total_rows,
                "시도청": sim_regions,
                "범죄대유형": sim_main_crimes,
                "피의자연령대": sim_ages,
            }
        )
        all_year_dfs.append(year_df)

    return pd.concat(all_year_dfs, ignore_index=True)


# ==========================================
# 🚀 메인 실행 및 예측/통계연보 시각화 루틴
# ==========================================
if __name__ == "__main__":
    target_files = [
        "2020-data.txt",
        "2021-data.txt",
        "2022-data.txt",
        "2023-data.txt",
        "2024-data.txt",
    ]
    stats_dict = load_and_combine_stats(target_files)

    SCALE_FACTOR = 0.02

    print("\n--- 1단계: 2020-2024 베이스라인 데이터셋 리샘플링 ---")
    df_historical = generate_historical_baseline(
        stats_dict, scale_factor=SCALE_FACTOR
    )

    print("\n--- 2단계: 선형 회귀 모형 기반 2025년 데이터 추정 및 합성 ---")
    historical_years = [2020, 2021, 2022, 2023, 2024]
    historical_counts = [
        stats_dict[str(y)]["전국"]["발생건수"] for y in historical_years
    ]

    slope, intercept = np.polyfit(historical_years, historical_counts, 1)
    predicted_2025_actual = int(slope * 2025 + intercept)
    predicted_2025_sample = int(predicted_2025_actual * SCALE_FACTOR)

    # 2025년 내부 분포는 2024년 데이터 비율 준용
    data_2024 = stats_dict["2024"]
    regions_25 = list(data_2024["시도청별"].keys())
    region_probs_25 = np.array(
        [data_2024["시도청별"][r]["발생건수"] for r in regions_25]
    ) / sum([data_2024["시도청별"][r]["발생건수"] for r in regions_25])

    crime_main_25, crime_counts_25 = [], []
    for main_cat, main_data in data_2024["전국"]["범죄유형"].items():
        crime_main_25.append(main_cat)
        crime_counts_25.append(main_data["발생건수"])
    crime_probs_25 = np.array(crime_counts_25) / sum(crime_counts_25)

    ages_25 = list(data_2024["피의자연령"].keys())
    age_probs_25 = np.array(
        [data_2024["피의자연령"][a] for a in ages_25]
    ) / sum([data_2024["피의자연령"][a] for a in ages_25])

    np.random.seed(42)
    sim_regions_25 = np.random.choice(
        regions_25, size=predicted_2025_sample, p=region_probs_25
    )
    crime_indices_25 = np.random.choice(
        len(crime_counts_25), size=predicted_2025_sample, p=crime_probs_25
    )
    sim_crimes_25 = np.array(crime_main_25)[crime_indices_25]
    sim_ages_25 = np.random.choice(
        ages_25, size=predicted_2025_sample, p=age_probs_25
    )

    df_2025_pred = pd.DataFrame(
        {
            "연도": ["2025 (예측)"] * predicted_2025_sample,
            "시도청": sim_regions_25,
            "범죄대유형": sim_crimes_25,
            "피의자연령대": sim_ages_25,
        }
    )

    df_extended_baseline = pd.concat(
        [df_historical, df_2025_pred], ignore_index=True
    )

    print("\n--- 3단계: SDV 메타데이터 설정 및 CTGAN 인공지능 학습 ---")
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=df_extended_baseline)
    for col in df_extended_baseline.columns:
        metadata.update_column(column_name=col, sdtype="categorical")

    # epochs=300
    synthesizer = CTGANSynthesizer(
        metadata, enforce_min_max_values=True, epochs=300, verbose=True
    )
    synthesizer.fit(df_extended_baseline)

    print("\n--- 4단계: 2020-2025 최종 통합 재현 데이터 마이닝 ---")
    df_ctgan_synthetic = synthesizer.sample(
        num_rows=len(df_extended_baseline)
    )

    print("\n--- 5단계: 2x2 종합 통계연보 그래프 시각화 시작 ---")
    # 그래프 편의성을 위해 2행 2열(총 4개) 구조로 확장 배치
    fig, axes = plt.subplots(2, 2, figsize=(22, 15))
    year_order = [
        "2020",
        "2021",
        "2022",
        "2023",
        "2024",
        "2025 (예측)",
    ]
    bar_width = 0.35

    # ----------------------------------------------------
    # [그래프 1] 좌측 상단 (0, 0): 연도별 총 발생건수 추이
    # ----------------------------------------------------
    base_year_counts = (
        df_extended_baseline["연도"].value_counts().reindex(year_order)
    )
    syn_year_counts = (
        df_ctgan_synthetic["연도"].value_counts().reindex(year_order)
    )
    x_indices = np.arange(len(year_order))

    rects1 = axes[0, 0].bar(
        x_indices - bar_width / 2,
        base_year_counts,
        bar_width,
        label="원본 및 추정 통계",
        color="midnightblue",
        alpha=0.85,
    )
    rects2 = axes[0, 0].bar(
        x_indices + bar_width / 2,
        syn_year_counts,
        bar_width,
        label="CTGAN 가상 재현",
        color="darkorange",
        alpha=0.85,
    )

    rects1[-1].set_hatch("//")
    rects1[-1].set_edgecolor("blue")
    rects2[-1].set_hatch("//")
    rects2[-1].set_edgecolor("chocolate")

    axes[0, 0].set_title(
        "① 연도별 총 발생건수 추이 (2025 예측포함)", fontsize=13, fontweight="bold"
    )
    axes[0, 0].set_xticks(x_indices)
    axes[0, 0].set_xticklabels(year_order, rotation=15)
    axes[0, 0].bar_label(
        rects1, padding=4, fmt="%d", fontsize=9, color="black"
    )
    axes[0, 0].bar_label(
        rects2, padding=4, fmt="%d", fontsize=9, color="black"
    )
    axes[0, 0].legend(loc="upper left")

    # ----------------------------------------------------
    # [그래프 2] 우측 상단 (0, 1): 범죄 대유형별 빈도 대조
    # ----------------------------------------------------
    base_crime_counts = df_extended_baseline["범죄대유형"].value_counts()
    syn_crime_counts = (
        df_ctgan_synthetic["범죄대유형"]
        .value_counts()
        .reindex(base_crime_counts.index)
    )
    y_indices_crime = np.arange(len(base_crime_counts.index))

    rects3 = axes[0, 1].barh(
        y_indices_crime - bar_width / 2,
        base_crime_counts,
        bar_width,
        label="원본 및 추정 통계",
        color="teal",
        alpha=0.85,
    )
    rects4 = axes[0, 1].barh(
        y_indices_crime + bar_width / 2,
        syn_crime_counts,
        bar_width,
        label="CTGAN 가상 재현",
        color="salmon",
        alpha=0.85,
    )

    axes[0, 1].set_title(
        "② 범죄 대유형별 빈도 대조 (2020-2025)", fontsize=13, fontweight="bold"
    )
    axes[0, 1].set_yticks(y_indices_crime)
    axes[0, 1].set_yticklabels(base_crime_counts.index, fontsize=10)
    axes[0, 1].bar_label(rects3, padding=4, fmt="%d", fontsize=9)
    axes[0, 1].bar_label(rects4, padding=4, fmt="%d", fontsize=9)
    axes[0, 1].legend(loc="lower right")

    # ----------------------------------------------------
    # ✨ [신규 그래프 3] 좌측 하단 (1, 0): 시도청별 발생건수 대조
    # ----------------------------------------------------
    base_region_counts = df_extended_baseline["시도청"].value_counts()
    syn_region_counts = (
        df_ctgan_synthetic["시도청"]
        .value_counts()
        .reindex(base_region_counts.index)
    )
    y_indices_region = np.arange(len(base_region_counts.index))

    rects5 = axes[1, 0].barh(
        y_indices_region - bar_width / 2,
        base_region_counts,
        bar_width,
        label="원본 및 추정 통계",
        color="indigo",
        alpha=0.85,
    )
    rects6 = axes[1, 0].barh(
        y_indices_region + bar_width / 2,
        syn_region_counts,
        bar_width,
        label="CTGAN 가상 재현",
        color="orchid",
        alpha=0.85,
    )

    axes[1, 0].set_title(
        "③ 시도청별 발생건수 대조 분포 (2020-2025)", fontsize=13, fontweight="bold"
    )
    axes[1, 0].set_yticks(y_indices_region)
    axes[1, 0].set_yticklabels(base_region_counts.index, fontsize=10)
    axes[1, 0].bar_label(rects5, padding=4, fmt="%d", fontsize=8)
    axes[1, 0].bar_label(rects6, padding=4, fmt="%d", fontsize=8)
    axes[1, 0].legend(loc="lower right")

    # ----------------------------------------------------
    # [그래프 4] 우측 하단 (1, 1): 연도별 사건 누적 발생 추이
    # ----------------------------------------------------
    base_cumulative = base_year_counts.cumsum()
    syn_cumulative = syn_year_counts.cumsum()

    axes[1, 1].plot(
        year_order,
        base_cumulative,
        marker="o",
        color="mediumblue",
        linewidth=2.5,
        label="원본 통계 누계",
    )
    axes[1, 1].plot(
        year_order,
        syn_cumulative,
        marker="s",
        color="darkorange",
        linewidth=2.5,
        linestyle="--",
        label="CTGAN 생성 누계",
    )

    for i in range(len(year_order)):
        axes[1, 1].annotate(
            f"{base_cumulative.iloc[i]:,}",
            (year_order[i], base_cumulative.iloc[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            color="mediumblue",
            weight="bold",
        )
        axes[1, 1].annotate(
            f"{syn_cumulative.iloc[i]:,}",
            (year_order[i], syn_cumulative.iloc[i]),
            textcoords="offset points",
            xytext=(0, -16),
            ha="center",
            fontsize=9,
            color="chocolate",
        )

    axes[1, 1].axvline(
        x=4.5, color="crimson", linestyle="-.", alpha=0.8, label="예측 시점 경계선"
    )
    axes[1, 1].set_title(
        "④ 연도별 사건 누적 발생 추이 (2020-2025)", fontsize=13, fontweight="bold"
    )
    axes[1, 1].set_xticklabels(year_order, rotation=15)
    axes[1, 1].legend(loc="upper left")

    # 전체 레이아웃 압축 정리 후 화면 표출
    plt.tight_layout()
    plt.show()
