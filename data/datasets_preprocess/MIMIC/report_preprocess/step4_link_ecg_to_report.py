import pandas as pd
import json


def save_subset(df, output_path, n=100):
    """保存前n条记录为子集"""
    subset = df.head(n)

    subset.to_csv(output_path, index=False, columns=['study_id', 'subject_id', 'hadm_id', 'report', 'path'])
    print(f"已保存前{n}条记录至 {output_path}")


def main():
    # 文件路径配置
    ECG_MAPPING_PATH = (
        "data/datasets_preprocess/MIMIC/report_files/ecg_to_admission_mapping.csv"
    )
    REPORT_PATH = "data/datasets_preprocess/MIMIC/report_files/hadm_report_full.csv"
    RECORD_LIST_PATH = "/data1/1shared/jinjiarui/datasets/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/record_list.csv"
    OUTPUT_PATH = (
        "data/datasets_preprocess/MIMIC/report_files/ecg_report_with_path.csv"
    )

    # 步骤1：加载基础数据
    ecg_core = pd.read_csv(
        ECG_MAPPING_PATH, usecols=["subject_id", "hadm_id", "study_id"]
    )
    reports = pd.read_csv(REPORT_PATH, usecols=["hadm_id", "report"])

    # 步骤2：加载ECG路径数据
    path_df = pd.read_csv(RECORD_LIST_PATH, usecols=["study_id", "path"]).astype(
        {"study_id": "int64"}
    )

    # 数据处理管道
    # 修改数据处理管道部分（原代码第24-27行）
    merged = (
        pd.merge(ecg_core, reports, on='hadm_id', how='inner')  # 改为内连接
        .merge(path_df, on='study_id', how='left')  # 保持左连接确保所有ECG路径保留
        .drop_duplicates('study_id')
        .reset_index(drop=True)
    )

    # 生成最终CSV
    merged.to_csv(OUTPUT_PATH, index=False, columns=['study_id', 'subject_id', 'hadm_id', 'report', 'path'])

    # 保存子集
    save_subset(
        merged,
        "data/datasets_preprocess/MIMIC/report_files/ecg_report_subset.csv",
        n=100,
    )


if __name__ == "__main__":
    main()
