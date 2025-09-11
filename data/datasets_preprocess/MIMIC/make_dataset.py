import gc
import os
import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import wfdb
from torch.utils.data import Dataset
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt
import json
from sklearn.model_selection import GroupShuffleSplit

mimic_leads = [
    "I",
    "II",
    "III",
    "aVR",
    "aVF",
    "aVL",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]

standard_leads = [
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]

def reorder_leads(data, current_order, target_order):
    """
    Reorder the leads of ECG data according to a target order.
    """
    # Create a mapping from lead names to their current indices
    index_map = {lead: idx for idx, lead in enumerate(current_order)}

    # Create a new order of indices based on the target order
    new_indices = [index_map[lead] for lead in target_order]

    # Reorder the data according to the new indices
    reordered_data = data[:, :, new_indices]

    return reordered_data


def normalize_data(train_data, val_data):
    """
    Normalize training and validation data using StandardScaler.

    Parameters:
    - train_data: Concatenated training data in 2D array (samples, features)
    - val_data: Concatenated validation data in 2D array (samples, features)

    Returns:
    - normalized_train_data: Normalized training data
    - normalized_val_data: Normalized validation data
    - scaler: Fitted StandardScaler
    """
    scaler = StandardScaler()# 对数据做Z-score标准化
    normalized_train_data = scaler.fit_transform(train_data)
    normalized_val_data = scaler.transform(val_data)

    return normalized_train_data, normalized_val_data, scaler


def resample_signals(signals, original_fs, target_fs):
    """
    Resample the signals from original_fs to target_fs and replace outliers.

    Parameters:
    - signals: numpy array of shape (samples, data_points, leads)
    - original_fs: Original sampling rate
    - target_fs: Target sampling rate

    Returns:
    - resampled_signals: numpy array of the resampled signals
    """
    # 提取输入信号的维度
    batch_size, num_original_samples, num_leads = signals.shape
    # 计算重采样后的时间点数（目标采样长度） 按照比例关系，原始时间长度 × (新采样率 / 原采样率)
    num_target_samples = int(num_original_samples * (target_fs / original_fs))
    #创建一个空数组来存放重采样后的信号数据
    resampled_signals = np.zeros((batch_size, num_target_samples, num_leads))
    #重采样 ECG 信号，也就是把原始的采样频率 original_fs 改成指定的新频率 target_fs
    for b in tqdm(range(batch_size), desc="Resampling signals"):
        for i in range(num_leads):
            resampled_signals[b, :, i] = signal.resample(
                signals[b, :, i], num_target_samples
            )
    return resampled_signals


def replace_nan_with_window_average(data, window_size=6):
    """
    Replace NaN values in the data array with the average of a window of surrounding data points,
    handling boundaries appropriately.

    Parameters:
    - data: numpy array of shape (samples, data_points, leads)
    - window_size: Number of elements to consider on each side of NaN

    Returns:
    - data: numpy array with NaN replaced
    """
    num_samples, num_data_points, num_leads = data.shape # 三个维度：样本数、时间点数量、导联数。

    for sample_index in tqdm(range(num_samples), desc="Replacing NaN values"):
        for lead_index in range(num_leads):
            nan_indices = np.where(np.isnan(data[sample_index, :, lead_index]))[0]#找出这个样本中这个导联所有是 NaN 的时间点的索引

            for idx in nan_indices:
                start_idx = max(0, idx - window_size) # 左窗口起点，防止越界
                end_idx = min(num_data_points, idx + window_size + 1) # 右窗口终点

                window = data[sample_index, start_idx:end_idx, lead_index] # 取出窗口数据（一个小段时间序列）

                valid_window = window[~np.isnan(window)] # 把窗口里 不是 NaN 的数据拿出来
                if valid_window.size > 0:
                    data[sample_index, idx, lead_index] = np.mean(valid_window) # 如果窗口中还有有效值，就用它们的平均数补 NaN
                else:
                    data[sample_index, idx, lead_index] = 0 # 如果全是 NaN（极端情况），就填 0（保守策略）

    return data # 返回处理完、没有 NaN 的数据


def process_ecg_batch(
    batch_data,
    original_fs,
    target_fs,
    is_norm=False,
):
    """
    Process a single batch of ECG data, resample it, filter it, and save it.
    """
    # batch_data = replace_nan_with_window_average(batch_data)
    batch_data = np.nan_to_num(batch_data, nan=0)   
    return batch_data


def process_data(
    record_list, # 一个包含 ECG 文件路径、报告等信息的 DataFrame
    dataset_path, # 数据所在的目录
    save_path,# 输出保存路径
    batch_size,
    period,
    original_fs,# 原始采样率
    target_fs,# 目标采样率（一般会降采样）
    is_norm=False,
):
    batch_index = 0
    batch_data = []# 当前这一批的数据
    batch_meta = [] # 当前这一批的元信息
    # batch_reports = []# 当前这一批的文本报告

    #遍历每一个记录（心电图 + 报告）
    for idx, row in tqdm(
        record_list.iterrows(), total=len(record_list), desc="Processing data"
    ):
        # 构造完整路径
        full_path = row["ecg_path"]  # 直接就是完整路径


        try:
            # 读取信号数据
            signals, _ = wfdb.rdsamp(full_path)# 使用 wfdb 读取信号
            batch_data.append(signals)# 加入当前批次数据列表
            # 收集元信息
            meta = {}
            for key in ["subject_id", "study_id", "hadm_id", "ecg_path", "report", "split"]:
                if key in row:
                    meta[key] = row[key]
            batch_meta.append(meta)
            # batch_reports.append({"study_id": study_id, "report": report_text})

            # Check if the current batch is full or if it is the last record
            if len(batch_data) >= batch_size or idx == len(record_list) - 1:
                batch_data_array = np.array(batch_data)
                #进入信号预处理
                batch_ecg_data = process_ecg_batch(
                    batch_data=batch_data_array,
                    original_fs=original_fs,
                    target_fs=target_fs,
                    is_norm=is_norm,
                )

                # 保存数据
                np.save(
                    os.path.join(save_path, f"{period}_data_{batch_index}.npy"),
                    batch_ecg_data,
                )

                # 保存元信息为 JSON 格式
                with open(
                    os.path.join(save_path, f"{period}_meta_{batch_index}.json"),
                    "w",
                    encoding="utf-8",
                ) as meta_file:
                    json.dump(batch_meta, meta_file, ensure_ascii=False, indent=4)


                del batch_data_array
                gc.collect()
                batch_data = []  # 清空当前批次数据
                batch_meta = []  # 清空当前批次元信息
                # batch_reports = []  # 清空当前批次报告
                batch_index += 1  # 批次编号递增

        except FileNotFoundError:
            print(f"File not found: {full_path}")
            continue


def main():
    
    dataset_path = "/data/0shared/MIMIC/physionet.org/files/mimic-iv-ecg/1.0/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0"
    save_path = "/data/0shared/guanxinyan/reconstruction/data/datasets/MIMIC"
    os.makedirs(save_path, exist_ok=True)

    original_fs = 500
    target_fs = 500
    batch_size = 200000
    is_norm = False

    #record_list = pd.read_csv(record_path)

    # 定义分组分割器 使用 GroupShuffleSplit 保证同一个 subject_id 不出现在 train 和 test 两边，防止数据泄漏。

    splitter = GroupShuffleSplit(test_size=0.2, random_state=42)

    # 原始 csv 路径
    csv_path = "/data/0shared/guanxinyan/reconstruction/data/datasets_preprocess/waveform_note_links.csv"
    csv_save_path = "/data/0shared/guanxinyan/reconstruction/data/datasets_preprocess/waveform_note_links_with_split.csv"

    # 读取 csv
    df = pd.read_csv(csv_path)

    # 保证 subject_id 是整数，并按其排序
    df["subject_id"] = df["subject_id"].astype(int)
    df_sorted = df.sort_values("subject_id").reset_index(drop=True)

    # 获取前 10% 的 subject_id（唯一的）
    unique_subjects = df_sorted["subject_id"].unique()
    n_test = int(len(unique_subjects) * 0.1)
    test_subjects = set(unique_subjects[:n_test])

    # 根据 subject_id 判断 split 类型
    df_sorted["split"] = df_sorted["subject_id"].apply(lambda x: "test" if x in test_subjects else "train")

    # 保存带 split 的新 csv
    df_sorted.to_csv(csv_save_path, index=False)
    print(f"新的 CSV 带 split 字段已保存至: {csv_save_path}")

    # 使用我们刚刚生成的新 CSV
    df_with_split = pd.read_csv(csv_save_path)

    # 拼接 ECG 的绝对路径（如果你需要）
    df_with_split["ecg_path"] = df_with_split["waveform_path"].apply(
        lambda x: os.path.join(dataset_path, x)
    )

    # 拆分
    train_list = df_with_split[df_with_split["split"] == "train"].reset_index(drop=True)
    test_list = df_with_split[df_with_split["split"] == "test"].reset_index(drop=True)

    print(f"Train: {len(train_list)}, Test: {len(test_list)}")

    train_list.to_csv("/data/0shared/guanxinyan/reconstruction/data/datasets_preprocess/MIMIC/train_records.csv", index=False)
    test_list.to_csv("/data/0shared/guanxinyan/reconstruction/data/datasets_preprocess/MIMIC/test_records.csv", index=False)

    train_list = train_list.reset_index(drop=True)
    test_list = test_list.reset_index(drop=True)


    # Process Train Data
    print("Loading and processing train data...")
    process_data(
        record_list=train_list,
        dataset_path=dataset_path,
        save_path=save_path,
        batch_size=batch_size,
        period="train",
        original_fs=original_fs,
        target_fs=target_fs,
        is_norm=is_norm,
    )

    # Process Test Data
    print("Loading and processing test data...")
    process_data(
        record_list=test_list,
        dataset_path=dataset_path,
        save_path=save_path,
        batch_size=batch_size,
        period="test",
        original_fs=original_fs,
        target_fs=target_fs,
        is_norm=is_norm,
    )


if __name__ == "__main__":
    main()