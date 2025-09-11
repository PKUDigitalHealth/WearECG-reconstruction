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

machine_measurements_dtype_dict = {
    "report_12": str,
    "report_13": str,
    "report_14": str,
    "report_15": str,
    "report_16": str,
    "report_17": str,
}


def reorder_leads(data, current_order, target_order):
    """
    Reorder the leads of ECG data according to a target order.

    Parameters:
    - data: numpy array of shape (samples, leads, data_points)
    - current_order: list of leads in the order they are currently in the data
    - target_order: list of leads in the desired order

    Returns:
    - reordered_data: numpy array with leads reordered according to target_order
    """
    # Create a mapping from lead names to their current indices
    index_map = {lead: idx for idx, lead in enumerate(current_order)}

    # Create a new order of indices based on the target order
    new_indices = [index_map[lead] for lead in target_order]

    # Reorder the data according to the new indices
    reordered_data = data[:, new_indices, :]

    return reordered_data


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Apply a bandpass filter to the data.

    Parameters:
    - data: numpy array of shape (samples, leads, data_points)
    - lowcut: Low cutoff frequency (Hz)
    - highcut: High cutoff frequency (Hz)
    - fs: Sampling rate (Hz)
    - order: Order of the filter (default is 5)

    Returns:
    - filtered_data: numpy array of the same shape as data
    """
    # 设计滤波器
    nyq = 0.5 * fs  # Nyquist Frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")

    # 应用滤波器
    filtered_data = np.zeros_like(
        data
    )  # 创建一个与输入数据相同形状的数组用于存放滤波后的数据
    for i in range(data.shape[0]):  # 遍历每个样本
        for j in range(data.shape[1]):  # 遍历每个通道
            filtered_data[i, j, :] = filtfilt(
                b, a, data[i, j, :]
            )  # 对单个通道数据进行滤波

    return filtered_data


def concatenate_leads_end_to_end(data):
    """
    Concatenate multi-lead ECG data end-to-end for each sample using numpy concatenate.

    Parameters:
    - data: numpy array of shape (samples, leads, data_points)

    Returns:
    - concatenated_data: numpy array where each sample is reshaped by concatenating leads end-to-end
    """
    num_samples = data.shape[0]
    concatenated_data = np.zeros(
        (num_samples, 0)
    )  # Start with empty array for concatenation

    # Loop over each lead and concatenate the data end-to-end
    for lead in range(data.shape[1]):
        # Stack horizontally on the second axis
        concatenated_data = np.hstack((concatenated_data, data[:, lead, :]))

    return concatenated_data


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
    scaler = StandardScaler()
    normalized_train_data = scaler.fit_transform(train_data)
    normalized_val_data = scaler.transform(val_data)

    return normalized_train_data, normalized_val_data, scaler


def restore_shape(normalized_data, original_shape):
    """
    Restore the normalized data back to its original multi-lead ECG shape.

    Parameters:
    - normalized_data: Normalized data in 2D array (samples, concatenated_leads_data)
    - original_shape: The original shape of the data before concatenation (samples, leads, data_points)

    Returns:
    - restored_data: Data restored to original shape
    """
    num_leads = original_shape[1]
    lead_length = original_shape[2]
    split_indices = [(i + 1) * lead_length for i in range(num_leads - 1)]

    restored_data = np.array(
        [np.split(sample, split_indices) for sample in normalized_data]
    )

    return restored_data


def replace_outliers(data):
    """
    Replace NaN or inf values in each data point of each lead with the average of up to six preceding and
    following data points.

    Parameters:
    - data: numpy array of shape (leads, data_points)

    Returns:
    - data: numpy array with NaN and inf replaced
    """
    num_leads, num_points = data.shape

    # 遍历每个导联和数据点
    for i in range(num_leads):
        for j in range(num_points):
            if np.isnan(data[i, j]) or np.isinf(data[i, j]):
                # 计算平均值用来替代的区间范围
                start_index = max(0, j - 6)
                end_index = min(num_points, j + 7)

                # 计算除了当前异常值外的平均值
                valid_data = data[i, start_index:end_index]
                valid_data = valid_data[
                    ~np.isnan(valid_data) & ~np.isinf(valid_data)
                ]  # 排除异常值

                if valid_data.size > 0:
                    data[i, j] = np.mean(valid_data)
                else:
                    # 如果附近没有有效数据，可以选择一个默认值或保持原样
                    data[i, j] = 0  # 例如，这里选择将值设为0
    return data


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
    batch_size, num_original_samples, num_leads = signals.shape
    num_target_samples = int(num_original_samples * (target_fs / original_fs))
    resampled_signals = np.zeros((batch_size, num_target_samples, num_leads))
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
    num_samples, num_data_points, num_leads = data.shape

    for sample_index in tqdm(range(num_samples), desc="Replacing NaN values"):
        for lead_index in range(num_leads):
            nan_indices = np.where(np.isnan(data[sample_index, :, lead_index]))[0]

            for idx in nan_indices:
                start_idx = max(0, idx - window_size)
                end_idx = min(num_data_points, idx + window_size + 1)

                window = data[sample_index, start_idx:end_idx, lead_index]

                valid_window = window[~np.isnan(window)]
                if valid_window.size > 0:
                    data[sample_index, idx, lead_index] = np.mean(valid_window)
                else:
                    data[sample_index, idx, lead_index] = 0

    return data


def prepare_report(row, report_columns):
    """
    根据指定的报告列提取有效内容，并合并为一段完整的报告。
    """
    valid_reports = []
    for col in report_columns:
        report = row[col]
        if pd.notnull(report):  # 检查报告是否非空
            report = report.strip()  # 去除首尾空格
            if not report.endswith("."):
                report += "."  # 如果末尾没有句号，则添加
            valid_reports.append(report)
    return " ".join(valid_reports)  # 合并成一段话


def process_ecg_batch(
    batch_data,
    original_fs,
    target_fs,
    is_norm=False,
):
    """
    Process a single batch of ECG data, resample it, filter it, and save it.
    """

    batch_data = replace_nan_with_window_average(batch_data)

    all_data_resampled = resample_signals(batch_data, original_fs, target_fs)

    # reorder_leads(all_data_resampled, mimic_leads, standard_leads)  # For generation tasks, we keep original settings

    return all_data_resampled


def process_data_by_study_id(
    record_list,
    machine_measurements,
    dataset_path,
    save_path,
    batch_size,
    period,
    original_fs,
    target_fs,
    is_norm=False,
):
    """
    根据唯一 study_id 逐条读取数据并获取报告。
    """
    report_columns = machine_measurements.columns[4:22]
    batch_index = 0
    batch_data = []
    batch_reports = []

    for idx, row in tqdm(
        record_list.iterrows(), total=len(record_list), desc="Processing data"
    ):
        study_id = row["study_id"]  # 获取 study_id
        file_name = row["path"]
        full_path = os.path.join(dataset_path, file_name)

        try:
            # 读取信号数据
            signals, _ = wfdb.rdsamp(full_path)
            batch_data.append(signals)

            # 在 machine_measurements 中查找对应的报告
            matching_row = machine_measurements.loc[
                machine_measurements["study_id"] == study_id
            ]
            if not matching_row.empty:
                report_text = prepare_report(matching_row.iloc[0], report_columns)
                batch_reports.append({"study_id": study_id, "report": report_text})
            else:
                batch_reports.append(
                    {"study_id": study_id, "report": "No machine measurement reports."}
                )

            # Check if the current batch is full or if it is the last record
            if len(batch_data) >= batch_size or idx == len(record_list) - 1:
                batch_data_array = np.array(batch_data)
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

                # 保存报告为 JSON 格式
                with open(
                    os.path.join(
                        save_path, f"{period}_reports_{batch_index}.json"
                    ),
                    "w",
                    encoding="utf-8",
                ) as json_file:
                    json.dump(batch_reports, json_file, ensure_ascii=False, indent=4)

                del batch_data_array
                gc.collect()
                batch_data = []  # 清空当前批次数据
                batch_reports = []  # 清空当前批次报告
                batch_index += 1  # 批次编号递增

        except FileNotFoundError:
            print(f"File not found: {full_path}")
            continue


def main():
    # Paths
    dataset_path = "/hot_data/jinjiarui/datasets/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/"
    save_path = "/data/0shared/jinjiarui/run/MLA-Diffusion/data/datasets/MIMIC"
    os.makedirs(save_path, exist_ok=True)

    original_fs = 500
    target_fs = 100
    batch_size = 200000
    is_norm = False

    record_list = pd.read_csv(os.path.join(dataset_path, "record_list.csv"))
    machine_measurements = pd.read_csv(
        os.path.join(
            dataset_path,
            "machine_measurements.csv",
        ),
        dtype=machine_measurements_dtype_dict,
    )

    train_list, test_list = train_test_split(
        record_list, test_size=0.2, random_state=42
    )
    train_list = train_list.reset_index(drop=True)
    test_list = test_list.reset_index(drop=True)

    # subset
    # train_list = train_list[:20000]
    # test_list = test_list[:2000]

    # Process Train Data
    print("Loading and processing train data...")
    process_data_by_study_id(
        record_list=train_list,
        machine_measurements=machine_measurements,
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
    process_data_by_study_id(
        record_list=test_list,
        machine_measurements=machine_measurements,
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
