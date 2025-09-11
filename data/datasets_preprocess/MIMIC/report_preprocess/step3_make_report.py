import pandas as pd
import os
import json

def convert_to_csv(file_path, output_dir=None):
    """将CSV文件处理并保存为新的CSV格式，report为完整文本"""
    df = pd.read_csv(file_path)
    
    if output_dir is None:
        output_dir = os.path.dirname(file_path)
    
    output_path = os.path.join(output_dir, "hadm_report_full.csv")
    
    # 创建新的DataFrame保存处理后的数据
    processed_data = []
    for _, row in df.iterrows():
        report_text = (
            "[Sex]\n" + str(row["sex"]) + "\n\n"
            "[History of present illness]\n" + str(row["history_of_present_illness"]) + "\n\n"
            "[Past medical history]\n" + str(row["past_medical_history"]) + "\n\n"
            "[Admission physical exam]\n" + str(row["admission_physical_exam"]) + "\n\n"
            "[Admission pertinent results]\n" + str(row["admission_pertinent_results"])
        )
        
        processed_data.append({
            "subject_id": row["subject_id"],
            "hadm_id": row["hadm_id"],
            "report": report_text
        })
    
    # 转换为DataFrame并保存为CSV
    processed_df = pd.DataFrame(processed_data)
    processed_df.to_csv(output_path, index=False)
    
    print(f"成功处理并保存CSV文件到: {output_path}")

def save_csv_subset(file_path, n=100, output_dir=None):
    """读取CSV文件并保存前n行为子集"""
    df = pd.read_csv(file_path)
    subset = df.head(n)
    
    if output_dir is None:
        output_dir = os.path.dirname(file_path)
    
    output_path = os.path.join(output_dir, "hadm_extracted_subset.csv")
    subset.to_csv(output_path, index=False)
    print(f"成功保存前{n}行数据到: {output_path}")

def print_by_hadm_id(file_path, hadm_id):
    """根据hadm_id打印每一列内容"""
    df = pd.read_csv(file_path)
    result = df[df['hadm_id'] == hadm_id]
    
    if result.empty:
        print(f"未找到hadm_id为{hadm_id}的记录")
    else:
        for col in result.columns:
            print(f"{col}: {result[col].values[0]}")

if __name__ == "__main__":
    csv_path = "data/datasets_preprocess/MIMIC/report_files/hadm_extracted_full.csv"

    # 保存子集
    # save_csv_subset(csv_path)

    # 示例：打印hadm_id为123的记录
    # print_by_hadm_id(csv_path, 27897940)
    
    # 转换为CSV格式
    convert_to_csv(csv_path)
