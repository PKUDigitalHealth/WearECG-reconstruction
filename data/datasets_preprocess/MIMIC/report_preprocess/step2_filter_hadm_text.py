import pandas as pd
import re
from tqdm import tqdm

# 正则表达式模式（新增PMH提取）
patterns = {
    "sex": re.compile(r"Sex:\s*([MF])\b", re.IGNORECASE),
    "hpi": re.compile(
        r"History\s+of\s+Present\s+Illness:\s*(.*?)(?=\n\s*\n[A-Z]{3,}\s)", re.DOTALL
    ),
    "pmh": re.compile(
        r"Past\s+Medical\s+History:\s*(.*?)(?=\n\s*\n[A-Z]{3,}\s)", re.DOTALL
    ),
    "admission_pe": re.compile(
        r"Physical Exam:\s*((?:.*?ADMISSION.*?(?:\n|$))+)", re.DOTALL | re.IGNORECASE
    ),
    "pertinent_results": re.compile(
        r"Pertinent Results:\s*(.*?)(?=\n\s*\n[A-Z]{3,}\s)", re.DOTALL | re.IGNORECASE
    ),
}


def extract_section(text, section_name, filter_keyword=None):
    """通用提取函数：从指定章节提取内容（可选关键词过滤）"""
    # 匹配章节
    section_match = re.search(
        rf"{section_name}\s*(.*?)(?=\n\s*\n[A-Z]{{3,}}\s)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if not section_match:
        return None

    section_content = section_match.group(1)

    # 如果需要关键词过滤
    if filter_keyword:
        paragraphs = re.findall(
            rf"(^{filter_keyword}[\s\S]*?)(?=\n\s*\n|\Z)",  # 关键修改点
            section_content,
            re.MULTILINE | re.IGNORECASE,
        )
        return "\n\n".join(paragraphs) if paragraphs else None
    else:
        return section_content.strip()


if __name__ == "__main__":
    # 路径设置
    discharge_path = "/data1/1shared/jinjiarui/datasets/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv"
    output_path = "data/datasets_preprocess/MIMIC/report_files/hadm_extracted_full.csv"

    # 读取数据
    df = pd.read_csv(discharge_path)

    # 初始化结果列表
    records = []
    # 处理每条记录
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        text = str(row["text"]).replace("\r", "\n")
        record = {
            "subject_id": row["subject_id"],
            "hadm_id": row["hadm_id"],
            "sex": (
                match.group(1).upper() if (match := patterns["sex"].search(text)) else None
            ),
            "history_of_present_illness": extract_section(text, "History of Present Illness:"),
            "past_medical_history": extract_section(text, "Past Medical History:"),
            "admission_physical_exam": extract_section(
                text, "Physical Exam:", filter_keyword="ADMISSION"
            ),
            "admission_pertinent_results": extract_section(
                text, "Pertinent Results:", filter_keyword="ADMISSION"
            ),
        }

        records.append(record)

    # 创建DataFrame并保存
    pd.DataFrame(records).to_csv(output_path, index=False)
    print(f"数据已保存至: {output_path}")
