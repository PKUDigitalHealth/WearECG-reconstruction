import pandas as pd
import json
from tqdm import tqdm


def convert_csv_to_jsonl_in_chunks(input_csv, output_file_base, chunk_size=50000):
    """
    从 CSV 文件中读取 hadm_id 和 past_medical_history，构造 JSONL 文件，每 chunk_size 条保存为一个文件。
    - custom_id 使用 hadm_id；
    - user 消息中的 content 使用 past_medical_history 字段；
    - system prompt 内容保持不变。
    """

    # 读取 CSV 文件
    df = pd.read_csv(input_csv)
    df = df.dropna(subset=["past_medical_history"])  # 确保没有空记录
    total_items = len(df)
    file_count = 0

    # 遍历并按 chunk_size 分段保存
    for i in range(0, total_items, chunk_size):
        file_count += 1
        chunk_df = df.iloc[i : i + chunk_size]
        output_file = f"{output_file_base}_{file_count}.jsonl"

        with open(output_file, "w", encoding="utf-8") as f_out:
            for _, row in tqdm(
                chunk_df.iterrows(),
                total=len(chunk_df),
                desc=f"Writing chunk {file_count}",
            ):
                hadm_id = row["hadm_id"]
                pmh = row["past_medical_history"]

                entry = {
                    "custom_id": str(hadm_id),
                    "method": "POST",
                    "url": "/v4/chat/completions",
                    "body": {
                        "model": "glm-4-plus",
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "Role: Medical Record Organizer\n"
                                    "Objective: Convert medical data fragments into grammatically correct English paragraphs with strict verbatim retention of all original terms, including potential typos, unknown abbreviations, and identifiers.\n\n"
                                    "Other than responding with the organized medical history content, do not reply with anything else.\n\n"
                                    "Non-Negotiable Rules:\n\n"
                                    "100% verbatim fidelity:\n\n"
                                    'Retain all original terms (e.g., "HED" → "HED", never expand or correct).\n\n'
                                    "Preserve identifiers ___, symbols, and punctuation exactly as written.\n\n"
                                    'Never flag, question, or explain potential errors (e.g., "HED" remains "HED").\n\n'
                                    "Subject standardization:\n\n"
                                    'Replace "Pt" with "the patient" once per sentence (e.g., "Pt HED" → "The patient has HED").\n\n'
                                    'Retain all other abbreviations (e.g., "DM" → "DM", "SOB" → "SOB").\n\n'
                                    "Special cases:\n\n"
                                    'Sentences with no actionable data (e.g., "HED.") → "The patient has HED."\n\n'
                                    "Output Format:\n\n"
                                    "Single plain-text paragraph.\n\n"
                                    "No markdown, bullet points, or line breaks.\n\n"
                                    "Examples:\n\n"
                                    "Input: Pt HED.\n"
                                    "Output: The patient has HED.\n\n"
                                    "Input: Pt c/o SOB. HLD___.\n"
                                    "Output: The patient c/o SOB. HLD___.\n\n"
                                    "Input: HR.\n"
                                    "Output: HR.\n\n"
                                    "Key Clarifications:\n\n"
                                    'No error handling: Treat "HED" as a valid term even if unrecognized.\n\n'
                                    'No contextualization: Avoid phrases like "diagnosed with" unless explicitly stated.\n\n'
                                    'Grammar limited to subject standardization: Only modify "Pt" → "the patient"; all other terms remain untouched.'
                                ),
                            },
                            {"role": "user", "content": str(pmh)},
                        ],
                        "temperature": 0.1,
                    },
                }

                f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✅ 生成完成：共 {file_count} 个 .jsonl 文件")


if __name__ == "__main__":
    input_csv = "/data1/1shared/jinjiarui/run/MLA-Diffusion/data/datasets_preprocess/MIMIC/report_files/hadm_pmh_extracted.csv"
    output_file_base = (
        "data/datasets_preprocess/MIMIC/report_files/batch_api_files/batch_input"
    )
    convert_csv_to_jsonl_in_chunks(input_csv, output_file_base, chunk_size=45000)
