from zhipuai import ZhipuAI
import sys
import os

project_root_dir = "/data1/1shared/jinjiarui/run/MLA-Diffusion"
sys.path.append(project_root_dir)
os.chdir(project_root_dir)

client = ZhipuAI(
    api_key="3054b366b2868b315f5a4cb6f5e5fae9.GVbQ1PQVYmgDzm4K"
)

# result = client.files.create(
#     file=open(
#         "data/datasets_preprocess/MIMIC/report_files/batch_api_files/batch_input_1.jsonl",
#         "rb",
#     ),
#     purpose="batch",
# )
# print(result.id)


# create = client.batches.create(
#     input_file_id="1742629441_e10081da04a8416aa5e1833b9aebd763",
#     endpoint="/v4/chat/completions",
#     auto_delete_input_file=True,
#     metadata={"description": "MIMIC-IV-Note Preprocess 1"},
# )
# print(create.id)  # 返回batch id


batch_job = client.batches.retrieve("batch_1903352883476037632")
print(batch_job)
