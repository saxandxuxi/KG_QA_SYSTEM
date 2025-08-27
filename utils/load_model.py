# from modelscope import snapshot_download
# # 下载英文嵌入模型（BGE英文版本）
# model_dir = snapshot_download(
#     model_id='BAAI/bge-large-en-v1.5',  # 英文版本模型ID
#     cache_dir='../models'
# )
# print(f"英文模型下载完成，路径：{model_dir}")
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# 模型名称
model_name = "Helsinki-NLP/opus-mt-zh-en"
# 本地保存路径（可自定义）
local_dir = "../models/opus-mt-zh-en"

# 下载并保存模型和分词器到本地
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=local_dir)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=local_dir)

# 保存模型和分词器（可选，确保文件完整保存）
model.save_pretrained(local_dir)
tokenizer.save_pretrained(local_dir)

print(f"模型已保存到：{local_dir}")

# 后续加载时，直接使用本地路径
loaded_model = AutoModelForSeq2SeqLM.from_pretrained(local_dir)
loaded_tokenizer = AutoTokenizer.from_pretrained(local_dir)
print("模型加载成功")
