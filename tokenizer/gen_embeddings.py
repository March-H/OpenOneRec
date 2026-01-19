import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import os
import pyarrow as pa
import pyarrow.parquet as pq

# ================= 你的配置 =================
# 1. 输入：你下载的 RecIF 原始文本数据
INPUT_FILE = "../data/raw_data/onerec_data/pid2caption.parquet"
# 2. 输出：Tokenizer 训练必须的中间文件
OUTPUT_FILE = "data/embeddings.parquet"
# 3. 模型：必须用官方指定的这个，否则空间不对齐
MODEL_NAME = "/root/share/models/Qwen3-8B"
DEVICE = "cuda"

class QwenEmbeddingModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state # (b,s,h)

        input_mask_expended = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float() # (b,s,h)
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expended, 1) # (b,h)
        sum_mask = torch.clamp(input_mask_expended.sum(1), min=1e-9) # (b,h)
        embeddings = sum_embeddings / sum_mask\

        # L2范数做归一化。x / sqrt(sum(x_i^2))。因为计算语义间相似度采用余弦相似度，这样算能使得余弦相似度的分母为1.A*B/(|A|*|B|)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

# ===========================================

def main():
    BATCH_SIZE = 64  # 根据显存调整
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"找不到输入文件: {INPUT_FILE}，请先去下载 RecIF 数据集")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    print(f"正在加载模型: {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    base_model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        dtype=torch.float16
    ).to(DEVICE)
    model = QwenEmbeddingModel(base_model)
    model.eval()

    if torch.cuda.device_count() > 1:
        print(f"启动DP进行加速，共{torch.cuda.device_count()}张卡")

        # 定义DP模型，会自动在batch维度上进行切分来进行DP
        model = nn.DataParallel(model)
        BATCH_SIZE = BATCH_SIZE * torch.cuda.device_count()
    else:
        print("未检测到多 GPU，将使用单卡运行。")
    model.to(DEVICE)
    WRITE_BATCH = BATCH_SIZE * 10

    print(f"正在读取数据: {INPUT_FILE}")
    try:
        df = pd.read_parquet(INPUT_FILE)
    except Exception as e:
        print("读取失败，尝试使用 fastparquet 引擎...")
        df = pd.read_parquet(INPUT_FILE, engine='fastparquet')

    print(f"数据量: {len(df)}")

    # 提取需要的列
    text_col = 'dense_caption' if 'dense_caption' in df.columns else 'text'
    # 确保是字符串
    texts = df[text_col].astype(str).tolist()
    pids = df['pid'].astype(str).tolist()

    schema = pa.schema([
        ('pid', pa.string()),
        ('embedding', pa.list_(pa.float32()))
    ])

    writer = pq.ParquetWriter(OUTPUT_FILE, schema)
    print(f"已创建流式写入器，写入到: {OUTPUT_FILE}")

    temp_pids = []
    temp_embeddings = []

    print("开始生成 Embeddings...")
    for i in tqdm(range(0, len(df), BATCH_SIZE)):
        batch_end = min(i + BATCH_SIZE, len(df))
        batch_texts = texts[i: batch_end]
        batch_pids = pids[i: batch_end]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=32768,
            return_tensors="pt"
        ).to(DEVICE)

        input_ids = inputs["input_ids"].to(DEVICE)
        attention_mask = inputs["attention_mask"].to(DEVICE)

        with torch.no_grad():
            batch_embeddings = model(input_ids, attention_mask)
            batch_embeddings = batch_embeddings.cpu().numpy().tolist()
            temp_embeddings.extend(batch_embeddings)
            temp_pids.extend(batch_pids)

        if len(temp_pids) >= WRITE_BATCH or i + BATCH_SIZE >= len(df):
            table = pa.Table.from_pydict({
                "pid": temp_pids,
                "embedding": temp_embeddings
            }, schema=schema)
            writer.write_table(table)
            temp_pids = []
            temp_embeddings = []
            import gc
            gc.collect()

    writer.close()
    print("生成完成！")


if __name__ == "__main__":
    main()