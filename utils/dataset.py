from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer

class ClassicalTranslationDataset(Dataset):
    def __init__(self, dataframe, tokenizer: PreTrainedTokenizer, max_length=64):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        source_text = self.data.iloc[idx]["modern"]
        target_text = self.data.iloc[idx]["classical"]

        inputs = self.tokenizer(
            source_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        targets = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": targets["input_ids"].squeeze()
        }

    
if __name__ == '__main__':
    # 读取 parquet 文件
    df = pd.read_parquet("/root/siton-data-412581749c3f4cfea0d7c972b8742057/proj/work_4_xzp/data/data/train-00000-of-00001-a082a3e0459a7949.parquet")
    tokenizer = T5Tokenizer.from_pretrained("/root/siton-data-412581749c3f4cfea0d7c972b8742057/proj/work_4_xzp/pre_trained/t5")
    
    # 划分数据集：90% 训练，10% 验证
    train_df, val_df = train_test_split(df, test_size=0.3, random_state=42)
    
    train_dataset = ClassicalTranslationDataset(train_df, tokenizer)
    print(train_dataset[0])