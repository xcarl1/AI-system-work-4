import os
from tqdm import tqdm
import time
import torch
import torch.distributed as dist
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from evaluate import load as evaluate_load
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
from torch.utils.tensorboard import SummaryWriter

from utils.dataset import ClassicalTranslationDataset

def setup(rank, world_size, backend):
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    
def cleanup():
    dist.destroy_process_group()

@torch.no_grad()
def evaluate(model, dataloader, tokenizer, rank):
    model.eval()
    predictions, references = [], []

    # Wrapping the dataloader with tqdm for progress display
    for batch in tqdm(dataloader, desc="Evaluating", position=0, leave=True):
        input_ids = batch["input_ids"].to(rank)
        attention_mask = batch["attention_mask"].to(rank)
        target_texts = batch["labels"]

        # Generate predictions
        generated_ids = model.module.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=64)
        preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        predictions.extend(preds)

        # Decode references and format them correctly
        target_texts = tokenizer.batch_decode(target_texts, skip_special_tokens=True)
        references.extend([[t] for t in target_texts])  # wrap in list for corpus BLEU

    # Load BLEU metric
    bleu = evaluate_load("sacrebleu")
    result = bleu.compute(predictions=predictions, references=references)
    
    return result["score"]
    

def train(args, rank, world_size, backend):
    setup(rank, world_size, backend)

    if rank == 0:
        log_dir = os.path.join("runs", args.log_dir if hasattr(args, 'log_dir') else "default")
        writer = SummaryWriter(log_dir=log_dir)
    else:
        writer = None
        
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name).to(rank)
    model = DDP(model, device_ids=[rank])
    
    df = pd.read_parquet(os.path.join(args.data_path, "data/train-00000-of-00001-a082a3e0459a7949.parquet"))
    
    temp_df, _ = train_test_split(df, test_size=1, random_state=42) # 为了减少时间只取前50%作为训练和测试
    train_df, val_df = train_test_split(temp_df, test_size=args.data_split, random_state=42)
    
    train_dataset = ClassicalTranslationDataset(train_df ,tokenizer)
    val_dataset = ClassicalTranslationDataset(val_df, tokenizer)
        
    if not args.eval_only:

        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

        train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)
        val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=args.eval_batch_size)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        global_step = 0

        start_time = time.time()
        for epoch in range(args.train_epochs):
            model.train()
            train_sampler.set_epoch(epoch)

            if rank == 0:
                train_iter = tqdm(train_loader, desc=f"Epoch {epoch}", position=0, leave=True)
            else:
                train_iter = train_loader  # Non-master processes don't use tqdm

            for batch in train_iter:
                input_ids = batch["input_ids"].to(rank)
                attention_mask = batch["attention_mask"].to(rank)
                labels = batch["labels"].to(rank)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if rank == 0:
                    writer.add_scalar("Loss/train_step", loss.item(), global_step)
                    writer.add_scalar("LR", optimizer.param_groups[0]["lr"], global_step)
                global_step += 1

            bleu_score = evaluate(model, val_loader, tokenizer, rank)
            
            if rank == 0:
                save_path = os.path.join(args.save_path, f"mengzi_t5_epoch{epoch}.pt")
                torch.save(model.module.state_dict(), save_path)
                # bleu_score = evaluate(model, val_loader, tokenizer, rank)
                print(f"Epoch {epoch} | Validation BLEU: {bleu_score:.2f}")
                writer.add_scalar("BLEU/val", bleu_score, epoch)

        if rank == 0:
            elapsed = time.time() - start_time
            print(f"Training completed in {elapsed / 60:.2f} minutes.")
            with open("training_time.txt", "a") as f:
                f.write(f"Training completed in {elapsed / 60:.2f} minutes.\n")
            writer.close()
    
    else:
        print("Evaluating only...")
        val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size)
    
        bleu_score = evaluate(model, val_loader, tokenizer, rank)
    
        if rank == 0:
            print(f"Validation BLEU: {bleu_score:.2f}")

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="nccl")
    parser.add_argument("--model_name", type=str, default="/root/siton-data-412581749c3f4cfea0d7c972b8742057/proj/work_4_xzp/pre_trained/t5")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--train_epochs", type=int, default=3)
    parser.add_argument("--log_dir", type=str, default="mengzi_t5_logs")
    parser.add_argument("--data_path", type=str, default="/root/siton-data-412581749c3f4cfea0d7c972b8742057/proj/work_4_xzp/data")
    parser.add_argument("--data_split", type=float, default=0.3)
    parser.add_argument("--save_path", type=str, default="checkpoints/")
    parser.add_argument("--eval_only", action="store_true", help="If set, only evaluate the model without training")
    parser.add_argument("--eval_batch_size", type=int, default=64)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    train(args, rank, world_size, args.backend)
