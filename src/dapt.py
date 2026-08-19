#!/usr/bin/env python3
"""
Domain-Adaptive Pre-training (DAPT) for DeBERTa-v3-base on resume corpus
Continues MLM pre-training on domain-specific text to improve downstream performance
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    get_linear_schedule_with_warmup,
    DataCollatorForLanguageModeling
)
from datasets import load_from_disk
import numpy as np
from tqdm.auto import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }

def load_corpus(corpus_path):
    """Load text corpus from file"""
    logger.info(f"Loading corpus from {corpus_path}")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    logger.info(f"Loaded {len(lines)} text lines")
    return lines

def main():
    parser = argparse.ArgumentParser(description="Domain-Adaptive Pre-training for DeBERTa-v3-base")
    parser.add_argument("--corpus", type=str, required=True, help="Path to corpus text file")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for saved model")
    parser.add_argument("--model-name", type=str, default="microsoft/deberta-v3-base", help="Base model name")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=0, help="Warmup steps (0 for 10% of total)")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--logging-steps", type=int, default=100, help="Log every N steps")
    parser.add_argument("--save-steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load tokenizer and model
    logger.info(f"Loading model and tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    model.to(device)

    # Load corpus
    texts = load_corpus(args.corpus)

    # Create dataset
    dataset = TextDataset(texts, tokenizer, args.max_length)

    # Data collator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator
    )

    # Calculate total steps
    total_steps = len(dataloader) * args.epochs // args.grad_acc
    if args.warmup_steps == 0:
        args.warmup_steps = int(0.1 * total_steps)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if args.fp16 else None

    # Training loop
    model.train()
    global_step = 0
    best_loss = float('inf')

    logger.info("Starting DAPT training...")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Gradient accumulation: {args.grad_acc}")
    logger.info(f"Effective batch size: {args.batch_size * args.grad_acc}")
    logger.info(f"Total steps: {total_steps}")
    logger.info(f"Warmup steps: {args.warmup_steps}")

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for step, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass with mixed precision
            with torch.cuda.amp.autocast() if args.fp16 else torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss / args.grad_acc  # Normalize loss for gradient accumulation

            # Backward pass
            if args.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            epoch_loss += loss.item()

            # Gradient accumulation
            if (step + 1) % args.grad_acc == 0:
                if args.fp16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % args.logging_steps == 0:
                    avg_loss = epoch_loss * args.grad_acc / (step + 1)
                    progress_bar.set_postfix({'loss': f'{avg_loss:.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})
                    logger.info(f"Step {global_step}/{total_steps} - Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")

                # Save checkpoint
                if global_step % args.save_steps == 0 and global_step > 0:
                    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
                    logger.info(f"Saved checkpoint to {checkpoint_dir}")

        # End of epoch
        avg_epoch_loss = epoch_loss * args.grad_acc / len(dataloader)
        logger.info(f"Epoch {epoch+1} completed - Average Loss: {avg_epoch_loss:.4f}")

        # Save epoch checkpoint
        epoch_dir = os.path.join(args.output_dir, f"epoch-{epoch+1}")
        os.makedirs(epoch_dir, exist_ok=True)
        model.save_pretrained(epoch_dir)
        tokenizer.save_pretrained(epoch_dir)
        logger.info(f"Saved epoch checkpoint to {epoch_dir}")

        # Save best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_dir = os.path.join(args.output_dir, "best_model")
            os.makedirs(best_dir, exist_ok=True)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            logger.info(f"Saved best model to {best_dir} with loss {best_loss:.4f}")

    # Save final model
    logger.info("Saving final model...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Training completed! Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()