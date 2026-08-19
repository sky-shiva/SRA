#!/usr/bin/env python3
"""
Semantic Matcher Training for HireSense
Trains bi-encoder (retrieval) + cross-encoder (re-ranking) for resume-JD matching
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup
)
from sentence_transformers import SentenceTransformer, losses, util
from datasets import load_from_disk
import numpy as np
from tqdm.auto import tqdm
import logging
from sklearn.metrics import roc_auc_score, accuracy_score
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BiEncoderDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        resume = example['resume']
        job_description = example['job_description']
        label = example.get('label', 1)  # Default to 1 for positive pairs

        # Tokenize resume
        resume_encoding = self.tokenizer(
            resume,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Tokenize job description
        jd_encoding = self.tokenizer(
            job_description,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'resume_input_ids': resume_encoding['input_ids'].flatten(),
            'resume_attention_mask': resume_encoding['attention_mask'].flatten(),
            'jd_input_ids': jd_encoding['input_ids'].flatten(),
            'jd_attention_mask': jd_encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.float)
        }

class BiEncoderModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Use [CLS] token embedding
        embeddings = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
        return embeddings

class CrossEncoderDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        resume = example['resume']
        job_description = example['job_description']
        label = example.get('label', 1)

        # Concatenate resume and JD with separator
        text = f"{resume} [SEP] {job_description}"

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
            'labels': torch.tensor(label, dtype=torch.long)
        }

class CrossEncoderModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size
        self.classifier = nn.Linear(self.hidden_size, 2)  # Binary classification

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Use [CLS] token embedding
        pooled_output = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))

        return {
            'loss': loss,
            'logits': logits
        }

def load_matching_dataset(pairs_path):
    """Load positive resume-JD pairs"""
    logger.info(f"Loading positive pairs from {pairs_path}")
    dataset = load_from_disk(pairs_path)
    return dataset

def generate_hard_negatives(resume_texts, jd_texts, bi_encoder_model, resume_tokenizer, jd_tokenizer, device, num_negatives=5):
    """Generate hard negatives using bi-encoder for mining"""
    logger.info("Generating hard negatives...")

    bi_encoder_model.eval()

    # Encode all resumes and JDs
    resume_embeddings = []
    jd_embeddings = []

    # Encode resumes in batches
    batch_size = 32
    for i in range(0, len(resume_texts), batch_size):
        batch_resumes = resume_texts[i:i+batch_size]
        encoding = resume_tokenizer(
            batch_resumes,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            outputs = bi_encoder_model.model(
                input_ids=encoding['input_ids'],
                attention_mask=encoding['attention_mask']
            )
            embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS]
            resume_embeddings.append(embeddings.cpu())

    # Encode JDs in batches
    for i in range(0, len(jd_texts), batch_size):
        batch_jds = jd_texts[i:i+batch_size]
        encoding = jd_tokenizer(
            batch_jds,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            outputs = bi_encoder_model.model(
                input_ids=encoding['input_ids'],
                attention_mask=encoding['attention_mask']
            )
            embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS]
            jd_embeddings.append(embeddings.cpu())

    resume_embeddings = torch.cat(resume_embeddings, dim=0)
    jd_embeddings = torch.cat(jd_embeddings, dim=0)

    # Compute similarity matrix
    similarity_matrix = torch.mm(resume_embeddings, jd_embeddings.t())

    # For each resume, find hardest negative JDs (highest similarity but not the true pair)
    hard_negatives = []

    for i in range(len(resume_texts)):
        # Get similarities for this resume
        similarities = similarity_matrix[i]

        # Sort by similarity (descending)
        sorted_indices = torch.argsort(similarities, descending=True)

        # Take top negatives excluding potential true positives (we don't know true pairs here)
        # In practice, you'd exclude known positives, but for simplicity we'll take top-k as hard negatives
        num_to_take = min(num_negatives, len(sorted_indices))
        hard_negative_indices = sorted_indices[:num_to_take]

        for jd_idx in hard_negative_indices:
            hard_negatives.append({
                'resume': resume_texts[i],
                'job_description': jd_texts[jd_idx.item()],
                'label': 0  # Negative pair
            })

    logger.info(f"Generated {len(hard_negatives)} hard negative pairs")
    return hard_negatives

def train_bi_encoder(args, device):
    """Train the bi-encoder model"""
    logger.info("=== Training Bi-Encoder (Retrieval) ===")

    # Load positive pairs
    positive_dataset = load_matching_dataset(args.positive_pairs)
    positive_texts = [(example['resume'], example['job_description']) for example in positive_dataset]

    logger.info(f"Loaded {len(positive_texts)} positive resume-JD pairs")

    # Extract resume and JD texts
    resume_texts = [pair[0] for pair in positive_texts]
    jd_texts = [pair[1] for pair in positive_texts]

    # Initialize model and tokenizer
    logger.info(f"Initializing bi-encoder with {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = BiEncoderModel(args.base_model)
    model.to(device)

    # Create dataset
    train_dataset = BiEncoderDataset(positive_dataset, tokenizer, args.max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Optimizer and scheduler
    total_steps = len(train_dataloader) * args.epochs // args.grad_acc
    if args.warmup_steps == 0:
        args.warmup_steps = int(0.1 * total_steps)

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

    scaler = torch.cuda.amp.GradScaler() if args.fp16 else None

    # Training loop
    model.train()
    global_step = 0
    best_loss = float('inf')

    logger.info("Starting bi-encoder training...")
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Bi-Encoder Epoch {epoch+1}/{args.epochs}")

        for step, batch in enumerate(progress_bar):
            # Move batch to device
            resume_input_ids = batch['resume_input_ids'].to(device)
            resume_attention_mask = batch['resume_attention_mask'].to(device)
            jd_input_ids = batch['jd_input_ids'].to(device)
            jd_attention_mask = batch['jd_attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            with torch.cuda.amp.autocast() if args.fp16 else torch.no_grad():
                # Get embeddings
                resume_embeds = model(resume_input_ids, resume_attention_mask)
                jd_embeds = model(jd_input_ids, jd_attention_mask)

                # Cosine similarity
                cosine_sim = torch.nn.functional.cosine_similarity(resume_embeds, jd_embeds)

                # Contrastive loss (simplified - in practice use multiple negatives)
                # For now, we'll use a simple binary classification loss on similarity
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(cosine_sim, labels) / args.grad_acc

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

                if global_step % args.logging_steps == 0:
                    avg_loss = epoch_loss * args.grad_acc / (step + 1)
                    progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})

        avg_epoch_loss = epoch_loss * args.grad_acc / len(train_dataloader)
        logger.info(f"Bi-Encoder Epoch {epoch+1} - Average Loss: {avg_epoch_loss:.4f}")

        # Save checkpoint
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            bi_encoder_dir = os.path.join(args.output_dir, "bi_encoder", "best_model")
            os.makedirs(bi_encoder_dir, exist_ok=True)
            model.model.save_pretrained(bi_encoder_dir)
            tokenizer.save_pretrained(bi_encoder_dir)
            logger.info(f"Saved best bi-encoder to {bi_encoder_dir}")

    # Save final bi-encoder
    bi_encoder_final_dir = os.path.join(args.output_dir, "bi_encoder")
    os.makedirs(bi_encoder_final_dir, exist_ok=True)
    model.model.save_pretrained(bi_encoder_final_dir)
    tokenizer.save_pretrained(bi_encoder_final_dir)

    logger.info(f"Bi-encoder training completed. Saved to {bi_encoder_final_dir}")
    return model, tokenizer

def train_cross_encoder(args, device, resume_texts, jd_texts):
    """Train the cross-encoder model with hard negative mining"""
    logger.info("=== Training Cross-Encoder (Re-ranking) ===")

    # Load positive pairs
    positive_dataset = load_matching_dataset(args.positive_pairs)
    positive_pairs = [(example['resume'], example['job_description']) for example in positive_dataset]

    logger.info(f"Loaded {len(positive_pairs)} positive resume-JD pairs")

    # Initialize bi-encoder for hard negative mining (use base model)
    logger.info("Initializing bi-encoder for hard negative mining...")
    bi_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    bi_model = BiEncoderModel(args.base_model)
    bi_model.to(device)
    bi_model.eval()

    # Generate hard negatives
    hard_negatives = generate_hard_negatives(
        resume_texts, jd_texts, bi_model, bi_tokenizer, bi_tokenizer,
        device, num_negatives=args.hard_negatives
    )

    # Combine positives and hard negatives
    all_pairs = []
    all_labels = []

    # Add positives
    for resume, jd in positive_pairs:
        all_pairs.append({'resume': resume, 'job_description': jd, 'label': 1})
        all_labels.append(1)

    # Add hard negatives
    for neg_pair in hard_negatives:
        all_pairs.append(neg_pair)
        all_labels.append(0)

    logger.info(f"Total training pairs: {len(all_pairs)} ({sum(all_labels)} positives, {len(all_labels)-sum(all_labels)} negatives)")

    # Create dataset
    cross_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    cross_dataset = CrossEncoderDataset(all_pairs, cross_tokenizer, args.max_length)
    cross_dataloader = DataLoader(cross_dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize cross-encoder model
    logger.info(f"Initializing cross-encoder with {args.base_model}")
    cross_model = CrossEncoderModel(args.base_model)
    cross_model.to(device)

    # Optimizer and scheduler
    total_steps = len(cross_dataloader) * args.epochs // args.grad_acc
    if args.warmup_steps == 0:
        args.warmup_steps = int(0.1 * total_steps)

    optimizer = torch.optim.AdamW(
        cross_model.parameters(),
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

    scaler = torch.cuda.amp.GradScaler() if args.fp16 else None

    # Training loop
    cross_model.train()
    global_step = 0
    best_loss = float('inf')

    logger.info("Starting cross-encoder training...")
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(cross_dataloader, desc=f"Cross-Encoder Epoch {epoch+1}/{args.epochs}")

        for step, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            with torch.cuda.amp.autocast() if args.fp16 else torch.no_grad():
                outputs = cross_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss / args.grad_acc

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
                    torch.nn.utils.clip_grad_norm_(cross_model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(cross_model.parameters(), max_norm=1.0)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.logging_steps == 0:
                    avg_loss = epoch_loss * args.grad_acc / (step + 1)
                    progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})

        avg_epoch_loss = epoch_loss * args.grad_acc / len(cross_dataloader)
        logger.info(f"Cross-Encoder Epoch {epoch+1} - Average Loss: {avg_epoch_loss:.4f}")

        # Save checkpoint
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            cross_encoder_dir = os.path.join(args.output_dir, "cross_encoder", "best_model")
            os.makedirs(cross_encoder_dir, exist_ok=True)
            cross_model.model.save_pretrained(cross_encoder_dir)
            cross_tokenizer.save_pretrained(cross_encoder_dir)
            logger.info(f"Saved best cross-encoder to {cross_encoder_dir}")

    # Save final cross-encoder
    cross_encoder_final_dir = os.path.join(args.output_dir, "cross_encoder")
    os.makedirs(cross_encoder_final_dir, exist_ok=True)
    cross_model.model.save_pretrained(cross_encoder_final_dir)
    cross_tokenizer.save_pretrained(cross_encoder_final_dir)

    logger.info(f"Cross-encoder training completed. Saved to {cross_encoder_final_dir}")
    return cross_model, cross_tokenizer

def main():
    parser = argparse.ArgumentParser(description="Train Semantic Matcher (Bi-Encoder + Cross-Encoder)")
    parser.add_argument("--positive-pairs", type=str, required=True, help="Path to positive resume-JD pairs dataset")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for saved models")
    parser.add_argument("--base-model", type=str, default="BAAI/bge-base-en-v1.5", help="Base model for bi-encoder")
    parser.add_argument("--cross-encoder-model", type=str, default="microsoft/deberta-v3-base", help="Model for cross-encoder")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate for bi-encoder")
    parser.add_argument("--cross-encoder-lr", type=float, default=2e-5, help="Learning rate for cross-encoder")
    parser.add_argument("--warmup-steps", type=int, default=0, help="Warmup steps")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--logging-steps", type=int, default=50, help="Log every N steps")
    parser.add_argument("--save-steps", type=int, default=200, help="Save checkpoint every N steps")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--hard-negatives", type=int, default=5, help="Number of hard negatives to mine per positive")

    args = parser.parse_args()

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data for hard negative mining
    logger.info("Loading data for training...")
    positive_dataset = load_matching_dataset(args.positive_pairs)
    resume_texts = [example['resume'] for example in positive_dataset]
    jd_texts = [example['job_description'] for example in positive_dataset]

    logger.info(f"Loaded {len(resume_texts)} resumes and {len(jd_texts)} job descriptions")

    # Train bi-encoder
    bi_encoder_model, bi_encoder_tokenizer = train_bi_encoder(args, device)

    # Temporarily override learning rate for cross-encoder
    original_lr = args.learning_rate
    args.learning_rate = args.cross_encoder_lr

    # Train cross-encoder
    cross_encoder_model, cross_encoder_tokenizer = train_cross_encoder(
        args, device, resume_texts, jd_texts
    )

    # Restore original learning rate
    args.learning_rate = original_lr

    # Save matcher configuration
    matcher_config = {
        'bi_encoder_model': args.base_model,
        'cross_encoder_model': args.cross_encoder_model,
        'max_length': args.max_length,
        'training_completed': True
    }

    import json
    with open(os.path.join(args.output_dir, "matcher_config.json"), 'w') as f:
        json.dump(matcher_config, f, indent=2)

    logger.info(f"Semantic matcher training completed!")
    logger.info(f"Models saved to {args.output_dir}")
    logger.info(f"Bi-encoder: {os.path.join(args.output_dir, 'bi_encoder')}")
    logger.info(f"Cross-encoder: {os.path.join(args.output_dir, 'cross_encoder')}")

if __name__ == "__main__":
    main()