#!/usr/bin/env python3
"""
Skill Extractor Training for HireSense
Trains DeBERTa-v3-base with NER head (BIO tagging) + Multi-label skill classification head
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
from datasets import load_from_disk
import numpy as np
from tqdm.auto import tqdm
import logging
from seqeval.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SkillExtractorModel(nn.Module):
    def __init__(self, base_model_name, num_ner_labels, num_skill_labels, dropout_rate=0.1):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(base_model_name)
        self.hidden_size = self.base_model.config.hidden_size

        # NER head (token-level classification)
        self.ner_dropout = nn.Dropout(dropout_rate)
        self.ner_classifier = nn.Linear(self.hidden_size, num_ner_labels)

        # Skill classification head (sequence-level multi-label)
        self.skill_dropout = nn.Dropout(dropout_rate)
        self.skill_classifier = nn.Linear(self.hidden_size, num_skill_labels)

    def forward(self, input_ids, attention_mask, ner_labels=None, skill_labels=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        sequence_output = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        pooled_output = outputs.pooler_output       # (batch_size, hidden_size)

        # NER predictions
        ner_sequence_output = self.ner_dropout(sequence_output)
        ner_logits = self.ner_classifier(ner_sequence_output)

        # Skill predictions
        skill_pooled_output = self.skill_dropout(pooled_output)
        skill_logits = self.skill_classifier(skill_pooled_output)

        loss = None
        if ner_labels is not None and skill_labels is not None:
            # NER loss
            loss_fct = nn.CrossEntropyLoss()
            active_loss = attention_mask.view(-1) == 1
            active_logits = ner_logits.view(-1, ner_logits.shape[-1])
            active_labels = torch.where(
                active_loss,
                ner_labels.view(-1),
                torch.tensor(loss_fct.ignore_index).type_as(ner_labels)
            )
            ner_loss = loss_fct(active_logits, active_labels)

            # Skill loss (multi-label binary cross-entropy)
            skill_loss_fct = nn.BCEWithLogitsLoss()
            skill_loss = skill_loss_fct(skill_logits, skill_labels.float())

            # Combined loss
            loss = ner_loss + skill_loss

        return {
            'loss': loss,
            'ner_logits': ner_logits,
            'skill_logits': skill_logits
        }

class SkillExtractionDataset(Dataset):
    def __init__(self, dataset, tokenizer, ner_label_to_id, skill_mlb, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.ner_label_to_id = ner_label_to_id
        self.skill_mlb = skill_mlb
        self.max_length = max_length
        self.id_to_ner_label = {v: k for k, v in ner_label_to_id.items()}

        # Define NER labels (BIO format for skills)
        # B-SKILL, I-SKILL, O
        # We'll create this dynamically from the data if needed

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        text = example['text']
        skills = example['skills']  # List of skill strings

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()

        # Create NER labels (BIO format)
        seq_len = len(input_ids)
        ner_labels = torch.full((seq_len,), self.ner_label_to_id['O'], dtype=torch.long)

        # For simplicity, we'll mark all tokens as O for now
        # In a real implementation, you would align skill spans with tokens
        # This is a simplified version - you'd need proper NER annotation

        # Create skill labels (multi-label binary)
        skill_labels = torch.zeros(len(self.skill_mlb.classes_), dtype=torch.float)
        if skills:
            # Transform skills to binary vector
            skill_binary = self.skill_mlb.transform([skills])
            skill_labels = torch.from_numpy(skill_binary[0]).float()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'ner_labels': ner_labels,
            'skill_labels': skill_labels
        }

def load_skill_datasets(train_path, val_path):
    """Load skill extraction datasets"""
    logger.info(f"Loading training data from {train_path}")
    train_dataset = load_from_disk(train_path)

    logger.info(f"Loading validation data from {val_path}")
    val_dataset = load_from_disk(val_path)

    return train_dataset, val_dataset

def extract_all_skills(dataset):
    """Extract all unique skills from dataset"""
    all_skills = set()
    for example in dataset:
        skills = example.get('skills', [])
        if isinstance(skills, list):
            all_skills.update(skills)
        elif isinstance(skills, str):
            # Handle case where skills might be a string
            all_skills.add(skills)
    return sorted(list(all_skills))

def main():
    parser = argparse.ArgumentParser(description="Train Skill Extractor model")
    parser.add_argument("--base-model", type=str, required=True, help="Path to DAPT model or base model name")
    parser.add_argument("--train-data", type=str, required=True, help="Path to training dataset")
    parser.add_argument("--val-data", type=str, required=True, help="Path to validation dataset")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for saved model")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=0, help="Warmup steps")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--logging-steps", type=int, default=50, help="Log every N steps")
    parser.add_argument("--save-steps", type=int, default=200, help="Save checkpoint every N steps")
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

    # Load datasets
    train_dataset, val_dataset = load_skill_datasets(args.train_data, args.val_data)

    # Extract all skills for multi-label classification
    logger.info("Extracting all unique skills...")
    train_skills = extract_all_skills(train_dataset)
    val_skills = extract_all_skills(val_dataset)
    all_skills = sorted(list(set(train_skills + val_skills)))

    logger.info(f"Found {len(all_skills)} unique skills")

    # Create MultiLabelBinarizer for skill classification
    skill_mlb = MultiLabelBinarizer()
    skill_mlb.fit([all_skills])  # Fit on all possible skills

    # Define NER labels (BIO format)
    ner_label_list = ['O', 'B-SKILL', 'I-SKILL']
    ner_label_to_id = {label: i for i, label in enumerate(ner_label_list)}
    num_ner_labels = len(ner_label_list)
    num_skill_labels = len(all_skills)

    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # Create datasets
    train_skill_dataset = SkillExtractionDataset(
        train_dataset, tokenizer, ner_label_to_id, skill_mlb, args.max_length
    )
    val_skill_dataset = SkillExtractionDataset(
        val_dataset, tokenizer, ner_label_to_id, skill_mlb, args.max_length
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_skill_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    val_dataloader = DataLoader(
        val_skill_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    # Initialize model
    logger.info(f"Initializing SkillExtractorModel from {args.base_model}")
    model = SkillExtractorModel(
        base_model_name=args.base_model,
        num_ner_labels=num_ner_labels,
        num_skill_labels=num_skill_labels,
        dropout_rate=0.1
    )
    model.to(device)

    # Calculate total steps
    total_steps = len(train_dataloader) * args.epochs // args.grad_acc
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
    best_val_f1 = 0.0

    logger.info("Starting Skill Extractor training...")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Gradient accumulation: {args.grad_acc}")
    logger.info(f"Effective batch size: {args.batch_size * args.grad_acc}")
    logger.info(f"Total steps: {total_steps}")
    logger.info(f"Warmup steps: {args.warmup_steps}")
    logger.info(f"Number of NER labels: {num_ner_labels}")
    logger.info(f"Number of skill labels: {num_skill_labels}")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")

        for step, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ner_labels = batch['ner_labels'].to(device)
            skill_labels = batch['skill_labels'].to(device)

            # Forward pass with mixed precision
            with torch.cuda.amp.autocast() if args.fp16 else torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    ner_labels=ner_labels,
                    skill_labels=skill_labels
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
                    current_lr = scheduler.get_last_lr()[0]
                    progress_bar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'lr': f'{current_lr:.2e}'
                    })
                    logger.info(f"Step {global_step}/{total_steps} - Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")

                # Save checkpoint
                if global_step % args.save_steps == 0 and global_step > 0:
                    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)

                    # Save skill MLB
                    with open(os.path.join(checkpoint_dir, "skill_mlb.json"), 'w') as f:
                        json.dump({
                            'classes': skill_mlb.classes_.tolist()
                        }, f)

                    logger.info(f"Saved checkpoint to {checkpoint_dir}")

        # End of epoch training
        avg_epoch_loss = epoch_loss * args.grad_acc / len(train_dataloader)
        logger.info(f"Epoch {epoch+1} training completed - Average Loss: {avg_epoch_loss:.4f}")

        # Validation
        logger.info(f"Running validation for epoch {epoch+1}...")
        val_loss, val_ner_f1, val_skill_f1 = validate_model(
            model, val_dataloader, device, ner_label_to_id, args.fp16
        )

        logger.info(f"Validation - Loss: {val_loss:.4f}, NER F1: {val_ner_f1:.4f}, Skill F1: {val_skill_f1:.4f}")

        # Save best model based on skill F1 score
        if val_skill_f1 > best_val_f1:
            best_val_f1 = val_skill_f1
            best_dir = os.path.join(args.output_dir, "best_model")
            os.makedirs(best_dir, exist_ok=True)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            logger.info(f"Saved best model to {best_dir} with Skill F1: {best_val_f1:.4f}")

        # Save epoch checkpoint
        epoch_dir = os.path.join(args.output_dir, f"epoch-{epoch+1}")
        os.makedirs(epoch_dir, exist_ok=True)
        model.save_pretrained(epoch_dir)
        tokenizer.save_pretrained(epoch_dir)
        logger.info(f"Saved epoch checkpoint to {epoch_dir}")

    # Save final model
    logger.info("Saving final model...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Training completed! Model saved to {args.output_dir}")
    logger.info(f"Best validation Skill F1: {best_val_f1:.4f}")

def validate_model(model, dataloader, device, ner_label_to_id, use_fp16):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    all_ner_preds = []
    all_ner_labels = []
    all_skill_preds = []
    all_skill_labels = []

    ner_id_to_label = {v: k for k, v in ner_label_to_id.items()}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ner_labels = batch['ner_labels'].to(device)
            skill_labels = batch['skill_labels'].to(device)

            # Forward pass
            with torch.cuda.amp.autocast() if use_fp16 else torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    ner_labels=ner_labels,
                    skill_labels=skill_labels
                )

                loss = outputs.loss
                ner_logits = outputs.ner_logits
                skill_logits = outputs.skill_logits

            total_loss += loss.item()

            # Get NER predictions
            ner_predictions = torch.argmax(ner_logits, dim=-1)

            # Get skill predictions (threshold at 0.5)
            skill_predictions = (torch.sigmoid(skill_logits) > 0.5).float()

            # Collect predictions and labels for metrics
            for i in range(input_ids.size(0)):
                # NER: only consider non-padding tokens
                mask = attention_mask[i] == 1
                true_ner_ids = ner_labels[i][mask].cpu().numpy()
                pred_ner_ids = ner_predictions[i][mask].cpu().numpy()

                true_ner_labels = [ner_id_to_label[id_] for id_ in true_ner_ids]
                pred_ner_labels = [ner_id_to_label[id_] for id_ in pred_ner_ids]

                all_ner_labels.append(true_ner_labels)
                all_ner_preds.append(pred_ner_labels)

                # Skill: collect for multi-label metrics
                all_skill_labels.append(skill_labels[i].cpu().numpy())
                all_skill_preds.append(skill_predictions[i].cpu().numpy())

    # Calculate average loss
    avg_loss = total_loss / len(dataloader)

    # Calculate NER F1 (seqeval format)
    try:
        ner_f1 = f1_score(all_ner_labels, all_ner_preds)
    except:
        ner_f1 = 0.0
        logger.warning("Could not compute NER F1 score")

    # Calculate Skill F1 (micro-average for multi-label)
    from sklearn.metrics import f1_score as sklearn_f1_score
    try:
        skill_f1 = sklearn_f1_score(
            np.array(all_skill_labels),
            np.array(all_skill_preds),
            average='micro',
            zero_division=0
        )
    except:
        skill_f1 = 0.0
        logger.warning("Could not compute Skill F1 score")

    return avg_loss, ner_f1, skill_f1

if __name__ == "__main__":
    main()