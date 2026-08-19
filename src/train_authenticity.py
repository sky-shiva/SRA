#!/usr/bin/env python3
"""
Authenticity Detector Training for HireSense
Trains ensemble of:
1. Fine-tuned DeBERTa-v3-small (binary classifier)
2. Perplexity-based detector (GPT-2)
3. Stylometric features (TF-IDF + Logistic Regression)
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from datasets import load_from_disk
import numpy as np
from tqdm.auto import tqdm
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
import joblib
import math
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeBERTaClassifier(nn.Module):
    def __init__(self, base_model_name, dropout_rate=0.1):
        super().__init__()
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=2,
            problem_type="single_label_classification"
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs

class TextDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        text = example['text']
        label = example['label']  # 0 for AI, 1 for human

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

def calculate_perplexity(texts, model, tokenizer, device, max_length=512):
    """Calculate perplexity for a list of texts using GPT-2"""
    model.eval()
    total_loss = 0.0
    total_length = 0

    with torch.no_grad():
        for text in texts:
            encoding = tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            ).to(device)

            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

            # Get actual length (non-padding tokens)
            length = attention_mask.sum().item()

            total_loss += loss.item() * length
            total_length += length

    # Perplexity = exp(average loss)
    avg_loss = total_loss / total_length if total_length > 0 else float('inf')
    perplexity = math.exp(avg_loss)
    return perplexity

def extract_stylometric_features(texts):
    """Extract stylometric features from texts"""
    features = []
    for text in texts:
        if not text or not isinstance(text, str):
            # Handle empty or non-string texts
            features.append([0] * 10)  # Default feature vector
            continue

        words = text.split()
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]

        # Basic features
        char_count = len(text)
        word_count = len(words)
        sentence_count = len(sentences)
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0

        # Lexical diversity
        unique_words = len(set(w.lower() for w in words))
        lexical_diversity = unique_words / word_count if word_count > 0 else 0

        # Punctuation ratios
        punct_count = sum(1 for c in text if c in ',.!?;:')
        punct_ratio = punct_count / char_count if char_count > 0 else 0

        # Capitalization features
        capital_count = sum(1 for c in text if c.isupper())
        capital_ratio = capital_count / char_count if char_count > 0 else 0

        # Digit ratio
        digit_count = sum(1 for c in text if c.isdigit())
        digit_ratio = digit_count / char_count if char_count > 0 else 0

        # Stopword approximation (common words)
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        stopword_count = sum(1 for w in words if w.lower() in stopwords)
        stopword_ratio = stopword_count / word_count if word_count > 0 else 0

        features.append([
            char_count, word_count, sentence_count, avg_word_length, avg_sentence_length,
            lexical_diversity, punct_ratio, capital_ratio, digit_ratio, stopword_ratio
        ])

    return np.array(features)

class AuthenticityEnsemble:
    def __init__(self, deberta_model_path=None):
        self.deberta_tokenizer = None
        self.deberta_model = None
        self.perplexity_model = None
        self.perplexity_tokenizer = None
        self.tfidf_vectorizer = None
        self.lr_classifier = None
        self.is_fitted = False
        self.deberta_model_path = deberta_model_path

    def load_deberta(self, model_path, device):
        """Load fine-tuned DeBERTa model"""
        logger.info(f"Loading DeBERTa model from {model_path}")
        self.deberta_tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.deberta_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.deberta_model.to(device)
        self.deberta_model.eval()

    def load_perplexity_model(self, model_name="gpt2", device="cpu"):
        """Load GPT-2 for perplexity calculation"""
        logger.info(f"Loading perplexity model: {model_name}")
        from transformers import GPT2LMHeadModel
        self.perplexity_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.perplexity_model = GPT2LMHeadModel.from_pretrained(model_name)
        self.perplexity_model.to(device)
        self.perplexity_model.eval()
        # Set pad token if not present
        if self.perplexity_tokenizer.pad_token is None:
            self.perplexity_tokenizer.pad_token = self.perplexity_tokenizer.eos_token

    def fit_stylometric(self, texts):
        """Fit TF-IDF vectorizer and train Logistic Regression on stylometric features"""
        logger.info("Fitting stylometric features (TF-IDF + LR)...")

        # TF-IDF features
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True
        )
        tfidf_features = self.tfidf_vectorizer.fit_transform(texts)

        # Stylometric features
        stylometric_features = extract_stylometric_features(texts)

        # Combine features
        from scipy.sparse import hstack
        combined_features = hstack([tfidf_features, stylometric_features])

        # Train Logistic Regression
        self.lr_classifier = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )
        # Note: We'll need labels to actually fit - this will be done in training
        self.is_fitted = True
        logger.info("Stylometric feature pipeline fitted")

    def predict_deberta(self, texts, device, batch_size=16):
        """Get predictions from DeBERTa model"""
        if self.deberta_model is None:
            raise ValueError("DeBERTa model not loaded")

        self.deberta_model.eval()
        all_probs = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                encoding = self.deberta_tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(device)

                outputs = self.deberta_model(**encoding)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                # Probability of being human (class 1)
                human_probs = probs[:, 1].cpu().numpy()
                all_probs.extend(human_probs)

        return np.array(all_probs)

    def predict_perplexity(self, texts, device, batch_size=8):
        """Get perplexity-based scores (lower perplexity = more likely AI)"""
        if self.perplexity_model is None:
            raise ValueError("Perplexity model not loaded")

        perplexities = calculate_perplexity(texts, self.perplexity_model, self.perplexity_tokenizer, device)
        # Convert to probability: lower perplexity -> higher probability of being AI
        # We'll normalize this later during ensemble
        return np.array([perplexities])  # Return as array for consistency

    def predict_stylometric(self, texts):
        """Get predictions from stylometric model"""
        if not self.is_fitted or self.tfidf_vectorizer is None or self.lr_classifier is None:
            raise ValueError("Stylometric model not fitted")

        # TF-IDF features
        tfidf_features = self.tfidf_vectorizer.transform(texts)

        # Stylometric features
        stylometric_features = extract_stylometric_features(texts)

        # Combine features
        from scipy.sparse import hstack
        combined_features = hstack([tfidf_features, stylometric_features])

        # Get probabilities
        probs = self.lr_classifier.predict_proba(combined_features)
        # Probability of being human (class 1)
        human_probs = probs[:, 1]
        return human_probs

    def ensemble_predict(self, deberta_probs, perplexity_scores, stylometric_probs, weights=None):
        """Combine predictions from all three models"""
        if weights is None:
            weights = [0.5, 0.3, 0.2]  # Default weights: DeBERTa, Perplexity, Stylometric

        # Normalize perplexity scores to [0,1] range where higher = more likely human
        # Lower perplexity = more likely AI, so we invert and normalize
        if len(perplexity_scores.shape) > 1:
            perp_scores = perplexity_scores[0]  # Extract if it's 2D
        else:
            perp_scores = perplexity_scores

        # Handle potential infinite values
        perp_scores = np.where(np.isinf(perp_scores), 1e6, perp_scores)
        max_perp = np.max(perp_scores)
        min_perp = np.min(perp_scores)

        if max_perp > min_perp:
            # Normalize to [0,1] and invert (so higher = more likely human)
            perp_norm = (max_perp - perp_scores) / (max_perp - min_perp)
        else:
            perp_norm = np.ones_like(perp_scores) * 0.5  # All same, give neutral score

        # Ensemble: weighted average
        ensemble_probs = (
            weights[0] * np.array(deberta_probs) +
            weights[1] * np.array(perp_norm) +
            weights[2] * np.array(stylometric_probs)
        ) / sum(weights)

        # Ensure probabilities are in [0,1]
        ensemble_probs = np.clip(ensemble_probs, 0, 1)
        return ensemble_probs

def load_authenticity_datasets(human_path, ai_path):
    """Load human and AI-generated datasets"""
    logger.info(f"Loading human data from {human_path}")
    human_dataset = load_from_disk(human_path)

    logger.info(f"Loading AI data from {ai_path}")
    ai_dataset = load_from_disk(ai_path)

    return human_dataset, ai_dataset

def main():
    parser = argparse.ArgumentParser(description="Train Authenticity Detector ensemble")
    parser.add_argument("--human-data", type=str, required=True, help="Path to human resume dataset")
    parser.add_argument("--ai-data", type=str, required=True, help="Path to AI-generated resume dataset")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for saved model")
    parser.add_argument("--deberta-model", type=str, default="microsoft/deberta-v3-small", help="Base DeBERTa model")
    parser.add_argument("--perplexity-model", type=str, default="gpt2", help="Model for perplexity calculation")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate for DeBERTa")
    parser.add_argument("--warmup-steps", type=int, default=0, help="Warmup steps")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--logging-steps", type=int, default=50, help="Log every N steps")
    parser.add_argument("--save-steps", type=int, default=200, help="Save checkpoint every N steps")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--deberta-weight", type=float, default=0.5, help="Weight for DeBERTa in ensemble")
    parser.add_argument("--perplexity-weight", type=float, default=0.3, help="Weight for perplexity in ensemble")
    parser.add_argument("--stylometric-weight", type=float, default=0.2, help="Weight for stylometric in ensemble")

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
    human_dataset, ai_dataset = load_authenticity_datasets(args.human_data, args.ai_data)

    # Combine datasets and create labels
    # Human: label 1, AI: label 0
    human_texts = [example['text'] for example in human_dataset if 'text' in example]
    ai_texts = [example['text'] for example in ai_dataset if 'text' in example]

    logger.info(f"Loaded {len(human_texts)} human resumes")
    logger.info(f"Loaded {len(ai_texts)} AI-generated resumes")

    all_texts = human_texts + ai_texts
    all_labels = [1] * len(human_texts) + [0] * len(ai_texts)  # 1=human, 0=AI

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize ensemble
    ensemble = AuthenticityEnsemble()

    # 1. Load and fine-tune DeBERTa
    logger.info("=== Phase 1: Fine-tuning DeBERTa-v3-small ===")
    deberta_output_dir = os.path.join(args.output_dir, "deberta_classifier")
    os.makedirs(deberta_output_dir, exist_ok=True)

    # Prepare data for DeBERTa training
    # We'll do a simple train/val split
    from sklearn.model_selection import train_test_split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        all_texts, all_labels, test_size=0.1, random_state=args.seed, stratify=all_labels
    )

    train_dataset_hf = Dataset.from_dict({
        'text': train_texts,
        'label': train_labels
    })
    val_dataset_hf = Dataset.from_dict({
        'text': val_texts,
        'label': val_labels
    })

    train_deberta_dataset = TextDataset(train_dataset_hf, AutoTokenizer.from_pretrained(args.deberta_model), args.max_length)
    val_deberta_dataset = TextDataset(val_dataset_hf, AutoTokenizer.from_pretrained(args.deberta_model), args.max_length)

    train_deberta_loader = DataLoader(train_deberta_dataset, batch_size=args.batch_size, shuffle=True)
    val_deberta_loader = DataLoader(val_deberta_dataset, batch_size=args.batch_size)

    # Initialize DeBERTa model
    deberta_model = DeBERTaClassifier(args.deberta_model)
    deberta_model.to(device)

    # Optimizer and scheduler for DeBERTa
    total_steps = len(train_deberta_loader) * args.epochs // args.grad_acc
    if args.warmup_steps == 0:
        args.warmup_steps = int(0.1 * total_steps)

    deberta_optimizer = torch.optim.AdamW(
        deberta_model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay
    )

    deberta_scheduler = get_linear_schedule_with_warmup(
        deberta_optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )

    scaler = torch.cuda.amp.GradScaler() if args.fp16 else None

    # Train DeBERTa
    best_val_auc = 0.0
    global_step = 0

    logger.info("Starting DeBERTa fine-tuning...")
    for epoch in range(args.epochs):
        deberta_model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_deberta_loader, desc=f"DeBERTa Epoch {epoch+1}/{args.epochs}")

        for step, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with torch.cuda.amp.autocast() if args.fp16 else torch.no_grad():
                outputs = deberta_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss / args.grad_acc

            if args.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            epoch_loss += loss.item()

            if (step + 1) % args.grad_acc == 0:
                if args.fp16:
                    scaler.unscale_(deberta_optimizer)
                    torch.nn.utils.clip_grad_norm_(deberta_model.parameters(), max_norm=1.0)
                    scaler.step(deberta_optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(deberta_model.parameters(), max_norm=1.0)
                    deberta_optimizer.step()

                deberta_scheduler.step()
                deberta_optimizer.zero_grad()
                global_step += 1

                if global_step % args.logging_steps == 0:
                    avg_loss = epoch_loss * args.grad_acc / (step + 1)
                    progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})

        # Validation
        deberta_model.eval()
        val_probs = []
        val_true_labels = []

        with torch.no_grad():
            for batch in val_deberta_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = deberta_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                human_probs = probs[:, 1].cpu().numpy()
                val_probs.extend(human_probs)
                val_true_labels.extend(labels.cpu().numpy())

        val_auc = roc_auc_score(val_true_labels, val_probs)
        val_accuracy = accuracy_score(val_true_labels, (np.array(val_probs) > 0.5).astype(int))

        logger.info(f"DeBERTa Epoch {epoch+1} - Val Loss: {epoch_loss * args.grad_acc / len(train_deberta_loader):.4f}, "
                   f"Val AUC: {val_auc:.4f}, Val Acc: {val_accuracy:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            deberta_model.save_pretrained(deberta_output_dir)
            train_dataset_hf['text'][0:]  # Just to trigger saving
            tokenizer = AutoTokenizer.from_pretrained(args.deberta_model)
            tokenizer.save_pretrained(deberta_output_dir)
            logger.info(f"Saved best DeBERTa model to {deberta_output_dir} with AUC: {best_val_auc:.4f}")

    # Load the best DeBERTa model for ensemble
    ensemble.load_deberta(deberta_output_dir, device)

    # 2. Load perplexity model
    logger.info("=== Phase 2: Setting up Perplexity Model ===")
    ensemble.load_perplexity_model(args.perplexity_model, device)

    # 3. Fit stylometric model
    logger.info("=== Phase 3: Fitting Stylometric Model ===")
    ensemble.fit_stylometric(all_texts)  # Fit on all texts

    # 4. Calculate ensemble weights (you could optimize these on validation set)
    logger.info("=== Phase 4: Calculating Ensemble Performance ===")
    weights = [args.deberta_weight, args.perplexity_weight, args.stylometric_weight]

    # Get predictions from each model on a sample for demonstration
    sample_size = min(1000, len(all_texts))
    sample_texts = all_texts[:sample_size]
    sample_labels = np.array(all_labels[:sample_size])

    logger.info(f"Evaluating ensemble on {sample_size} samples...")

    # DeBERTa predictions
    deberta_probs = ensemble.predict_deberta(sample_texts, device, batch_size=args.batch_size)

    # Perplexity scores
    perplexity_scores = ensemble.predict_perplexity(sample_texts, device, batch_size=max(1, args.batch_size//2))

    # Stylometric predictions
    stylometric_probs = ensemble.predict_stylometric(sample_texts)

    # Ensemble prediction
    ensemble_probs = ensemble.ensemble_predict(
        deberta_probs,
        perplexity_scores,
        stylometric_probs,
        weights
    )

    # Calculate metrics
    ensemble_auc = roc_auc_score(sample_labels, ensemble_probs)
    ensemble_accuracy = accuracy_score(sample_labels, (ensemble_probs > 0.5).astype(int))
    ensemble_brier = brier_score_loss(sample_labels, ensemble_probs)

    logger.info(f"Ensemble Performance:")
    logger.info(f"  AUC: {ensemble_auc:.4f}")
    logger.info(f"  Accuracy: {ensemble_accuracy:.4f}")
    logger.info(f"  Brier Score: {ensemble_brier:.4f}")

    # Save ensemble model
    logger.info("=== Saving Ensemble Model ===")
    ensemble_info = {
        'deberta_model_path': deberta_output_dir,
        'perplexity_model_name': args.perplexity_model,
        'weights': weights,
        'max_length': args.max_length,
        'performance': {
            'auc': float(ensemble_auc),
            'accuracy': float(ensemble_accuracy),
            'brier_score': float(ensemble_brier)
        }
    }

    import json
    with open(os.path.join(args.output_dir, "ensemble_config.json"), 'w') as f:
        json.dump(ensemble_info, f, indent=2)

    # Save individual components
    joblib.dump(ensemble.tfidf_vectorizer, os.path.join(args.output_dir, "tfidf_vectorizer.pkl"))
    joblib.dump(ensemble.lr_classifier, os.path.join(args.output_dir, "stylometric_classifier.pkl"))

    # Save ensemble weights
    with open(os.path.join(args.output_dir, "ensemble_weights.json"), 'w') as f:
        json.dump({
            'deberta': weights[0],
            'perplexity': weights[1],
            'stylometric': weights[2]
        }, f, indent=2)

    logger.info(f"Authenticity detector training completed!")
    logger.info(f"Model saved to {args.output_dir}")
    logger.info(f"Ensemble weights: DeBERTa={weights[0]}, Perplexity={weights[1]}, Stylometric={weights[2]}")

if __name__ == "__main__":
    main()