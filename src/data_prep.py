#!/usr/bin/env python3
"""
Data Preparation Script for HireSense NLP Pipeline

Downloads and processes all required datasets from Hugging Face:
1. lang-uk/recruitment-dataset-candidate-profiles-english (210,250 CVs)
2. lang-uk/recruitment-dataset-job-descriptions-english (141,897 jobs)
3. TechWolf/Synthetic-ESCO-skill-sentences (138,260 sentences)
4. imocha-ai-org/ssf-skill-extraction-pairs (21,958 pairs)
5. datasetmaster/resumes (~5,000 resumes)
6. mounimzad/brainhr-plus (15 annotated pairs)

Also generates AI resume corpus for authenticity detector training.
"""

import os
import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass
class DatasetConfig:
    """Configuration for each dataset to download."""
    hf_id: str
    local_name: str
    purpose: str
    split: str = "train"
    sample_size: Optional[int] = None  # None = full dataset


# Dataset configurations
DATASETS = {
    "candidate_profiles": DatasetConfig(
        hf_id="lang-uk/recruitment-dataset-candidate-profiles-english",
        local_name="candidate_profiles",
        purpose="DAPT corpus, skill extraction corpus, authenticity positive samples",
        split="train",
        sample_size=None  # Use full 210,250
    ),
    "job_descriptions": DatasetConfig(
        hf_id="lang-uk/recruitment-dataset-job-descriptions-english",
        local_name="job_descriptions",
        purpose="Semantic matching corpus, JD encoding",
        split="train",
        sample_size=None  # Use full 141,897
    ),
    "esco_sentences": DatasetConfig(
        hf_id="TechWolf/Synthetic-ESCO-skill-sentences",
        local_name="esco_sentences",
        purpose="Skill taxonomy alignment, NER weak supervision",
        split="train",
        sample_size=None  # Use full 138,260
    ),
    "skill_extraction_pairs": DatasetConfig(
        hf_id="imocha-ai-org/ssf-skill-extraction-pairs",
        local_name="skill_extraction_pairs",
        purpose="Gold standard skill extraction supervision",
        split="train",
        sample_size=None  # Use full 21,958
    ),
    "resume_corpus": DatasetConfig(
        hf_id="datasetmaster/resumes",
        local_name="resume_corpus",
        purpose="Resume parsing, section segmentation",
        split="train",
        sample_size=None
    ),
    "brainhr_plus": DatasetConfig(
        hf_id="mounimzad/brainhr-plus",
        local_name="brainhr_plus",
        purpose="Gold standard evaluation (NEVER for training)",
        split="train",
        sample_size=None  # Only 15 pairs
    ),
}


def get_device_info():
    """Get GPU/CPU info for logging."""
    import torch
    if torch.cuda.is_available():
        return f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB"
    return "CPU only"


def download_dataset(config: DatasetConfig, output_dir: Path, cache_dir: Optional[Path] = None) -> Dataset:
    """Download a single dataset from Hugging Face."""
    print(f"\n{'='*60}")
    print(f"Downloading: {config.hf_id}")
    print(f"Purpose: {config.purpose}")
    print(f"{'='*60}")

    try:
        # Load dataset with cache_dir if provided
        if cache_dir:
            dataset = load_dataset(config.hf_id, split=config.split, cache_dir=str(cache_dir))
        else:
            dataset = load_dataset(config.hf_id, split=config.split)

        print(f"Loaded {len(dataset)} samples")

        # Sample if specified
        if config.sample_size and config.sample_size < len(dataset):
            dataset = dataset.shuffle(seed=42).select(range(config.sample_size))
            print(f"Sampled to {config.sample_size} samples")

        # Save to local directory
        local_path = output_dir / config.local_name
        local_path.mkdir(parents=True, exist_ok=True)

        # Save as Arrow format for fast loading
        dataset.save_to_disk(str(local_path))
        print(f"Saved to: {local_path}")

        # Also save a Parquet sample for inspection
        sample_path = local_path / "sample.parquet"
        if len(dataset) > 100:
            sample_df = dataset.select(range(100)).to_pandas()
        else:
            sample_df = dataset.to_pandas()
        sample_df.to_parquet(sample_path, index=False)
        print(f"Sample saved to: {sample_path}")

        return dataset

    except Exception as e:
        print(f"ERROR downloading {config.hf_id}: {e}")
        raise


def inspect_dataset(dataset: Dataset, name: str, num_samples: int = 3):
    """Print dataset structure and sample records."""
    print(f"\n--- {name} Structure ---")
    print(f"Features: {dataset.features}")
    print(f"Number of samples: {len(dataset)}")

    print(f"\nFirst {num_samples} samples:")
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        print(f"\n  Sample {i}:")
        for key, value in sample.items():
            if isinstance(value, str) and len(value) > 200:
                value = value[:200] + "..."
            print(f"    {key}: {value}")


def create_skill_extraction_dataset(
    imocha_dataset: Dataset,
    esco_dataset: Dataset,
    output_dir: Path
) -> Dataset:
    """
    Create unified skill extraction training dataset.
    Combines imocha (gold) + ESCO sentences (weak supervision).
    """
    print("\n" + "="*60)
    print("Creating unified skill extraction dataset")
    print("="*60)

    # Process imocha pairs (text -> skills)
    imocha_records = []
    for sample in tqdm(imocha_dataset, desc="Processing imocha pairs"):
        # The dataset has 'text' and 'skills' fields
        text = sample.get('text', '')
        skills = sample.get('skills', [])

        if text and skills:
            imocha_records.append({
                'text': text,
                'skills': skills,
                'source': 'imocha_gold',
                'has_ner_labels': False  # Will be generated later
            })

    print(f"Processed {len(imocha_records)} imocha records")

    # Process ESCO sentences (sentence -> skill)
    esco_records = []
    for sample in tqdm(esco_dataset, desc="Processing ESCO sentences"):
        sentence = sample.get('sentence', '') or sample.get('text', '')
        skill = sample.get('skill', '') or sample.get('label', '')

        if sentence and skill:
            esco_records.append({
                'text': sentence,
                'skills': [skill],
                'source': 'esco_weak',
                'has_ner_labels': False
            })

    print(f"Processed {len(esco_records)} ESCO records")

    # Combine
    all_records = imocha_records + esco_records
    print(f"Total skill extraction records: {len(all_records)}")

    # Create dataset
    combined_dataset = Dataset.from_list(all_records)

    # Save
    output_path = output_dir / "skill_extraction_combined"
    output_path.mkdir(parents=True, exist_ok=True)
    combined_dataset.save_to_disk(str(output_path))
    print(f"Saved combined dataset to: {output_path}")

    return combined_dataset


def create_dapt_corpus(candidate_profiles: Dataset, output_dir: Path, max_samples: int = 50000):
    """
    Create DAPT corpus from candidate profiles.
    Extracts text from profiles for domain-adaptive pre-training.
    """
    print("\n" + "="*60)
    print("Creating DAPT corpus")
    print("="*60)

    # Sample if needed
    if len(candidate_profiles) > max_samples:
        candidate_profiles = candidate_profiles.shuffle(seed=42).select(range(max_samples))
        print(f"Sampled {max_samples} profiles for DAPT")

    texts = []
    for sample in tqdm(candidate_profiles, desc="Extracting text for DAPT"):
        # Try different possible text fields
        text = ""
        for field in ['text', 'profile', 'resume', 'cv', 'content', 'description']:
            if field in sample and sample[field]:
                text = sample[field]
                break

        if text and len(text) > 50:  # Minimum length
            texts.append(text)

    print(f"Extracted {len(texts)} texts for DAPT")

    # Save as text file (one document per line)
    dapt_path = output_dir / "dapt_corpus.txt"
    with open(dapt_path, 'w', encoding='utf-8') as f:
        for text in texts:
            # Clean and write
            cleaned = ' '.join(text.split())  # Normalize whitespace
            f.write(cleaned + '\n')

    print(f"DAPT corpus saved to: {dapt_path}")

    # Also save as dataset for flexibility
    dapt_dataset = Dataset.from_dict({'text': texts})
    dapt_dataset.save_to_disk(str(output_dir / "dapt_corpus"))

    return texts


def create_authenticity_dataset(
    candidate_profiles: Dataset,
    output_dir: Path,
    num_ai_samples: int = 20000
) -> Dataset:
    """
    Create authenticity detector dataset.
    Positive: Human resumes from candidate profiles
    Negative: AI-generated resumes (to be generated)

    Note: AI generation requires API keys - this creates the structure
    and placeholder for human samples.
    """
    print("\n" + "="*60)
    print("Creating authenticity dataset structure")
    print("="*60)

    # Sample human resumes
    human_samples = candidate_profiles.shuffle(seed=42).select(range(min(15000, len(candidate_profiles))))

    human_records = []
    for sample in tqdm(human_samples, desc="Processing human resumes"):
        text = ""
        # Try candidate_profiles specific fields first
        for field in ['CV', 'cv', 'text', 'profile', 'resume', 'content', 'description']:
            if field in sample and sample[field]:
                text = sample[field]
                break

        if text and len(text) > 100:
            human_records.append({
                'text': text,
                'label': 1,  # Human
                'source': 'languk_human',
                'generation_method': 'original'
            })

    print(f"Prepared {len(human_records)} human resume samples")
    if len(human_records) == 0:
        print("WARNING: No human samples found! Check field names in candidate profiles dataset.")
        print("Available fields:", list(candidate_profiles[0].keys()) if len(candidate_profiles) > 0 else "No samples available")

    print(f"Target AI samples: {num_ai_samples}")
    print("NOTE: AI resume generation requires LLM API access (OpenAI, Anthropic, etc.)")
    print("For free alternatives, we'll use rule-based text transformations instead.")
    print("Run scripts/generate_ai_resumes_free.py after updating this script")

    print(f"Prepared {len(human_records)} human resume samples")
    print(f"Target AI samples: {num_ai_samples}")
    print("NOTE: AI resume generation requires LLM API access (OpenAI, Anthropic, etc.)")
    print("Run scripts/generate_ai_resumes.py after setting up API keys")

    # Save human portion
    human_dataset = Dataset.from_list(human_records)
    human_dataset.save_to_disk(str(output_dir / "authenticity_human"))

    # Create template for AI samples
    ai_template = {
        'text': '',
        'label': 0,  # AI
        'source': 'generated',
        'generation_method': '',  # 'pure', 'rewrite', 'hybrid', 'adversarial'
        'model_used': '',  # 'gpt-4o', 'claude-3.5-sonnet', 'llama-3.1-70b', etc.
        'prompt_template': ''
    }

    # Save template
    template_path = output_dir / "authenticity_ai_template.json"
    with open(template_path, 'w') as f:
        json.dump(ai_template, indent=2)

    print(f"Human data saved. AI template saved to: {template_path}")

    return human_dataset


def create_matching_dataset(
    candidate_profiles: Dataset,
    job_descriptions: Dataset,
    output_dir: Path,
    num_positive: int = 20000
) -> Dataset:
    """
    Create resume-JD matching dataset.
    Positive pairs: Same domain profiles + JDs
    Negative pairs: Random + hard negatives (mined later)
    """
    print("\n" + "="*60)
    print("Creating semantic matching dataset")
    print("="*60)

    # For now, create a basic structure
    # Hard negative mining will be done in training script after bi-encoder training

    # Sample profiles and JDs
    profiles_sample = candidate_profiles.shuffle(seed=42).select(range(min(10000, len(candidate_profiles))))
    jds_sample = job_descriptions.shuffle(seed=42).select(range(min(10000, len(job_descriptions))))

    # Extract text from profiles
    profile_texts = []
    for sample in tqdm(profiles_sample, desc="Extracting profile texts"):
        text = ""
        for field in ['text', 'profile', 'resume', 'cv', 'content', 'description']:
            if field in sample and sample[field]:
                text = sample[field]
                break
        if text and len(text) > 100:
            profile_texts.append(text)

    # Extract text from JDs
    jd_texts = []
    for sample in tqdm(jds_sample, desc="Extracting JD texts"):
        text = ""
        # Try job_descriptions specific fields first
        for field in ['Long Description', 'text', 'description', 'job_description', 'jd', 'content']:
            if field in sample and sample[field]:
                text = sample[field]
                break
        if text and len(text) > 100:
            jd_texts.append(text)

    print(f"Extracted {len(profile_texts)} profiles and {len(jd_texts)} JDs")

    # Create positive pairs (simple heuristic: pair by index for now)
    # Real pairing will use domain matching in training script
    positive_pairs = []
    min_len = min(len(profile_texts), len(jd_texts), num_positive)
    for i in range(min_len):
        positive_pairs.append({
            'resume': profile_texts[i],
            'job_description': jd_texts[i],
            'label': 1,
            'pair_type': 'positive_same_domain'
        })

    print(f"Created {len(positive_pairs)} positive pairs")
    print("Hard negative mining will be performed in train_matcher.py")

    # Save
    matching_dataset = Dataset.from_list(positive_pairs)
    matching_dataset.save_to_disk(str(output_dir / "matching_positive_pairs"))

    return matching_dataset


def create_train_val_test_splits(
    skill_dataset: Dataset,
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
):
    """Create stratified train/val/test splits."""
    print("\n" + "="*60)
    print("Creating train/val/test splits")
    print("="*60)

    # Shuffle
    dataset = skill_dataset.shuffle(seed=42)
    n = len(dataset)

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    splits = {
        'train': dataset.select(range(train_end)),
        'val': dataset.select(range(train_end, val_end)),
        'test': dataset.select(range(val_end, n))
    }

    for split_name, split_dataset in splits.items():
        split_path = output_dir / f"skill_extraction_{split_name}"
        split_dataset.save_to_disk(str(split_path))
        print(f"  {split_name}: {len(split_dataset)} samples -> {split_path}")

    # Save split indices for reproducibility
    split_info = {
        'train_indices': list(range(train_end)),
        'val_indices': list(range(train_end, val_end)),
        'test_indices': list(range(val_end, n)),
        'total': n,
        'ratios': {'train': train_ratio, 'val': val_ratio, 'test': test_ratio}
    }

    with open(output_dir / "split_info.json", 'w') as f:
        json.dump(split_info, indent=2)

    print("Split info saved to split_info.json")

    return splits


def download_additional_datasets(output_dir: Path, cache_dir: Optional[Path] = None):
    """Download additional datasets for specific purposes."""
    print("\n" + "="*60)
    print("Downloading additional datasets")
    print("="*60)

    additional = {
        "ai_detection_baseline": "cl-tohoku/ai-generated-text-detection",
        "stsb_hard_negatives": "mteb/stsb-hard-negatives",
    }

    for name, hf_id in additional.items():
        try:
            print(f"\nDownloading {hf_id}...")
            dataset = load_dataset(hf_id, split="train", cache_dir=str(cache_dir) if cache_dir else None)
            local_path = output_dir / name
            local_path.mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(str(local_path))
            print(f"Saved {len(dataset)} samples to {local_path}")
        except Exception as e:
            print(f"WARNING: Could not download {hf_id}: {e}")

def get_device_info():
    """Get GPU/CPU info for logging."""
    try:
        import torch
        if torch.cuda.is_available():
            return f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB"
        return "CPU only"
    except ImportError:
        return "CPU only (torch not installed)"

def main():
    parser = argparse.ArgumentParser(description="Download and prepare HireSense datasets")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/processed",
        help="Output directory for processed datasets"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./data/raw",
        help="Cache directory for HF downloads"
    )
    parser.add_argument(
        "--dapt-samples",
        type=int,
        default=50000,
        help="Number of samples for DAPT corpus"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading, only process existing data"
    )
    parser.add_argument(
        "--download-additional",
        action="store_true",
        help="Also download additional datasets (AI detection, STS hard negatives)"
    )
    parser.add_argument(
        "--max-samples-per-dataset",
        type=int,
        default=None,
        help="Maximum samples to download per dataset for testing (None = full dataset)"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("HireSense Data Preparation")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Cache directory: {cache_dir}")
    print(f"Device: {get_device_info()}")
    if args.max_samples_per_dataset:
        print(f"Limiting to {args.max_samples_per_dataset} samples per dataset for testing")

    # Track downloaded datasets
    downloaded = {}

    if not args.skip_download:
        # Download all primary datasets
        for key, config in DATASETS.items():
            try:
                # Override sample size if testing limit is set
                test_config = DatasetConfig(
                    hf_id=config.hf_id,
                    local_name=config.local_name,
                    purpose=config.purpose,
                    split=config.split,
                    sample_size=args.max_samples_per_dataset if args.max_samples_per_dataset and args.max_samples_per_dataset < (config.sample_size or float('inf')) else config.sample_size
                )
                dataset = download_dataset(test_config, output_dir, cache_dir)
                downloaded[key] = dataset
                inspect_dataset(dataset, key)
            except Exception as e:
                print(f"FAILED to download {key}: {e}")
                # Continue with other datasets

        # Download additional datasets if requested
        if args.download_additional:
            download_additional_datasets(output_dir, cache_dir)
    else:
        # Load from local
        print("Loading from local cache...")
        for key, config in DATASETS.items():
            local_path = output_dir / config.local_name
            if local_path.exists():
                downloaded[key] = Dataset.load_from_disk(str(local_path))
                print(f"Loaded {key}: {len(downloaded[key])} samples")
            else:
                print(f"WARNING: {key} not found at {local_path}")

    # Process datasets into training-ready formats
    if "skill_extraction_pairs" in downloaded and "esco_sentences" in downloaded:
        skill_dataset = create_skill_extraction_dataset(
            downloaded["skill_extraction_pairs"],
            downloaded["esco_sentences"],
            output_dir
        )

        # Create train/val/test splits
        create_train_val_test_splits(skill_dataset, output_dir)

    if "candidate_profiles" in downloaded:
        # Create DAPT corpus
        create_dapt_corpus(
            downloaded["candidate_profiles"],
            output_dir,
            max_samples=args.dapt_samples if not args.max_samples_per_dataset else min(args.dapt_samples, args.max_samples_per_dataset)
        )

        # Create authenticity dataset structure
        create_authenticity_dataset(
            downloaded["candidate_profiles"],
            output_dir
        )

        # Create matching dataset if JDs also available
        if "job_descriptions" in downloaded:
            create_matching_dataset(
                downloaded["candidate_profiles"],
                downloaded["job_descriptions"],
                output_dir
            )

    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE")
    print("="*60)
    print(f"\nAll processed data saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Run DAPT: python src/dapt.py --corpus ./data/processed/dapt_corpus.txt")
    print("2. Generate AI resumes: python scripts/generate_ai_resumes_free.py (free alternative)")
    print("3. Train Skill Extractor: python src/train_skill_extractor.py")
    print("4. Train Authenticity: python src/train_authenticity.py")
    print("5. Train Matcher: python src/train_matcher.py")


if __name__ == "__main__":
    main()