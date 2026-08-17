#!/usr/bin/env python3
"""
Free AI Resume Generator for HireSense
Creates synthetic resume variations using rule-based text transformations
instead of expensive LLM API calls.
"""

import os
import json
import random
from pathlib import Path
from typing import List, Dict
from datasets import load_dataset, Dataset, DatasetDict
import tqdm
import numpy as np


def load_human_resumes(data_dir: Path) -> List[str]:
    """Load human resumes from the processed authenticity dataset."""
    human_data_path = data_dir / "authenticity_human"
    if not human_data_path.exists():
        raise FileNotFoundError(f"Human data not found at {human_data_path}. Run data_prep.py first.")

    dataset = Dataset.load_from_disk(str(human_data_path))
    resumes = [sample['text'] for sample in dataset if 'text' in sample]
    print(f"Loaded {len(resumes)} human resumes")
    return resumes


def apply_synonym_replacement(text: str) -> str:
    """Replace words with synonyms to create variations."""
    # Simple synonym dictionary for resume-related terms
    synonyms = {
        'experienced': ['skilled', 'proficient', 'accomplished', 'seasoned'],
        'developed': ['created', 'built', 'designed', 'implemented', 'engineered'],
        'managed': ['led', 'directed', 'supervised', 'oversaw', 'controlled'],
        'responsible': ['accountable', 'in charge of', 'tasked with'],
        '_improved': ['enhanced', 'optimized', 'upgraded', 'boosted', 'increased'],
        'worked': ['collaborated', 'partnered', 'teamed up', 'cooperated'],
        'helped': ['assisted', 'supported', 'aided', 'facilitated'],
        'used': ['utilized', 'employed', 'leveraged', 'applied'],
        'showed': ['demonstrated', 'exhibited', 'displayed', 'revealed'],
        'made': ['produced', 'generated', 'created', 'constructed'],
        'got': ['obtained', 'received', 'secured', 'acquired'],
        'help': ['assist', 'support', 'aid', 'facilitate'],
        'need': ['require', 'necessitate', 'demand', 'entail'],
        'use': ['utilize', 'employ', 'leverage', 'apply'],
        'get': ['obtain', 'receive', 'secure', 'acquire'],
        'make': ['produce', 'generate', 'create', 'construct'],
        'know': ['understand', 'comprehend', 'grasp', 'master'],
        'see': ['observe', 'notice', 'detect', 'identify'],
        'come': ['arrive', 'appear', 'emerge', 'surface'],
        'want': ['desire', 'wish', 'seek', 'pursue'],
        'look': ['search', 'seek', 'hunt', 'scout'],
        'feel': ['sense', 'detect', 'experience', 'undergo'],
        'give': ['provide', 'supply', 'offer', 'present'],
        'find': ['discover', 'locate', 'detect', 'uncover'],
        'tell': ['inform', 'notify', 'relate', 'communicate'],
        'become': ['turn into', 'grow into', 'develop into', 'evolve into'],
        'leave': ['depart', 'exit', 'vacate', 'abandon'],
        'put': ['place', 'position', 'locate', 'deposit'],
        'mean': ['signify', 'indicate', 'imply', 'suggest'],
        'keep': ['retain', 'maintain', 'preserve', 'continue'],
        'let': ['allow', 'permit', 'enable', 'authorize'],
        'begin': ['start', 'commence', 'initiate', 'launch'],
        'seem': ['appear', 'look', 'sound', 'appear'],
        'talk': ['speak', 'discuss', 'converse', 'communicate'],
        'turn': ['rotate', 'revolve', 'spin', 'twist'],
        'move': ['shift', 'relocate', 'transfer', 'reposition'],
        'like': ['enjoy', 'appreciate', 'value', 'favor'],
        'run': ['operate', 'manage', 'direct', 'conduct'],
        'believe': ['think', 'consider', 'suppose', 'assume'],
        'hold': ['grasp', 'grip', 'clutch', 'seize'],
        'bring': ['fetch', 'carry', 'deliver', 'transport'],
        'happen': ['occur', 'take place', 'transpire', 'materialize'],
        'write': ['compose', 'draft', 'author', 'pen'],
        'sit': ['rest', 'pause', 'relax', 'repose'],
        'stand': ['rise', 'ascend', 'climb', 'mount'],
        'lose': ['misplace', 'forfeit', 'surrender', 'yield'],
        'pay': ['compensate', 'remunerate', 'settle', 'clear'],
        'meet': ['encounter', 'come across', 'run into', 'bump into'],
        'include': ['contain', 'comprise', 'encompass', 'embrace'],
        'continue': ['proceed', 'persist', 'endure', 'last'],
        'set': ['establish', 'fix', 'determine', 'define'],
        'learn': ['study', 'master', 'acquire', 'grasp'],
        'change': ['alter', 'modify', 'adjust', 'revise'],
        'lead': ['guide', 'direct', 'steer', 'conduct'],
        'understand': ['comprehend', 'grasp', 'apprehend', ' apprehend'],
        'offer': ['provide', 'present', 'extend', 'proffer'],
        'need': ['require', 'necessitate', 'demand', 'entail'],
        'feel': ['sense', 'detect', 'experience', 'undergo'],
        'become': ['turn into', 'grow into', 'develop into', 'evolve into'],
        'leave': ['depart', 'exit', 'vacate', 'abandon'],
        'put': ['place', 'position', 'locate', 'deposit'],
        'mean': ['signify', 'indicate', 'imply', 'suggest'],
        'keep': ['retain', 'maintain', 'preserve', 'continue'],
        'let': ['allow', 'permit', 'enable', 'authorize'],
        'begin': ['start', 'commence', 'initiate', 'launch'],
        'seem': ['appear', 'look', 'sound', 'appear'],
        'talk': ['speak', 'discuss', 'converse', 'communicate'],
        'turn': ['rotate', 'revolve', 'spin', 'twist'],
        'move': ['shift', 'relocate', 'transfer', 'reposition'],
        'like': ['enjoy', 'appreciate', 'value', 'favor'],
        'run': ['operate', 'manage', 'direct', 'conduct'],
        'believe': ['think', 'consider', 'suppose', 'assume'],
        'hold': ['grasp', 'grip', 'clutch', 'seize'],
        'bring': ['fetch', 'carry', 'deliver', 'transport'],
        'happen': ['occur', 'take place', 'transpire', 'materialize'],
        'write': ['compose', 'draft', 'author', 'pen'],
        'sit': ['rest', 'pause', 'relax', 'repose'],
        'stand': ['rise', 'ascend', 'climb', 'mount'],
        'lose': ['misplace', 'forfeit', 'surrender', 'yield'],
        'pay': ['compensate', 'remunerate', 'settle', 'clear'],
        'meet': ['encounter', 'come across', 'run into', 'bump into'],
        'include': ['contain', 'comprise', 'encompass', 'embrace'],
        'continue': ['proceed', 'persist', 'endure', 'last'],
        'set': ['establish', 'fix', 'determine', 'define'],
        'learn': ['study', 'master', 'acquire', 'grasp'],
        'change': ['alter', 'modify', 'adjust', 'revise'],
        'lead': ['guide', 'direct', 'steer', 'conduct'],
        'understand': ['comprehend', 'grasp', 'apprehend', 'understand'],
    }

    words = text.split()
    new_words = []

    for word in words:
        # Clean word of punctuation for lookup
        clean_word = word.lower().strip('.,!?;:"\'()[]{}')
        if clean_word in synonyms and random.random() < 0.3:  # 30% chance to replace
            replacement = random.choice(synonyms[clean_word])
            # Preserve original capitalization
            if word[0].isupper():
                replacement = replacement.capitalize()
            # Preserve trailing punctuation
            suffix = word[len(clean_word):]
            new_words.append(replacement + suffix)
        else:
            new_words.append(word)

    return ' '.join(new_words)


def apply_sentence_shuffling(text: str) -> str:
    """Shuffle sentences within paragraphs to create variation."""
    paragraphs = text.split('\n\n')
    new_paragraphs = []

    for para in paragraphs:
        sentences = para.split('. ')
        if len(sentences) > 3:  # Only shuffle if enough sentences
            # Shuffle middle sentences, keep first and last
            if len(sentences) > 2:
                middle = sentences[1:-1]
                random.shuffle(middle)
                sentences = [sentences[0]] + middle + [sentences[-1]]
            new_paragraphs.append('. '.join(sentences))
        else:
            new_paragraphs.append(para)

    return '\n\n'.join(new_paragraphs)


def apply_section_reordering(text: str) -> str:
    """Reorder resume sections to create variation."""
    # Common resume section headers
    section_headers = [
        'EXPERIENCE', 'WORK EXPERIENCE', 'EMPLOYMENT',
        'EDUCATION', 'SKILLS', 'TECHNICAL SKILLS',
        'PROJECTS', 'CERTIFICATIONS', 'ACHIEVEMENTS',
        'SUMMARY', 'OBJECTIVE', 'PROFILE',
        'CONTACT', 'REFERENCES'
    ]

    lines = text.split('\n')
    # Find sections
    sections = {}
    current_section = 'HEADER'
    current_content = []

    for line in lines:
        line_upper = line.strip().upper()
        is_header = any(header in line_upper for header in section_headers) and len(line.strip()) < 50

        if is_header and current_content:
            sections[current_section] = '\n'.join(current_content)
            current_section = line.strip()
            current_content = []
        else:
            current_content.append(line)

    if current_content:
        sections[current_section] = '\n'.join(current_content)

    # Reorder sections (keep HEADER first)
    if 'HEADER' in sections:
        header = sections.pop('HEADER')
        section_items = list(sections.items())
        random.shuffle(section_items)

        # Reconstruct text
        result_lines = [header]
        for section_name, section_content in section_items:
            if section_name.strip():  # Skip empty sections
                result_lines.append(section_name)
                result_lines.append(section_content)

        return '\n'.join(result_lines)

    return text


def apply_minor_edits(text: str) -> str:
    """Apply minor edits like typos, formatting changes, etc."""
    # Sometimes add minor "errors" to seem more human-like
    if random.random() < 0.1:  # 10% chance
        # Randomly duplicate a word
        words = text.split()
        if len(words) > 5:
            idx = random.randint(0, len(words)-2)
            words.insert(idx+1, words[idx])
            text = ' '.join(words)

    if random.random() < 0.05:  # 5% chance
        # Randomly remove a comma
        if ',' in text:
            parts = text.split(',', 1)
            if len(parts) > 1 and random.random() < 0.5:
                text = parts[0] + parts[1]
            else:
                text = parts[0] + ',' + ','.join(parts[1:]) if len(parts) > 2 else parts[0] + ',' + parts[1]

    return text


def generate_ai_resume(human_resume: str, method: str) -> str:
    """Generate an AI resume variation based on the specified method."""
    if method == 'synonym':
        return apply_synonym_replacement(human_resume)
    elif method == 'shuffle':
        return apply_sentence_shuffling(human_resume)
    elif method == 'reorder':
        return apply_section_reordering(human_resume)
    elif method == 'mixed':
        # Apply multiple transformations
        text = apply_synonym_replacement(human_resume)
        text = apply_sentence_shuffling(text)
        if random.random() < 0.3:
            text = apply_section_reordering(text)
        return apply_minor_edits(text)
    elif method == 'export_import':
        # Simulate export/import artifacts (slight formatting changes)
        lines = human_resume.split('\n')
        # Randomly adjust spacing
        if len(lines) > 10 and random.random() < 0.4:
            idx = random.randint(0, len(lines)-1)
            lines[idx] = '  ' + lines[idx]  # Add extra spaces
        return '\n'.join(lines)
    else:
        return human_resume  # Fallback


def create_free_ai_dataset(human_resumes: List[str], num_samples: int = 20000, output_dir: Path = Path("./data/processed")):
    """Create AI resume dataset using free rule-based methods."""
    print(f"Generating {num_samples} AI resume samples using free methods...")

    # Define generation methods with weights
    methods = [
        ('synonym', 0.3),
        ('shuffle', 0.2),
        ('reorder', 0.2),
        ('mixed', 0.2),
        ('export_import', 0.1)
    ]

    method_choices, method_weights = zip(*methods)

    ai_records = []

    for i in tqdm.tqdm(range(num_samples), desc="Generating AI resumes"):
        # Select random human resume as base
        base_resume = random.choice(human_resumes)
        # Select generation method
        method = random.choices(method_choices, weights=method_weights)[0]

        # Generate AI variation
        ai_text = generate_ai_resume(base_resume, method)

        ai_records.append({
            'text': ai_text,
            'label': 0,  # AI-generated
            'source': 'rule_based_generated',
            'generation_method': method,
            'base_resume_index': human_resumes.index(base_resume) if base_resume in human_resumes else -1
        })

        # Progress indicator
        if (i + 1) % 5000 == 0:
            print(f"Generated {i + 1}/{num_samples} AI resumes")

    print(f"Generated {len(ai_records)} AI resume samples")

    # Create dataset
    ai_dataset = Dataset.from_list(ai_records)

    # Save dataset
    ai_dir = output_dir / "authenticity_ai"
    ai_dir.mkdir(parents=True, exist_ok=True)
    ai_dataset.save_to_disk(str(ai_dir))
    print(f"AI dataset saved to: {ai_dir}")

    # Also save combined dataset (human + AI)
    human_data_path = output_dir / "authenticity_human"
    human_dataset = Dataset.load_from_disk(str(human_data_path))

    # Add source field to human dataset for consistency
    human_records = []
    for sample in human_dataset:
        record = dict(sample)
        record['source'] = 'languk_human'
        record ['generation_method'] = 'original'
        human_records.append(record)

    combined_records = human_records + ai_records
    combined_dataset = Dataset.from_list(combined_records)

    combined_dir = output_dir / "authenticity_combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    combined_dataset.save_to_disk(str(combined_dir))
    print(f"Combined dataset saved to: {combined_dir}")
    print(f"Total samples: {len(combined_dataset)} ({len(human_records)} human, {len(ai_records)} AI)")

    return ai_dataset, combined_dataset


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate free AI resume variations")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/processed",
        help="Directory containing processed data from data_prep.py"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20000,
        help="Number of AI resume samples to generate"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)

    print("="*60)
    print("Free AI Resume Generator for HireSense")
    print("="*60)

    # Load human resumes
    human_resumes = load_human_resumes(data_dir)

    if len(human_resumes) == 0:
        print("ERROR: No human resumes found. Please run data_prep.py first.")
        return

    # Generate AI dataset
    ai_dataset, combined_dataset = create_free_ai_dataset(
        human_resumes=human_resumes,
        num_samples=args.num_samples,
        output_dir=data_dir
    )

    print("\n" + "="*60)
    print("AI RESUME GENERATION COMPLETE")
    print("="*60)
    print(f"Generated {len(ai_dataset)} AI resume samples")
    print(f"Combined dataset has {len(combined_dataset)} total samples")
    print("\nNext steps:")
    print("1. Train Authenticity Detector: python src/train_authenticity.py")
    print("2. The detector will use the combined human/AI dataset for training")


if __name__ == "__main__":
    main()