# HireSense - NLP Pipeline for Recruitment

## Project Overview
**HireSense** is an end-to-end NLP pipeline that extracts and verifies skills from resumes to help students understand their fit and help recruiters find authentic, qualified candidates.

### Core Problem Statement

#### Student Perspective
- Students spend hours customizing resumes but get rejected with **no feedback**
- Their real skills go unnoticed because all resumes look the same (AI-generated)
- They don't know **what skills they're missing** or **why they're not getting interviews**

#### Recruiter Perspective
- Recruiters receive **1000+ applications per job posting**
- Most are **AI-generated fluff** (62% of companies hired someone who faked skills using AI)
- They spend **80% of their time filtering** instead of interviewing
- Cannot verify skill authenticity at scale

---

## Three NLP Components

| Component | Purpose | Input | Output |
|-----------|---------|-------|--------|
| **Skill Extractor** | Extract skills from resume text | Resume text | Structured skills: Programming Languages, Frameworks, Tools, Soft Skills, Domain Knowledge |
| **Authenticity Detector** | Detect AI-generated vs human-written resumes | Resume text | Authenticity score (0-100%) |
| **Semantic Matcher** | Match resumes to job descriptions | Resume + Job Description | Match score (0-100%) with explainability |

---

## Dataset Strategy

### Primary Datasets (from Hugging Face)

| Dataset | Size | Purpose | License |
|---------|------|---------|---------|
| `lang-uk/recruitment-dataset-candidate-profiles-english` | 210,250 CVs | Skill extraction corpus, authenticity training | CC-BY-4.0 |
| `lang-uk/recruitment-dataset-job-descriptions-english` | 141,897 jobs | Semantic matching, JD corpus | CC-BY-4.0 |
| `TechWolf/Synthetic-ESCO-skill-sentences` | 138,260 sentences | Skill taxonomy alignment, NER training | CC-BY-4.0 |
| `imocha-ai-org/ssf-skill-extraction-pairs` | 21,958 pairs | Skill extraction supervision (text → skills) | Apache-2.0 |
| `datasetmaster/resumes` | ~5,000 resumes | Resume parsing, section segmentation | MIT |
| `mounimzad/brainhr-plus` | 15 annotated pairs | Gold-standard evaluation | CC-BY-4.0 |

### Additional Datasets to Acquire

| Dataset | Purpose | Source |
|---------|---------|--------|
| AI-generated resume corpus | Authenticity detector training | Generate via LLMs + human resumes |
| `cl-tohoku/ai-generated-text-detection` | General AI detection baseline | Hugging Face |
| `mteb/stsb-hard-negatives` | Hard negative mining for matching | Hugging Face |
| ESCO/O*NET skill taxonomy | Skill normalization | Official sources |

---

## Model Architecture

### 1. Skill Extractor
```
Input: Resume text (tokenized, max 512 tokens)
         │
         ▼
┌─────────────────────────────────────────┐
│  DeBERTa-v3-base (microsoft/deberta-v3-base)  │
│  Pre-trained on resume corpus (DAPT)        │
│  Fine-tuned with:                             │
│  - Token-level NER head (BIO tags)            │
│  - Multi-label skill classification head      │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  Post-processing:                        │
│  - Span merging                          │
│  - Skill normalization to ESCO taxonomy  │
│  - Category classification (PL/FW/TL/SS/DK) │
└─────────────────────────────────────────┘
         │
         ▼
Output: List of skills with categories, confidence scores
```

**Training Data Construction:**
- `imocha-ai-org/ssf-skill-extraction-pairs` → Direct supervision (text + skill labels)
- `TechWolf/Synthetic-ESCO-skill-sentences` → Weak supervision (skill sentences)
- `lang-uk/recruitment-dataset-candidate-profiles-english` → Domain-adaptive pre-training (DAPT)
- Silver labels via rule-based + LLM annotation on candidate profiles

**Loss Function:**
```python
loss = λ_ner * L_NER + λ_cls * L_MultiLabelCls + λ_con * L_Contrastive
```

### 2. Authenticity Detector
```
Input: Resume text (tokenized, max 512 tokens)
         │
         ▼
┌─────────────────────────────────────────┐
│  Ensemble of:                            │
│  1. Fine-tuned DeBERTa-v3-small         │
│     (classifier head, binary)            │
│  2. Perplexity-based detector (GPT-2)   │
│  3. Stylometric features (TF-IDF + LR)  │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  Calibration (Platt scaling)             │
│  Ensemble weighting (learned)            │
└─────────────────────────────────────────┘
         │
         ▼
Output: Authenticity score 0-100%
        + Explanation (which sections suspicious)
```

**Training Data Construction:**
- **Positive (Human)**: `lang-uk/recruitment-dataset-candidate-profiles-english` (sampled)
- **Negative (AI-generated)**: Generate using GPT-4, Claude, Llama-3 with prompts:
  - "Write a resume for a [role] with [skills]"
  - "Rewrite this resume to be more impressive"
  - "Generate a fake resume for [role]"
- **Hard negatives**: Human resumes lightly edited by AI
- **Mix ratios**: 40% pure AI, 30% human-AI hybrid, 30% pure human

### 3. Semantic Matcher
```
Input: Resume text + Job Description text
         │
         ├──► Resume Encoder ──────────►│
         │                              │
         ├──► JD Encoder ──────────────►│  Bi-Encoder (Retrieval)
         │         (shared weights)     │  │
         ▼                              ▼
┌─────────────────────────────────────────┐
│  Sentence Transformer:                   │
│  BGE-base-en-v1.5 (BAAI/bge-base-en-v1.5) │
│  OR                                      │
│  E5-base-v2 (intfloat/e5-base-v2)        │
│  Fine-tuned on resume-JD pairs           │
└─────────────────────────────────────────┘
         │
         ├──► Bi-encoder score (cosine similarity) - FAST
         │
         ▼ (for top-k)
┌─────────────────────────────────────────┐
│  Cross-Encoder (Re-ranking):             │
│  DeBERTa-v3-base cross-encoder           │
│  Fine-tuned on hard negatives            │
└─────────────────────────────────────────┘
         │
         ▼
Output: Match score 0-100%
        + Matched skills
        + Missing skills
        + Explanation
```

**Training Data Construction:**
- **Positive pairs**: `lang-uk` candidate profiles ↔ job descriptions (same domain)
- **Negative pairs**: Random profile-JD pairs + hard negatives
- **Hard negatives**: 
  - Same skills, different seniority
  - Similar domain, missing key skills
  - Generated via cross-encoder mining

---

## Training Pipeline

### Phase 1: Data Preparation (`data_prep.py`)
```python
# 1. Load all datasets
# 2. Clean & normalize text (PDF → text, sections)
# 3. Build skill taxonomy from ESCO + O*NET
# 4. Create NER labels (BIO tagging)
# 5. Generate AI-resume corpus for authenticity
# 6. Build resume-JD pairs with labels for matching
# 7. Split train/val/test (stratified by domain)
# 8. Save processed datasets to ./data/processed/
```

### Phase 2: Domain-Adaptive Pre-training (`dapt.py`)
```python
# 1. Load microsoft/deberta-v3-base
# 2. Continue MLM on 210k candidate profiles (lang-uk)
# 3. Save as ./models/deberta-v3-base-resume-dapt
```

### Phase 3: Skill Extractor Training (`train_skill_extractor.py`)
```python
# 1. Load DAPT model
# 2. Add NER head + classification head
# 3. Train on imocha pairs + silver labels
# 4. Multi-task: NER + skill category classification
# 5. Evaluate on brainhr-plus (gold)
# 6. Save best to ./models/skill_extractor/
```

### Phase 4: Authenticity Detector Training (`train_authenticity.py`)
```python
# 1. Generate AI resume corpus (50k samples)
# 2. Fine-tune DeBERTa-v3-small classifier
# 3. Train perplexity + stylometric baselines
# 4. Ensemble with learned weights
# 5. Calibrate on held-out set
# 6. Save to ./models/authenticity_detector/
```

### Phase 5: Semantic Matcher Training (`train_matcher.py`)
```python
# 1. Bi-encoder: Fine-tune BGE-base on resume-JD pairs
#    - Contrastive loss with hard negatives
# 2. Cross-encoder: Fine-tune DeBERTa on top-k hard negatives
#    - Binary classification loss
# 3. Evaluate on brainhr-plus + held-out pairs
# 4. Save to ./models/semantic_matcher/
```

---

## Directory Structure

```
HireSense/
├── CLAUDE.md                    # This file
├── spec/
│   └── sra-plan.md             # Detailed specification
├── data/
│   ├── raw/                    # Downloaded HF datasets
│   └── processed/              # Prepared training data
├── models/                     # Saved model checkpoints
│   ├── deberta-v3-base-resume-dapt/
│   ├── skill_extractor/
│   ├── authenticity_detector/
│   └── semantic_matcher/
├── src/
│   ├── data_prep.py
│   ├── dapt.py
│   ├── train_skill_extractor.py
│   ├── train_authenticity.py
│   ├── train_matcher.py
│   ├── inference_pipeline.py
│   └── utils/
│       ├── resume_parser.py
│       ├── skill_taxonomy.py
│       ├── metrics.py
│       └── gpu_utils.py
├── configs/
│   ├── skill_extractor.yaml
│   ├── authenticity.yaml
│   └── matcher.yaml
├── scripts/
│   ├── download_data.sh
│   ├── run_all_training.sh
│   └── evaluate.py
├── tests/
│   ├── test_data_prep.py
│   ├── test_inference.py
│   └── fixtures/
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## GPU Requirements & Fallback

| Component | VRAM (Training) | VRAM (Inference) | CPU Fallback |
|-----------|----------------|------------------|--------------|
| DAPT | 16GB (batch=8, grad_acc=4) | 4GB | ✅ Slow |
| Skill Extractor | 12GB (batch=16) | 2GB | ✅ |
| Authenticity | 8GB (batch=32) | 1GB | ✅ Fast |
| Bi-encoder | 12GB (batch=64) | 2GB | ✅ |
| Cross-encoder | 16GB (batch=16) | 4GB | ⚠️ Slow |

**GPU Detection:**
```python
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
```

---

## Evaluation Metrics

### Skill Extractor
- **Entity-level**: Precision, Recall, F1 (exact match)
- **Skill-level**: Precision@k, Recall@k, NDCG@k
- **Category-level**: Per-category F1 (PL, FW, TL, SS, DK)

### Authenticity Detector
- **Binary**: AUC-ROC, AUC-PR, Accuracy@0.5
- **Calibration**: ECE (Expected Calibration Error), Brier score
- **Per-domain**: Performance by role/type

### Semantic Matcher
- **Ranking**: MRR, Recall@10, NDCG@10
- **Classification**: Accuracy, F1 (match/no-match threshold)
- **Skill-aware**: Skill coverage, missing skill precision

---

## Key Technical Decisions

1. **DeBERTa-v3 over BERT**: Better performance on NER, handles long contexts
2. **DAPT on resume corpus**: Critical for domain adaptation (proven in HR-NLP literature)
3. **Bi-encoder + Cross-encoder**: Balance speed (retrieval) and accuracy (ranking)
4. **Ensemble authenticity**: Single detectors fail on cross-model generalization
5. **ESCO taxonomy normalization**: Enables skill transferability, gap analysis
6. **Multi-task skill extraction**: Joint NER + category improves both tasks
6. **Hard negative mining**: Essential for semantic matching quality

---

## Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| AI resume detection arms race | High | High | Ensemble, continuous retraining, human-in-loop |
| Skill taxonomy coverage gaps | Medium | Medium | ESCO + O*NET + custom, active learning |
| Domain shift (new tech stacks) | High | Medium | DAPT refresh, few-shot adaptation |
| Bias in training data | High | High | Stratified sampling, fairness audits |
| PDF parsing failures | Medium | Medium | Multiple parsers, fallback to OCR |

---

## Next Steps (After Pipeline)

1. **API Layer**: FastAPI with batch inference
2. **UI**: Streamlit/React for student/recruiter dashboards
3. **Feedback Loop**: Collect recruiter corrections → retrain
4. **Explainability**: SHAP/LIME for match explanations
5. **Multilingual**: Extend to non-English resumes
6. **Active Learning**: Prioritize uncertain samples for annotation

