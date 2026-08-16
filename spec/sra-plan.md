# HireSense - Detailed Specification Document

## 1. Executive Summary

**Project**: HireSense - NLP Pipeline for Recruitment Intelligence  
**Version**: 1.0  
**Date**: 2026-08-16  
**Status**: Design Phase  

### 1.1 Vision
Build a production-ready NLP pipeline that solves the fundamental asymmetry in recruitment: students can't prove their skills, recruiters can't verify them.

### 1.2 Success Criteria
| Metric | Target |
|--------|--------|
| Skill Extraction F1 | ≥ 0.85 |
| Authenticity AUC-ROC | ≥ 0.95 |
| Semantic Matching MRR | ≥ 0.75 |
| Inference Latency (single resume) | < 500ms |
| Training Time (full pipeline) | < 8 hours on 1×A100 |

---

## 2. Problem Analysis (Deep Dive)

### 2.1 Student Pain Points (Research-Backed)

| Problem | Evidence | Impact |
|---------|----------|--------|
| **No feedback loop** | 73% of candidates never hear back (CareerBuilder 2023) | Cannot improve |
| **AI resume homogenization** | 46% of job seekers use AI for resumes (ResumeBuilder 2024) | Real skills buried |
| **Skill gap blindness** | 68% don't know what skills employers want (LinkedIn 2023) | Misdirected upskilling |
| **ATS keyword gaming** | 75% of resumes rejected by ATS before human sees (TopResume) | Qualified candidates filtered |

**Core Insight**: Students need *actionable* feedback: "You're missing Skill X which appears in 80% of target job descriptions."

### 2.2 Recruiter Pain Points (Research-Backed)

| Problem | Evidence | Impact |
|---------|----------|--------|
| **Volume overload** | Avg 250 applications/role; 1000+ for tech (Greenhouse 2024) | 80% time on screening |
| **AI fraud epidemic** | 62% companies hired someone who faked skills via AI (Checkster 2024) | Bad hires cost 30% salary |
| **Keyword matching failure** | 88% of recruiters say ATS misses qualified candidates (Harris Poll) | False negatives |
| **Skill verification vacuum** | No scalable way to verify claimed skills | Trust but cannot verify |
| **Bias in screening** | Name/education bias in first 6 seconds (NBER 2023) | DEI violations, legal risk |

**Core Insight**: Recruiters need *verified* skill evidence + *explainable* match scores, not black-box rankings.

### 2.3 The Core NLP Problems

```
┌─────────────────────────────────────────────────────────────────┐
│                    RESUME TEXT (Unstructured)                   │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
       ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
       │   SKILL     │ │ AUTHENTICITY│ │   MATCHING  │
       │ EXTRACTION  │ │  DETECTION  │ │  (vs JD)    │
       └─────────────┘ └─────────────┘ └─────────────┘
              │               │               │
              ▼               ▼               ▼
       Structured    0-100% Score    0-100% Score
       Skills List   + Explanation   + Gap Analysis
```

**Why this is hard:**
1. **Resume format variance**: PDF, DOCX, LaTeX, plain text, creative layouts
2. **Skill polymorphism**: "PyTorch", "torch", "pytorch", "PyTorch 2.0" = same skill
3. **Implicit skills**: "Built scalable APIs" → implies FastAPI, Docker, AWS, system design
4. **AI mimicry**: LLMs generate perfectly formatted but hollow resumes
5. **Cross-domain transfer**: "Project management" in construction ≠ software
6. **Taxonomy alignment**: ESCO has 13,485 skills; O*NET has 1,100; company-specific taxonomies

---

## 3. Dataset Analysis & Strategy

### 3.1 Verified Dataset Inventory

| Dataset | HF ID | Samples | Quality | License | Use Case |
|---------|-------|---------|---------|---------|----------|
| Candidate Profiles | `lang-uk/recruitment-dataset-candidate-profiles-english` | 210,250 | High (structured) | CC-BY-4.0 | DAPT, Skill Ext. corpus, Auth+ |
| Job Descriptions | `lang-uk/recruitment-dataset-job-descriptions-english` | 141,897 | High | CC-BY-4.0 | Matching corpus, JD encoding |
| ESCO Skill Sentences | `TechWolf/Synthetic-ESCO-skill-sentences` | 138,260 | Synthetic but clean | CC-BY-4.0 | Skill NER weak supervision |
| Skill Extraction Pairs | `imocha-ai-org/ssf-skill-extraction-pairs` | 21,958 | Human-annotated | Apache-2.0 | **Gold standard for Skill Ext.** |
| Resume Corpus | `datasetmaster/resumes` | ~5,000 | Mixed | MIT | Parsing, section segmentation |
| BrainHR+ | `mounimzad/brainhr-plus` | 15 pairs | Expert-annotated | CC-BY-4.0 | **Gold evaluation only** |

### 3.2 Dataset Gaps & Acquisition Plan

| Gap | Solution | Effort |
|-----|----------|--------|
| **AI-generated resume labels** | Generate 50k synthetic resumes via GPT-4/Claude/Llama with varied prompts | Medium |
| **Resume-JD match labels** | Use lang-uk pairs (same domain = positive) + hard negative mining | Low |
| **Skill normalization labels** | Map extracted skills to ESCO v1.2 via fuzzy matching + manual review | Medium |
| **Section segmentation labels** | Annotate 1k resumes with sections (experience, education, skills, projects) | Medium |
| **Multilingual support** | Not in v1 scope | - |

### 3.3 Data Splits Strategy

```python
# All splits stratified by: domain, seniority, resume_length
SPLITS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}

# Special: BrainHR+ (15 pairs) ONLY for final evaluation, NEVER for training
# Special: imocha pairs - use 80/10/10 split, ensure no skill leakage
# Special: lang-uk - sample 50k for DAPT, rest for silver labels
```

---

## 4. Component Specifications

### 4.1 Skill Extractor

#### 4.1.1 Architecture Details
```
Model: microsoft/deberta-v3-base (86M params) → DAPT → Fine-tune
Input: Tokenized resume text (max_len=512, stride=128 for long docs)
Outputs:
  1. NER logits: [batch, seq_len, num_labels]  # BIO tags
  2. Skill category logits: [batch, num_categories]  # Multi-label
  3. Skill embeddings: [batch, hidden_dim]  # For contrastive loss
```

#### 4.1.2 Label Schema (BIO Tags)
```python
# 5 Categories × 2 (B/I) + O = 11 labels
LABELS = [
    "O",                           # Outside
    "B-PROG_LANG", "I-PROG_LANG",  # Python, Java, Rust...
    "B-FRAMEWORK", "I-FRAMEWORK",  # React, Django, Spring...
    "B-TOOL", "I-TOOL",            # Docker, Git, Kubernetes...
    "B-SOFT_SKILL", "I-SOFT_SKILL",# Leadership, Communication...
    "B-DOMAIN_KNOW", "I-DOMAIN_KNOW" # FinTech, Healthcare, CV...
]
```

#### 4.1.3 Skill Category Definitions
| Category | ESCO Mapping | Examples |
|----------|--------------|----------|
| PROG_LANG | `programming language` | Python, Java, JavaScript, Go, Rust, SQL |
| FRAMEWORK | `software framework` | React, Django, Spring Boot, FastAPI, Next.js |
| TOOL | `development tool` | Docker, Git, Kubernetes, Jenkins, Terraform |
| SOFT_SKILL | `transversal skill` | Leadership, Communication, Problem Solving |
| DOMAIN_KNOW | `domain-specific knowledge` | FinTech, Healthcare, Computer Vision, NLP |

#### 4.1.4 Training Recipe
```yaml
# configs/skill_extractor.yaml
model:
  name: "microsoft/deberta-v3-base"
  dapt_checkpoint: "./models/deberta-v3-base-resume-dapt"
  max_length: 512
  num_labels: 11  # BIO tags
  num_categories: 5  # Multi-label

training:
  batch_size: 16
  grad_accumulation: 2
  learning_rate: 3e-5
  weight_decay: 0.01
  num_epochs: 10
  warmup_ratio: 0.1
  lr_scheduler: "cosine"
  mixed_precision: "fp16"
  gradient_checkpointing: true

loss:
  ner_weight: 1.0
  category_weight: 0.5
  contrastive_weight: 0.3
  temperature: 0.07

data:
  train_sources: ["imocha", "silver_languk", "esco_sentences"]
  val_source: "imocha_val"
  test_source: "brainhr_plus"
  oversample_rare_skills: true
```

#### 4.1.5 Post-Processing Pipeline
```python
def post_process_skills(ner_predictions, category_predictions, tokens):
    # 1. Merge B-I spans
    spans = merge_bio_spans(ner_predictions, tokens)
    
    # 2. Normalize to canonical form
    normalized = []
    for span in spans:
        canonical = normalize_skill(span.text)  # "PyTorch" → "PyTorch"
        esc_id = map_to_esco(canonical, span.category)
        confidence = span.confidence * category_predictions[span.category]
        normalized.append({
            "skill": canonical,
            "category": span.category,
            "esco_id": esc_id,
            "confidence": confidence,
            "span": (span.start, span.end)
        })
    
    # 3. Deduplicate (keep highest confidence)
    # 4. Filter by confidence threshold
    return deduplicate(normalized, threshold=0.5)
```

### 4.2 Authenticity Detector

#### 4.2.1 Architecture: Ensemble Approach
```
┌────────────────────────────────────────────────────────────────┐
│                        ENSEMBLE                                 │
├──────────────┬──────────────┬──────────────┬──────────────────┤
│  Classifier  │  Perplexity  │ Stylometric  │  Calibration     │
│  (DeBERTa)   │  (GPT-2)     │  (TF-IDF+LR) │  (Platt/Diag)    │
│  Weight: 0.5 │  Weight: 0.2 │  Weight: 0.1 │  Weight: learned │
└──────────────┴──────────────┴──────────────┴──────────────────┘
```

#### 4.2.2 AI Resume Generation Strategy
```python
GENERATION_PROMPTS = [
    # Pure generation
    "Write a professional resume for a {role} with {years} years experience. Skills: {skills}",
    "Create an impressive resume for a {role} at a top tech company.",
    
    # Rewriting (human → AI)
    "Rewrite this resume to be more impressive and ATS-friendly: {resume}",
    "Enhance this resume with stronger action verbs and quantified achievements: {resume}",
    
    # Hybrid (human base + AI sections)
    "Add a compelling professional summary to this resume: {resume}",
    "Rewrite the skills section to be more comprehensive: {resume}",
    
    # Adversarial (trying to fool detector)
    "Write a resume that looks completely human-written for a {role}. Include specific project details, minor imperfections, and natural language variations.",
]

MODELS_TO_USE = [
    "gpt-4o", "gpt-4o-mini",      # OpenAI
    "claude-3-5-sonnet",           # Anthropic
    "llama-3.1-70b", "llama-3.1-8b", # Meta
    "mistral-large", "mixtral-8x7b", # Mistral
]
```

#### 4.2.3 Training Data Composition
| Source | Count | Label | Notes |
|--------|-------|-------|-------|
| lang-uk profiles (sampled) | 15,000 | Human (1) | Diverse domains, seniorities |
| Generated (pure AI) | 10,000 | AI (0) | 5 models × 2k each |
| Generated (rewrites) | 10,000 | AI (0) | Human base → AI rewrite |
| Generated (hybrid) | 5,000 | AI (0) | Partial AI |
| Hard negatives (light edit) | 5,000 | AI (0) | Human + minor AI touches |
| **Total** | **45,000** | **Balanced** | Stratified split |

#### 4.2.4 Training Recipe
```yaml
# configs/authenticity.yaml
classifier:
  model: "microsoft/deberta-v3-small"  # 44M params, faster
  max_length: 512
  num_labels: 2

training:
  batch_size: 32
  learning_rate: 2e-5
  num_epochs: 5
  warmup_ratio: 0.1
  class_weights: [1.0, 1.0]  # Balanced
  label_smoothing: 0.1

perplexity:
  model: "gpt2"
  threshold_tuning: true

stylometric:
  features: "tfidf_char_ngram_3_5"  # Character n-grams capture AI patterns
  classifier: "logistic_regression"
  c: 1.0

ensemble:
  method: "stacking"  # Meta-learner on validation predictions
  calibration: "platt_scaling"
```

#### 4.2.5 Explainability Output
```python
def explain_authenticity(resume_text, score, ensemble_details):
    return {
        "score": score,  # 0-100
        "label": "Likely Human" if score > 50 else "Likely AI",
        "confidence": abs(score - 50) * 2,  # Distance from boundary
        "section_scores": {
            "summary": section_score(resume_text, "summary"),
            "experience": section_score(resume_text, "experience"),
            "skills": section_score(resume_text, "skills"),
            "education": section_score(resume_text, "education"),
        },
        "indicators": [
            {"type": "perplexity", "value": ppl, "suspicious": ppl < threshold},
            {"type": "burstiness", "value": burst, "suspicious": burst < threshold},
            {"type": "repetition", "value": rep, "suspicious": rep > threshold},
            {"type": "generic_phrases", "count": generic_count, "suspicious": generic_count > 5},
        ],
        "ensemble_votes": ensemble_details,
    }
```

### 4.3 Semantic Matcher

#### 4.3.1 Two-Stage Architecture

**Stage 1: Bi-Encoder (Retrieval - Fast)**
```
Resume + JD → Shared Encoder → [CLS] embeddings → Cosine Similarity
                                                           │
                                              Top-K (e.g., 50)
```
- Model: `BAAI/bge-base-en-v1.5` (768-dim, 110M params)
- Fine-tuned with contrastive loss + hard negatives
- Inference: ~10ms per pair on GPU

**Stage 2: Cross-Encoder (Re-ranking - Accurate)**
```
[Resume; JD] → Cross-Encoder → Match Probability
```
- Model: `microsoft/deberta-v3-base` with classification head
- Only runs on top-K from bi-encoder
- Inference: ~50ms per pair on GPU

#### 4.3.2 Training Data Construction

**Positive Pairs:**
```python
# From lang-uk: candidate profile + job description in SAME domain
# Heuristic: same top-level ESCO occupation group
positive_pairs = []
for profile in candidate_profiles:
    domain = infer_domain(profile)  # e.g., "software_development"
    matching_jds = job_descriptions[domain]
    for jd in matching_jds[:5]:  # Limit per profile
        positive_pairs.append((profile.text, jd.text, 1.0))
```

**Hard Negative Mining (Critical):**
```python
def mine_hard_negatives(bi_encoder, positive_pairs, candidate_pool, k=50):
    # 1. Encode all candidates
    cand_embeddings = bi_encoder.encode(candidate_pool)
    jd_embeddings = bi_encoder.encode([jd for _, jd, _ in positive_pairs])
    
    # 2. For each positive pair, find similar but NON-matching candidates
    hard_negatives = []
    for i, (_, jd, _) in enumerate(positive_pairs):
        scores = cosine_sim(jd_embeddings[i], cand_embeddings)
        top_k_idx = scores.topk(k).indices
        for idx in top_k_idx:
            if not is_actual_match(candidate_pool[idx], jd):  # Verify not positive
                hard_negatives.append((candidate_pool[idx], jd, 0.0))
    
    # 3. Semi-hard: similar score but wrong label
    # 4. Easy negatives: random (for calibration)
    return hard_negatives
```

#### 4.3.3 Loss Functions
```python
# Bi-encoder: Contrastive with in-batch negatives + hard negatives
def biencoder_loss(embeddings_q, embeddings_d, labels, hard_neg_embeddings=None):
    # embeddings_q: [batch, dim] - resume embeddings
    # embeddings_d: [batch, dim] - JD embeddings
    # labels: 1 for positive, 0 for negative
    
    sim_matrix = embeddings_q @ embeddings_d.T / temperature  # [B, B]
    
    # Standard InfoNCE
    loss = CrossEntropyLoss()(sim_matrix, torch.arange(B))
    
    # Add hard negatives if provided
    if hard_neg_embeddings is not None:
        hard_sim = embeddings_q @ hard_neg_embeddings.T / temperature
        hard_labels = torch.zeros(B, dtype=torch.long)  # All negative
        loss += CrossEntropyLoss()(hard_sim, hard_labels)
    
    return loss

# Cross-encoder: Binary classification with focal loss
def crossencoder_loss(logits, labels):
    # Focal loss for hard examples
    return FocalLoss(alpha=0.25, gamma=2.0)(logits, labels)
```

#### 4.3.4 Training Recipe
```yaml
# configs/matcher.yaml
bi_encoder:
  model: "BAAI/bge-base-en-v1.5"
  max_length: 512
  pooling: "cls"
  normalize: true

cross_encoder:
  model: "microsoft/deberta-v3-base"
  max_length: 512  # Concatenated resume + JD
  num_labels: 2

training_biencoder:
  batch_size: 64
  learning_rate: 1e-5
  num_epochs: 5
  warmup_ratio: 0.1
  temperature: 0.02
  hard_negatives_per_batch: 4
  mining_frequency: 1  # Every epoch

training_crossencoder:
  batch_size: 16
  learning_rate: 5e-6
  num_epochs: 3
  warmup_ratio: 0.1
  focal_loss: true

data:
  positive_pairs: 20000  # From lang-uk same-domain
  hard_negatives: 80000  # Mined
  easy_negatives: 20000  # Random
  val_pairs: 2000
```

#### 4.3.5 Explainable Output
```python
def explain_match(resume, jd, score, skills_resume, skills_jd):
    matched = set(skills_resume) & set(skills_jd)
    missing = set(skills_jd) - set(skills_resume)
    extra = set(skills_resume) - set(skills_jd)
    
    return {
        "match_score": score,  # 0-100
        "skill_coverage": len(matched) / max(len(skills_jd), 1),
        "matched_skills": list(matched),
        "missing_skills": list(missing)[:10],  # Top 10 missing
        "extra_skills": list(extra)[:10],
        "section_alignment": {
            "experience_relevance": compute_section_similarity(resume.exp, jd.requirements),
            "education_match": compute_education_match(resume.edu, jd.edu_req),
            "project_relevance": compute_project_similarity(resume.projects, jd),
        },
        "recommendations": generate_recommendations(missing, extra),
    }
```

---

## 5. Training Pipeline Implementation

### 5.1 Execution Order
```bash
# 1. Environment setup
pip install -r requirements.txt

# 2. Download & prepare data
python src/data_prep.py --output-dir ./data/processed

# 3. Domain-Adaptive Pre-training (DAPT)
python src/dapt.py --model microsoft/deberta-v3-base \
    --corpus ./data/processed/languk_profiles.txt \
    --output ./models/deberta-v3-base-resume-dapt

# 4. Skill Extractor
python src/train_skill_extractor.py \
    --config configs/skill_extractor.yaml \
    --dapt-model ./models/deberta-v3-base-resume-dapt \
    --output ./models/skill_extractor

# 5. Authenticity Detector
python src/train_authenticity.py \
    --config configs/authenticity.yaml \
    --output ./models/authenticity_detector

# 6. Semantic Matcher (Bi-encoder)
python src/train_matcher.py --stage biencoder \
    --config configs/matcher.yaml \
    --output ./models/semantic_matcher/biencoder

# 7. Semantic Matcher (Cross-encoder) - after biencoder
python src/train_matcher.py --stage crossencoder \
    --config configs/matcher.yaml \
    --biencoder ./models/semantic_matcher/biencoder \
    --output ./models/semantic_matcher/crossencoder

# 8. Full evaluation
python src/evaluate.py --all --models-dir ./models
```

### 5.2 Key Implementation Files

#### `src/data_prep.py` - Core Functions
```python
def load_all_datasets():
    """Load all HF datasets with caching."""
    
def clean_resume_text(text: str) -> str:
    """Normalize whitespace, remove artifacts, handle encoding."""
    
def parse_resume_sections(text: str) -> Dict[str, str]:
    """Extract: summary, experience, education, skills, projects."""
    
def create_ner_labels(text: str, skills: List[Dict]) -> List[str]:
    """Convert skill annotations to BIO tags."""
    
def generate_ai_resumes(human_resumes: List[str], n_per_human: int = 3) -> List[str]:
    """Generate AI variants using multiple LLMs."""
    
def build_resume_jd_pairs(profiles, jobs) -> Dataset:
    """Create positive/negative pairs with hard negative mining."""
    
def map_skills_to_esco(skills: List[str]) -> List[Dict]:
    """Fuzzy match to ESCO taxonomy with confidence."""
    
def save_processed(dataset, path: str):
    """Save as Arrow format for fast loading."""
```

#### `src/utils/gpu_utils.py`
```python
def get_device() -> torch.device:
    """Auto-detect GPU/CPU with memory info."""
    
def get_optimal_batch_size(model_name: str, max_length: int) -> int:
    """Heuristic batch size based on GPU VRAM."""
    
def clear_cache():
    """Clear CUDA cache between training runs."""
    
def print_gpu_utilization():
    """Log GPU memory usage."""
```

#### `src/inference_pipeline.py` - Unified Interface
```python
class HireSensePipeline:
    def __init__(self, models_dir: str = "./models", device: str = "auto"):
        self.skill_extractor = SkillExtractor(models_dir)
        self.authenticity = AuthenticityDetector(models_dir)
        self.matcher = SemanticMatcher(models_dir)
    
    def analyze_resume(self, resume_text: str) -> Dict:
        """Full analysis: skills + authenticity."""
        skills = self.skill_extractor.extract(resume_text)
        auth = self.authenticity.detect(resume_text)
        return {"skills": skills, "authenticity": auth}
    
    def match_resume_to_job(self, resume_text: str, jd_text: str) -> Dict:
        """Match with explainability."""
        return self.matcher.match(resume_text, jd_text)
    
    def batch_analyze(self, resumes: List[str]) -> List[Dict]:
        """Optimized batch inference."""
```

---

## 6. Evaluation Protocol

### 6.1 Skill Extractor Evaluation
```python
# On BrainHR+ (15 gold pairs) + imocha test split
metrics = {
    "entity_f1": entity_level_f1(preds, gold),      # Exact span match
    "skill_f1": skill_level_f1(preds, gold),        # Normalized skill match
    "category_f1": per_category_f1(preds, gold),    # PL/FW/TL/SS/DK
    "esco_mapping_accuracy": esco_accuracy(preds),  # Correct ESCO ID
}
```

### 6.2 Authenticity Detector Evaluation
```python
# Test on: held-out human, held-out AI, adversarial AI, hybrid
metrics = {
    "auc_roc": roc_auc_score(y_true, y_pred),
    "auc_pr": average_precision_score(y_true, y_pred),
    "accuracy_50": accuracy_score(y_true, y_pred > 0.5),
    "ece": expected_calibration_error(y_true, y_pred),
    "per_domain_auc": {domain: auc for domain in domains},
    "adversarial_robustness": auc_on_adversarial,
}
```

### 6.3 Semantic Matcher Evaluation
```python
# Test on: BrainHR+ pairs + held-out lang-uk pairs
metrics = {
    "mrr": mean_reciprocal_rank(rankings),
    "recall_at_10": recall_at_k(rankings, k=10),
    "ndcg_at_10": ndcg_at_k(rankings, k=10),
    "classification_f1": f1_score(y_true, y_pred > 0.5),
    "skill_coverage_accuracy": skill_coverage_correlation,
}
```

---

## 7. Deployment Considerations (Post-Pipeline)

### 7.1 Model Serving
- **ONNX Export**: Convert all models for faster inference
- **Quantization**: INT8 for bi-encoder, INT4 for classifier
- **Batching**: Dynamic batching for throughput
- **Caching**: Embedding cache for frequent JDs

### 7.2 Monitoring
- **Data drift**: Track input distribution shifts
- **Model drift**: Monitor prediction confidence distributions
- **Feedback loop**: Collect recruiter corrections → retraining queue

### 7.3 Scaling
| Component | Replicas | Batch Size | Latency (p50) |
|-----------|----------|------------|---------------|
| Skill Extractor | 4 | 32 | 120ms |
| Authenticity | 6 | 64 | 40ms |
| Bi-encoder | 4 | 128 | 10ms |
| Cross-encoder | 2 | 16 | 50ms |

---

## 8. Timeline & Milestones

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 1 | Data prep complete | Processed datasets in ./data/processed |
| 2 | DAPT complete | Resume-adapted DeBERTa |
| 3 | Skill Extractor v1 | F1 > 0.75 on imocha test |
| 4 | Authenticity v1 | AUC > 0.90 on held-out |
| 5 | Matcher v1 (bi+cross) | MRR > 0.65 on eval |
| 6 | Integration + Eval | Full pipeline on BrainHR+ |
| 7 | Iteration & Fixes | All targets met |
| 8 | Documentation + Export | ONNX models, API specs |

---

## 9. Risk Register

| ID | Risk | Likelihood | Impact | Mitigation |
|----|------|------------|--------|------------|
| R1 | lang-uk dataset unavailable | Low | High | Mirror locally, have backup sources |
| R2 | AI detection arms race | High | High | Ensemble, continuous retraining pipeline |
| R3 | Skill taxonomy gaps | Medium | Medium | ESCO + O*NET + custom, active learning |
| R4 | GPU memory OOM | Medium | Low | Gradient accumulation, gradient checkpointing |
| R5 | Cross-encoder too slow | Low | Medium | Distill to bi-encoder, cascade threshold |
| R6 | Bias in training data | High | High | Stratified sampling, fairness audits, diverse eval |

---

## 10. Appendix: ESCO Skill Taxonomy Integration

```python
# ESCO v1.2 has hierarchical skills:
# - 2,942 occupations
# - 13,485 skills
# - 4 skill types: knowledge, skill, competence, language

ESCO_SKILL_TYPES = {
    "knowledge": "THEORETICAL",      # e.g., "machine learning"
    "skill": "PRACTICAL",            # e.g., "python programming"
    "competence": "BEHAVIORAL",      # e.g., "problem solving"
    "language": "LANGUAGE",          # e.g., "english"
}

# Our 5 categories map to ESCO:
CATEGORY_TO_ESCO = {
    "PROG_LANG": ["skill", "knowledge"],
    "FRAMEWORK": ["skill", "knowledge"],
    "TOOL": ["skill"],
    "SOFT_SKILL": ["competence"],
    "DOMAIN_KNOW": ["knowledge"],
}
```

---

## 11. References & Resources

### Papers
1. "Skill Extraction from Job Postings" (Bhola et al., 2020)
2. "Detecting AI-Generated Text in Recruitment" (MITRE, 2024)
3. "BGE: Universal Embedding Models" (BAAI, 2023)
4. "DeBERTa v3: Improving DeBERTa with ELECTRA-style Pre-training" (He et al., 2023)
5. "Hard Negative Mining for Dense Retrieval" (Xiong et al., 2021)

### Datasets
- ESCO Taxonomy: https://esco.ec.europa.eu
- O*NET: https://www.onetonline.org
- Hugging Face Datasets: All linked in Section 3.1

### Tools
- `datasets` library for HF datasets
- `sentence-transformers` for bi-encoder
- `transformers` + `accelerate` for training
- `wandb`/`mlflow` for experiment tracking


