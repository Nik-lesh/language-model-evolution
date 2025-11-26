# Language Model Evolution: From RNN to Transformer

A comprehensive deep learning project demonstrating the evolution of language model architectures, trained on financial text to understand the impact of model architecture, tokenization strategy, and dataset composition on performance.

## 🎯 Project Summary

This project systematically compares three generations of neural network architectures (RNN → LSTM → Transformer) across multiple tokenization strategies and dataset scales, providing insights into when each approach excels.

**Key Achievement:** Successfully scaled from 640 KB to 1 GB balanced corpus while demonstrating that data quality and composition matter as much as architecture choice.

## 🏆 Complete Results Overview

| Experiment   | Model       | Dataset             | Tokenization | Val Loss      | Text Quality              | Key Learning                                   |
| ------------ | ----------- | ------------------- | ------------ | ------------- | ------------------------- | ---------------------------------------------- |
| **Baseline** | Simple RNN  | 640 KB              | Character    | 1.6482        | Poor                      | Vanishing gradients limit performance          |
| **Phase 1**  | **LSTM**    | 640 KB              | Character    | **1.4711** 🥇 | **Excellent**             | **Gates solve long-term dependencies**         |
| Comparison   | Transformer | 640 KB              | Character    | 1.5523        | Good                      | Works well, LSTM slightly better on small data |
| **Phase 2A** | Transformer | 132K words          | Word         | 6.2291        | Broken                    | Vocabulary too sparse (13.6 samples/word)      |
| **Phase 2B** | Transformer | 22M words (103 MB)  | Word         | 5.3214        | Archaic 18th century      | 83% classical texts → classical output         |
| Over-cleaned | Transformer | 111M words (571 MB) | Word         | 7.5487        | Gibberish                 | Aggressive cleaning destroyed sentences        |
| **Phase 2C** | Transformer | 168M words (1 GB)   | Word         | **4.0100** ✅ | **Modern conversational** | **87% modern content → modern style**          |

## 📊 Key Findings

### 1. Data Composition > Architecture > Dataset Size

**Most Important Discovery:**

The **composition and quality** of training data determines output style more than model architecture or dataset size.

**Evidence:**

- 640 KB focused modern corpus → 1.47 loss, excellent modern text (LSTM)
- 103 MB classical-heavy corpus → 5.32 loss, 18th-century prose (Transformer)
- 571 MB over-cleaned corpus → 7.55 loss, gibberish (Transformer)
- 1 GB balanced modern corpus → 4.01 loss, modern conversational text (Transformer)

**Lesson:** Small, well-curated, balanced data > large, imbalanced, or poorly processed data

### 2. Character-Level vs Word-Level Trade-offs

| Aspect                | Character-Level      | Word-Level                                |
| --------------------- | -------------------- | ----------------------------------------- |
| **Min Data Required** | 500 KB - 10 MB       | 100 MB+                                   |
| **Vocabulary**        | 50-200 chars         | 10K-50K words                             |
| **Best Architecture** | LSTM                 | Transformer                               |
| **Training Speed**    | Slower (more tokens) | Faster (fewer tokens)                     |
| **Small Dataset**     | Wins (1.47 loss)     | Fails (6.23 loss)                         |
| **Large Dataset**     | Good                 | Better (4.01 loss)                        |
| **Rare Words**        | Handles naturally    | Needs large vocab or subword tokenization |

**Optimal Use Cases:**

- **Character-level:** Datasets <10 MB, languages with complex morphology, handling typos/abbreviations
- **Word-level:** Datasets >100 MB, semantic understanding, modern NLP applications

### 3. Tokenization Density Requirements

**Critical Metric:** Samples per vocabulary item

| Dataset | Tokenization | Vocab | Total Tokens | Samples/Vocab | Result                                 |
| ------- | ------------ | ----- | ------------ | ------------- | -------------------------------------- |
| 640 KB  | Character    | 113   | 652K         | 5,775         | ✅ Excellent (1.47 loss)               |
| 640 KB  | Word         | 10K   | 132K         | 13.6          | ❌ Severe underfitting (6.23 loss)     |
| 103 MB  | Word         | 20K   | 22M          | 916           | ⚠️ Learning but imbalanced (5.32 loss) |
| 1 GB    | Word         | 20K   | 168M         | 5,570         | ✅ Good density (4.01 loss)            |

**Minimum Threshold:** >500 samples per vocabulary item for stable learning

### 4. The Adam Smith Effect: Dataset Composition Determines Output Style

**Experiment:** Trained Transformer on 103 MB corpus (83% classical economics texts)

**Result:** Model generated perfect 18th-century prose:

```
"the whole capital of consumption is always at the price of all sorts.
the quantity of silver which it will purchase, in exchange for which
all other improvements are purchased at the same time..."
```

**Analysis:**

- Adam Smith's "Wealth of Nations" (1776) heavily represented in corpus
- Model learned classical economic terminology and sentence structures
- Grammatically perfect but completely archaic for modern applications
- Modern sources (17%) drowned out by classical majority

**Lesson:** Model output will mirror the era and style of training data, regardless of architecture sophistication

### 5. Data Preprocessing Impact

**Experiment:** Aggressive cleaning removed 44% of data

| Metric           | Raw Corpus      | Aggressively Cleaned | Impact              |
| ---------------- | --------------- | -------------------- | ------------------- |
| **Size**         | 1,020 MB        | 571 MB               | 44% removed         |
| **Sentences**    | Intact          | 22% fragments        | Structure destroyed |
| **Val Loss**     | 4.01            | 7.55                 | 88% worse           |
| **Text Quality** | Modern coherent | Gibberish            | Unusable            |

**What Went Wrong:**

- Removed lines with >50% numbers (deleted financial data!)
- Broke sentences mid-word
- Created 559K sentence fragments
- Destroyed contextual relationships

**Optimal Cleaning:**

- Remove only: HTML tags, URLs, excessive whitespace
- Preserve: Numbers, financial symbols ($, %), sentence structure
- Result: 5-10% removed, quality preserved

### 6. Hyperparameters Secondary to Data Quality

**Experiment:** Fine-tuning after identifying data issues

| Optimization                               | Improvement           | Conclusion                      |
| ------------------------------------------ | --------------------- | ------------------------------- |
| Lower learning rate (0.001 → 0.0001)       | 0.5%                  | Minimal                         |
| Higher dropout (0.2 → 0.4)                 | <1%                   | Marginal                        |
| Early stopping                             | Prevented degradation | Useful but doesn't improve peak |
| **Data rebalancing (83% → 13% classical)** | **24%**               | **Primary factor**              |

**Lesson:** Fix data problems before extensive hyperparameter tuning

### 7. Training From Scratch vs Transfer Learning

**Observation:**

- Custom Transformer: 4.01 loss after 7.5 hours TPU training
- GPT-2 fine-tuned (expected): <2.0 perplexity in 2-3 hours
- Difference: GPT-2 pre-trained on 40 GB modern text

**Lesson:** For production applications, fine-tuning pre-trained models superior to training from scratch

## 📝 Sample Text Generations

### Character-Level LSTM (1.47 loss) - Best Overall

**Prompt:** "Money is"

```
Money is always right to seek by less than investments. When the result
is that the poor and the drivers that are high-specialized because I
didn't want to work for money. The power of being set them to be...
```

**Quality Assessment:**

- ✅ Modern language (2017-2018 style)
- ✅ Financial vocabulary: "investments", "specialized"
- ✅ Semantic coherence within sentences
- ✅ No unknown tokens
- ⚠️ Some grammatical quirks

**Best for:** Small datasets, production deployment on limited resources

---

### Word-Level Transformer - 103 MB Classical Corpus (5.32 loss)

**Prompt:** "Money is"

```
money is meant. but the whole capital of consumption is always at the price
of all sorts. the quantity of silver which it will purchase, in exchange for
which all other improvements are purchased at the same time, is the same, not
of a greater value than, but of which it can purchase...
```

**Quality Assessment:**

- ✅ Grammatically flawless
- ✅ Economically accurate (classical theory)
- ❌ 18th-century Adam Smith prose style
- ❌ Archaic terminology ("quantity of silver", "whole capital of consumption")
- ❌ Unsuitable for modern financial advice

**Demonstrates:** Data composition determines output style

---

### Word-Level Transformer - 1 GB Balanced Corpus (4.01 loss)

**Prompt:** "Money is"

```
money is really a little more like to manage the money. you know, listen
to your money, you can't run a competitive game. you can't have the
financial situation, but you can't get the handle on the stock...
```

**Quality Assessment:**

- ✅ Modern conversational style (2018 news)
- ✅ Contractions ("can't", "you're")
- ✅ Colloquial phrases ("you know")
- ✅ Financial vocabulary appropriate for general audience
- ⚠️ Some semantic drift between sentences
- ⚠️ Occasional `<UNK>` tokens (vocabulary limit)

**Prompt:** "The best investment strategy"

```
...recommends portfolio of products...there's a much better investment
approach than at berkshire hathaway, it's better to find a new way to
manage your finances
```

**Quality Assessment:**

- ✅ Mentions real companies (Berkshire Hathaway - Warren Buffett)
- ✅ Modern investment terminology
- ✅ Practical advice framing
- ✅ Natural language flow

**Best for:** Large-scale applications requiring modern language

## 🔬 Technical Specifications

### Models Implemented

#### Simple RNN

```python
Architecture:
  Embedding(113, 128)
  RNN(128, 256, num_layers=2, dropout=0.3)
  Linear(256, 113)

Parameters: 273,905
Val Loss: 1.6482
Training: 50 epochs, 3.8 min CPU
```

#### LSTM (Best for Small Data)

```python
Architecture:
  Embedding(113, 256)
  LSTM(256, 512, num_layers=2, dropout=0.3)
  Linear(512, 113)

Parameters: 3,765,105
Val Loss: 1.4711 🥇
Training: 50 epochs, 39.5 min CPU
```

#### Transformer (Best for Large Balanced Data)

```python
Architecture:
  Embedding(20000, 512)
  PositionalEncoding(512, dropout=0.2)
  TransformerEncoder(
    d_model=512,
    nhead=8,
    num_layers=6,
    dim_feedforward=2048,
    dropout=0.2
  )
  Linear(512, 20000)

Parameters: ~12,000,000
Val Loss: 4.0100 (1GB corpus)
Training: 30 epochs, 7.5 hours TPU
```

### Training Infrastructure

**Hardware Progression:**

- Phase 1: Apple Silicon M-series CPU (char-level models)
- Phase 2A-B: Google Colab V100 GPU (103 MB corpus)
- Phase 2C: Google Colab TPU v2 (1 GB corpus)

**Training Times:**
| Model | Dataset | Hardware | Time |
|-------|---------|----------|------|
| Simple RNN | 640 KB | CPU | 3.8 min |
| LSTM | 640 KB | CPU | 39.5 min |
| Transformer | 640 KB | CPU | 103 min |
| Transformer | 103 MB | V100 GPU | 1.5 hrs |
| Transformer | 1 GB | TPU v2 | 7.5 hrs |

### Optimization Techniques Applied

**Learning Rate Scheduling:**

- ReduceLROnPlateau (factor=0.5, patience=3)
- Adaptive learning from 0.0003 → 0.000037 over training
- Prevented divergence on large dataset

**Regularization:**

- Dropout: 0.2-0.3 depending on model size
- Gradient clipping: max_norm=5.0
- Early stopping: Monitored validation loss

**Hardware Optimization:**

- Batch size scaled with hardware: 64 (CPU) → 128 (GPU) → 512 (TPU)
- Mixed precision training on GPU (not needed on TPU)
- Gradient accumulation for memory-limited setups

## 📚 Dataset Construction

### Final Balanced Corpus (1 GB)

**Composition:**

- Modern financial news (2015-2024): 750 MB (73.5%)
- Stock market analysis: 240 MB (23.5%)
- Classical economics (sampled): 35 MB (3.4%)
- Wikipedia/Academic: 5 MB (0.5%)

**Sources:**

- Kaggle datasets: 300K+ articles from CNBC, Reuters, Bloomberg, MarketWatch
- Project Gutenberg: Curated selection of 30 foundational economics texts
- HuggingFace: Financial sentiment and classification datasets
- Total: 379 unique sources

**Statistics:**

- Total size: 1,020 MB raw → 598 MB cleaned
- Total words: 168,174,387
- Vocabulary: 20,000 most common words
- Unique characters: 5,818
- Modern content: 87% (by volume)

### Data Collection Process

**Phase 1:** Initial corpus (2 books, 640 KB)

- Manual PDF extraction
- Text cleaning and normalization

**Phase 2:** Classical expansion (229 books, 103 MB)

- Automated Project Gutenberg scraping
- Discovered data composition problem

**Phase 3:** Modern rebalancing (379 sources, 1 GB)

- Kaggle financial news datasets (manual download due to CLI issues)
- HuggingFace open datasets
- Wikipedia finance articles
- Balanced to 87% modern content

**Phase 4:** Cleaning and quality control

- Tested aggressive vs minimal cleaning
- Found over-cleaning destroys sentence structure
- Optimal: Remove only HTML, URLs, excessive whitespace

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Nik-lesh/language-model-evolution.git
cd language-model-evolution

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Train Models Locally

**Character-level (CPU, ~40 minutes):**

```bash
python src/train/train.py lstm
```

**Analyze results:**

```bash
python src/analyze/analyze_training.py lstm
python src/analyze/generate_samples.py
```

### Train on Large Dataset (GPU/TPU Required)

Requires Google Colab Pro for reasonable training times.

**Steps:**

1. Prepare 1 GB balanced corpus (see `DATA_COLLECTION_PLAN.md`)
2. Generate word-level dataset: `python src/scripts/utils/prepare_word_level_data.py data/balanced_corpus.txt`
3. Upload `mega_word_dataset.pkl` to Google Drive (2.2 GB)
4. Open `notebooks/train_on_tpu.ipynb` in Google Colab
5. Select TPU v2 runtime
6. Run all cells (~7.5 hours training time)

### Run API Backend (Local)

```bash
cd backend
pip install -r requirements.txt
python main.py

# API available at http://localhost:8000
# Documentation at http://localhost:8000/docs
```

### Run Frontend (Local)

```bash
cd frontend
npm install
npm start

# Opens at http://localhost:3000
```

## 📁 Project Structure

```
language-model-evolution/
├── src/
│   ├── models/                    # Neural network implementations
│   │   ├── simple_rnn.py          # Vanilla RNN with hidden state
│   │   ├── lstm.py                # LSTM with forget/input/output gates
│   │   └── transformer.py         # Multi-head self-attention
│   ├── train/                     # Training pipelines
│   │   ├── train.py               # Character-level training
│   │   └── train_word_level.py    # Word-level training
│   ├── analyze/                   # Analysis and visualization
│   │   ├── analyze_training.py
│   │   ├── compare_models.py
│   │   └── generate_samples.py
│   └── scripts/utils/             # Data preparation utilities
│       ├── prepare_data.py        # Character-level tokenization
│       └── prepare_word_level_data.py  # Word-level tokenization
│
├── data/                          # Training data (excluded from git)
│   ├── books/                     # Source texts by category
│   ├── news/                      # Modern financial news
│   ├── balanced_corpus.txt        # 1 GB combined corpus
│   ├── dataset.pkl                # Character-level (5.7 MB)
│   └── mega_word_dataset.pkl      # Word-level (2.2 GB)
│
├── checkpoints/                   # Trained models (excluded from git)
│   ├── simple_rnn_best.pth        # 1.1 MB
│   ├── lstm_best.pth              # 14.3 MB - BEST MODEL
│   ├── transformer_best.pth       # 12.2 MB
│   └── transformer_1gb_balanced_best.pth  # 168 MB
│
├── results/                       # Visualizations and reports (in git)
│   ├── training_curves/
│   ├── comparison_plots/
│   ├── MEGA_CORPUS_EXPERIMENT.md
│   └── word_level/README.md
│
├── notebooks/                     # Jupyter/Colab notebooks
│   └── train_on_tpu.ipynb         # TPU training workflow
│
├── backend/                       # FastAPI backend (Phase 4)
│   ├── main.py                    # API server
│   ├── services/model_service.py  # Model inference
│   ├── models/schemas.py          # Request/response models
│   └── config/settings.py         # Configuration
│
├── frontend/                      # React frontend (Phase 4)
│   └── src/components/ChatInterface.jsx
│
├── scripts/                       # Utility scripts
│   ├── create_balanced_final_corpus.py
│   ├── upload_to_huggingface.py
│   └── clean_balanced_corpus.py
│
├── requirements.txt               # Python dependencies
├── README.md
└── .gitignore
```

## 🎓 Educational Insights

### Architecture Evolution Understanding

**RNN (1980s):**

- Sequential processing
- Hidden state carries context
- **Problem:** Vanishing gradients over long sequences
- **Result:** 1.65 loss, forgets context beyond ~10 characters

**LSTM (1997):**

- Gating mechanisms (forget, input, output)
- Separate cell state for long-term memory
- **Innovation:** Controlled information flow prevents vanishing gradients
- **Result:** 1.47 loss, handles dependencies across entire sequence
- **Winner** for small, focused datasets

**Transformer (2017):**

- Parallel processing via self-attention
- Positional encoding for sequence order
- **Innovation:** Each token attends to all others simultaneously
- **Trade-off:** Needs more data to shine
- **Result:** 4.01 loss on 1GB, competitive with proper scale and balance

### When Each Architecture Excels

**Use Simple RNN when:**

- Prototyping quickly
- Extremely limited compute
- Educational purposes
- Not recommended for production

**Use LSTM when:**

- Dataset: 1 MB - 100 MB
- Sequential dependencies important
- Limited training time/compute
- Character-level tokenization
- **Best choice:** 1.47 loss achieved on focused dataset

**Use Transformer when:**

- Dataset: >100 MB, ideally >500 MB
- Well-balanced, modern content
- Sufficient compute (GPU/TPU)
- Word-level or subword tokenization
- **Best choice:** Modern language applications with scale

### Cross-Entropy Loss Interpretation

**Important:** Loss values not directly comparable across different vocabulary sizes

**Example:**

- Character-level (113 vocab): 1.47 loss = exp(1.47) = 4.35 perplexity
- Word-level (20K vocab): 4.01 loss = exp(4.01) = 55.1 perplexity

Random baseline:

- Character: log(113) = 4.73 loss
- Word: log(20000) = 9.90 loss

**Better metric:** Perplexity (lower is better) or human evaluation of text quality

## 🛠️ Technologies Used

**Core ML:**

- PyTorch 2.x (deep learning framework)
- NumPy (numerical computing)
- TQDM (progress tracking)

**Data Processing:**

- pdfplumber, PyPDF2 (PDF text extraction)
- pandas (data manipulation)
- BeautifulSoup (web scraping)
- HuggingFace Datasets (pre-built datasets)

**Visualization:**

- Matplotlib, Seaborn (plotting)
- Jupyter notebooks (interactive analysis)

**Production:**

- FastAPI (backend API)
- React + Material-UI (frontend)
- HuggingFace Hub (model hosting)
- Google Colab (cloud compute)

**Development:**

- Git + GitHub (version control)
- VS Code (IDE)
- Virtual environments (dependency isolation)

## 📊 Training Details

### Hyperparameter Configurations

**Character-Level LSTM (Best):**

```python
{
    'embedding_dim': 256,
    'hidden_dim': 512,
    'num_layers': 2,
    'dropout': 0.3,
    'learning_rate': 0.001,
    'batch_size': 64,
    'seq_length': 100,
    'epochs': 50
}
```

**Word-Level Transformer (1GB):**

```python
{
    'd_model': 512,
    'nhead': 8,
    'num_layers': 6,
    'dim_feedforward': 2048,
    'dropout': 0.3,
    'learning_rate': 0.0003,  # Adaptive
    'batch_size': 512,  # TPU
    'seq_length': 50,
    'epochs': 30
}
```

### Computational Costs

**Total Project Compute:**

- CPU training: ~3 hours
- GPU training: ~5 hours (V100)
- TPU training: ~8 hours (v2)
- **Total cost:** ~$10 (Google Colab Pro subscription)

**Energy Efficiency:**

- Local CPU: Minimal cost
- Cloud GPU: ~$0.50/hour
- Cloud TPU: Included in Colab Pro ($10/month)

## 📦 Pre-trained Models

Models available on HuggingFace Hub: [Nikilesh9/financial-language-model](https://huggingface.co/Nikilesh9/financial-language-model)

**Available Files:**

- `lstm_char_level.pth` (14 MB) - Character-level LSTM, 1.47 loss
- `transformer_1gb_balanced_best.pth` (168 MB) - Word Transformer, 4.01 loss
- `char_dataset.pkl` (5.7 MB) - Character-level dataset
- `mega_word_dataset.pkl` (2.2 GB) - Word-level dataset (1GB corpus)

**Usage:**

```python
from huggingface_hub import hf_hub_download
import torch
import pickle

# Download model
model_path = hf_hub_download(
    repo_id="Nikilesh9/financial-language-model",
    filename="lstm_char_level.pth"
)

# Download dataset
dataset_path = hf_hub_download(
    repo_id="Nikilesh9/financial-language-model",
    filename="char_dataset.pkl"
)

# Load and use
with open(dataset_path, 'rb') as f:
    dataset = pickle.load(f)

checkpoint = torch.load(model_path, map_location='cpu')
# ... create model and generate text
```

## 🎯 Future Work

### Immediate Improvements

**1. Increase Word-Level Vocabulary (Quick Win)**

- Current: 20K words → Target: 40K words
- Expected: 4.01 → 3.5 loss
- Fewer `<UNK>` tokens
- Time: 6-8 hours retraining

**2. Subword Tokenization (BPE/WordPiece)**

- Better handling of rare words
- Vocabulary: 30-50K subword units
- Expected: 2.5-3.5 loss
- Production-standard approach

**3. Fine-Tune GPT-2 (Best Quality)**

- Start with pre-trained GPT-2 (40 GB modern text)
- Fine-tune on 1 GB finance corpus
- Expected: <2.0 perplexity
- Time: 2-3 hours on GPU
- **Recommended path for production**

### Research Extensions

**Architecture Experiments:**

- GPT-style decoder-only Transformer
- BERT-style bidirectional encoding
- Transformer-XL (longer context)
- Mixture of Experts (MoE)

**Training Enhancements:**

- Beam search decoding
- Top-k and nucleus (top-p) sampling
- Reinforcement learning from human feedback (RLHF)
- Curriculum learning (easy→hard examples)

**Production Features:**

- Retrieval-Augmented Generation (RAG) with vector database
- Multi-task learning (generation + classification)
- Model distillation (large→small for deployment)
- Quantization (INT8) for faster inference

## 📖 Documentation

### Detailed Reports

- `results/MEGA_CORPUS_EXPERIMENT.md` - 103 MB classical corpus analysis
- `results/word_level/README.md` - Word-level tokenization experiments
- `DATA_COLLECTION_PLAN.md` - Dataset assembly strategy
- `checkpoints/MODEL_SUMMARY.md` - Complete model catalog

### Notebooks

- `notebooks/train_on_tpu.ipynb` - Cloud TPU training workflow
- Interactive analysis notebooks (planned)

## 🤝 Contributing

This research project demonstrates practical ML engineering. Contributions welcome:

**Areas for contribution:**

- Additional architectures (GRU, Mamba, RWKV)
- Different domains (legal, medical, code)
- Improved data preprocessing pipelines
- Enhanced evaluation metrics
- Production deployment examples

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

**Educational Resources:**

- "Attention Is All You Need" (Vaswani et al., 2017)
- Stanford CS231n, CS224n courses
- PyTorch documentation and tutorials

**Data Sources:**

- Project Gutenberg (public domain books)
- Kaggle community (financial news datasets)
- HuggingFace (open datasets)
- Wikipedia (CC-BY-SA licensed articles)

**Computational Resources:**

- Google Colab Pro (GPU/TPU access)
- Personal hardware (Apple Silicon for development)

**Tools and Assistance:**

- GitHub (version control and collaboration)
- Anthropic's Claude (debugging and code assistance)
- VS Code (development environment)

## 📊 Project Statistics

**Development Timeline:** 4 weeks  
**Code Written:** ~4,000 lines Python  
**Models Trained:** 8 distinct experiments  
**Total Training Time:** ~45 hours  
**Data Collected:** 1 GB across 379 sources  
**Experiments Conducted:** 6 major experiments  
**Key Insights:** 7 critical learnings documented

**Best Model:** Character-level LSTM (1.47 val loss)  
**Most Interesting:** Word Transformer on balanced corpus (demonstrates data importance)  
**For Production:** Recommend GPT-2 fine-tuning (best quality/effort ratio)

## 🎓 Skills Demonstrated

This project showcases:

**Machine Learning:**

- ✅ Neural network architecture implementation from scratch
- ✅ Training optimization (learning rates, schedulers, regularization)
- ✅ Hyperparameter tuning and ablation studies
- ✅ Model evaluation beyond simple metrics
- ✅ Understanding architecture-data fit

**Data Engineering:**

- ✅ Large-scale data collection (1 GB from 379 sources)
- ✅ PDF extraction and text processing
- ✅ Data cleaning and quality control
- ✅ Dataset balancing and composition analysis
- ✅ Tokenization strategies (character vs word-level)

**Software Engineering:**

- ✅ Clean, modular code organization
- ✅ Version control with Git (meaningful commits, branching)
- ✅ API development (FastAPI with OpenAPI docs)
- ✅ Frontend development (React + Material-UI)
- ✅ Cloud deployment (HuggingFace Hub integration)

**Research Skills:**

- ✅ Experimental design and methodology
- ✅ Systematic comparison and ablation studies
- ✅ Root cause analysis (diagnosing data vs architecture issues)
- ✅ Clear documentation of findings
- ✅ Honest reporting of failures and learnings

**Production Readiness:**

- ✅ Model hosting and versioning
- ✅ API design and implementation
- ✅ Frontend user interface
- ✅ Deployment pipeline (attempted Railway/Render)
- ✅ Understanding of production constraints (memory, latency)

## 🎬 Demo

**Video Demonstration:** [Coming Soon - Demo Video Link]

Shows:

- Model training process
- Text generation examples
- Comparison across architectures
- Full-stack application interface

**Live Demo:** [Planned - Will deploy after optimization]

## 📞 Contact & Links

**Project Repository:** https://github.com/Nik-lesh/language-model-evolution  
**HuggingFace Models:** https://huggingface.co/Nikilesh9/financial-language-model  
**Author:** Nikhilesh (Northeastern University)  
**Purpose:** Educational demonstration of language model evolution and practical ML engineering

## 🏆 Key Takeaways

1. **Data quality matters more than model size** - 640 KB focused > 103 MB imbalanced
2. **Match architecture to data scale** - LSTM for <100 MB, Transformer for >500 MB
3. **Dataset composition determines output style** - 83% classical → classical prose
4. **Character-level surprisingly competitive** - 1.47 loss vs 4.01 with 1600x less data
5. **Over-engineering is real** - Simple approaches often win
6. **Shipping > Perfection** - Working demo more valuable than perfect metrics
7. **Transfer learning > training from scratch** - Fine-tune GPT-2 for production

---

**Current Status:** ✅ Research Complete | ⏳ Deployment In Progress | 🎯 Production Optimization Planned

**Best Model for Production:** Character-level LSTM (1.47 loss, 14 MB, modern style)  
**Most Interesting Finding:** Data composition matters more than architecture choice  
**Next Milestone:** Full-stack deployment with optimized model serving

**Last Updated:** November 23, 2025  
**Project Phase:** Complete research, moving to production deployment
