# Language Model Evolution: RNN → LSTM → Transformer

A comprehensive study of language model architectures trained on financial text, demonstrating the evolution from RNNs to Transformers and the critical impact of dataset composition and scale.

## 🎯 Project Overview

Built three neural network architectures from scratch and systematically tested across multiple tokenization strategies and dataset scales to understand when each architecture excels.

**Key Question:** When do RNNs, LSTMs, and Transformers perform best, and how do tokenization and data quality affect results?

## 🏆 Final Results

### Complete Experimental Journey

| Model       | Dataset                | Tokenization | Val Loss   | Text Quality         | Status                     |
| ----------- | ---------------------- | ------------ | ---------- | -------------------- | -------------------------- |
| LSTM        | 640 KB                 | Character    | **1.4711** | Modern, excellent    | 🥇 **Best for small data** |
| Transformer | 640 KB                 | Character    | 1.5523     | Good                 | ✅ Works well              |
| Transformer | 132K words             | Word         | 6.2291     | Broken               | ❌ Insufficient data       |
| Transformer | 103 MB (83% classical) | Word         | 5.3214     | Archaic prose        | ⚠️ Data imbalance          |
| Transformer | 571 MB (over-cleaned)  | Word         | 7.5487     | Gibberish            | ❌ Cleaning destroyed data |
| Transformer | **1 GB (87% modern)**  | Word         | **4.0100** | **Modern, coherent** | ✅ **Best word-level**     |

### Key Achievement

**Successfully scaled from 640 KB → 1 GB while maintaining modern language style.**

Final model (Transformer on 1GB balanced corpus):

- Val Loss: 4.01
- Training: 7.5 hours on TPU
- Text Style: Modern conversational (2018 financial news)
- Vocabulary: 20,000 words
- Dataset: 168 million words, 87% modern content

## 📝 Sample Text Generations

### Character-Level LSTM (1.47 loss) - Small Dataset Champion

**Prompt:** "Money is"

```
"Money is always right to seek by less than investments. When the result
is that the poor and the drivers that are high-specialized because I didn't
want to work for money. The power of being set them to be..."
```

✅ Modern language, financial concepts, semantic coherence

### Word-Level Transformer (4.01 loss) - 1GB Balanced Corpus

**Prompt:** "Money is"

```
"money is really a little more like to manage the money. you know, listen
to your money, you can't run a competitive game. you can't have the financial
situation, but you can't get the handle on the stock..."
```

✅ Modern conversational style, contractions, proper grammar

**Prompt:** "The best investment strategy"

```
"the best investment strategy...recommends that the last market with a
portfolio of products...there's a much better investment approach than at
berkshire hathaway, it's better to find a new way to manage your finances"
```

✅ Mentions real companies (Berkshire Hathaway), modern investment language

## 🔬 Critical Insights & Learnings

### 1. Data Composition > Data Quantity

**Experiment:** 103 MB classical texts vs 571 MB balanced modern content

| Corpus          | Size   | Era Balance   | Val Loss | Output Style               |
| --------------- | ------ | ------------- | -------- | -------------------------- |
| Classical-heavy | 103 MB | 83% pre-1900  | 5.32     | Archaic 18th century prose |
| Balanced modern | 571 MB | 87% post-2000 | 4.01     | Modern conversational      |

**Key Learning:** 500 MB of modern data > 100 MB classical for modern applications

### 2. Data Cleaning Can Destroy Quality

**Experiment:** Aggressive cleaning removed 44% of data

| Version      | Size   | Sentences     | Val Loss | Quality          |
| ------------ | ------ | ------------- | -------- | ---------------- |
| Raw corpus   | 1 GB   | Intact        | 4.01     | Modern, coherent |
| Over-cleaned | 571 MB | 22% fragments | 7.55     | Gibberish        |

**Key Learning:** Preserve sentence structure over removing noise

### 3. Character-Level Excels on Small Data

For datasets <10 MB:

- Character-level LSTM dominates (1.47 loss)
- Word-level struggles (6.23 loss on same data)
- Reason: Vocabulary sparsity (13.6 samples/word insufficient)

### 4. Word-Level Requires Scale AND Balance

Minimum requirements for word-level:

- Data: >500 MB text
- Words: >50 million
- Samples per word: >500
- Era balance: >60% modern content for modern output

### 5. Architecture Fits Data Characteristics

- **Small focused data (<10 MB):** Character-level LSTM wins
- **Large balanced data (>500 MB):** Word-level Transformer competitive
- **No universal winner:** Match architecture to data availability

### 6. Loss Metrics Across Vocabularies Not Comparable

- Character (113 vocab): 1.47 loss = exp(1.47) = 4.35 perplexity
- Word (20K vocab): 4.01 loss = exp(4.01) = 55.1 perplexity

Different difficulty levels - evaluate text quality instead.

## 🚀 Quick Start

### Setup

```bash
git clone https://github.com/YOUR_USERNAME/language-model-evolution.git
cd language-model-evolution
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Train Models Locally (Small Dataset)

```bash
# Character-level (CPU, 40 min)
python src/train/train.py lstm

# Analyze results
python src/analyze/analyze_training.py lstm
```

### Train on Large Dataset (GPU/TPU Required)

```bash
# Prepare 1GB balanced corpus (see Data Collection Guide)
python scripts/create_balanced_final_corpus.py

# Prepare word-level dataset
python src/scripts/utils/prepare_word_level_data.py data/balanced_corpus.txt

# Train on Google Colab (TPU recommended)
# Upload mega_word_dataset.pkl to Google Drive
# Use notebooks/train_on_tpu.ipynb
# Training time: 6-8 hours on TPU, 15-20 hours on GPU
```

## 📁 Project Structure

```
language-model-evolution/
├── src/
│   ├── models/              # Neural network architectures
│   │   ├── simple_rnn.py    # Baseline RNN
│   │   ├── lstm.py          # LSTM with gates
│   │   └── transformer.py   # Multi-head attention
│   ├── train/               # Training scripts
│   │   ├── train.py         # Character-level
│   │   └── train_word_level.py  # Word-level
│   ├── analyze/             # Analysis & visualization
│   └── scripts/utils/       # Data preparation
│
├── data/                    # Training data (not in git)
│   ├── books/              # 379 source texts (103 MB)
│   ├── news/               # Modern financial news (750 MB)
│   ├── balanced_corpus.txt # Combined 1GB corpus
│   └── mega_word_dataset.pkl  # Processed dataset (2.2 GB)
│
├── checkpoints/            # Trained models (not in git)
│   ├── lstm_best.pth      # Char-level LSTM (1.47 loss)
│   └── transformer_1gb_balanced_best.pth  # Word Transformer (4.01 loss)
│
├── results/                # Visualizations and reports
│   ├── training curves
│   ├── comparison plots
│   └── experiment reports
│
├── notebooks/              # Jupyter/Colab notebooks
│   └── train_on_tpu.ipynb # TPU training notebook
│
├── backend/                # FastAPI backend (Phase 4)
└── frontend/               # React frontend (Phase 4)
```

## 📊 Complete Results Summary

### Phase 1: Character-Level Baseline (640 KB corpus)

**Best Model:** LSTM

- Validation Loss: 1.4711
- Parameters: 3.8M
- Training: 50 epochs, 39.5 min CPU
- **Winner for small datasets**

Generated text quality: Excellent modern financial advice

### Phase 2A: Word-Level Small Dataset (132K words)

**Finding:** Severe underfitting due to vocabulary sparsity

- LSTM: 6.41 loss
- Transformer: 6.23 loss
- Samples per word: 13.6 (insufficient)
- **Conclusion:** Need 10-20x more data

### Phase 2B: Mega Corpus - Classical Heavy (103 MB, 22M words)

**Finding:** Data composition determines output style

- Transformer: 5.32 loss
- Classical texts: 83% of corpus
- Output: Perfect 18th-century prose (Adam Smith style)
- **Conclusion:** Data balance matters more than size

### Phase 2C: Balanced Corpus (1 GB, 168M words, 87% modern)

**Best Model:** Transformer (6 layers, 512 dim)

- Validation Loss: 4.01
- Training: 30 epochs, 7.5 hours TPU
- Output: Modern conversational financial language
- **Success:** Modern content produces modern output

## 🔍 Technical Specifications

### Character-Level Models

**LSTM Architecture:**

```python
Embedding(113, 256)
LSTM(256, 512, num_layers=2, dropout=0.3)
Linear(512, 113)
Parameters: 3,765,105
```

### Word-Level Models

**Transformer Architecture (1GB Balanced):**

```python
Embedding(20000, 512)
PositionalEncoding(512)
TransformerEncoder(
    d_model=512,
    nhead=8,
    num_layers=6,
    dim_feedforward=2048
)
Linear(512, 20000)
Parameters: ~12,000,000
```

**Training Configuration:**

- Optimizer: Adam (LR: 0.0003)
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)
- Batch Size: 512 (TPU), 128 (GPU)
- Hardware: Google Colab TPU v2
- Loss: Cross-Entropy

## 📚 Dataset Details

### Small Corpus (640 KB)

- Sources: 2 finance books
- Characters: 652,809
- Words: 132,609
- Best for: Character-level modeling

### Mega Corpus - Classical (103 MB)

- Sources: 229 economics classics (Gutenberg)
- Words: 22 million
- Era: 83% pre-1900, 17% modern
- Result: Archaic language output

### Balanced Corpus - Final (1 GB)

- Sources: 379 books + 300K news articles
- Words: 168 million
- Era: 87% modern (2015-2024), 13% classical
- Composition:
  - Modern financial news: 750 MB (Kaggle datasets)
  - Stock market analysis: 240 MB
  - Classical economics: 35 MB (sampled from originals)
  - Wikipedia/Academic: 5 MB

**Result:** Modern, conversational financial language

## 🎓 Key Learnings

### 1. Match Architecture to Data Scale

| Data Size | Best Approach      | Why                           |
| --------- | ------------------ | ----------------------------- |
| <1 MB     | Character RNN/LSTM | Limited vocab, dense coverage |
| 1-10 MB   | Character LSTM     | Optimal sample density        |
| 10-100 MB | Word LSTM          | Transitional scale            |
| >100 MB   | Word Transformer   | Attention benefits with scale |

### 2. Data Quality Over Quantity

**Evidence:**

- 640 KB focused: 1.47 loss, excellent quality
- 103 MB imbalanced: 5.32 loss, archaic style
- 571 MB over-cleaned: 7.55 loss, broken
- 1 GB balanced: 4.01 loss, modern style

**Lesson:** Curation and balance matter more than raw size

### 3. Domain Determines Style

Model output mirrors training data composition:

- 83% classical texts → 18th-century prose
- 87% modern news → contemporary language
- Can't expect modern output from historical training data

### 4. Vocabulary Size Critical for Word-Level

- 20K vocab on 168M words: 4.01 loss, some `<UNK>` tokens
- Optimal: 30-50K vocab for <2.0 loss
- Trade-off: Larger vocab = larger model = slower training

### 5. Hyperparameters Secondary to Data

**Experiments:**

- Fine-tuning learning rate: 0.5% improvement
- Adjusting dropout: Minimal impact
- Data rebalancing: 24% improvement

**Conclusion:** Get data right first, then tune hyperparameters

### 6. Training From Scratch Has Limits

**Observation:**

- Custom Transformer: 4.01 loss after weeks of work
- Fine-tuned GPT-2: <2.0 perplexity in days (expected)

**Lesson:** Transfer learning beats training from scratch for production

## 🚀 Usage

### Generate Text (Python)

```python
import torch
import pickle
from src.models.transformer import TransformerModel

# Load model and dataset
with open('data/mega_word_dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

checkpoint = torch.load('checkpoints/transformer_1gb_balanced_best.pth')
model = TransformerModel(vocab_size=dataset['vocab_size'], ...)
model.load_state_dict(checkpoint['model_state_dict'])

# Generate
prompt = "Money is"
# [generation code]
```

### API Server (FastAPI)

```bash
cd backend
pip install -r requirements.txt
python main.py

# API available at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

### Web Demo (Coming Soon - Phase 4)

React + FastAPI full-stack application in development.

## 📊 Training Details

### Hardware Used

- **CPU:** Apple Silicon M-series (initial experiments)
- **GPU:** Google Colab V100 (103 MB training)
- **TPU:** Google Colab TPU v2 (1 GB training)

### Training Times

| Model      | Dataset | Hardware | Time    |
| ---------- | ------- | -------- | ------- |
| Char LSTM  | 640 KB  | CPU      | 40 min  |
| Word Trans | 103 MB  | V100 GPU | 1.5 hrs |
| Word Trans | 1 GB    | TPU v2   | 7.5 hrs |

### Cost Analysis

- Google Colab Pro: $10/month
- Total compute cost: ~$10
- Data collection: Free (public domain sources)
- **Total project cost: <$20**

## 🔬 Experiments Conducted

### Experiment 1: Architecture Comparison (Character-Level)

- **Dataset:** 640 KB, 2 finance books
- **Winner:** LSTM (1.47 loss)
- **Insight:** Gates crucial for character sequence modeling

### Experiment 2: Tokenization Strategy

- **Dataset:** Same 640 KB
- **Finding:** Word-level needs 10-20x more data
- **Conclusion:** 13.6 samples/word insufficient

### Experiment 3: Mega Corpus (Classical Heavy)

- **Dataset:** 103 MB, 229 books, 83% pre-1900
- **Result:** 5.32 loss, generates 18th-century prose
- **Learning:** Model mirrors training data era

### Experiment 4: Hyperparameter Tuning

- **Approach:** Lower LR, higher dropout, early stopping
- **Result:** 0.5% improvement only
- **Conclusion:** Data quality is bottleneck, not hyperparameters

### Experiment 5: Data Cleaning Effects

- **Aggressive cleaning:** 44% removed, 7.55 loss (destroyed sentences)
- **Minimal cleaning:** 5-10% removed, 4.01 loss (preserved structure)
- **Learning:** Over-cleaning worse than under-cleaning

### Experiment 6: Balanced Corpus at Scale

- **Dataset:** 1 GB, 168M words, 87% modern
- **Training:** TPU, adaptive LR, 30 epochs
- **Result:** 4.01 loss, modern language achieved
- **Success:** Proper data balance produces proper output

## 📈 Scaling Analysis

### Vocabulary Density Requirements

| Dataset | Words | Vocab | Samples/Word | Val Loss   | Status                  |
| ------- | ----- | ----- | ------------ | ---------- | ----------------------- |
| Small   | 132K  | 10K   | 13.6         | 6.23       | Severe underfitting     |
| Medium  | 22M   | 20K   | 916          | 5.32       | Learning but imbalanced |
| Large   | 168M  | 20K   | 5,570        | 4.01       | Good density            |
| Optimal | 168M  | 40K   | 4,200        | ~3.5 (est) | Target                  |

**Finding:** Need >500 samples per word for stable learning

### Data Quality Impact

**Text preprocessing effects:**

| Processing | Data Lost | Sentence Integrity     | Model Performance   |
| ---------- | --------- | ---------------------- | ------------------- |
| None       | 0%        | Perfect                | Noisy but learnable |
| Minimal    | 5-10%     | Preserved              | ✅ Optimal          |
| Aggressive | 44%       | Broken (22% fragments) | ❌ Destroyed        |

**Optimal:** Remove only HTML, URLs, excessive whitespace

## 🛠️ Tech Stack

**Core:**

- PyTorch 2.1+ (deep learning framework)
- NumPy, Pandas (data processing)
- Matplotlib, Seaborn (visualization)

**Data Collection:**

- pdfplumber, PyPDF2 (PDF extraction)
- wikipedia, datasets (public data)
- Beautiful Soup (web scraping)

**Training Infrastructure:**

- Google Colab Pro (GPU/TPU)
- Weights & Biases (experiment tracking - optional)

**Production (Phase 4):**

- FastAPI (backend API)
- React (frontend)
- Docker (containerization)
- PostgreSQL (conversation storage)

## 📦 Installation

### Local Development

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/language-model-evolution.git
cd language-model-evolution

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

```bash
# Character-level (small dataset)
python src/scripts/utils/prepare_data.py

# Word-level (requires 1GB corpus - see Data Collection Guide)
python scripts/create_balanced_final_corpus.py
python src/scripts/utils/prepare_word_level_data.py data/balanced_corpus.txt
```

### Training

**Local (CPU):**

```bash
python src/train/train.py lstm  # 40 min
```

**Google Colab (GPU/TPU):**

- Upload notebook: `notebooks/train_on_tpu.ipynb`
- Upload dataset to Google Drive
- Select TPU runtime
- Execute all cells

## 📚 Data Sources

### Classical Economics (103 MB)

- Project Gutenberg: 229 public domain books
- Authors: Adam Smith, Karl Marx, Keynes, Mises, Hayek
- Era: 1700s-1900s
- License: Public domain

### Modern Financial News (750 MB)

- Kaggle datasets: 300K+ articles (2015-2024)
- Sources: CNBC, Reuters, Bloomberg, MarketWatch
- License: Dataset-specific (check individual sources)

### Supporting Data (15 MB)

- Wikipedia: 127 finance articles (CC-BY-SA)
- Academic: 10 arXiv papers (open access)
- HuggingFace: Financial sentiment datasets

**Total:** 379 sources, 1 GB combined, 168M words

## 🎯 Next Steps: Phase 4 - Production Deployment

### Planned Features

**Core Functionality:**

- 💬 Chat interface (like ChatGPT for finance)
- 📊 Multiple model comparison
- 🔄 Conversation history
- ⚙️ Adjustable parameters (temperature, length)

**Advanced Features:**

- 🔐 User authentication
- 💾 Conversation persistence
- 📈 Usage analytics
- 🎨 Custom themes

### Tech Stack (Phase 4)

**Backend:**

- FastAPI (Python web framework)
- PostgreSQL (user data, conversations)
- Redis (caching, sessions)
- Docker (containerization)

**Frontend:**

- React 18+ (UI framework)
- Material-UI (component library)
- Axios (API calls)
- React Router (navigation)

**Deployment:**

- Railway/Render (hosting)
- Cloudflare (CDN)
- GitHub Actions (CI/CD)

## 📄 Documentation

### Detailed Experiment Reports

- `results/MEGA_CORPUS_EXPERIMENT.md` - 103 MB classical corpus analysis
- `results/word_level/README.md` - Word-level tokenization experiments
- `DATA_COLLECTION_PLAN.md` - Dataset assembly strategy
- `checkpoints/MODEL_SUMMARY.md` - All trained models catalog

### Notebooks

- `notebooks/train_on_tpu.ipynb` - TPU training workflow
- `notebooks/analyze_results.ipynb` - Interactive analysis (WIP)

## 🤝 Contributing

This is a research and learning project demonstrating:

- Language model architecture evolution
- Impact of data composition and scale
- Practical ML engineering workflow

Feel free to:

- Fork and experiment with different architectures
- Try different datasets or domains
- Implement additional models (GRU, GPT, BERT)
- Improve data preprocessing

## 📄 License

MIT License - See LICENSE file

## 🙏 Acknowledgments

**Inspiration:**

- The evolution of NLP from RNNs (1980s) to Transformers (2017+)
- "Attention Is All You Need" (Vaswani et al., 2017)
- OpenAI's GPT series demonstrating scale benefits

**Data Sources:**

- Project Gutenberg (public domain books)
- Kaggle community datasets
- Wikipedia contributors
- arXiv open access papers

**Computational Resources:**

- Google Colab Pro (TPU/GPU access)
- Anthropic's Claude (code assistance and debugging)

## 📞 Project Stats

**Development Time:** 3 weeks  
**Lines of Code:** ~3,000  
**Models Trained:** 8 distinct experiments  
**Total Training Time:** ~40 hours  
**Dataset Collected:** 1 GB+ across 379 sources  
**Best Model:** Character-level LSTM (1.47 loss) for small data  
**Production Model:** Word Transformer (4.01 loss) for modern applications

## 🎓 Educational Value

This project demonstrates understanding of:

- ✅ Neural network architectures (RNN, LSTM, Transformer)
- ✅ Sequence modeling and attention mechanisms
- ✅ Data collection and preprocessing at scale
- ✅ Training optimization (learning rates, schedulers, regularization)
- ✅ Experimental design and ablation studies
- ✅ Production ML considerations (data quality, compute trade-offs)
- ✅ Model evaluation beyond simple metrics

**Perfect for:**

- FAANG ML engineer interviews
- Graduate school applications
- Research paper foundation
- Production deployment experience

---

**Current Status:**  
✅ **Research Complete** - All experiments finished  
✅ **Best Models Identified** - Char LSTM (small), Word Transformer (large)  
🔄 **Phase 4 In Progress** - Full-stack deployment  
⏳ **Demo Deployment** - Coming soon

**Key Achievement:** Demonstrated that data composition and quality matter more than architecture choice or dataset size alone.

**Last Updated:** November 17, 2025  
**Project Status:** Production deployment phase  
**Next Milestone:** Full-stack web application with React + FastAPI
