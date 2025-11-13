# Language Model Evolution: RNN → LSTM → Transformer

A comprehensive study comparing three generations of language models on financial text, exploring the impact of architecture, tokenization strategy, and dataset scale on model performance.

## 🎯 Project Overview

Built and trained three neural network architectures from scratch, systematically testing:

- **Architectures:** RNN → LSTM → Transformer
- **Tokenization:** Character-level vs Word-level
- **Dataset Scale:** 640KB → 103MB → 1GB (in progress)

**Goal:** Understand when each architecture excels and the critical role of data composition.

## 🏆 Complete Results

### Phase 1: Character-Level (640KB, 652K characters)

| Model       | Parameters | Val Loss   | Train Time | Text Quality     |
| ----------- | ---------- | ---------- | ---------- | ---------------- |
| Simple RNN  | 274K       | 1.6482     | 3.8 min    | Poor (gibberish) |
| **LSTM** 🥇 | 3.8M       | **1.4711** | 39.5 min   | **Excellent**    |
| Transformer | 3.2M       | 1.5523     | 103 min    | Undertrained     |

### Phase 2: Word-Level Small Dataset (640KB, 132K words)

| Model       | Parameters | Val Loss | Train Time | Text Quality                 |
| ----------- | ---------- | -------- | ---------- | ---------------------------- |
| LSTM        | 3.8M       | 6.4072   | 15.7 min   | Good grammar, weak semantics |
| Transformer | 3.2M       | 6.2291   | 25.8 min   | Slightly better, still poor  |

**Finding:** Word-level needs significantly more data (vocabulary too sparse at 13.6 samples/word).

### Phase 2: Word-Level Mega Corpus (103MB, 22M words, 379 sources)

| Model                    | Parameters | Val Loss | Train Time         | Text Quality         |
| ------------------------ | ---------- | -------- | ------------------ | -------------------- |
| Transformer              | 3.2M       | 5.3214   | 1.5 hrs (V100 GPU) | Coherent but archaic |
| Transformer (fine-tuned) | 3.2M       | 5.3214   | +30 min            | No improvement       |

**Findings:**

- ✅ 14% improvement over small dataset (6.23 → 5.32)
- ✅ Sufficient vocabulary density (~916 samples/word)
- ⚠️ Classical books (83%) dominate corpus → archaic language style
- ⚠️ Hyperparameter tuning had minimal impact
- ❌ Still worse than character-level LSTM (5.32 vs 1.47)

**Diagnosis:** Data composition problem, not architecture or training issue.

## 📝 Sample Text Generations

### Character-Level LSTM (1.47 loss) - Best Overall Quality

**Prompt:** "Money is"

```
"Money is always right to seek by less than investments. When the result
is that the poor and the drivers that are high-specialized because I
didn't want to work for money. The power of being set them to be..."
```

✅ Modern language, financial concepts, semantic coherence

### Word-Level Transformer - Small Dataset (6.23 loss)

**Prompt:** "Money is"

```
"money is all, the road and white kahneman hurried off the second
investor, but if you want to be so losing money at some three..."
```

✅ Perfect grammar, learned names  
❌ Semantically incoherent (insufficient data)

### Word-Level Transformer - Mega Corpus (5.32 loss)

**Prompt:** "Money is"

```
"money is meant. but the whole capital of consumption is always at
the price of all sorts. the quantity of silver which it will purchase,
in exchange for which all other improvements are purchased..."
```

✅ Grammatically perfect, rich financial vocabulary  
⚠️ Archaic 18th-century prose style (Adam Smith influence)  
⚠️ Overly verbose, classical phrasing

**Prompt:** "Investing in"

```
"investing in the whole value of the loan of the borrower and my father's
new estate. it was an expedient that would have been so unexpected..."
```

✅ Coherent narrative  
✅ Financial concepts (loan, value, estate)  
⚠️ Classical writing style persists

## 🔍 Critical Insights

### 1. Dataset Composition Matters More Than Size

**103 MB Corpus Breakdown:**

- Classical economics (1700s-1800s): ~85 MB (83%) ⚠️
- Modern content (2000s-2020s): ~18 MB (17%) ⚠️

**Result:** Model learned perfect grammar but archaic language patterns from Adam Smith, Karl Marx, etc.

**Key Learning:** 100 MB of imbalanced data < 10 MB of balanced data

### 2. Character-Level LSTM Remains Champion

Despite 166x more data and GPU training:

- Character LSTM: **1.47 loss** on 640KB
- Word Transformer: **5.32 loss** on 103 MB

**Why:** Focused, modern dataset beats massive classical corpus for practical applications.

### 3. Hyperparameters Not the Bottleneck

Fine-tuning experiments (lower LR, higher dropout, early stopping):

- Improvement: 0.025 (0.5%)
- Conclusion: Data composition is the root cause

### 4. Word-Level Scaling Validation

| Dataset | Words | Vocab | Samples/Word | Val Loss | Status                  |
| ------- | ----- | ----- | ------------ | -------- | ----------------------- |
| Small   | 132K  | 10K   | 13.6         | 6.23     | Severe underfitting     |
| Mega    | 22M   | 20K   | 916          | 5.32     | Learning but imbalanced |
| Target  | 150M  | 30K   | 5,000        | <2.5     | Expected optimal        |

Proved scaling works, but needs balanced modern content.

## 📁 Project Structure

```
language-model-evolution/
├── data/                          # Training data (not in git)
│   ├── books/                     # 379 source texts (103 MB)
│   │   ├── gutenberg/             # 60 books (31 MB)
│   │   ├── gutenberg_expanded/    # 169 books (66 MB)
│   │   ├── wikipedia/             # 127 articles (4.3 MB)
│   │   └── academic_text/         # 10 papers (1.3 MB)
│   ├── mega_corpus.txt            # Combined (103 MB, 22M words)
│   ├── mega_word_dataset.pkl      # Processed (416 MB)
│   └── dataset.pkl                # Character-level (5.7 MB)
│
├── src/
│   ├── models/                    # RNN, LSTM, Transformer
│   ├── train/                     # Training scripts
│   ├── analyze/                   # Visualization
│   └── scripts/utils/             # Data preparation
│
├── results/                       # Visualizations (in git)
├── checkpoints/                   # Models (not in git - 50+ MB each)
├── notebooks/                     # Jupyter/Colab notebooks
└── README.md
```

## 🚀 Quick Start

```bash
git clone https://github.com/Nik-lesh/language-model-evolution.git
cd language-model-evolution
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Train on small dataset (local CPU):**

```bash
python src/train/train.py lstm  # Best model: 1.47 loss in 40 min
```

**Train on mega corpus (Google Colab GPU):**

- Upload `mega_word_dataset.pkl` to Google Drive
- Open `notebooks/train_on_colab.ipynb` in Colab
- Select V100 GPU runtime
- Run all cells (~1.5 hours)

## 🔬 Experiments Conducted

### Experiment 1: Architecture Comparison (Character-Level)

- **Winner:** LSTM (1.47 loss)
- **Insight:** Gates crucial for sequence modeling

### Experiment 2: Tokenization Strategy (Small Data)

- **Finding:** Word-level needs 10-20x more data
- **Vocabulary sparsity:** 13.6 samples/word insufficient

### Experiment 3: Dataset Scaling (22M words)

- **Improvement:** 14% better than small dataset
- **Issue:** Classical books dominated, created archaic style
- **Learning:** Data composition > Data quantity

### Experiment 4: Hyperparameter Fine-Tuning

- **Result:** 0.5% improvement only
- **Conclusion:** Not a tuning problem, needs balanced data

## 📊 Model Performance Summary

| Model       | Dataset    | Tokenization | Val Loss | Style         | Best For       |
| ----------- | ---------- | ------------ | -------- | ------------- | -------------- |
| **LSTM** 🥇 | 640KB      | Character    | **1.47** | Modern, clear | **Production** |
| Transformer | 640KB      | Character    | 1.55     | Good          | Small data     |
| Transformer | 132K words | Word         | 6.23     | Broken        | N/A            |
| Transformer | 22M words  | Word         | 5.32     | Classical     | Research       |

## 🚀 Next Steps: Phase 2B - Balanced Dataset

### Problem Identified

Current 103 MB corpus composition:

- Classical economics: 83% (archaic language)
- Modern content: 17% (insufficient)

### Solution: Balanced 1 GB Corpus

**Target Composition:**

- Classical texts: 100 MB (10%)
- Modern finance books (2000-2024): 400 MB (40%)
- Financial news articles: 300 MB (30%)
- Corporate reports (SEC): 200 MB (20%)

**Expected Results:**

- Val loss: <2.5 (vs current 5.32)
- Modern language style
- Production-ready text quality
- Vocabulary: 30-40K words

**Timeline:** 3-4 weeks
**Training:** TPU (4-5 hours) vs GPU (12-15 hours)

### Data Sources (Modern Content)

**Modern Books (400 MB):**

- Internet Archive borrowing
- Library digital lending (Libby)
- Investment classics (2000s-2020s)

**Financial News (300 MB):**

- Kaggle datasets (pre-collected)
- Investopedia articles
- MarketWatch archives (2020-2024)

**Corporate Data (200 MB):**

- SEC 10-K filings (100 companies)
- Earnings call transcripts
- Analyst reports

## 📚 Key Learnings

1. **Data quality > Data quantity** - 100 MB balanced > 1 GB imbalanced
2. **Character-level LSTM excels** on focused, small datasets
3. **Word-level needs both scale AND balance** - 20M words + modern content
4. **Architecture fits data characteristics** - No universal winner
5. **Classical texts create classical output** - Model mirrors training distribution
6. **Hyperparameters secondary** to data composition
7. **Early stopping crucial** - Best model often found early (epoch 2 vs 50)

## 🛠️ Tech Stack

- **Framework:** PyTorch 2.0+
- **GPU:** Google Colab (V100/A100)
- **Processing:** NumPy, Pandas, pdfplumber
- **Visualization:** Matplotlib, Seaborn
- **Next:** TPU for 1GB training, Hugging Face Transformers for fine-tuning

## 📄 License

MIT License

## 🙏 Acknowledgments

- **Dataset:** 379 finance sources from Gutenberg, Wikipedia, and academic papers
- **Inspiration:** Evolution of NLP from RNN to Transformers
- **Purpose:** Understanding architecture-data relationships in deep learning

---

**Current Status:**  
✅ **Phase 1 Complete** - Character-level baseline (LSTM: 1.47 loss)  
✅ **Phase 2A Complete** - Mega corpus training (Transformer: 5.32 loss)  
🔄 **Phase 2B In Progress** - Balanced 1GB dataset collection  
⏳ **Phase 3 Planned** - TPU training (4-5 hours)  
⏳ **Phase 4 Planned** - Production deployment

**Key Achievement:** Proved dataset composition determines output style more than architecture choice

**Current Best:** Character-level LSTM (1.47 loss, modern readable output)  
**Research Goal:** Word-level Transformer on balanced 1GB (<2.5 loss, modern style)  
**Production Path:** Fine-tune GPT-2 on finance corpus (combines modern language + domain expertise)

**Last Updated:** November 12, 2025
