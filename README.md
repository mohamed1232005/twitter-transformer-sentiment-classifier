# Twitter Transformer Sentiment Classifier (Experimental)

## 📌 Project Overview

This repository contains an **experimental implementation of a Transformer-based sentiment classifier** applied to the **Twitter Sentiment Analysis dataset**.  
The objective of this project was to build a **Transformer architecture from scratch** (without relying on pre-trained HuggingFace models) and train it directly on the provided dataset.  

⚠️ **Disclaimer**:  
This project was conducted as an **experiment**.  
The model works, but the results were **not perfect**, largely due to the **limited dataset**, simplified preprocessing, and training constraints.  
The repository serves more as a **learning project** for implementing a Transformer pipeline end-to-end rather than a production-ready sentiment classifier.

---

---

## 📊 Dataset

The dataset used is a **Twitter sentiment dataset**, split into **training** and **testing** files.

### Columns
- **id** → unique identifier of the sample  
- **topic** → the subject or category of the tweet  
- **label** → the sentiment label (`Positive`, `Negative`, `Neutral`, or `Irrelevant`)  
- **text** → the raw tweet text  

### Example

| id   | topic   | label      | text                                |
|------|---------|------------|-------------------------------------|
| 1234 | Amazon  | Positive   | "I love the fast delivery service!" |
| 5678 | Tesla   | Negative   | "Worst customer service ever..."    |
| 9101 | Google  | Neutral    | "Google announced a new feature."   |

---

## 🔎 Workflow & Approach

Unlike traditional approaches that use **pre-trained embeddings** (like BERT or GPT models), here we **designed and trained a Transformer from scratch** on Twitter sentiment data.

The workflow followed these steps:

1. **Data Exploration (EDA)**  
   - Checked dataset size, duplicates, and missing values.  
   - Analyzed label distribution (imbalanced dataset).  
   - Tokenized tweet lengths and inspected samples with extreme lengths.  

2. **Data Preprocessing**  
   - Normalized text (lowercasing, removing URLs, mentions, hashtags).  
   - Removed stopwords, digits, and emojis.  
   - Tokenized sentences and applied **vectorization** using a `TextVectorization` layer.  

3. **Model Design: Custom Transformer**  
   - Implemented **Positional Encoding** and **Multi-Head Self-Attention**.  
   - Built a **stack of Transformer encoder blocks** with residual connections.  
   - Added **GlobalAveragePooling** and a **Dense softmax layer** for classification.  

   ### Transformer Block (Simplified Equation):

   - **Scaled Dot-Product Attention**:  
     \[
     \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
     \]

   - **Feed-Forward Network**:  
     \[
     FFN(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2
     \]

   - **Final Encoder Layer**:  
     \[
     \text{LayerNorm}(x + \text{Dropout}(FFN(\text{Attention}(x))))
     \]

4. **Training Setup**  
   - Optimizer: `Adam` with learning rate scheduling.  
   - Loss: `SparseCategoricalCrossentropy`.  
   - Metrics: `Accuracy` and `Macro-F1` (via `tensorflow-addons`).  
   - Training: 10 epochs, 128 batch size, early stopping applied.  

5. **Evaluation & Testing**  
   - Saved model (`.keras`) and vectorizer (`.pkl`) artifacts.  
   - Loaded the trained model in a separate notebook for predictions.  

---

## 📈 Experimental Results

The experimental Transformer achieved **moderate performance** on the training data but struggled with **generalization** due to dataset and preprocessing constraints.

### Example Metrics (Train/Validation)

| Metric        | Training | Validation |
|---------------|----------|------------|
| Accuracy      | ~72%     | ~65%       |
| Macro-F1      | ~0.68    | ~0.61      |
| Loss          | ↓        | Plateaued  |

---

## ⚠️ Limitations

This project is **experimental** and has several limitations:

1. **Data Limitations**  
   - Small dataset size compared to standard transformer needs.  
   - Imbalanced label distribution (e.g., fewer `Irrelevant` samples).  

2. **Model Complexity**  
   - Custom Transformer lacked the scale of pre-trained models (BERT, RoBERTa).  
   - Few encoder layers → underfitting on complex language patterns.  

3. **Preprocessing Gaps**  
   - Limited handling of slang, sarcasm, and emoji-heavy tweets.  
   - No pretrained embeddings (GloVe/Word2Vec/BERT).  

4. **Computational Resources**  
   - Training was constrained to a limited environment, restricting hyperparameter tuning.  

---

## 🚀 Future Enhancements

To improve results, future work could focus on:

1. **Better Data**  
   - Collect a larger, more diverse Twitter dataset.  
   - Apply **data augmentation** (e.g., back-translation, synonym replacement).  

2. **Advanced Preprocessing**  
   - Use pretrained embeddings for initialization.  
   - Better emoji/sarcasm handling with context-aware preprocessing.  

3. **Model Improvements**  
   - Increase transformer depth (more encoder blocks).  
   - Use **transformer-decoder** or hybrid architectures.  
   - Transfer learning from HuggingFace models (BERT, DistilBERT).  

4. **Training Enhancements**  
   - Hyperparameter optimization (learning rate, dropout).  
   - Longer training with learning rate scheduling.  
   - Mixed-precision training for efficiency.  

---

## 🛠️ Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/twitter-transformer-sentiment-classifier.git
cd twitter-transformer-sentiment-classifier
```
## 📌 Conclusion

This repository demonstrates a Transformer sentiment classifier from scratch on Twitter data.
While the results were not state-of-the-art, the project highlights the end-to-end pipeline of building, training, and evaluating a custom Transformer.

It should be viewed as a learning experiment rather than a production-ready solution.
With better data, preprocessing, and model design, this approach can be extended and improved significantly in future work.
