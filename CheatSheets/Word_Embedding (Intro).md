# 🧠 Word Embeddings

Word Embeddings are:

> A way to represent words as dense numerical vectors that capture semantic meaning.

---

# 🧠 The Core Idea

Earlier in NLP, we used:

* Bag of Words
* TF-IDF

These create:

# ❌ Sparse vectors

Example:

```text id="e1"
Vocabulary:
["cat", "dog", "apple", "car", "music"]

"dog" → [0, 1, 0, 0, 0]
```

Problems:

* mostly zeros
* no meaning
* no similarity understanding

---

# 🚨 Huge Problem with TF-IDF / BoW

To TF-IDF:

```text id="e2"
king
queen
apple
car
```

are just independent words.

It does NOT know:

* king and queen are related
* car and vehicle are similar
* good and excellent are close in meaning

---

# 🚀 Word Embeddings Solve This

Instead of sparse vectors:

```text id="e3"
dog → [0,0,1,0,0]
```

we learn:

# ✅ Dense vectors

Example:

```text id="e4"
dog → [0.21, -0.88, 0.44, 0.91]
cat → [0.19, -0.82, 0.40, 0.89]
car → [-0.71, 0.33, -0.91, 0.11]
```

---

# 🧠 KEY IDEA

Similar words get similar vectors.

So:

```text id="e5"
dog ≈ cat
king ≈ queen
car ≈ vehicle
```

because their vectors become close in vector space.

---

# 🧠 What Does “Embedding” Mean?

Embedding means:

> embedding something into a continuous mathematical space.

In simple words:

```text id="e6"
word → coordinates in high-dimensional space
```

Like:

```text id="e7"
king → [1.2, -0.4, 0.9, ...]
queen → [1.1, -0.3, 0.8, ...]
```

---

# 🧠 Why Dense Vectors Are Powerful

Dense vectors can capture:

* meaning
* similarity
* context
* relationships

Example:

```text id="e8"
Paris - France + Italy ≈ Rome
```

This actually works in good embeddings.

---

# 🚀 Dense vs Sparse

| Sparse (TF-IDF) | Dense (Embeddings)        |
| --------------- | ------------------------- |
| mostly zeros    | mostly meaningful numbers |
| huge vectors    | compact vectors           |
| no meaning      | semantic meaning          |
| counts words    | learns relationships      |

---

# 🧠 How Are Embeddings Learned?

Models learn embeddings by:

> observing which words appear near each other in huge text corpora.

Example:

```text id="e9"
The dog barked loudly
The cat meowed softly
```

Model notices:

* dog/cat appear in similar contexts
* both are animals
* both appear near verbs

So vectors become similar.

---

# 🚀 Famous Embedding Models

## 🟢 Word2Vec

Google’s famous embedding model.

Two training methods:

* CBOW
* Skip-Gram

---

## 🟢 GloVe

Uses global word statistics.

---

## 🟢 FastText

Handles subwords/character information.

---

## 🟢 Modern Transformers

BERT, GPT also learn contextual embeddings.

---

# 🧠 Static vs Contextual Embeddings

## 🟢 Static Embeddings

Same vector every time.

Example:

```text id="e10"
bank → same vector always
```

Problem:

* river bank
* money bank

same vector ❌

---

## 🟢 Contextual Embeddings (BERT/GPT)

Vector changes depending on sentence.

```text id="e11"
river bank
bank account
```

Different meanings → different vectors ✔

---

# 🧠 Typical Embedding Size

Usually:

```text id="e12"
50
100
200
300
768
1024
4096
```

dimensions.

Example:

```text id="e13"
dog → 300-dimensional vector
```

---

# 🧠 Visualization Intuition

Imagine:

* similar words cluster together
* unrelated words stay far apart

Like:

```text id="e14"
animals cluster together
countries cluster together
sports cluster together
```

inside mathematical space.

---

# 🚀 Why Embeddings Changed NLP Forever

Before embeddings:

* NLP was mostly counting words

After embeddings:

* NLP started understanding semantic similarity

This led to:

* machine translation
* chatbots
* semantic search
* transformers
* modern LLMs

including models like ChatGPT.

---

# 🧠 Final Intuition

TF-IDF says:

> “How important is this word?”

Embeddings say:

> “What does this word MEAN?”
