## Unit II — Word Embeddings and Vector Representations

This unit is the bridge between **classical sparse NLP** and **modern deep NLP**.
In Unit I, words were represented mostly as **counts** or **weighted counts**. That works, but it fails to capture meaning. In this unit, the central goal is to learn **dense vector representations** where words with similar meaning end up near each other in vector space.

The key idea is:

> **A word is known by the company it keeps.**

If two words appear in similar contexts, they probably share related meaning. Word embeddings turn that linguistic idea into a learnable mathematical representation.

---

# 1) Vector Space Models

## 1.1 Core intuition

A **vector space model** represents words, documents, or sentences as vectors in a high-dimensional space.

The foundational assumption is that **meaning can be approximated by geometry**:

* similar words → nearby vectors
* dissimilar words → far apart vectors

This is a huge shift from symbolic representation. Instead of saying:

* `king` is just token #4312

we say:

* `king = [0.21, -0.84, 1.07, ...]`

Now the model can reason using distances, angles, and linear algebra.

---

## 1.2 Sparse vs dense representations

### Sparse vectors

Classical models like BoW and TF-IDF create vectors with:

* dimension = vocabulary size
* most values = 0

Example:
If vocabulary size is 50,000, one word may be represented by a 50,000-dimensional one-hot vector.

This is:

* interpretable
* simple
* sparse
* memory-heavy
* semantically weak

### Dense vectors

Embeddings compress meaning into a much smaller space, such as:

* 50 dimensions
* 100 dimensions
* 300 dimensions
* 768 dimensions and beyond in modern transformers

Dense vectors:

* are compact
* capture semantic similarity
* are learned from data
* are much better for generalization

---

## 1.3 One-hot vectors and why they are weak

A one-hot vector for the word “cat” might look like:

[
[0,0,1,0,0,\dots,0]
]

Every word is equally far from every other word in one-hot space.
That means:

* “cat” and “dog” are as different as “cat” and “airplane”
* no notion of similarity exists
* no shared structure exists

This is the main reason vector space models evolved beyond one-hot encoding.

---

## 1.4 Distributional hypothesis

The distributional hypothesis is the linguistic foundation of embeddings.

### Statement

**Words that occur in similar contexts tend to have similar meanings.**

Examples:

* “doctor” and “physician” appear in similar contexts
* “big” and “large” occur in similar contexts
* “cat” and “dog” often share neighbors like “pet”, “furry”, “animal”

This hypothesis allows machine learning to infer meaning from usage statistics.

### Why this matters

You do not need an explicit dictionary definition to learn semantics.
The surrounding text provides weak but very rich supervision.

---

## 1.5 Geometric interpretation

In vector space models:

* **direction** can encode semantic relations
* **distance** can encode similarity
* **angles** can encode relational closeness

Common similarity measure:

### Cosine similarity

[
\cos(\theta) = \frac{v \cdot w}{|v||w|}
]

Where:

* (v \cdot w) is the dot product
* (|v|) and (|w|) are vector norms

Cosine similarity is preferred over Euclidean distance in many NLP tasks because it focuses on **orientation**, not magnitude.

---

# 2) Dense Word Embeddings

## 2.1 What is a dense word embedding?

A dense word embedding is a learned low-dimensional vector that represents a word’s meaning and usage.

Example:

* `king → [0.12, -0.44, 0.98, ...]`
* `queen → [0.15, -0.40, 0.95, ...]`

These vectors are learned so that words with similar contexts have similar vectors.

---

## 2.2 Why dense embeddings work better than sparse vectors

Dense embeddings solve three big problems:

### A. Sparsity

Instead of a 50,000-dimensional sparse vector, a word may be represented in 100 dimensions.

### B. Generalization

Similar words share nearby regions in space, so knowledge transfers.

### C. Semantic structure

The vector space itself becomes meaningful:

* gender directions
* tense-like directions
* plural/singular patterns
* similarity neighborhoods

---

## 2.3 What embeddings really store

An embedding is not a dictionary definition.
It stores **usage patterns**.

For example:

* “bank” near “loan”, “money”, “credit” gets one region
* “bank” near “river”, “flood”, “shore” gets another region

This means a single word embedding may still struggle with polysemy, which is one reason contextual embeddings later became important.

---

## 2.4 Word embedding matrix

Suppose:

* vocabulary size = (V)
* embedding dimension = (d)

Then the embedding matrix is:

[
E \in \mathbb{R}^{V \times d}
]

Each row corresponds to one word vector.

If word (w_i) has index (i), its embedding is:

[
e_i = E[i]
]

During training, the model learns the values in (E).

---

## 2.5 Engineering reality

Embeddings are useful because they:

* reduce dimensionality
* improve performance on downstream tasks
* encode semantic relations

But they also have limitations:

* one vector per word cannot fully represent multiple senses
* rare words may be poorly learned
* domain mismatch can degrade quality
* static embeddings cannot adapt to context automatically

---

# 3) Word2Vec

Word2Vec is one of the most influential methods in NLP.
It learns dense embeddings by predicting words from context or context from words.

It comes in two main architectures:

* **CBOW**: Continuous Bag of Words
* **Skip-Gram**

Both are shallow neural networks, but they learn very powerful representations.

---

## 3.1 Core intuition behind Word2Vec

Instead of directly labeling words, Word2Vec uses **self-supervision** from text itself.

Example sentence:

> “The cat sat on the mat”

From this, we can create training pairs:

* context → target
* target → context

This is powerful because large text corpora are abundant, so the model can learn from enormous unlabeled data.

---

## 3.2 Basic structure of Word2Vec

Word2Vec typically uses:

* input layer
* embedding/projection layer
* output layer

The hidden layer is the embedding we care about.

### Key idea

The network is trained to do a prediction task, but the learned hidden weights become the word vectors.

---

# 4) CBOW (Continuous Bag of Words)

## 4.1 Core intuition

CBOW predicts the **center word** from its surrounding context words.

Example sentence:

> “The cat sat on the mat”

If the target word is “sat”, the context may be:

* “the”, “cat”, “on”, “the”

CBOW uses these surrounding words to predict the missing center word.

### Why it is called “bag of words”

CBOW typically ignores word order in the context window.
It only cares which words are present, not their exact sequence.

---

## 4.2 Architecture

Assume:

* vocabulary size = (V)
* embedding dimension = (d)
* context window size = (m)

Each context word is represented as one-hot, then mapped into embeddings.

### Step 1: Input

Context words:
[
w_{t-2}, w_{t-1}, w_{t+1}, w_{t+2}
]

### Step 2: Lookup embeddings

Each context word index selects a row from the embedding matrix (E).

### Step 3: Combine context vectors

Typically, the embeddings are averaged or summed:

[
h = \frac{1}{C}\sum_{i=1}^{C} e_i
]

where (C) is the number of context words.

### Step 4: Predict target word

The hidden representation (h) is passed to the output layer to produce a probability over vocabulary:

[
p(w_t \mid context) = \frac{\exp(u_{w_t}^\top h)}{\sum_{j=1}^{V} \exp(u_j^\top h)}
]

Here:

* (u_j) is the output vector for word (j)

This is a softmax classifier over the vocabulary.

---

## 4.3 Training objective

CBOW maximizes the probability of the center word given the context.

For a corpus:
[
\max \sum_{t} \log p(w_t \mid w_{t-m}, \dots, w_{t-1}, w_{t+1}, \dots, w_{t+m})
]

Equivalent to minimizing negative log-likelihood.

---

## 4.4 Why CBOW works well

### Strengths

* fast to train
* stable
* works well on frequent words
* efficient with large corpora

### Weaknesses

* tends to smooth over rare words
* context averaging loses order information
* less effective for learning subtle distinctions than Skip-Gram in some cases

---

# 5) Skip-Gram Model

## 5.1 Core intuition

Skip-Gram does the reverse of CBOW.

Given a center word, it predicts the surrounding context words.

Example:
Center word: “sat”
Context words: “the”, “cat”, “on”, “the”

This is especially useful because one word can generate multiple training pairs, making the representation learn rich contextual information.

---

## 5.2 Architecture

### Step 1: Input

One center word (w_t)

### Step 2: Embedding lookup

Map center word to vector (h)

### Step 3: Predict each context word

For each surrounding word (w_{t+j}), predict:

[
p(w_{t+j} \mid w_t) = \frac{\exp(u_{w_{t+j}}^\top h)}{\sum_{k=1}^{V} \exp(u_k^\top h)}
]

The total objective across the context window is:

[
\max \sum_{t}\sum_{j=-m, j \neq 0}^{m} \log p(w_{t+j} \mid w_t)
]

---

## 5.3 Why Skip-Gram is useful

### Strengths

* often better for rare words
* learns good semantic relations
* produces high-quality embeddings

### Weaknesses

* slower than CBOW
* more training pairs per token
* full softmax is expensive for large vocabularies

---

# 6) Softmax Bottleneck and Efficiency Tricks

## 6.1 Why naive Word2Vec softmax is expensive

The denominator of the softmax sums over the entire vocabulary:

[
\sum_{j=1}^{V} \exp(u_j^\top h)
]

If (V) is huge, this is expensive.

### Common solutions

* hierarchical softmax
* negative sampling
* subsampling frequent words

These make training practical on large corpora.

---

## 6.2 Negative sampling

Instead of predicting the full distribution over all words, the model learns to distinguish:

* true context word pairs
* random noise pairs

For a positive pair ((w, c)), maximize:
[
\log \sigma(v_w^\top v_c)
]

For negative samples (n_1, n_2, \dots, n_k), maximize:
[
\sum_{i=1}^{k} \log \sigma(-v_{n_i}^\top v_c)
]

where:

* (\sigma) is the sigmoid function

This turns multi-class prediction into several binary classification tasks.

### Why this matters

Negative sampling is one reason Word2Vec became scalable and practical.

---

## 6.3 Subsampling frequent words

Very frequent words like:

* the
* of
* and
* is

carry less semantic information.

Word2Vec often down-samples them so training focuses more on informative words.

This improves:

* efficiency
* quality
* training balance

---

# 7) GloVe Embeddings

GloVe stands for **Global Vectors**.

Unlike Word2Vec, which is a predictive model, GloVe is based on **global co-occurrence statistics**.

---

## 7.1 Core intuition

Word meaning can be captured by how often words co-occur across the entire corpus.

If:

* “ice” co-occurs with “cold”
* “king” co-occurs with “royal”
* “doctor” co-occurs with “hospital”

then those co-occurrence patterns encode semantics.

GloVe tries to build embeddings such that dot products reflect co-occurrence likelihoods.

---

## 7.2 Co-occurrence matrix

Let (X_{ij}) be the number of times word (j) appears in the context of word (i).

This creates a large co-occurrence matrix:
[
X \in \mathbb{R}^{V \times V}
]

This matrix is huge, but sparse.

---

## 7.3 GloVe objective

GloVe learns vectors (w_i) and (w_j), plus biases, such that:

[
w_i^\top \tilde{w}_j + b_i + \tilde{b}*j \approx \log X*{ij}
]

The training objective is:

[
J = \sum_{i,j} f(X_{ij})\left(w_i^\top \tilde{w}_j + b_i + \tilde{b}*j - \log X*{ij}\right)^2
]

where:

* (f(X_{ij})) is a weighting function that reduces the influence of extremely frequent or rare co-occurrences

### Why log counts?

Logarithms reduce the impact of very large counts and often linearize multiplicative relationships.

---

## 7.4 How GloVe differs from Word2Vec

### Word2Vec

* predictive
* local context-based
* trained on window prediction tasks

### GloVe

* count-based with global co-occurrence
* matrix factorization flavored
* uses corpus-wide statistics

Both produce dense embeddings and often perform similarly well, but they arise from different principles.

---

# 8) Capturing Semantic Similarity

## 8.1 What semantic similarity means

Semantic similarity means two words are related in meaning.

Examples:

* car / automobile
* doctor / physician
* happy / joyful

Embedding spaces aim to place such words close together.

---

## 8.2 Similarity metrics

### Cosine similarity

Most common:
[
\cos(\theta) = \frac{v \cdot w}{|v||w|}
]

### Euclidean distance

[
|v-w|_2
]

Cosine similarity is often preferred in NLP because vector direction matters more than absolute size.

---

## 8.3 Why embeddings capture similarity

The training objective forces words appearing in similar contexts to get similar vectors.

For example, if:

* “dog” and “cat” appear near “pet”, “owner”, “food”
* their vectors must become similar to help prediction

Thus similarity is not manually encoded; it emerges from the training objective.

---

## 8.4 Distributional similarity vs true meaning

A crucial limitation:

> Words with similar contexts are not always true synonyms.

Example:

* “doctor” and “hospital” may be close
* but they are not synonyms

So embeddings capture **distributional similarity**, which is related to semantics but not identical to it.

---

# 9) Analogy Relationships

One of the most famous properties of word embeddings is analogy solving through vector arithmetic.

## 9.1 Example

[
\text{king} - \text{man} + \text{woman} \approx \text{queen}
]

This suggests that embeddings encode some relations as approximately linear directions.

---

## 9.2 Why this happens

If a relation is consistently represented across many examples, the vector space may organize that relation as a direction.

For example:

* male ↔ female
* singular ↔ plural
* present ↔ past
* country ↔ capital

These may form geometric patterns.

---

## 9.3 Algebraic analogy task

Given:

* (a) is to (b) as (c) is to ?

We compute:
[
v_b - v_a + v_c
]

Then search for the nearest word vector to that result, often by cosine similarity.

Example:
[
v_{\text{queen}} \approx v_{\text{king}} - v_{\text{man}} + v_{\text{woman}}
]

---

## 9.4 Limitations of analogy performance

Not all relations are linear.
Not all embeddings preserve the same relational geometry.
Analogy success depends on:

* training data
* embedding quality
* frequency
* subword structure
* polysemy

So analogy is an interesting diagnostic, not proof of full understanding.

---

# 10) Visualization of Embedding Spaces

Embedding spaces are high-dimensional, so we often use dimensionality reduction for visualization.

---

## 10.1 Why visualize embeddings?

Visualization helps us inspect:

* clusters of related words
* semantic neighborhoods
* anomalies
* analogical structure
* training quality

It is a debugging and interpretation tool.

---

## 10.2 PCA (Principal Component Analysis)

PCA is a linear dimensionality reduction method.

### Core idea

Find directions of maximum variance in the data.

Given embeddings in high-dimensional space, PCA projects them onto the top principal components.

If embeddings are (X), PCA finds directions (u_1, u_2) such that:

* (u_1) captures the most variance
* (u_2) captures the next most variance
* and so on

### Why PCA is useful

* fast
* stable
* interpretable
* preserves global variance structure reasonably well

### Limitation

PCA is linear, so it may not reveal complex nonlinear neighborhood structures.

---

## 10.3 t-SNE (t-distributed Stochastic Neighbor Embedding)

t-SNE is a nonlinear visualization method designed to preserve local neighborhoods.

### Core intuition

Points that are close in high-dimensional space should remain close in 2D/3D.

t-SNE focuses on:

* local cluster structure
* neighborhood preservation

### Why it is popular

It often produces visually beautiful clusters for word embeddings.

### Limitation

* not ideal for global distance interpretation
* can distort large-scale geometry
* sensitive to hyperparameters
* mainly used for visualization, not downstream modeling

---

## 10.4 PCA vs t-SNE

### PCA

* linear
* preserves global variance
* faster
* more interpretable

### t-SNE

* nonlinear
* preserves local neighborhoods
* visually cluster-friendly
* slower and less globally faithful

Use PCA when you want a simple linear projection.
Use t-SNE when you want to inspect clustering structure.

---

## 10.5 Engineering caution

A 2D t-SNE plot may look impressive, but do not over-interpret it.
Cluster separation in t-SNE does not always mean true semantic separation in the original space.

---

# 11) Practical Use Cases of Word Embeddings

## 11.1 Text classification

Word embeddings improve sentiment, topic, and intent models by giving input words meaningful vectors.

## 11.2 Information retrieval

Semantic matching between queries and documents is better than exact word match.

## 11.3 Machine translation

Better source-language representations improve alignment and translation quality.

## 11.4 Named entity tasks

Embeddings help models recognize patterns like:

* person names
* locations
* organizations

## 11.5 Similarity search

Embeddings support semantic search, nearest-neighbor retrieval, and clustering.

---

# 12) Strengths and Limitations of Static Embeddings

## Strengths

* compact
* semantically rich
* learnable from raw text
* useful across many tasks
* support transfer learning

## Limitations

### 1. Static representation

One word has one vector, even if it has multiple meanings.

Example:

* “bank” gets one embedding, but “river bank” and “financial bank” are different senses.

### 2. Context ignorance

Embeddings do not change with sentence context.

### 3. Rare word weakness

Very rare words may not get good vectors.

### 4. Domain mismatch

Vectors trained on news may not work well for medical or legal text.

### 5. Subtle meaning gaps

Embeddings capture similarity, but not full reasoning or truth conditions.

---

# 13) How This Unit Connects to Later Units

This unit is the foundation for sequence models and transformers.

Word embeddings provide:

* dense input representations for RNNs, LSTMs, GRUs
* semantically meaningful token vectors
* a transition away from sparse classical features

Later units will extend this idea by making representations:

* contextual instead of static
* sequence-aware instead of isolated
* learned end-to-end inside deep architectures

---

# 14) Quick Concept Map

* **Vector space models**: represent language geometrically
* **Dense embeddings**: compact learned word vectors
* **Word2Vec**: prediction-based embedding learning
* **CBOW**: predict center word from context
* **Skip-Gram**: predict context from center word
* **GloVe**: learn from global co-occurrence statistics
* **Semantic similarity**: nearby vectors mean related meaning
* **Analogies**: linear relationships in embedding space
* **PCA / t-SNE**: visualize high-dimensional embeddings

---

# 15) End-of-Unit NLP Master Challenge

## Problem 1

A Word2Vec model trained on a large news corpus places “doctor” near “hospital” and “nurse” near “clinic.”
Explain why this happens mathematically and linguistically.
Then explain why this does not mean the model truly “understands” medicine.

## Problem 2

Given a sentence corpus, CBOW learns better representations for frequent function words, while Skip-Gram often performs better on rare words.
Explain the training dynamics behind this difference.

## Problem 3

A student says: “If `king - man + woman = queen`, then embeddings have solved language understanding.”
Critically analyze this claim.
What does analogy success show, and what does it not show?

## Problem 4

You are tasked with visualizing 300-dimensional embeddings of 2,000 words.
When would you use PCA, and when would you use t-SNE?
Explain what each method preserves and what it may distort.

---

When you are ready, I can teach **Unit III** in the same deep-dive style.
