## Unit V — Transformers and Pretrained Language Models

This unit is the centerpiece of modern NLP.

If RNNs and LSTMs model sequences by carrying state forward through time, **Transformers** replace recurrence with **attention over the entire sequence at once**. That shift changes everything:

* training becomes parallelizable,
* long-range dependencies become easier to model,
* representation learning becomes much richer,
* pretrained language models become practical at scale.

The unit has two tightly connected parts:

1. **Transformers**

   * transformer architecture
   * self-attention
   * multi-head attention
   * positional encoding
   * transformer encoder and decoder blocks
   * tokenization methods
   * Byte-Pair Encoding and WordPiece

2. **Pretrained Language Models**

   * pretrained transformer models
   * BERT, GPT, T5
   * masked language modeling
   * next sentence prediction
   * causal language modeling
   * transfer learning for NLP tasks
   * fine-tuning for text classification, named entity recognition, question answering
   * HuggingFace Transformers

---

# Part I — Transformers

## 1) Transformer Architecture

### Core intuition & linguistic context

The Transformer was built to solve a key weakness of recurrent models:

> How can a model capture relationships between any two words in a sentence without processing tokens one by one?

In language, dependency is often non-local.

Examples:

* Subject-verb agreement across many words
* Coreference between distant mentions
* Translation correspondences between far-apart words
* Long-range discourse cues in summarization

A Transformer lets each token directly interact with every other token through attention, so it does not need to pass information through many recurrent steps.

This makes it especially powerful for:

* translation,
* summarization,
* classification,
* question answering,
* generation,
* multilingual understanding.

---

### Architecture, math & mechanics

The Transformer is built from two main building blocks:

* **encoder stack**
* **decoder stack**

Each stack is made of repeated layers.

A typical Transformer layer contains:

* self-attention
* feed-forward network
* residual connections
* layer normalization

The original Transformer uses:

* **encoder-only** layers for input encoding
* **decoder-only** layers for autoregressive generation
* **encoder-decoder** architecture for seq2seq tasks

---

### Big picture data flow

Suppose the input sequence has length (T).

1. Convert tokens to embeddings.
2. Add positional information.
3. Pass through attention and feed-forward sublayers.
4. Produce contextualized token representations.
5. Optionally decode outputs token by token, using masked attention in the decoder.

---

### Engineering reality & pitfalls

Transformers are powerful, but they are not magic:

* they are data-hungry,
* memory intensive,
* expensive on long sequences,
* sensitive to tokenization,
* prone to hallucination in generation settings.

Their success comes from the combination of:

* global token interaction,
* large-scale pretraining,
* scalable optimization,
* subword tokenization.

---

# 2) Self-Attention

## Core intuition & linguistic context

Self-attention is the heart of the Transformer.

For each token, the model asks:

> Which other tokens in the same sequence should I look at to understand this token?

That is essential in language because meaning is contextual.

Examples:

* “bank” depends on nearby words
* “it” depends on antecedents
* “not” can invert nearby sentiment
* verbs need subjects and objects
* noun phrases often depend on adjectives, determiners, and modifiers

Self-attention allows a token to build a context-aware representation by attending to all other tokens.

---

## Architecture, math & mechanics

Each input token embedding is projected into three vectors:

* **Query** (Q)
* **Key** (K)
* **Value** (V)

For an input matrix (X \in \mathbb{R}^{T \times d मॉडल}), we compute:

[
Q = XW^Q,\quad K = XW^K,\quad V = XW^V
]

where:

* (W^Q, W^K, W^V) are learned projection matrices
* each has shape ((d_{model}, d_k)), ((d_{model}, d_k)), ((d_{model}, d_v))

### Attention score

The similarity between a query and a key is:

[
\text{score}(q_i, k_j) = q_i^\top k_j
]

Scaled dot-product attention uses:

[
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
]

### Why the scaling factor?

Without division by (\sqrt{d_k}), dot products can become too large as dimensionality increases, causing softmax to saturate and gradients to become small or unstable.

---

### Interpreting the mechanics

For token (i):

1. compute similarity to every token (j),
2. normalize those scores into attention weights,
3. take a weighted sum of all value vectors.

So token (i) gets a new representation:

[
\mathbf{z}*i = \sum*{j=1}^{T} \alpha_{ij}\mathbf{v}_j
]

where:

[
\alpha_{ij} = \frac{\exp(q_i^\top k_j/\sqrt{d_k})}{\sum_{m=1}^{T}\exp(q_i^\top k_m/\sqrt{d_k})}
]

This produces a context-dependent token vector.

---

### Why self-attention is so powerful

Unlike recurrence:

* every token can directly access every other token,
* long-distance dependencies do not require many steps,
* computation can be parallelized across tokens.

Self-attention is also content-based:

* tokens attend based on similarity, not fixed position.

That means the model can learn patterns like:

* subject ↔ verb agreement
* pronoun ↔ antecedent
* modifier ↔ head noun
* question word ↔ answer span

---

### Engineering reality & pitfalls

Self-attention has a cost:

* time and memory scale as (O(T^2)) in sequence length because each token attends to all others.

That is manageable for short sequences but expensive for long documents.

Another pitfall:

* attention is learned from data and may not always correspond to human-meaningful explanations.

---

# 3) Multi-Head Attention

## Core intuition & linguistic context

A single attention mechanism may learn only one type of relationship at a time. But language contains many simultaneous relations:

* syntax,
* coreference,
* semantic similarity,
* discourse relation,
* positional pattern,
* local phrase structure.

Multi-head attention lets the model attend to the sequence in multiple subspaces simultaneously.

Think of it as several “attention experts” looking at the same sentence for different patterns.

---

## Architecture, math & mechanics

Instead of one attention computation, we use (h) heads.

For head (r):

[
\text{head}_r = \text{Attention}(QW_r^Q, KW_r^K, VW_r^V)
]

Then concatenate all heads:

[
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,\dots,\text{head}_h)W^O
]

where (W^O) is a learned output projection.

### Why multiple heads help

Each head can specialize:

* one head might track local syntactic relations,
* another long-distance dependencies,
* another positional patterns,
* another semantic similarity.

This gives the model a richer representational capacity than one monolithic attention map.

---

## Engineering reality & pitfalls

Advantages:

* richer representational diversity
* better handling of different linguistic relations
* improved expressivity

Limitations:

* more parameters
* more compute
* interpretability still limited
* some heads may become redundant

In practice, multi-head attention is one of the key reasons Transformers outperform simpler attention models.

---

# 4) Positional Encoding

## Core intuition & linguistic context

Self-attention alone is permutation-invariant.

If you shuffle tokens, pure self-attention does not inherently know the order changed.

But order matters enormously in language:

* “dog bites man” ≠ “man bites dog”
* syntax depends on position
* local phrases depend on adjacency

Therefore, Transformers need explicit positional information.

---

## Architecture, math & mechanics

Positional encodings inject sequence order into token representations.

If token embedding is (\mathbf{e}_t), the final input to the Transformer is often:

[
\mathbf{x}_t = \mathbf{e}_t + \mathbf{p}_t
]

where (\mathbf{p}_t) is the positional encoding.

### Sinusoidal positional encoding

The original Transformer uses deterministic sinusoidal encodings:

[
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
]

[
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
]

where:

* (pos) is the position index,
* (i) is the dimension index.

### Why sinusoids?

They allow the model to infer relative positions from linear combinations of the encodings, and they generalize to unseen lengths better than purely learned absolute vectors in some settings.

---

### Learned positional embeddings

Some models learn a positional embedding table instead of using fixed sinusoids.

If sequence positions go from (1) to (T_{max}), then:

[
P \in \mathbb{R}^{T_{max} \times d_{model}}
]

and each position gets a learned vector.

---

## Engineering reality & pitfalls

Without positional encoding:

* the model cannot know token order,
* sentence structure collapses,
* generation and classification degrade severely.

Pitfalls:

* absolute positions may not generalize well to longer sequences than seen in training
* positional choices interact with tokenization and maximum context length
* long-context modeling remains a design challenge

---

# 5) Transformer Encoder and Decoder Blocks

This is where the architecture becomes concrete.

---

## 5.1 Encoder Block

### Core intuition & linguistic context

An encoder block transforms raw token embeddings into deeply contextualized representations.

Each token representation is refined by:

* attending to all other tokens,
* passing through a feed-forward network,
* maintaining stable gradients with residuals and normalization.

The encoder is best thought of as a **contextual feature extractor**.

---

### Architecture, math & mechanics

A standard encoder layer contains:

1. Multi-head self-attention
2. Add & layer norm
3. Feed-forward network
4. Add & layer norm

Let the input be (X).

#### Step 1: self-attention

[
A = \text{MultiHead}(X,X,X)
]

#### Step 2: residual connection + layer norm

[
X' = \text{LayerNorm}(X + A)
]

#### Step 3: feed-forward network

A position-wise MLP applied to each token independently:

[
FFN(x) = W_2 \sigma(W_1 x + b_1) + b_2
]

This is applied to each token vector separately, but with shared weights across positions.

#### Step 4: residual + layer norm

[
Y = \text{LayerNorm}(X' + FFN(X'))
]

---

### Why residual connections matter

They help:

* gradient flow,
* training stability,
* information preservation.

### Why layer normalization matters

It stabilizes hidden activations and makes optimization easier.

---

## 5.2 Decoder Block

### Core intuition & linguistic context

The decoder generates tokens autoregressively:

* it predicts the next token based on previous generated tokens,
* it must not peek at future tokens,
* it may also attend to encoder outputs in seq2seq tasks.

That causal constraint is essential for generation.

---

### Architecture, math & mechanics

A Transformer decoder layer contains:

1. **Masked self-attention**
2. **Encoder-decoder attention** (in seq2seq models)
3. **Feed-forward network**
4. Residuals + layer norms

#### Masked self-attention

The decoder can only attend to earlier positions, not future ones.

If the target sequence is (y_1,\dots,y_U), position (u) can attend only to positions (\leq u).

This is enforced with an attention mask:

[
M_{ij} =
\begin{cases}
0 & j \le i \
-\infty & j > i
\end{cases}
]

Then attention becomes:

[
\text{softmax}\left(\frac{QK^\top + M}{\sqrt{d_k}}\right)V
]

This ensures causal generation.

---

### Encoder-decoder attention in the decoder

In seq2seq Transformers:

* decoder queries come from decoder states,
* keys and values come from encoder outputs.

This lets the decoder condition on the source sequence.

---

## 5.3 Why the block design works

Each layer performs:

* global token interaction through attention,
* local nonlinear transformation through FFN,
* stabilized gradient propagation through residuals and normalization.

Stacking many such layers allows the model to build increasingly abstract linguistic features:

* lexical patterns,
* phrase structure,
* dependencies,
* discourse relations,
* task-specific abstractions.

---

## Engineering reality & pitfalls

Common issues:

* memory usage grows with sequence length
* deep stacks need careful optimization
* masking mistakes can leak future information
* layer norm and residual placement affect performance
* decoder generation is slow at inference because outputs are generated one token at a time

---

# 6) Tokenization Methods

Modern Transformer performance depends heavily on tokenization.

Why? Because the model operates on tokens, not raw characters or words. Tokenization determines:

* vocabulary size,
* OOV behavior,
* sequence length,
* handling of rare words,
* efficiency.

---

## 6.1 Word-level tokenization

### Core intuition

Each word is a token.

### Problems

* huge vocabulary
* OOV words
* morphological explosion
* poor handling of typos and rare forms

Word-level tokenization was common in older NLP systems but is less favored for modern Transformers.

---

## 6.2 Character-level tokenization

### Core intuition

Each character becomes a token.

### Benefits

* no OOV problem at the token level
* handles arbitrary strings
* morphology is naturally visible

### Problems

* sequences become much longer
* models need more steps to represent meaning
* computation becomes heavier

---

## 6.3 Subword tokenization

This is the modern sweet spot.

Subwords balance:

* vocabulary size,
* coverage,
* sequence length,
* morphological flexibility.

Words like:

* “unhappiness” may be split into `un`, `happi`, `ness`
* “internationalization” into meaningful chunks

This lets the model share representations across word forms.

---

## Engineering reality & pitfalls

Tokenization must match the pretrained model’s tokenizer exactly. If you use the wrong tokenizer:

* token IDs will be wrong,
* embeddings will not match,
* performance collapses.

Tokenization also determines maximum context length in practice because longer token sequences consume more memory.

---

# 7) Byte-Pair Encoding (BPE)

## Core intuition & linguistic context

Byte-Pair Encoding is a subword tokenization method that starts with small units and merges frequent adjacent pairs.

Its goal is to build a vocabulary that can:

* represent common words as compact units,
* still decompose rare words into meaningful parts,
* avoid OOV issues.

---

## Architecture, math & mechanics

### Training procedure

1. Start with a vocabulary of individual characters or bytes.
2. Count adjacent symbol pairs in a corpus.
3. Merge the most frequent pair into a new symbol.
4. Repeat for a fixed number of merges.

Example:

* `l`, `o`, `w`
* merge `l` + `o` → `lo`
* merge `lo` + `w` → `low`

Then:

* “lowest” might become `low` + `est`

### Why it works

Frequent substrings become single tokens, so the model learns efficient representations for common morphological patterns and frequent words.

---

## Engineering reality & pitfalls

Benefits:

* reduces OOV issues
* improves parameter sharing
* keeps vocabulary manageable
* widely used in practice

Limitations:

* token boundaries are not always linguistically perfect
* rare or unusual forms may still fragment awkwardly
* merges are frequency-based, not meaning-based

BPE is a practical compromise, not a linguistic theory.

---

# 8) WordPiece

## Core intuition & linguistic context

WordPiece is another subword tokenization method used widely in Transformer models.

It also breaks words into subword units, but with a slightly different training logic than BPE.

The key idea is still:

* represent common words or morphemes as tokens,
* split rare words into subwords,
* maximize coverage with a fixed vocabulary.

---

## Architecture, math & mechanics

WordPiece chooses subword units to maximize likelihood of the training corpus under the tokenization scheme. Conceptually, it tries to build a vocabulary that makes the corpus probable and tokenization efficient.

A typical WordPiece tokenization uses markers like:

* `##ing`
* `##ed`
* `##tion`

These markers indicate that a subword occurs after the beginning of a word.

Example:

* `playing` → `play`, `##ing`

### Why the `##` notation matters

It tells the model whether a subword starts a word or continues one.

That gives the tokenizer extra structure and helps with reconstructing words.

---

## BPE vs WordPiece

### BPE

* frequency-based pair merging
* easy to understand
* common in many autoregressive LLM pipelines

### WordPiece

* likelihood-oriented vocabulary building
* often associated with BERT-style systems

Both are subword methods; the key difference is how the vocabulary is constructed.

---

## Engineering reality & pitfalls

Like BPE, WordPiece:

* reduces OOV problems
* handles morphology better than pure word tokenization
* increases sequence length relative to word tokenization
* can split rare words into many pieces

For transformers, tokenization is not a minor implementation detail. It directly shapes model capacity and efficiency.

---

# Part II — Pretrained Language Models

## 9) Pretrained Transformer Models

## Core intuition & linguistic context

Pretraining changes the NLP workflow.

Instead of training a model from scratch on a single task, we first train a large model on massive text corpora to learn general language structure, then adapt it to downstream tasks.

This works because language has reusable regularities:

* syntax,
* morphology,
* semantics,
* discourse,
* lexical relations,
* phrase patterns.

A pretrained model becomes a general-purpose language feature extractor or generator.

---

## Architecture, math & mechanics

The standard approach is:

1. Choose a Transformer architecture.
2. Train on a large self-supervised objective.
3. Transfer the learned parameters to downstream tasks.
4. Fine-tune or adapt as needed.

The pretrained model has learned:

* token embeddings,
* syntax-aware contextual layers,
* semantic regularities,
* task-agnostic representational capacity.

This is one of the biggest shifts in machine learning:

> from task-specific training to reusable language foundations.

---

## Engineering reality & pitfalls

Advantages:

* better accuracy with less labeled data
* faster convergence on downstream tasks
* richer representations
* strong transfer across tasks

Limitations:

* large compute requirements
* domain mismatch can still hurt
* pretraining objective may not perfectly align with downstream task
* fine-tuning can overfit on small datasets

---

# 10) BERT

## Core intuition & linguistic context

BERT is designed to produce **deep bidirectional contextual representations** of text.

That means each token representation can depend on both left and right context.

This is particularly good for:

* understanding tasks,
* classification,
* tagging,
* question answering,
* extraction.

BERT is not designed primarily for free-form left-to-right generation.

---

## Architecture, math & mechanics

BERT uses:

* Transformer encoder stack
* bidirectional self-attention
* special tokens like `[CLS]` and `[SEP]`

The `[CLS]` token is often used as an aggregate representation for classification.

### Masked Language Modeling (MLM)

BERT is trained by masking some input tokens and predicting them from context.

Given a sentence with masked positions, the model learns:

[
P(x_i \mid x_{\setminus i})
]

for selected masked positions.

This forces the model to use both left and right context.

---

## Why BERT is bidirectional

Unlike causal models, BERT can attend to tokens on both sides of the masked token because the full sentence is visible except for the masked positions.

That is why BERT is highly effective for understanding tasks.

---

## Engineering reality & pitfalls

Strengths:

* excellent contextual understanding
* strong on classification and extraction
* useful as a general encoder backbone

Limitations:

* not naturally generative
* MLM pretraining creates a train-test gap because mask tokens do not appear the same way at inference
* max context length constraints still apply
* fine-tuning needs care to avoid catastrophic forgetting or overfitting

---

# 11) GPT

## Core intuition & linguistic context

GPT is a decoder-style Transformer trained autoregressively.

It predicts the next token based on previous tokens only.

This makes it ideal for:

* text generation,
* dialogue,
* completion,
* few-shot prompting,
* instruction-following variants later on.

GPT models are fundamentally **causal language models**.

---

## Architecture, math & mechanics

GPT uses:

* masked self-attention in decoder blocks
* no bidirectional access
* next-token prediction objective

For tokens (x_1,\dots,x_T):

[
P(x_1,\dots,x_T) = \prod_{t=1}^{T} P(x_t \mid x_{<t})
]

This is the causal language modeling objective.

The model learns to continue text one token at a time.

---

## Why GPT is powerful

A causal LM can serve as:

* a generator,
* a completion engine,
* a conditional text model,
* a zero-shot/few-shot learner when scaled and instructed properly.

Because it models the full distribution over continuations, it can generate flexible outputs.

---

## Engineering reality & pitfalls

Strengths:

* excellent generation ability
* natural fit for open-ended text prediction
* can be adapted to instruction following

Limitations:

* unidirectional context only during generation
* hallucination risk
* token-by-token inference can be slow
* long outputs can drift
* correctness is not guaranteed by fluency

GPT-style models are often fluent but need careful control for factual tasks.

---

# 12) T5

## Core intuition & linguistic context

T5 reframes every NLP task as a **text-to-text** problem.

Instead of designing task-specific heads for classification, tagging, translation, or question answering, the model takes text in and produces text out.

This is conceptually elegant because it unifies NLP tasks under one interface.

Examples:

* classification: “sentence: ...” → “positive”
* translation: “translate English to French: ...” → French output
* QA: “question: ... context: ...” → answer text

---

## Architecture, math & mechanics

T5 uses an encoder-decoder Transformer.

It is trained with a denoising-style objective where input text is corrupted and the model learns to reconstruct the missing pieces.

A common T5-style pretraining setup uses **span corruption**:

* contiguous spans are replaced with sentinel tokens,
* the model predicts the missing spans.

This teaches the model:

* understanding,
* generation,
* reconstruction,
* sequence-to-sequence mapping.

---

## Why T5 matters

T5 makes task formulation uniform:

* all tasks become generation,
* all outputs are text,
* task-specific heads are less necessary.

This simplifies fine-tuning and multi-task learning.

---

## Engineering reality & pitfalls

Strengths:

* unified framework
* flexible for many tasks
* strong seq2seq behavior

Limitations:

* generation can be slower than classification heads
* output formatting matters
* training and prompt design can strongly affect performance
* text-to-text is elegant, but sometimes overkill for simple classification tasks

---

# 13) Masked Language Modeling (MLM)

## Core intuition & linguistic context

MLM trains the model to predict missing tokens from surrounding context.

This is a powerful self-supervised objective because you do not need labels. The text itself supplies the supervision.

---

## Architecture, math & mechanics

Given a sequence, some tokens are masked:

[
x = (x_1,\dots,x_T)
]

Choose a subset (M) of positions to mask. The model predicts:

[
P(x_i \mid x_{\setminus M}) \quad \text{for } i \in M
]

Loss:

[
L = -\sum_{i \in M} \log P(x_i \mid x_{\setminus M})
]

### Why it works

The model must infer hidden words from context, so it learns:

* syntax,
* semantics,
* collocations,
* broader sentence structure.

---

## Engineering reality & pitfalls

Strengths:

* excellent for representation learning
* bidirectional contextualization
* strong downstream transfer

Limitations:

* pretrain-finetune mismatch from mask tokens
* not ideal for generative text in the same way causal LM is
* masking strategy affects results
* predicting multiple masked tokens can be hard if too much is masked

---

# 14) Next Sentence Prediction (NSP)

## Core intuition & linguistic context

NSP was introduced to help BERT learn relationships between sentences.

The model is trained to predict whether sentence B follows sentence A in the original corpus.

This was intended to support:

* discourse modeling,
* sentence-pair tasks,
* textual coherence.

---

## Architecture, math & mechanics

Input format:

* `[CLS] sentence A [SEP] sentence B [SEP]`

The model predicts a binary label:

* IsNext
* NotNext

The objective is to classify sentence pairs using the `[CLS]` representation.

---

## Engineering reality & pitfalls

The idea is linguistically sensible, but NSP has been debated because:

* some implementations found it less useful than expected,
* sentence-pair coherence may require stronger objectives,
* later models used alternative pretraining strategies.

For the syllabus, the key point is that NSP is a sentence-level relational pretraining task layered on top of MLM-style token prediction.

---

# 15) Causal Language Modeling (CLM)

## Core intuition & linguistic context

Causal language modeling is the autoregressive objective used by GPT-style models.

The model predicts the next token using only previous tokens.

This matches the natural process of generation.

---

## Architecture, math & mechanics

For tokens (x_1,\dots,x_T):

[
L = -\sum_{t=1}^{T} \log P(x_t \mid x_{<t})
]

The attention mask prevents attending to future tokens.

This is exactly the language modeling objective from classic NLP, scaled up with Transformer capacity.

---

## Engineering reality & pitfalls

Strengths:

* natural for generation
* simple objective
* strong transfer in large models

Limitations:

* unidirectional context only
* may be weaker on pure understanding tasks than bidirectional encoders
* autoregressive decoding is slower at inference

---

# 16) Transfer Learning for NLP Tasks

## Core intuition & linguistic context

Transfer learning means:

> Learn general linguistic knowledge from one objective, then reuse it for downstream tasks.

This is one of the biggest reasons modern NLP works so well.

A pretrained model already knows:

* syntax patterns,
* subword composition,
* semantic associations,
* discourse regularities,
* some world-statistical priors.

Then a small labeled dataset can adapt the model to a specific task.

---

## Architecture, math & mechanics

Typical transfer workflow:

1. **Pretrain** on large unlabeled corpus.
2. **Fine-tune** on labeled downstream data.
3. Optionally freeze some layers or use parameter-efficient adaptation.
4. Evaluate on the target task.

The pretrained parameters act as an initialization that is far better than random initialization.

This drastically improves sample efficiency.

---

## Why transfer learning works

Language has shared structure across tasks:

* a classifier still needs to know what nouns, negation, and sentiment cues look like
* NER still benefits from knowing entity-like patterns
* QA still benefits from sentence understanding and token-level salience

Pretraining learns reusable representations.

---

## Engineering reality & pitfalls

Advantages:

* better performance with less labeled data
* faster convergence
* strong baselines

Pitfalls:

* domain mismatch
* overfitting during fine-tuning
* catastrophic forgetting if the learning rate is too high
* mismatch between pretraining objective and downstream task

---

# 17) Fine-Tuning for Text Classification

## Core intuition & linguistic context

Text classification is one of the most common fine-tuning use cases.

The model takes a text and predicts a label:

* sentiment,
* topic,
* spam,
* toxicity,
* intent,
* stance,
* emotion.

A pretrained encoder already understands language structure, so we only need a task-specific classifier on top.

---

## Architecture, math & mechanics

A typical BERT-style classifier:

1. Tokenize text.
2. Encode through Transformer.
3. Take `[CLS]` representation.
4. Pass through linear classification head.

If (\mathbf{h}_{CLS}) is the final hidden state for `[CLS]`, then:

[
\hat{\mathbf{y}} = \text{softmax}(W\mathbf{h}_{CLS} + b)
]

For binary classification, use sigmoid instead.

Loss is cross-entropy.

---

## Engineering reality & pitfalls

Advantages:

* strong performance with minimal architecture changes
* simple pipeline
* excellent transfer from pretrained encoders

Pitfalls:

* dataset size may be small
* class imbalance
* calibration issues
* tokenization can affect results
* model may overfit quickly if fine-tuning is too aggressive

Best practice:

* small learning rates,
* careful regularization,
* early stopping,
* appropriate metrics like F1 when classes are imbalanced.

---

# 18) Fine-Tuning for Named Entity Recognition (NER)

## Core intuition & linguistic context

NER is a token-level sequence labeling task.

We want to label each token as:

* PER
* LOC
* ORG
* MISC
* O

This task depends heavily on context:

* “Apple” can be a company or fruit
* “Jordan” can be a person or location
* entity boundaries may span multiple tokens

Pretrained Transformers are highly effective here because they give contextual token embeddings.

---

## Architecture, math & mechanics

For each token representation (\mathbf{h}_t), predict a label:

[
\hat{\mathbf{y}}_t = \text{softmax}(W\mathbf{h}_t + b)
]

This can be done token-wise.

For sequence labeling, some models also add a CRF layer on top, though that is beyond the current syllabus emphasis.

### Why contextual embeddings help

The token “Washington” has a different representation in:

* “Washington was elected”
* “Washington is a state”
* “located in Washington”

The model can use surrounding context to determine the correct entity type.

---

## Engineering reality & pitfalls

NER challenges:

* class imbalance
* label boundary errors
* nested entities
* tokenization alignment with subwords
* inconsistent annotation standards

Using subword tokenization requires care:

* some subtokens may share the label of the original word,
* only the first subtoken may be supervised,
* or labels may be aligned across all subtokens depending on implementation.

Evaluation usually relies on entity-level precision/recall/F1, not only token accuracy.

---

# 19) Fine-Tuning for Question Answering

## Core intuition & linguistic context

Question answering comes in several forms, but in many Transformer tutorials and benchmarks, the task is:

* given a question and a context passage,
* predict the answer span in the passage.

This is a precise token-level extraction problem, not just classification.

The model must understand:

* what the question asks,
* where the answer is located in the context,
* how to align semantic roles and entities.

---

## Architecture, math & mechanics

Input format often looks like:

[
[\text{CLS}] \ \text{question} \ [\text{SEP}] \ \text{context} \ [\text{SEP}]
]

The model predicts:

* start index of answer span
* end index of answer span

For each token (t), the model outputs:

* start score
* end score

For example:

[
P_{start}(t) = \text{softmax}(W_s \mathbf{h}*t)
]
[
P*{end}(t) = \text{softmax}(W_e \mathbf{h}_t)
]

The predicted answer is the span with the highest joint probability, subject to constraints.

---

## Engineering reality & pitfalls

Challenges:

* long contexts may exceed max length
* answer may not be extractable if not present
* multiple spans may be plausible
* subword boundaries complicate span reconstruction
* models can be confident but wrong

Question answering is a strong test of contextual understanding, but span extraction is still limited relative to open-domain generative QA.

---

# 20) HuggingFace Transformers

## Core intuition & linguistic context

HuggingFace Transformers is an ecosystem that made pretrained Transformer models practical to use.

It provides:

* tokenizers,
* pretrained checkpoints,
* model classes,
* fine-tuning utilities,
* pipelines,
* training integration.

The key benefit is engineering standardization:

> use the same model family across tasks without rebuilding everything from scratch.

---

## Architecture, math & mechanics

In practice, you typically work with:

* tokenizer
* model
* task-specific head
* datasets / dataloaders
* training loop or trainer

Conceptually, the library maps:

* raw text → token IDs → Transformer → logits → predictions

This standardization makes experimentation much faster and reduces implementation errors.

---

## Engineering reality & pitfalls

Advantages:

* huge model zoo
* strong interoperability
* standardized tokenization and checkpoints
* easy fine-tuning and inference
* wide community adoption

Pitfalls:

* you must still choose the right tokenizer/model pair
* sequence length limits matter
* different architectures behave differently
* simple “pipeline” usage can hide important details like truncation, padding, and label alignment

A good NLP engineer must understand what happens under the hood, not just call a convenience API.

---

# 21) How the Pieces Fit Together

This unit is not a loose set of Transformer facts. It forms a coherent stack:

1. **Transformer architecture** replaces recurrence with attention.
2. **Self-attention** lets tokens contextualize themselves.
3. **Multi-head attention** lets multiple linguistic relations be modeled at once.
4. **Positional encoding** restores order information.
5. **Encoder and decoder blocks** build deep contextual representations and causal generation.
6. **Tokenization** defines the symbolic interface between raw text and the model.
7. **BPE and WordPiece** solve vocabulary and OOV problems with subword segmentation.
8. **Pretraining** learns reusable linguistic knowledge from unlabeled data.
9. **BERT** is encoder-based and bidirectional.
10. **GPT** is decoder-based and autoregressive.
11. **T5** reframes all tasks as text-to-text.
12. **MLM, NSP, CLM** are self-supervised objectives that shape representation learning.
13. **Transfer learning and fine-tuning** adapt pretrained models to classification, NER, QA, and more.
14. **HuggingFace Transformers** operationalizes all of this in a usable toolkit.

The central philosophical shift is:

> We no longer train a separate model from scratch for every task.
> We pretrain a general language model and adapt it.

That is the modern NLP paradigm.

---

# 22) Practical Engineering Guidance

## Choosing between BERT, GPT, and T5

### BERT

Best when you need:

* understanding,
* classification,
* token labeling,
* extraction,
* bidirectional context.

### GPT

Best when you need:

* generation,
* continuation,
* dialogue,
* open-ended text production.

### T5

Best when you want:

* a unified text-to-text interface,
* flexible seq2seq behavior,
* multi-task transfer.

---

## Choosing a tokenization method

* **Word-level**: too brittle for modern deep NLP
* **Character-level**: robust but long and expensive
* **Subword-level**: best practical tradeoff for most Transformer systems

---

## Fine-tuning advice

* use task-appropriate metrics
* ensure input truncation is sensible
* handle padding and masking correctly
* align labels carefully for token tasks
* use small learning rates and regularization
* inspect errors manually, not just aggregate numbers

---

# 23) Common Mistakes and Failure Modes

1. **Ignoring tokenization**
   Tokenization is part of the model, not a preprocessing footnote.

2. **Using the wrong attention mask**
   A masking bug can break causality or leak future information.

3. **Assuming attention weights are explanations**
   They are useful signals, but not guaranteed explanations.

4. **Fine-tuning too aggressively**
   Large learning rates can erase pretrained knowledge.

5. **Evaluating generation only with one metric**
   BLEU, ROUGE, exact match, and human judgments each capture different things.

6. **Using the same architecture for every task without thought**
   Encoder-only, decoder-only, and encoder-decoder models serve different purposes.

7. **Ignoring sequence length limits**
   Many practical failures come from truncation or context window constraints.

---

# 24) Summary of the Unit

This unit explained the architecture and ecosystem that define modern NLP.

* **Transformers** use attention instead of recurrence.
* **Self-attention** gives each token direct access to all others.
* **Multi-head attention** lets the model learn diverse linguistic relationships in parallel.
* **Positional encoding** injects word order.
* **Encoder and decoder blocks** build deep contextual representations and controlled generation.
* **BPE** and **WordPiece** solve tokenization and OOV issues using subwords.
* **Pretrained language models** learn general linguistic knowledge from large corpora.
* **BERT** is bidirectional and ideal for understanding.
* **GPT** is causal and ideal for generation.
* **T5** unifies tasks as text-to-text.
* **MLM, NSP, and CLM** are foundational self-supervised objectives.
* **Transfer learning** and **fine-tuning** make pretrained models useful for classification, NER, QA, and many other tasks.
* **HuggingFace Transformers** makes this entire ecosystem accessible in practice.

The deepest lesson is:

> Modern NLP is not about hand-crafting features for each task.
> It is about learning general language representations, then adapting them efficiently.

---

# 25) End-of-Unit NLP Master Challenge

## Problem 1: Self-attention shape reasoning

A Transformer encoder receives a batch of tokenized sentences with:

* batch size (B = 16)
* sequence length (T = 64)
* model dimension (d_{model} = 512)

What are the shapes of:

1. the input embeddings,
2. the (Q, K, V) matrices for one head if (d_k = 64),
3. the attention score matrix,
4. the final output of the attention block?

Explain each shape in words.

## Problem 2: Encoder-only vs decoder-only vs encoder-decoder

You need to build:

1. a sentiment classifier,
2. a chat assistant,
3. a translation system.

Which Transformer family would you choose for each, and why? Discuss how the objective and masking differ.

## Problem 3: BERT vs GPT

A product review classification task has both positive and negative cues distributed across the sentence.
Would BERT or GPT be more natural as the backbone, and why?
Then explain why GPT may still be useful for certain generation-style review tasks.

## Problem 4: Tokenization tradeoffs

A low-resource language has many long inflected words and a small training corpus.
Would you prefer word-level, character-level, BPE, or WordPiece tokenization? Justify your answer using OOV behavior, sequence length, and parameter sharing.

## Problem 5: Fine-tuning for QA

A BERT-based QA model predicts the correct answer span only when the answer appears early in the context, but fails when the answer is later in the passage.
Explain possible causes, including truncation, attention limitations, and training data bias.

## Problem 6: MLM vs CLM

Explain why masked language modeling is well suited to bidirectional understanding tasks, while causal language modeling is naturally suited to generation.
What is the role of the attention mask in each case?

---

When you are ready, send **Unit VI** and I will continue with **Generative NLP and LLMs** at the same depth.
