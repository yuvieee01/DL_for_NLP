## Unit III — Deep Learning Sequence Models for NLP

This unit is where NLP becomes truly **sequence-aware**.
In Units I and II, text was turned into tokens and then into vectors. But a sentence is not just a set of words. It is an **ordered sequence** where meaning depends on what came before and what comes after. Sequence models are designed to process that order explicitly.

The big question this unit answers is:

> **How do we build neural networks that can read text one token at a time, remember useful past information, and make predictions over sequences?**

---

# 1) Sequential Text Data

## 1.1 What makes text sequential?

Text is not an unordered bag of words. Word order changes meaning.

Examples:

* “Dog bites man”
* “Man bites dog”

Same words, completely different meaning.

Sequence models are built to handle:

* ordered tokens
* varying-length inputs
* dependencies between earlier and later words
* prefix context and suffix context

---

## 1.2 Representing text as a sequence

Suppose a sentence has tokens:

[
x_1, x_2, x_3, \dots, x_T
]

where:

* (T) = number of tokens
* (x_t) = token at position (t)

If each token is converted to an embedding vector, then the input becomes:

[
X = [x_1, x_2, \dots, x_T]
]

where each (x_t \in \mathbb{R}^d).

So the full sequence tensor usually has shape:

[
(T, d)
]

for one example, or

[
(B, T, d)
]

for a batch of size (B).

---

## 1.3 Why sequential modeling is hard

Text is hard because:

* sequence lengths vary
* dependencies can be short or long
* the model must preserve order
* relevant clues may appear far apart
* padding and masking are required in minibatches

Example:

> “The movie, although slow at first, was actually extremely enjoyable.”

The word “enjoyable” may be the key sentiment signal, but the model must understand the whole prefix and the reversal introduced by “although”.

---

## 1.4 Types of sequence prediction tasks

### A. One-to-one

One input sequence → one label

Example:

* sentiment classification
* topic classification

### B. Many-to-one

Many tokens → one output

This is essentially the same as one-to-one for text classifiers.

### C. One-to-many

One input → sequence output

Example:

* image captioning
* text generation from a prompt

### D. Many-to-many

Sequence input → sequence output

Examples:

* machine translation
* POS tagging
* named entity recognition
* summarization

This unit focuses on the neural sequence foundations that support these tasks.

---

# 2) Recurrent Neural Networks (RNNs)

## 2.1 Core intuition

A recurrent neural network processes a sequence **step by step**, maintaining a hidden memory of what it has seen so far.

The idea is simple:

* read token (x_1)
* update memory to hidden state (h_1)
* read token (x_2)
* update memory to hidden state (h_2)
* continue until the end

The hidden state acts like a compact summary of the past.

---

## 2.2 Why recurrence is useful

Language depends on context.

Example:

* “The food was not good”
* “The food was good”

The word “not” flips meaning, but its effect must persist until “good” appears. An RNN is designed to carry information forward through time.

---

## 2.3 RNN architecture

At time step (t), an RNN takes:

* current input vector (x_t)
* previous hidden state (h_{t-1})

and computes a new hidden state:

[
h_t = \phi(W_{xh}x_t + W_{hh}h_{t-1} + b_h)
]

where:

* (W_{xh}): input-to-hidden weights
* (W_{hh}): hidden-to-hidden weights
* (b_h): bias
* (\phi): activation function, usually (\tanh) or ReLU

The output may be:

[
y_t = \psi(W_{hy}h_t + b_y)
]

where (\psi) may be softmax for classification.

---

## 2.4 Shape intuition

If:

* batch size = (B)
* sequence length = (T)
* input embedding size = (d)
* hidden size = (H)

Then:

* input tensor: ((B, T, d))
* hidden state at each step: ((B, H))
* output sequence: often ((B, T, H)) or ((B, T, C)) depending on task

---

## 2.5 Unrolling through time

An RNN can be visualized as the same cell repeated across time steps.

The parameters are shared across all positions:

* same (W_{xh})
* same (W_{hh})
* same biases

This weight sharing is important:

* fewer parameters
* position-invariant processing
* ability to handle variable sequence lengths

---

## 2.6 Why RNNs are better than feedforward networks for text

A feedforward network requires fixed-size input.
An RNN can naturally process:

* short sentences
* long sentences
* variable-length documents

It also maintains sequential context, which feedforward BoW models cannot do.

---

## 2.7 Problems with vanilla RNNs

Vanilla RNNs look elegant, but in practice they suffer from major issues.

### A. Vanishing gradients

When backpropagating through many time steps, gradients can become very small.

This makes it hard to learn long-range dependencies.

### B. Exploding gradients

Gradients can also become too large, causing unstable training.

### C. Memory bottleneck

The hidden state is a fixed-size vector.
A single vector must compress everything from the entire past.

That is often too restrictive.

### D. Weak long-term dependency learning

RNNs struggle to remember information over long spans.

Example:

> “The book that the professor who the student admired recommended was excellent.”

The relevant subject may be far away from the verb.

---

## 2.8 Vanishing gradients: the mechanism

During backpropagation through time, gradient terms are multiplied repeatedly by Jacobians and weights across many steps.

If those multiplications are smaller than 1 in magnitude, the gradient shrinks exponentially.

This means early tokens get weak learning signals.

That is why simple RNNs are often not enough for real-world NLP.

---

## 2.9 Engineering reality

RNNs:

* are conceptually important
* are useful for understanding sequence learning
* can still be effective on short sequences

But in many modern NLP systems, they were replaced by LSTMs, GRUs, and later transformers because of their limitations.

---

# 3) Long Short-Term Memory Networks (LSTMs)

## 3.1 Core intuition

LSTMs were designed to solve the memory problem of vanilla RNNs.

Their key idea is to maintain a **cell state** that acts like a controlled memory highway through time.

Instead of forcing the hidden state to carry everything, LSTMs regulate:

* what to forget
* what to write
* what to expose

This is done using gates.

---

## 3.2 LSTM components

An LSTM has:

* hidden state (h_t)
* cell state (c_t)
* forget gate (f_t)
* input gate (i_t)
* candidate cell update (\tilde{c}_t)
* output gate (o_t)

These gates control information flow.

---

## 3.3 LSTM equations

Given input (x_t) and previous hidden state (h_{t-1}):

### Forget gate

[
f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)
]

This decides what fraction of previous memory to keep.

### Input gate

[
i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)
]

This decides how much new information to write.

### Candidate memory

[
\tilde{c}*t = \tanh(W_c [h*{t-1}, x_t] + b_c)
]

This proposes new memory content.

### Cell state update

[
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
]

where (\odot) is elementwise multiplication.

### Output gate

[
o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)
]

### Hidden state

[
h_t = o_t \odot \tanh(c_t)
]

---

## 3.4 Intuition behind the gates

### Forget gate

“Do I keep the old memory?”

If (f_t) is close to 1, preserve memory.
If close to 0, discard it.

### Input gate

“How much new information should I store?”

### Output gate

“What part of my memory should I reveal as the hidden state?”

This controlled structure makes LSTMs much more powerful than vanilla RNNs.

---

## 3.5 Why the cell state helps gradients

The cell state offers a path where information can flow more smoothly across time.

Because it is updated additively:

[
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
]

the model can preserve information for long spans with less gradient decay than a plain RNN.

This is why LSTMs are good at:

* long-range dependencies
* sequence memory
* time-series patterns
* text where context is distributed

---

## 3.6 LSTM vs vanilla RNN

### Vanilla RNN

* simple recurrence
* prone to vanishing/exploding gradients
* weak long-term memory

### LSTM

* gated memory
* better gradient flow
* learns long dependencies much better
* more parameters and compute

---

## 3.7 Engineering reality

LSTMs are often strong when:

* data is sequential and medium-scale
* dependencies matter
* the sequence lengths are not too massive
* interpretability of gates is useful

But they are heavier than GRUs and slower than transformers on many tasks.

---

# 4) Gated Recurrent Units (GRUs)

## 4.1 Core intuition

GRUs simplify the LSTM while keeping its ability to learn long dependencies.

They remove the separate cell state and combine memory control into fewer gates.

This makes GRUs:

* simpler
* faster
* often competitive with LSTMs

---

## 4.2 GRU components

A GRU has:

* reset gate (r_t)
* update gate (z_t)
* hidden state (h_t)

---

## 4.3 GRU equations

Given input (x_t) and previous hidden state (h_{t-1}):

### Update gate

[
z_t = \sigma(W_z [h_{t-1}, x_t] + b_z)
]

This decides how much of the old hidden state to keep.

### Reset gate

[
r_t = \sigma(W_r [h_{t-1}, x_t] + b_r)
]

This decides how much of the past to ignore when computing the new candidate state.

### Candidate hidden state

[
\tilde{h}*t = \tanh(W_h [r_t \odot h*{t-1}, x_t] + b_h)
]

### Final hidden state

[
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
]

---

## 4.4 Intuition behind GRU gates

### Update gate

“How much old memory should I preserve versus replace?”

### Reset gate

“When forming a new candidate, should I forget the past?”

GRUs are elegant because they directly mix old and new information without a separate memory cell.

---

## 4.5 GRU vs LSTM

### GRU advantages

* fewer parameters
* faster training
* simpler architecture
* often works very well in practice

### LSTM advantages

* more explicit memory structure
* sometimes better when complex gating is needed

### Practical reality

Either can work well.
The best choice depends on:

* task
* dataset size
* compute budget
* sequence length
* required performance

---

# 5) Bidirectional RNNs

## 5.1 Core intuition

A normal RNN reads text left to right only.

But in language, both left and right context matter.

Example:

* “The bank was crowded.”
* “The river bank was crowded.”

The meaning of “bank” depends on future words as well as previous ones.

A bidirectional RNN processes the sequence in **both directions**.

---

## 5.2 Architecture

There are two recurrent chains:

* forward RNN: reads (x_1 \to x_T)
* backward RNN: reads (x_T \to x_1)

At each time step (t):

* forward hidden state ( \overrightarrow{h_t} )
* backward hidden state ( \overleftarrow{h_t} )

These are combined, often by concatenation:

[
h_t = [\overrightarrow{h_t}; \overleftarrow{h_t}]
]

---

## 5.3 Why bidirectionality helps

It gives the model:

* past context
* future context
* richer token representations

This is especially useful for:

* POS tagging
* named entity recognition
* chunking
* sequence labeling
* sentence classification

---

## 5.4 Engineering reality

Bidirectional models are excellent when the full sequence is available before prediction.

But they are not suitable for strict real-time autoregressive generation, because the “future” is unavailable while generating tokens one by one.

So:

* great for encoding and labeling
* not used the same way for causal generation

---

# 6) Sequence Modeling Applications

Sequence models are used across many NLP tasks.

---

## 6.1 Sentiment classification

Given a review or sentence, predict sentiment:

* positive
* negative
* neutral
* multi-class sentiment scales

Example:

> “The plot was slow, but the ending was beautiful.”

A sequence model can learn that “but” changes the importance of clauses and that “beautiful” may dominate the final sentiment.

### Why sequence models help

They can capture:

* negation
* contrast
* intensifiers
* clause structure
* longer sentiment dependencies

---

## 6.2 Text classification

Sequence models can classify:

* spam vs ham
* topic category
* toxicity
* intent
* emotion
* author style

Unlike BoW models, RNN-based models can use order and context.

Example:

* “not acceptable” should not be treated the same as “acceptable”

---

## 6.3 Sequence labeling

Some NLP tasks require a label for every token.

Examples:

* POS tagging
* NER
* chunking
* slot filling

Input:
[
x_1, x_2, \dots, x_T
]

Output:
[
y_1, y_2, \dots, y_T
]

This is a many-to-many setting.

Bidirectional RNNs are especially useful here because each token label can depend on both left and right context.

---

## 6.4 Language modeling perspective

A sequence model may learn:
[
p(w_t \mid w_1, \dots, w_{t-1})
]

This means it predicts the next token from previous tokens.

This idea will become extremely important in later units, especially transformers and large language models.

---

# 7) Sentiment Classification in Detail

## 7.1 Problem formulation

Given a text sequence (X), predict label (y).

[
X = (x_1, x_2, \dots, x_T)
]
[
y \in { \text{positive}, \text{negative}, \text{neutral} }
]

---

## 7.2 Common sequence model pipeline

### Step 1: Tokenize text

### Step 2: Convert tokens to embeddings

### Step 3: Feed embeddings to an RNN/LSTM/GRU

### Step 4: Use final hidden state or pooled representation

### Step 5: Apply classification layer

### Step 6: Softmax or sigmoid for prediction

---

## 7.3 Using the final hidden state

For a many-to-one task, the final hidden state may summarize the entire sequence.

If (h_T) is the last hidden state, then:

[
\hat{y} = \text{softmax}(W h_T + b)
]

or for binary classification:

[
\hat{y} = \sigma(W h_T + b)
]

---

## 7.4 Why this works

The final hidden state is a learned summary of the text, ideally capturing:

* topic
* tone
* negation
* emphasis
* context

But if the sequence is very long, the final state may forget early information. That is one reason bidirectional and attention-based approaches became popular later.

---

## 7.5 Engineering pitfalls in sentiment classification

* sarcasm is hard
* long reviews dilute important clues
* label noise is common
* domain shift hurts performance
* class imbalance can bias predictions

Example:

* “This phone is sick” may be positive in slang, negative in another context.

---

# 8) Text Classification in Detail

Text classification is broader than sentiment analysis.

Possible classes:

* sports
* politics
* finance
* legal
* spam
* intent categories

Sequence models help by learning patterns over token order rather than isolated keywords.

---

## 8.1 Why RNN-based models can outperform BoW

BoW may see:

* “not useful” and “useful” as very similar if stop words are removed or weights are weak.

Sequence models preserve:

* order
* local composition
* modifier scope

That makes them stronger for classification tasks where word order matters.

---

## 8.2 Using pooling over hidden states

Instead of only using (h_T), one may pool over all hidden states:

* mean pooling
* max pooling
* concatenation of last forward and backward states in bidirectional models

This can improve classification by using information from all positions.

---

# 9) Sequence Training Techniques

Training sequence models is not the same as training simple feedforward networks.
The temporal structure introduces special training problems.

This section covers the main techniques listed in the syllabus:

* teacher forcing
* truncated backpropagation through time
* sequence evaluation metrics

---

# 10) Teacher Forcing

## 10.1 Core intuition

Teacher forcing is used when training models that generate sequences.

At time step (t), instead of feeding the model’s own previous prediction into the next step, we feed the **true previous token** from the training data.

This stabilizes training and speeds convergence.

---

## 10.2 Why it is needed

During generation, the model predicts one token at a time.

Suppose the correct sequence is:

* “I love NLP”

At time step 1, the model predicts “I”.
At time step 2, instead of using its predicted output from step 1, teacher forcing gives the true token “I” as input for predicting the next token.

This reduces error accumulation during training.

---

## 10.3 Formal view

For a target sequence (y_1, y_2, \dots, y_T), the model learns:

[
p(y_t \mid y_1, \dots, y_{t-1}, x)
]

With teacher forcing, the input at step (t) is the ground truth (y_{t-1}).

---

## 10.4 Advantages

* faster training
* more stable gradients
* better early learning
* easier optimization

---

## 10.5 Limitation: exposure bias

During training, the model sees correct previous tokens.
During inference, it sees its own predicted tokens.

This mismatch is called **exposure bias**.

Consequences:

* training and inference conditions differ
* small early mistakes may accumulate
* generation quality may drop at test time

This is one of the most important practical problems in sequence generation.

---

# 11) Truncated Backpropagation Through Time (TBPTT)

## 11.1 Core intuition

Backpropagation through time (BPTT) unrolls an RNN over all time steps and backpropagates gradients through the entire sequence.

For long sequences, this is expensive and unstable.

Truncated BPTT limits the number of steps over which gradients are backpropagated.

---

## 11.2 Why truncation is used

For a sequence of length (T), full BPTT can require:

* large memory
* long compute time
* unstable gradients

TBPTT processes the sequence in chunks of length (k).

Example:

* full sequence length = 1000
* truncation length = 50

The model backpropagates only across 50 steps at a time.

---

## 11.3 How TBPTT works

The hidden state is carried forward across chunks, but the gradient is cut when moving to the previous chunk boundary.

This means:

* memory state continues
* gradient history is truncated

### Benefit

You can train on long sequences without storing everything.

### Cost

The model may fail to learn dependencies longer than the truncation window.

---

## 11.4 Engineering reality

TBPTT is a compromise:

* practical for long sequences
* cheaper than full BPTT
* but less capable of long-range credit assignment

It was especially common in RNN training before transformers reduced the need for recurrence in many NLP settings.

---

# 12) Evaluation Metrics for Sequence Tasks

Sequence models require careful evaluation depending on the task.

---

## 12.1 Accuracy

For classification tasks:
[
Accuracy = \frac{\text{correct predictions}}{\text{total predictions}}
]

Useful for:

* sentiment classification
* topic classification
* spam detection

### Limitation

Accuracy can be misleading under class imbalance.

Example:
If 95% of samples are negative, a trivial negative classifier gets 95% accuracy and still fails badly.

---

## 12.2 Precision, Recall, F1

Especially important for imbalanced classification.

### Precision

Of the predicted positives, how many are correct?

[
Precision = \frac{TP}{TP + FP}
]

### Recall

Of the actual positives, how many did we find?

[
Recall = \frac{TP}{TP + FN}
]

### F1 score

Harmonic mean of precision and recall:

[
F1 = \frac{2PR}{P+R}
]

These are crucial in tasks like:

* toxicity detection
* spam detection
* medical text classification
* entity extraction

---

## 12.3 Token-level metrics for sequence labeling

For tasks like NER or POS tagging, evaluation is often at the token level or span level.

### Token accuracy

How many token labels were correct?

### Span-level F1

For NER, exact entity spans are often evaluated because partial overlap is not enough.

Example:

* predicted: “New York”
* gold: “New York City”

That may count as incorrect under strict span evaluation.

---

## 12.4 Perplexity

Perplexity is a standard metric for language modeling.

If the model predicts a sequence with low uncertainty, perplexity is low.

For a language model with average negative log-likelihood (L):

[
Perplexity = e^{L}
]

or sometimes:
[
Perplexity = 2^{L}
]
depending on the log base.

### Interpretation

Lower perplexity means the model better predicts the next token.

### Limitation

Low perplexity does not always mean better downstream task performance or better generated text quality.

---

## 12.5 Why evaluation is task-dependent

A single metric cannot judge every sequence task.

Examples:

* sentiment classification → accuracy, F1
* NER → span F1
* language modeling → perplexity
* generation tasks → BLEU-like metrics, human judgment

Even within the same task, metric choice matters.

---

# 13) Practical Applications of Sequence Models

## 13.1 Sentiment analysis

RNNs/LSTMs/GRUs can model:

* negation
* contrast
* emphasis
* context-dependent polarity

## 13.2 Topic classification

Useful when class depends on sequence patterns and not just keywords.

## 13.3 Sequence tagging

Common in:

* POS tagging
* NER
* chunking

## 13.4 Dialogue and intent modeling

Utterance order matters heavily.

## 13.5 Time-sensitive text and event streams

Useful for logs, user interactions, and temporal text.

---

# 14) Engineering Reality: Strengths and Limitations of Sequence Models

## Strengths

* handle ordered input naturally
* model context over time
* work well on many NLP sequence tasks
* learn representations automatically

## Limitations

* RNNs are sequential, so training is hard to parallelize
* long-range memory is difficult
* gradients can vanish or explode
* teacher forcing causes exposure bias
* TBPTT cuts long-term credit assignment
* fixed hidden state can become a bottleneck

---

# 15) Concept Map

* **Sequential text data**: text as ordered token streams
* **RNN**: recurrent hidden state over time
* **LSTM**: gated memory with cell state
* **GRU**: simplified gated recurrent architecture
* **Bidirectional RNN**: processes both past and future context
* **Sentiment classification**: many-to-one sequence understanding
* **Text classification**: broader classification of text sequences
* **Teacher forcing**: training with true previous tokens
* **TBPTT**: gradient truncation for long sequences
* **Evaluation metrics**: accuracy, precision, recall, F1, perplexity

---

# 16) End-of-Unit NLP Master Challenge

## Problem 1

A vanilla RNN is trained on long reviews, but it fails to remember a negation appearing near the beginning of the review.
Explain this failure using the idea of vanishing gradients and hidden-state bottlenecks.
Then explain why an LSTM would likely perform better.

## Problem 2

You are building a POS tagger for sentences.
Why is a bidirectional RNN more suitable than a unidirectional RNN for this task?
Explain what information each direction contributes.

## Problem 3

A sequence generator is trained with teacher forcing and works well during training, but during inference its outputs quickly become incoherent.
Explain exposure bias and why this mismatch occurs.

## Problem 4

You train a sentiment classifier and get 92% accuracy, but the dataset is 90% negative samples.
Is the model actually good?
Explain why accuracy alone is misleading and which metrics you should inspect instead.

---

When you are ready, I can teach **Unit IV** in the same deep, master-class style.
