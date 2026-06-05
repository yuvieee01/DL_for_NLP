## Unit IV — Sequence-to-Sequence Models and Attention Mechanisms

This unit solves one of the central problems in NLP:

> How do we map one variable-length sequence to another variable-length sequence when the output may depend on the entire input in complex, non-local ways?

That is the core job of **sequence-to-sequence (seq2seq)** modeling.

Examples:

* machine translation: English → French
* summarization: long article → short summary
* dialogue response generation: user utterance → reply
* text simplification: complex sentence → simpler sentence

The great leap from Unit III is that now the model is not merely classifying a sequence or labeling tokens. It is **generating an entire new sequence** conditioned on an input sequence.

---

# 1) Encoder–Decoder Architectures for NLP

## Core intuition & linguistic context

The encoder–decoder architecture separates the task into two phases:

1. **Encode** the input sequence into a latent representation.
2. **Decode** that representation into an output sequence.

This is useful whenever input and output lengths differ and alignments are not one-to-one.

For example, in translation:

* one English sentence may map to a longer or shorter French sentence,
* word order changes,
* some words are dropped or expanded,
* grammatical structure changes.

A simple classifier cannot do this. A seq2seq system can.

---

## Architecture, math & mechanics

Let the input be:

[
x = (x_1, x_2, \dots, x_T)
]

and the output be:

[
y = (y_1, y_2, \dots, y_U)
]

where (T) and (U) may differ.

### Encoder

The encoder reads the input sequence and produces hidden states:

[
\mathbf{h}_t = f(\mathbf{x}*t, \mathbf{h}*{t-1})
]

The encoder may output:

* the final hidden state (\mathbf{h}_T), or
* all hidden states (\mathbf{h}_1, \dots, \mathbf{h}_T)

In classical seq2seq, the final hidden state is treated as a compressed summary of the entire input.

### Decoder

The decoder generates output tokens one step at a time:

[
\mathbf{s}*u = g(\mathbf{y}*{u-1}, \mathbf{s}_{u-1}, \mathbf{c})
]

where:

* (\mathbf{s}_u) is the decoder hidden state,
* (\mathbf{c}) is the context vector from the encoder,
* (\mathbf{y}_{u-1}) is the previously generated token.

The probability of the next token is:

[
P(y_u \mid y_{<u}, x) = \text{softmax}(W\mathbf{s}_u + b)
]

The full conditional likelihood is:

[
P(y \mid x) = \prod_{u=1}^{U} P(y_u \mid y_{<u}, x)
]

Training usually minimizes negative log-likelihood:

[
L = -\sum_{u=1}^{U} \log P(y_u^{*} \mid y_{<u}^{*}, x)
]

where (y_u^{*}) is the ground-truth target token.

---

## The fixed-length bottleneck

In the original encoder-decoder design, the entire input sentence is compressed into one vector, often the encoder’s last hidden state.

That creates a **information bottleneck**:

* short sentences may be encoded adequately,
* long sentences lose detail,
* rare words and long-distance dependencies are hard to preserve.

This limitation is one of the main motivations for attention.

---

## Engineering reality & pitfalls

Encoder-decoder models are elegant but fragile when:

* input sequences are long,
* output needs fine-grained alignment,
* the decoder must reference specific source words,
* the latent vector compresses too much information.

In classical seq2seq, the decoder must rely heavily on a single summary vector. That is often too restrictive for natural language.

---

# 2) Sequence-to-Sequence Models for Machine Translation and Summarization

## 2.1 Machine Translation

### Core intuition & linguistic context

Machine translation requires mapping one linguistic system into another. This is not word substitution. The model must handle:

* word order differences,
* idioms,
* morphology,
* agreement,
* omitted subjects or pronouns,
* culture-specific phrasing.

Example:

* English: “I am reading a book.”
* Hindi may reorder the structure.
* French may require different agreement patterns.
* Some languages encode information with suffixes rather than separate words.

Translation is therefore both a sequence modeling and linguistic transformation problem.

---

### Architecture, math & mechanics

A seq2seq translation model learns:

[
P(\text{target sentence} \mid \text{source sentence})
]

The encoder processes the source sentence. The decoder generates the target sentence token by token.

Training uses **teacher forcing**, where the decoder receives the true previous target token during training.

For a source sentence (x) and target sentence (y):

[
P(y \mid x) = \prod_{u=1}^{U} P(y_u \mid y_{<u}, x)
]

The decoder typically starts from a special token like `<SOS>` and stops at `<EOS>`.

---

### Engineering reality & pitfalls

Translation is difficult because:

* the mapping is not word-to-word,
* a source token may align with multiple target tokens,
* one target token may depend on multiple source tokens,
* syntax and morphology can differ strongly between languages.

Classical seq2seq models fail especially on:

* long sentences,
* rare named entities,
* precise alignment,
* multiword expressions,
* reordering.

Attention is what made seq2seq translation truly competitive.

---

## 2.2 Summarization

### Core intuition & linguistic context

Summarization asks the model to compress a long source text into a shorter version that preserves meaning.

There are two broad forms:

* **Extractive summarization**: select important sentences or spans from the source.
* **Abstractive summarization**: generate a new summary in novel wording.

Seq2seq models are especially important for abstractive summarization because the output should not merely copy the input; it should rephrase and compress it.

---

### Architecture, math & mechanics

Given a document (x), the model generates a summary (y):

[
P(y \mid x) = \prod_{u=1}^{U} P(y_u \mid y_{<u}, x)
]

The encoder reads the full document. The decoder generates salient summary tokens.

Attention is essential here because the summary may need to refer to different parts of the source at different times:

* one sentence for the topic,
* another for the result,
* another for names or dates.

---

### Engineering reality & pitfalls

Summarization is harder than it looks because:

* the output must be concise,
* key facts must be preserved,
* unsupported hallucinated content must be avoided,
* source content may be long and multi-topic.

Common failure modes:

* over-compression,
* copying too much,
* factual hallucination,
* missing salient details,
* repetitive phrasing.

Classical seq2seq summarizers often struggle because the source-to-summary alignment is diffuse and non-local.

---

# 3) Attention in Deep NLP

## Core intuition & linguistic context

Attention was introduced to solve the bottleneck of fixed-length encoding.

The key idea:

> The decoder should not rely on one single summary vector. Instead, it should dynamically look back at different parts of the input when generating each output token.

This is much closer to how humans process translation or summarization:

* when translating a word, we attend to the relevant source words,
* when generating a summary phrase, we consult the relevant source segments.

Attention is a learned soft alignment mechanism.

---

## Architecture, math & mechanics

At each decoder step (u), attention computes a weighted combination of encoder hidden states:

[
\mathbf{c}*u = \sum*{t=1}^{T} \alpha_{u,t}\mathbf{h}_t
]

where:

* (\mathbf{h}_t) are encoder states,
* (\alpha_{u,t}) are attention weights,
* (\sum_t \alpha_{u,t} = 1).

The context vector (\mathbf{c}_u) is then used by the decoder to generate token (y_u).

### Why this helps

Instead of one fixed context vector for the entire sentence, the decoder gets a different context vector at each step:

* one for generating a subject,
* another for a verb,
* another for a named entity,
* another for a closing phrase.

This allows dynamic source access.

---

## Soft attention

### Core intuition

Soft attention assigns fractional weights to all source positions, rather than selecting one source position discretely.

If the decoder needs to focus mostly on one source word but partially on neighboring words, soft attention can express that naturally.

---

### Architecture, math & mechanics

Compute alignment scores (e_{u,t}) between the decoder state at step (u) and encoder state at time (t):

[
e_{u,t} = \text{score}(\mathbf{s}_{u-1}, \mathbf{h}_t)
]

Then normalize with softmax:

[
\alpha_{u,t} = \frac{\exp(e_{u,t})}{\sum_{k=1}^{T}\exp(e_{u,k})}
]

Then compute the context vector:

[
\mathbf{c}*u = \sum*{t=1}^{T}\alpha_{u,t}\mathbf{h}_t
]

This is differentiable end-to-end, which means the whole system can be trained with gradient descent.

---

### Engineering reality & pitfalls

Advantages:

* differentiable
* stable to train
* naturally handles soft alignments
* improves long-sequence performance dramatically

Limitations:

* computational cost scales with source length
* may diffuse attention too broadly
* not guaranteed to be interpretable in a strict sense
* attention weights are not always faithful explanations

Soft attention is a major advance, but it is still a learned statistical mechanism, not a perfect linguistic alignment engine.

---

# 4) Bahdanau Attention

Bahdanau attention is one of the first successful neural attention mechanisms for seq2seq.

It is often called **additive attention**.

## Core intuition & linguistic context

Bahdanau attention lets the decoder learn which encoder states matter for the current output token. It was especially important in neural machine translation because it removed the rigid bottleneck of fixed-vector seq2seq.

The decoder can now say, in effect:

* “For this output word, look at these source positions.”
* “For the next output word, look somewhere else.”

---

## Architecture, math & mechanics

Bahdanau attention computes an alignment score using a small feedforward neural network.

For decoder state (\mathbf{s}_{u-1}) and encoder state (\mathbf{h}_t):

[
e_{u,t} = \mathbf{v}^\top \tanh(W_s \mathbf{s}_{u-1} + W_h \mathbf{h}_t)
]

where:

* (W_s) and (W_h) are learned matrices,
* (\mathbf{v}) is a learned vector,
* (\tanh) introduces nonlinearity.

Then:

[
\alpha_{u,t} = \text{softmax}(e_{u,t})
]

and:

[
\mathbf{c}*u = \sum_t \alpha*{u,t}\mathbf{h}_t
]

This context vector is fed into the decoder.

### Why “additive” attention?

Because the decoder and encoder states are linearly transformed and added before applying nonlinearity.

---

## Strengths of Bahdanau attention

* handles rich nonlinear matching
* effective for translation and generation
* made seq2seq models much more usable

## Weaknesses

* computationally heavier than simpler score functions
* still requires computing alignment against all source positions
* not ideal for very long sequences without further optimization

---

# 5) Luong Attention

Luong attention is often called **multiplicative attention** or **dot-product style attention**.

It was designed to be simpler and faster than Bahdanau attention.

## Core intuition & linguistic context

Luong attention follows the same basic idea:

> The decoder should attend to relevant encoder states at each step.

The difference lies in how the alignment score is computed.

---

## Architecture, math & mechanics

Common Luong scoring functions include:

### 1. Dot score

[
e_{u,t} = \mathbf{s}_u^\top \mathbf{h}_t
]

### 2. General score

[
e_{u,t} = \mathbf{s}_u^\top W \mathbf{h}_t
]

### 3. Concat score

[
e_{u,t} = \mathbf{v}^\top \tanh(W[\mathbf{s}_u ; \mathbf{h}_t])
]

Then:
[
\alpha_{u,t} = \text{softmax}(e_{u,t})
]
[
\mathbf{c}*u = \sum_t \alpha*{u,t}\mathbf{h}_t
]

Luong attention also introduces the idea of combining the context vector with the decoder hidden state before output prediction.

A common formulation is:

[
\tilde{\mathbf{h}}_u = \tanh(W_c[\mathbf{c}_u ; \mathbf{s}_u])
]

then:

[
P(y_u \mid y_{<u}, x) = \text{softmax}(W_o \tilde{\mathbf{h}}_u)
]

---

## Bahdanau vs Luong

### Bahdanau

* additive attention
* score uses a small neural network
* often uses previous decoder state in scoring
* historically influential and flexible

### Luong

* multiplicative or dot-product attention
* simpler and faster
* often easier to compute efficiently

Both are soft attention mechanisms. The choice often comes down to architecture style and efficiency.

---

# 6) Integrating Attention into Encoder–Decoder Networks

## Core intuition & linguistic context

Attention is not a separate model. It is a **component** embedded inside the encoder-decoder architecture.

The decoder should not just get a fixed summary vector. It should get:

* its own recurrent state,
* a dynamic source-side context vector,
* optionally previous attention behavior.

This creates a much more expressive generation process.

---

## Architecture, math & mechanics

At decoder step (u):

1. Use previous decoder state and previous output token.
2. Compute attention over all encoder states.
3. Form context vector (\mathbf{c}_u).
4. Combine (\mathbf{c}_u) with decoder hidden state (\mathbf{s}_u).
5. Predict next token distribution.

A generic formulation:

[
\mathbf{s}*u = \text{RNN}(\mathbf{s}*{u-1}, \mathbf{y}*{u-1}, \mathbf{c}*{u-1})
]

[
\mathbf{c}*u = \sum_t \alpha*{u,t} \mathbf{h}_t
]

[
\mathbf{o}_u = \text{softmax}(W[\mathbf{s}_u ; \mathbf{c}_u] + b)
]

where (\mathbf{o}_u) is the output probability distribution over the vocabulary.

---

## Why integration matters

The decoder no longer has to “remember everything.”
It can:

* store local generation state,
* retrieve source information on demand,
* improve alignment,
* support copying and reordering behavior better than classical seq2seq.

This is especially important in translation and summarization, where the decoder often needs specific source details at different times.

---

## Engineering reality & pitfalls

Even with attention:

* the model may still hallucinate,
* attention can spread too thinly,
* long documents may still be challenging,
* source and target alignments may be noisy,
* repeated attention to the same source span can cause repetition in output.

Attention helps, but it is not a complete solution to generation reliability.

---

# 7) Evaluation Techniques

Evaluation for seq2seq systems is much harder than for classification because there are often multiple valid outputs.

For example:

* multiple translations can be correct,
* multiple summaries can be acceptable,
* wording may vary while meaning remains similar.

So evaluation must consider both automatic metrics and human judgment.

---

## Core evaluation dimensions

### 1. Fluency

Is the output grammatical and natural?

### 2. Adequacy

Does the output preserve the meaning of the input?

### 3. Fidelity

Does the output stay faithful to the source?

### 4. Relevance

Does the output include the most important content?

### 5. Coverage

Does it capture enough of the source?

### 6. Conciseness

For summarization, is it short without losing important information?

---

## Why evaluation is hard

A generated sentence can be:

* semantically correct but phrased differently from the reference,
* factually right but lexically different,
* partially correct,
* fluent but wrong,
* faithful but awkward.

This means simple exact-match metrics are usually insufficient.

---

## Human evaluation

Human judges can score:

* fluency
* relevance
* adequacy
* factual consistency
* coherence

Human evaluation is valuable because it directly assesses what automatic metrics may miss.

But it is:

* expensive,
* slow,
* subjective,
* difficult to scale.

Therefore, automatic metrics are used as proxies.

---

# 8) BLEU Score

BLEU is the canonical metric for machine translation evaluation.

## Core intuition & linguistic context

BLEU tries to measure how much the generated translation overlaps with one or more reference translations using n-gram precision.

The idea is:

* good translations should share many local n-grams with references,
* but they should not be rewarded for being excessively long,
* so a brevity penalty is added.

---

## Architecture, math & mechanics

BLEU computes modified n-gram precision for n = 1,2,3,4 typically.

For each n-gram order (n):

[
p_n = \frac{\sum_{\text{ngram} \in \text{candidate}} \text{clipped count}}{\sum_{\text{ngram} \in \text{candidate}} \text{count}}
]

Then BLEU is:

[
BLEU = BP \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)
]

where:

* (w_n) are weights, often uniform,
* (BP) is the brevity penalty.

### Brevity penalty

If the candidate translation is too short, BLEU penalizes it:

[
BP =
\begin{cases}
1 & \text{if } c > r \
\exp(1-r/c) & \text{if } c \le r
\end{cases}
]

where:

* (c) = candidate length
* (r) = reference length

This discourages trivial short outputs.

---

## Strengths of BLEU

* easy to compute
* widely used historically
* useful for comparing translation systems on the same dataset

## Weaknesses of BLEU

* n-gram overlap does not equal semantic correctness
* paraphrases may score poorly
* penalizes valid alternative phrasing
* insensitive to factual meaning if n-grams overlap superficially
* can be misleading when used alone

BLEU is a proxy, not truth.

---

# 9) ROUGE Score

ROUGE is commonly used for summarization evaluation.

## Core intuition & linguistic context

ROUGE emphasizes **recall of n-grams or sequences from the reference**. This is appropriate for summarization because a good summary should capture key content from the reference summary.

---

## Architecture, math & mechanics

Common ROUGE variants include:

### ROUGE-N

Measures overlap of n-grams between candidate and reference.

[
ROUGE\text{-}N = \frac{\text{overlap of n-grams}}{\text{total n-grams in reference}}
]

This is recall-oriented.

### ROUGE-L

Based on the **Longest Common Subsequence (LCS)** between candidate and reference. It measures how much of the reference sequence is captured in order, without requiring contiguity.

This is useful because summaries may paraphrase while preserving order partially.

### Other variants

* ROUGE-1: unigram overlap
* ROUGE-2: bigram overlap
* ROUGE-L: subsequence overlap

---

## Why ROUGE is used for summarization

Summaries should ideally preserve the important content of a reference summary, so recall matters. ROUGE rewards models that include key reference units.

---

## Weaknesses of ROUGE

* lexical overlap is not the same as semantic quality
* paraphrases can be under-rewarded
* factual errors may go unnoticed if n-grams overlap
* reference summaries are not unique, so a candidate may be good but score poorly

ROUGE is useful, but it is not a full measure of summary quality.

---

# 10) Limitations of Classical Seq2Seq Models

This is one of the most important sections in the unit.

Classical seq2seq models, even with RNNs and attention, have serious limitations compared with later transformer-based systems and modern large language models.

---

## 10.1 Fixed or weak memory in the encoder

Without attention, the encoder compresses too much into one vector. Even with attention, the recurrent encoder may struggle to preserve very long dependencies.

---

## 10.2 Sequential computation

RNN-based encoders and decoders process tokens step by step, which:

* limits parallelization,
* slows training,
* makes large-scale optimization harder.

---

## 10.3 Exposure bias

Because of teacher forcing, the model sees gold history during training but its own predictions during inference. Errors accumulate at test time.

---

## 10.4 Error propagation

One wrong early output can corrupt all later decoding steps. This is especially serious in translation and summarization.

---

## 10.5 Difficulty with long-range dependencies

Even with attention, recurrent models may struggle when the input is very long or the target depends on distant source positions.

---

## 10.6 Hallucination and faithfulness issues

In abstractive generation, the model may produce fluent but unsupported content.

This is especially problematic in summarization:

* the output sounds plausible,
* but may not be grounded in the source.

---

## 10.7 Weak global coherence

Generated sequences may:

* repeat themselves,
* lose topic consistency,
* drift off-topic,
* fail to maintain discourse structure.

---

## 10.8 Limited handling of multiple valid outputs

Seq2seq models are trained against one reference output, but there may be many valid ones. Metrics and training objectives may underrepresent this variability.

---

## 10.9 Vocabulary and OOV issues

Classical seq2seq systems often rely on fixed vocabularies, so rare words and named entities are difficult to generate precisely.

This was later improved with subword tokenization and copy mechanisms.

---

# 11) How the Pieces Fit Together

The unit has a clean progression:

1. **Encoder-decoder architecture** gives the basic input-to-output sequence mapping.
2. **Machine translation and summarization** show why this mapping matters.
3. **Attention** fixes the fixed-vector bottleneck.
4. **Soft attention** makes alignment differentiable and trainable.
5. **Bahdanau attention** provides additive learned alignment.
6. **Luong attention** provides a faster multiplicative alternative.
7. **Integration into seq2seq** turns attention into a full conditional generation system.
8. **BLEU and ROUGE** provide evaluation tools.
9. **Classical seq2seq limitations** explain why the field moved toward transformers and more advanced generative models.

The deep insight is this:

> Sequence generation is not just about remembering the input; it is about dynamically deciding what part of the input matters at each output step.

That is what attention operationalizes.

---

# 12) Practical Engineering Guidance

## Use encoder-decoder with attention when:

* you need a variable-length output sequence,
* the input-output alignment is nontrivial,
* you are doing translation, summarization, or structured generation.

## Choose Bahdanau attention when:

* you want a flexible, nonlinear attention scorer,
* training cost is acceptable,
* you want the classic additive formulation.

## Choose Luong attention when:

* you want a simpler, often faster attention computation,
* dot-product style scoring is sufficient.

## Evaluate translation with BLEU, but not only BLEU

Use BLEU for a rough automatic measure, but inspect:

* adequacy,
* fluency,
* human judgments,
* error categories.

## Evaluate summarization with ROUGE, but not only ROUGE

ROUGE is useful, but also examine:

* factual consistency,
* readability,
* coverage,
* hallucination rate.

---

# 13) Common Mistakes and Failure Modes

1. **Assuming attention is explanation**
   Attention weights are not always faithful explanations of model reasoning.

2. **Using BLEU or ROUGE as the sole metric**
   They do not capture all aspects of quality.

3. **Ignoring exposure bias**
   Good training loss does not guarantee good generation.

4. **Overlooking factuality**
   Especially in summarization, fluency can mask hallucination.

5. **Treating seq2seq as word substitution**
   Translation is not word replacement; it is structured re-expression.

6. **Ignoring the bottleneck in classical encoder-decoder models**
   Fixed-vector representations are often insufficient for long inputs.

---

# 14) Summary of the Unit

This unit covered the classic neural framework for conditional sequence generation.

* **Encoder-decoder architectures** map one sequence to another.
* **Machine translation** and **summarization** are canonical seq2seq tasks.
* **Attention** removes the fixed-vector bottleneck by letting the decoder consult source states dynamically.
* **Soft attention** is differentiable and trainable end-to-end.
* **Bahdanau attention** uses additive scoring.
* **Luong attention** uses multiplicative or dot-product scoring.
* Attention is integrated directly into encoder-decoder decoding.
* **BLEU** measures n-gram overlap for translation.
* **ROUGE** measures overlap, especially recall, for summarization.
* Classical seq2seq models are limited by sequential computation, exposure bias, compression bottlenecks, and faithfulness issues.

This unit is the conceptual predecessor to transformers. Transformers replace recurrence with self-attention, but the attention ideas introduced here remain foundational.

---

# 15) End-of-Unit NLP Master Challenge

## Problem 1: Fixed-vector bottleneck

A classical seq2seq model performs well on short translation pairs but fails badly on long sentences.
Explain why this happens in terms of encoder compression, and describe how attention solves the issue.

## Problem 2: Bahdanau vs Luong attention

You are designing a translation model and must choose between additive and dot-product attention.
Explain the mathematical difference between Bahdanau and Luong attention, and discuss one practical reason to choose each.

## Problem 3: BLEU interpretation

System A gets a higher BLEU score than System B on a translation benchmark, but human judges prefer System B because it is more fluent and semantically faithful.
How can this happen? What does this tell you about BLEU’s limitations?

## Problem 4: ROUGE and summarization

A summarization model copies long spans from the source and gets a strong ROUGE score, but the summary feels poor to humans.
Why can ROUGE reward this behavior, and what additional evaluation would you add?

## Problem 5: Teacher forcing and inference

A seq2seq model is trained with teacher forcing and appears to converge well. At inference, it begins producing repetitive or incoherent sequences.
Explain the train-test mismatch and name one decoding or training strategy that may reduce the problem.

---

When you are ready, send **Unit V** and I will continue with **Transformers and Pretrained Language Models** at the same depth.
