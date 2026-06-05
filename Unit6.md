## Unit VI — Generative NLP and LLMs

This unit is about the most visible and most misunderstood part of modern NLP: **generation**.

In earlier units, the model predicted labels, tags, spans, or translations. Here, the model is asked to **produce text itself**. That sounds similar, but it is a much harder problem because the output space is open-ended, the target can be multi-modal, and the model can be fluent yet wrong.

The unit has six major ideas:

1. **Generative NLP models**
2. **Text generation strategies**
3. **Greedy search, beam search, top-k, nucleus sampling**
4. **Instruction-tuned and large language models**
5. **Model behaviors in summarization, dialogue generation, and reasoning**
6. **Evaluation metrics, perplexity, human judgment, explainability, hallucination**

---

# 1) Generative NLP Models

## Core intuition & linguistic context

Generative NLP models do not merely classify language; they **model the probability distribution over possible continuations** of text.

Instead of asking:

* “What label should this sentence get?”

they ask:

* “What should come next?”

That next step could be:

* a word,
* a phrase,
* a sentence,
* or an entire document.

This makes generative NLP relevant for:

* text completion,
* dialogue systems,
* summarization,
* translation,
* story generation,
* question answering,
* code generation,
* instruction following.

Language is inherently generative because humans continually compose novel sentences. A generative model tries to approximate that creative, conditional distribution.

---

## Architecture, math & mechanics

A language model defines a probability over token sequences:

[
P(x_1, x_2, \dots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_{<t})
]

This is the central factorization of autoregressive generation.

If conditioned on an input (c) such as a prompt, source sentence, or dialogue history:

[
P(y \mid c) = \prod_{t=1}^{T} P(y_t \mid y_{<t}, c)
]

The model is trained to maximize the likelihood of observed text or target outputs.

### What the model actually learns

A generative model learns:

* syntax,
* lexical patterns,
* stylistic tendencies,
* contextual dependencies,
* some world regularities,
* task-following behavior if fine-tuned appropriately.

A modern generative Transformer outputs a probability distribution over the vocabulary at each step:

[
\hat{\mathbf{p}}_t = \text{softmax}(W\mathbf{h}_t + b)
]

and generation proceeds by selecting the next token from that distribution.

---

## Engineering reality & pitfalls

Generative models are powerful because they are flexible, but that flexibility comes with risk.

They can:

* produce fluent text,
* imitate style,
* answer questions,
* summarize documents,
* generate explanations.

But they can also:

* hallucinate facts,
* ramble,
* repeat themselves,
* become unsafe,
* overfit prompt patterns,
* follow instructions incorrectly.

A key engineering lesson:

> Fluency is not correctness.

A model can sound excellent and still be wrong.

---

# 2) Text Generation Strategies

## Core intuition & linguistic context

Once the model predicts a probability distribution over the next token, we still need a rule for **choosing** the token.

That choice is not trivial. The highest-probability token is not always the best long-term choice, because generation is a sequential decision process. A locally optimal choice can lead to globally poor text.

Thus decoding is a balance between:

* determinism,
* diversity,
* coherence,
* factuality,
* creativity.

---

## Architecture, math & mechanics

At time step (t), the model gives:

[
P(x_t \mid x_{<t})
]

We must choose the next token (x_t) according to a decoding strategy.

The main strategies in this unit are:

* greedy search
* beam search
* top-k sampling
* nucleus sampling

These differ in how they trade off:

* probability maximization,
* exploration,
* diversity,
* computational cost.

---

# 3) Greedy Search

## Core intuition & linguistic context

Greedy search always picks the most probable next token at each step.

This is the simplest possible decoding rule:

> “Take the best local choice right now.”

It is attractive because it is fast and deterministic.

---

## Architecture, math & mechanics

At step (t), choose:

[
x_t = \arg\max_{v \in V} P(v \mid x_{<t})
]

where (V) is the vocabulary.

Then feed that chosen token back into the model and continue until:

* an end token is generated,
* a length limit is reached.

### Why it is greedy

It makes no lookahead beyond the current step. It does not consider whether the locally best token may hurt future generation.

---

## Engineering reality & pitfalls

### Advantages

* very fast
* simple
* deterministic
* easy to debug

### Weaknesses

* can get stuck in repetitive loops
* often produces bland text
* may miss globally better sequences
* may prematurely choose safe but unhelpful continuations

Greedy decoding is useful as a baseline, but it is usually too rigid for high-quality open-ended generation.

Example failure:

* the model keeps selecting safe high-probability phrases and produces generic or repetitive output.

---

# 4) Beam Search

## Core intuition & linguistic context

Beam search improves on greedy search by keeping multiple candidate continuations alive at each step.

Instead of committing to one path immediately, it explores several high-probability partial sequences.

This is especially useful in:

* machine translation,
* summarization,
* structured generation,
* any task where global sequence quality matters.

---

## Architecture, math & mechanics

Let beam width be (B).

At each time step:

1. Keep the top (B) partial sequences.
2. Expand each with all possible next tokens.
3. Score all expansions.
4. Keep the top (B) overall.
5. Repeat until completion.

### Sequence score

A candidate sequence (y = (y_1,\dots,y_T)) is often scored by log probability:

[
\log P(y \mid x) = \sum_{t=1}^{T} \log P(y_t \mid y_{<t}, x)
]

Because raw probabilities get very small as sequences grow, log-space is used.

### Length normalization

Beam search can favor shorter sequences because multiplying many probabilities makes long sequences look worse. So practical systems often use length normalization:

[
score(y) = \frac{1}{|y|^\alpha} \sum_{t=1}^{|y|} \log P(y_t \mid y_{<t}, x)
]

where (\alpha) is a tuning parameter.

---

## Why beam search helps

Beam search reduces the chance that a single early wrong choice destroys the entire sequence. It is a compromise between:

* exhaustive search, which is impossible,
* greedy search, which is too narrow.

---

## Engineering reality & pitfalls

### Strengths

* better than greedy for many seq2seq tasks
* explores multiple promising hypotheses
* improves translation and summarization in many settings

### Weaknesses

* more computationally expensive
* can still produce generic or repetitive output
* may over-prefer high-probability but dull sequences
* does not guarantee semantic correctness

In open-ended generation, beam search can sometimes make outputs less diverse and more conservative. In translation, that may be acceptable; in creative tasks, it may be undesirable.

---

# 5) Top-k Sampling

## Core intuition & linguistic context

Sampling-based decoding introduces randomness so the model can generate more diverse text.

Top-k sampling limits the next-token distribution to the (k) most probable tokens, then samples from that restricted set.

This helps avoid both:

* overly deterministic output,
* and extremely low-probability noise.

---

## Architecture, math & mechanics

At step (t):

1. Compute next-token probabilities.
2. Keep only the top (k) tokens.
3. Renormalize probabilities over those (k) tokens.
4. Sample from the filtered distribution.

Formally, if (V_k) is the set of top-(k) tokens, then:

[
P'(v) = \frac{P(v)}{\sum_{u \in V_k} P(u)} \quad \text{for } v \in V_k
]

and (P'(v)=0) for tokens outside (V_k).

Then sample:

[
x_t \sim P'
]

---

## Why top-k helps

It preserves some randomness while preventing the model from sampling from the entire long tail of the vocabulary.

This is useful when:

* you want diversity,
* but you do not want the model to wander into garbage tokens too easily.

---

## Engineering reality & pitfalls

### Strengths

* more diverse than greedy or beam search
* easy to implement
* useful for creative and conversational generation

### Weaknesses

* fixed (k) is not adaptive
* can still include poor tokens if the distribution is flat
* can be too restrictive if (k) is small
* can be too permissive if (k) is large

Top-k is a blunt but effective tool.

---

# 6) Nucleus Sampling

## Core intuition & linguistic context

Nucleus sampling, also called top-p sampling, is often better than top-k because it adapts to the uncertainty of the model.

Instead of keeping a fixed number of tokens, it keeps the smallest set of tokens whose cumulative probability exceeds a threshold (p).

This means:

* if the model is confident, only a few tokens are kept,
* if the model is uncertain, more tokens are allowed.

That makes the decoding behavior more adaptive.

---

## Architecture, math & mechanics

At step (t):

1. Sort tokens by descending probability.
2. Select the smallest set (S) such that:

[
\sum_{v \in S} P(v) \ge p
]

3. Renormalize over (S).
4. Sample from the filtered distribution.

So the candidate set size is dynamic.

---

## Why nucleus sampling is powerful

Language is not equally certain at every step. Sometimes the next word is obvious; sometimes many continuations are reasonable.

Nucleus sampling respects that variability better than fixed-k methods.

It often gives a better balance of:

* coherence,
* diversity,
* naturalness,
* novelty.

---

## Engineering reality & pitfalls

### Strengths

* adaptive
* strong diversity without full randomness
* often produces more natural open-ended text than beam search
* widely used in dialogue and LLM generation

### Weaknesses

* still stochastic, so outputs vary
* can generate incoherent text if the threshold is too loose
* needs tuning
* less deterministic for evaluation

Nucleus sampling is often preferred for chat and creative generation, while beam search is often preferred for translation-like tasks.

---

# 7) Decoding Strategy Comparison

## Greedy search

* fastest
* deterministic
* narrow
* often repetitive or overly conservative

## Beam search

* better global search
* deterministic
* more expensive
* may reduce diversity

## Top-k sampling

* stochastic
* diverse
* simple
* fixed-size probability filter

## Nucleus sampling

* stochastic
* adaptive
* often best practical diversity-quality tradeoff
* requires threshold tuning

A useful way to think about them:

* **Greedy**: “best immediate choice”
* **Beam**: “best few sequences so far”
* **Top-k**: “sample from the most likely k options”
* **Nucleus**: “sample from the probable mass, however many tokens that takes”

---

# 8) Instruction-Tuned Models

## Core intuition & linguistic context

Pretrained language models know a lot of language structure, but they do not automatically know how to follow human instructions well.

Instruction tuning adapts a model to respond appropriately to prompts like:

* “Summarize this”
* “Translate this sentence”
* “Explain this concept”
* “Write code to do X”
* “Answer in one paragraph”

This transforms a raw language model into a more usable assistant.

The key shift is from:

* general continuation
  to
* task-following behavior.

---

## Architecture, math & mechanics

Instruction tuning usually involves supervised fine-tuning on datasets of:

* instruction,
* input,
* output.

The model learns a mapping of the form:

[
P(\text{response} \mid \text{instruction}, \text{context})
]

The training objective is still generally next-token prediction, but now over instruction-response pairs.

If the input is:

* “Translate to French: The house is blue.”

the target is:

* “La maison est bleue.”

### Why this works

The model learns patterns of:

* obeying task phrasing,
* producing helpful formats,
* answering questions directly,
* following style constraints.

---

## Engineering reality & pitfalls

Instruction-tuned models are better at:

* chat,
* helpful explanation,
* formatting outputs,
* multi-step responses,
* task generalization.

But they can still fail by:

* refusing incorrectly,
* being overly verbose,
* hallucinating confidently,
* misunderstanding ambiguous prompts,
* following the wrong instruction in a multi-part prompt.

Instruction tuning improves behavior, but it does not guarantee truthfulness or reasoning correctness.

---

# 9) Large Language Models (LLMs)

## Core intuition & linguistic context

LLMs are very large pretrained language models, usually Transformer-based, trained on massive corpora and often instruction-tuned or aligned further for practical use.

They are not just bigger versions of earlier models. Their scale changes behavior:

* they become much better at in-context learning,
* few-shot prompting becomes effective,
* emergent capabilities appear,
* they can generalize across many tasks with little task-specific training.

The term “large” typically refers to:

* many parameters,
* large training data,
* high compute scale.

---

## Architecture, math & mechanics

An LLM may be:

* decoder-only,
* encoder-only,
* encoder-decoder,
* or a variant designed for scaling and efficiency.

Their training objective is generally one of:

* causal language modeling,
* masked denoising,
* seq2seq denoising,
* instruction-response tuning,
* reinforcement learning from human feedback or related alignment methods in later systems.

At runtime, the model takes a prompt and generates continuations token by token.

---

## Why scale matters

Large models tend to:

* represent more linguistic patterns,
* memorize more facts,
* solve more tasks via prompting,
* exhibit stronger generalization,
* support more nuanced style control.

But scale also introduces:

* greater training cost,
* inference cost,
* safety concerns,
* hallucination risk,
* bias amplification,
* evaluation complexity.

---

## Engineering reality & pitfalls

LLMs are useful, but not omniscient.

They can:

* sound authoritative without evidence,
* fabricate citations or facts,
* obey prompt instructions inconsistently,
* struggle with arithmetic or precise symbolic reasoning unless specifically supported.

They are powerful pattern learners, not guaranteed truth engines.

---

# 10) Model Behaviors in Summarization

## Core intuition & linguistic context

Summarization is one of the most important tests of LLM behavior because it requires balancing:

* compression,
* fidelity,
* coherence,
* salience,
* factual correctness.

A summary must not merely be short; it must preserve the main meaning.

---

## Typical behaviors

### Good summarization behavior

* identifies salient points
* compresses redundant detail
* preserves entities and relationships
* maintains coherent flow
* uses concise language

### Common failure modes

* omitting critical facts
* over-compressing nuance
* copying too much verbatim
* inventing unsupported facts
* flattening distinctions
* producing vague summaries

---

## Architecture, math & mechanics

A summarization model estimates:

[
P(\text{summary} \mid \text{document})
]

With LLMs, this may be done via:

* direct prompting,
* fine-tuned generation,
* instruction-tuned summarization behavior.

The model must learn what content is salient. In practice, salience is statistical and task-dependent, not explicitly encoded as a rule.

---

## Engineering reality & pitfalls

Summarization is especially vulnerable to hallucination because:

* the model may try to produce a coherent summary even when uncertain,
* compression encourages omission and paraphrase,
* missing one detail can distort the meaning,
* user expectation is often factual exactness.

A summary that is fluent but incorrect is worse than a rough summary that is faithful.

---

# 11) Model Behaviors in Dialogue Generation

## Core intuition & linguistic context

Dialogue generation is not just text generation. It requires:

* response relevance,
* persona consistency,
* conversational coherence,
* turn-taking,
* politeness,
* safety,
* context tracking.

The model must answer in a way that fits the dialogue state.

---

## Typical behaviors

### Good dialogue behavior

* responds directly to the user
* stays on topic
* maintains context across turns
* uses appropriate tone
* asks clarifying questions when needed
* gives useful follow-up content

### Failure modes

* generic responses like “I’m not sure”
* repetition
* topic drift
* hallucination
* over-refusal
* unsafe or inappropriate output
* forgetting earlier context

---

## Engineering reality & pitfalls

Dialogue is difficult because the correct response is often underdetermined:

* many valid replies exist,
* tone matters,
* context is incomplete,
* user intent may be ambiguous,
* safety constraints matter.

A dialogue model must balance:

* helpfulness,
* correctness,
* safety,
* brevity,
* creativity.

Sampling strategy matters a lot here:

* greedy decoding tends to be dull,
* beam search can be overly rigid,
* nucleus sampling often gives more natural responses.

---

# 12) Model Behaviors in Reasoning Tasks

## Core intuition & linguistic context

Reasoning tasks require more than fluent pattern completion. They ask the model to:

* infer,
* chain steps,
* apply rules,
* manipulate symbols,
* compare alternatives,
* maintain intermediate state.

Examples:

* math word problems
* logical deduction
* code reasoning
* multi-hop QA
* planning
* causal inference-style prompts

---

## Why reasoning is hard for LLMs

LLMs are trained to predict tokens, not to explicitly execute logic. They can still appear to reason because:

* training data contains many reasoning-like patterns,
* scale improves pattern abstraction,
* the model can imitate chain-of-thought structures.

But this is not the same as guaranteed symbolic correctness.

---

## Typical behaviors

### Strengths

* can solve many short reasoning problems
* can produce stepwise explanations
* can recognize common reasoning templates
* can use context examples effectively

### Weaknesses

* brittle on exact arithmetic
* sensitive to prompt phrasing
* may produce plausible but incorrect chains of thought
* can fail on compositional generalization
* may confuse correlation with inference

---

## Engineering reality & pitfalls

Reasoning is one of the clearest places where fluency and competence diverge.

A model may:

* write a convincing proof-like answer,
* but make a hidden logical error.

This is why reasoning evaluation must be rigorous and not rely solely on style.

---

# 13) Evaluation Metrics

Generative NLP evaluation is difficult because there is no single universally correct output in many tasks. A good metric must match the task and the failure mode you care about.

---

## 13.1 Accuracy and exact match

For tasks with a single acceptable answer, exact match is useful.

Examples:

* extraction QA
* constrained generation
* multiple-choice style tasks

Exact match checks whether the output string matches the reference exactly.

### Limitation

Too strict for open-ended tasks because paraphrases may be correct but counted wrong.

---

## 13.2 BLEU and ROUGE revisited

Although introduced earlier, they often appear in generative evaluation.

* **BLEU**: better suited to translation, n-gram precision, brevity penalty.
* **ROUGE**: better suited to summarization, n-gram recall, subsequence overlap.

### Limitation

Both are overlap-based. They do not fully measure:

* factual correctness,
* coherence,
* semantic equivalence,
* utility,
* safety.

---

## 13.3 Perplexity

### Core intuition

Perplexity measures how well the model predicts the next tokens in a dataset.

[
PPL = \exp\left(-\frac{1}{T}\sum_{t=1}^{T}\log P(x_t \mid x_{<t})\right)
]

Lower perplexity means better next-token prediction.

### Why useful

* language modeling quality
* comparison on held-out corpora
* training progress monitoring

### Limitation

A model can have low perplexity and still produce:

* dull text,
* factual errors,
* unsafe answers,
* poor reasoning,
* repetitive generations.

Perplexity is about predictive fit, not usefulness.

---

## 13.4 Human judgment

Human evaluation is still essential for generative models because humans care about:

* helpfulness,
* correctness,
* coherence,
* style,
* factuality,
* safety,
* relevance.

Human ratings may involve:

* fluency
* faithfulness
* adequacy
* relevance
* coherence
* harmlessness
* overall preference

### Limitations

* expensive
* slow
* subjective
* inconsistent across annotators
* hard to scale

Even so, for open-ended generation, human judgment is often the most meaningful evaluation signal.

---

## 13.5 Task-specific metrics

### Summarization

* ROUGE
* factuality measures
* human preference
* hallucination rate

### Dialogue

* human preference
* helpfulness
* safety
* diversity
* consistency

### Reasoning

* exact match
* accuracy
* step consistency
* robustness to paraphrase

### Generation quality

* diversity metrics
* repetition rates
* calibration-related analyses

There is no universal metric for all generative tasks.

---

# 14) Explainability in LLMs

## Core intuition & linguistic context

Explainability asks:

> Why did the model produce this output?

This matters for debugging, trust, safety, and scientific understanding.

For LLMs, explainability is especially hard because:

* the model has many layers and parameters,
* the computation is distributed,
* attention patterns alone do not fully explain outputs,
* emergent behaviors can be difficult to attribute.

---

## Forms of explainability

### 1. Attention visualization

Show which tokens attended to which others.

Useful for:

* rough inspection,
* alignment intuition,
* debugging.

But attention weights are not always faithful explanations.

### 2. Gradient-based attribution

Estimate which input tokens most influenced the output via gradients or saliency.

This gives some sense of token importance.

### 3. Probing and representation analysis

Train diagnostic classifiers on hidden states to see what information is encoded.

### 4. Counterfactual analysis

Change the prompt slightly and see how output changes.

This is often one of the most revealing methods.

### 5. Chain-of-thought style reasoning traces

Models may produce intermediate steps, but these are not always reliable explanations of internal computation. They may be helpful, but must be treated cautiously.

---

## Engineering reality & pitfalls

Explainability tools can mislead if interpreted too literally:

* attention may look meaningful but not be causal,
* saliency can be noisy,
* verbal explanations may be post-hoc rationalizations,
* hidden representations are distributed and nonlinear.

Still, explainability is useful for:

* safety audits,
* error analysis,
* bias inspection,
* hallucination diagnosis,
* prompt debugging.

---

# 15) Hallucination in LLMs

## Core intuition & linguistic context

Hallucination is when a model generates information that is:

* unsupported,
* incorrect,
* fabricated,
* or inconsistent with the prompt/source.

This is one of the most important practical problems in LLM deployment.

Examples:

* inventing a citation,
* fabricating a fact,
* misreporting a date,
* making up an entity,
* inventing a medical or legal claim.

The model can sound extremely confident while being wrong.

---

## Why hallucination happens

Hallucination arises because LLMs are trained to maximize text likelihood, not truth.

Key contributing factors:

* next-token prediction objective
* incomplete or conflicting training data
* prompt ambiguity
* generative pressure to always answer
* exposure to patterns of plausible-sounding text
* weak grounding in external reality
* decoding randomness
* overgeneralization

The model often prefers a fluent continuation over admitting uncertainty.

---

## Types of hallucination

### 1. Intrinsic hallucination

The output conflicts with the input source.

Example:

* summary changes a fact from the source.

### 2. Extrinsic hallucination

The output introduces new unsupported information.

Example:

* adds a detail not present in the source text.

### 3. Fabricated specificity

The model gives precise-looking but ungrounded details, such as fake references, dates, or names.

---

## Mitigation strategies

### Better prompting

* ask for source-grounded answers
* request uncertainty when unsure
* constrain output format

### Retrieval augmentation

* fetch supporting documents before generation

### Fine-tuning

* train on grounded examples
* include refusal or uncertainty behavior

### Decoding control

* lower randomness for factual tasks
* use constrained generation when possible

### Verification

* post-generation fact checking
* citation validation
* consistency checks against source text

### Human oversight

* essential in high-stakes domains

---

## Engineering reality & pitfalls

Hallucination is not a rare edge case. It is a core modeling risk in generative NLP.

Important principle:

> A fluent answer is not automatically a reliable answer.

For high-stakes tasks like medicine, law, finance, or policy, hallucination control is as important as raw accuracy.

---

# 16) How the Pieces Fit Together

This unit forms a complete generative NLP pipeline:

1. **Generative models** define token sequence probabilities.
2. **Decoding strategies** choose how to sample or search those probabilities.
3. **Greedy search** is simple but narrow.
4. **Beam search** explores multiple promising continuations.
5. **Top-k** and **nucleus sampling** add controlled randomness for diverse generation.
6. **Instruction tuning** turns raw language models into more helpful assistants.
7. **LLMs** scale language modeling to large parameter and data regimes.
8. **Summarization, dialogue, and reasoning** show how generation behaves in practical applications.
9. **Perplexity, BLEU, ROUGE, and human judgment** evaluate different aspects of output quality.
10. **Explainability and hallucination analysis** are essential for trust and deployment.

The deepest lesson is this:

> Generation is not only about producing text.
> It is about balancing probability, diversity, grounding, usefulness, and safety.

That balance is the real art of modern NLP engineering.

---

# 17) Practical Engineering Guidance

## Choosing a decoding strategy

### Use greedy search when:

* you need speed,
* output must be deterministic,
* you are debugging.

### Use beam search when:

* the task is structured generation,
* faithfulness and likelihood matter,
* translation-like behavior is desired.

### Use top-k or nucleus sampling when:

* you want creative, diverse, human-like generation,
* you are building dialogue or open-ended generation systems.

A common rule of thumb:

* deterministic tasks → beam search or greedy
* creative/open-ended tasks → nucleus sampling

---

## Choosing an evaluation approach

### For factual tasks

Use:

* exact match
* human verification
* source-based factuality checks
* constrained metrics

### For translation

Use:

* BLEU
* human evaluation
* adequacy checks

### For summarization

Use:

* ROUGE
* factuality analysis
* human preference
* hallucination detection

### For dialogue

Use:

* human preference
* helpfulness and safety judgments
* conversation-level metrics
* qualitative inspection

---

## Working with LLMs responsibly

* do not treat fluent output as verified truth
* inspect high-stakes outputs manually
* prefer grounded generation when possible
* tune decoding for the task
* use retrieval or verification for factual domains
* be skeptical of perfect-sounding explanations

---

# 18) Common Mistakes and Failure Modes

1. **Using beam search for everything**
   Beam search can reduce diversity and make chat outputs bland.

2. **Using sampling for factual extraction tasks**
   Randomness is dangerous when exactness matters.

3. **Trusting perplexity as a full quality measure**
   It measures predictive fit, not usefulness or truth.

4. **Assuming instruction tuning guarantees correctness**
   It improves behavior, not omniscience.

5. **Treating model explanations as ground truth**
   Explanations may be post-hoc and incomplete.

6. **Ignoring hallucination in summarization**
   This is one of the most harmful failure modes in production systems.

7. **Forgetting that decoding changes behavior**
   The same model can behave very differently under greedy, beam, or sampling-based decoding.

---

# 19) Summary of the Unit

This unit taught the machinery and behavior of modern text generation.

* **Generative NLP models** model the probability of text continuations.
* **Decoding strategies** determine how outputs are selected from probability distributions.
* **Greedy search** is fast but narrow.
* **Beam search** explores multiple promising sequences.
* **Top-k** and **nucleus sampling** inject controlled randomness for diversity.
* **Instruction-tuned models** are adapted to follow human instructions.
* **LLMs** scale these ideas to large parameter counts and broad capabilities.
* LLMs behave differently in **summarization**, **dialogue**, and **reasoning** tasks.
* **Perplexity**, **BLEU**, **ROUGE**, and **human judgment** each measure different aspects of quality.
* **Explainability** helps inspect internal behavior, but it is imperfect.
* **Hallucination** remains one of the most important unresolved deployment challenges.

The core idea is simple but profound:

> Modern generative NLP is a probability engine trained on language, but deployed as an assistant, writer, reasoner, and sometimes as a knowledge interface.
> That makes calibration, grounding, and evaluation just as important as raw generation ability.

---

# 20) End-of-Unit NLP Master Challenge

## Problem 1: Decoding strategy selection

You are building:

1. a machine translation system,
2. a customer support chatbot,
3. a creative story generator.

Which decoding strategy would you choose for each: greedy, beam, top-k, or nucleus sampling? Justify each choice in terms of coherence, diversity, and factual reliability.

## Problem 2: Beam search pathology

A seq2seq model using beam search produces very short outputs that look incomplete.
Explain why beam search can favor short sequences and describe how length normalization addresses the issue.

## Problem 3: Perplexity vs usefulness

Model A has lower perplexity than Model B on a validation set, but human users prefer Model B’s responses.
Explain why this can happen and why perplexity alone is insufficient for generative evaluation.

## Problem 4: Hallucination diagnosis

A summarization model produces a fluent summary that changes one key numerical value from the source article.
Classify the hallucination type and propose two mitigation strategies.

## Problem 5: Instruction tuning behavior

A pretrained model can complete text well but performs poorly when asked, “Explain this in simple terms.” After instruction tuning, it improves.
Explain what changed in the training signal and why the model’s behavior changed even though the underlying architecture stayed the same.

## Problem 6: Reasoning reliability

An LLM gives a correct final answer on a reasoning problem but an incorrect intermediate explanation.
What does this tell you about explainability and chain-of-thought-style outputs?

---

If you want, I can now turn the **entire syllabus into a clean exam-prep package** with:

* unit-wise condensed revision notes,
* formula sheets,
* likely exam questions,
* and a 30-day study plan.
