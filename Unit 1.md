## Unit I — Foundations of NLP and Text Processing

Natural Language Processing (NLP) sits at the intersection of **linguistics, computer science, and machine learning**. The core problem is simple to state but hard to solve: **how can a machine understand, represent, and generate human language?** Language is messy. It is ambiguous, highly contextual, often ungrammatical, and full of variation. This unit builds the foundation for everything that comes later in deep learning for NLP.

---

# 1) Foundations of NLP

## 1.1 Origin of NLP

NLP did not begin with deep learning. It evolved through several major phases.

### A. Rule-based era

Early NLP systems were built using **hand-written rules** created by linguists and engineers.

**Idea:**

* If a word ends with “-ed”, maybe it is past tense.
* If a sentence matches a grammar pattern, parse it using that rule.
* If a phrase contains “not”, invert sentiment.

**Strengths**

* Transparent and interpretable.
* Good for controlled domains.

**Weaknesses**

* Extremely brittle.
* Hard to scale to real language.
* Cannot handle the full variety of human expression.

### B. Statistical NLP era

As digital text became available, NLP shifted toward **probabilistic models**.

**Idea:**

* Learn patterns from data instead of encoding every rule manually.
* Estimate probabilities of words, tags, parses, and sequences.

Examples:

* N-gram language models
* Hidden Markov Models
* Maximum Entropy models
* Conditional Random Fields

**Key change:** language was treated as something to be **learned from corpora**.

### C. Machine learning era

Feature-based machine learning improved NLP dramatically.

Instead of writing the full rule, engineers created features like:

* word identity
* suffixes
* capitalization
* surrounding words
* part-of-speech tags

Then models such as:

* Naive Bayes
* Logistic Regression
* SVMs
* CRFs

learned the mapping from features to labels.

### D. Deep learning era

Deep learning reduced manual feature design by learning representations automatically:

* Word embeddings
* RNNs / LSTMs / GRUs
* Seq2Seq + attention
* Transformers
* Large language models

**Key change:** language is now represented in dense vectors and learned end-to-end.

### E. Why this history matters

Knowing the origin of NLP helps you understand:

* why preprocessing exists,
* why sparse vectors were important,
* why embeddings replaced BoW,
* why transformers became dominant.

---

## 1.2 Language and Grammar

Language is not just a list of words. It is a structured system that operates at multiple levels.

### What is language?

Language is a **symbolic communication system** used by humans to express meaning, intention, and relationships.

Important properties:

* **Discrete**: words are distinct units
* **Compositional**: meaning comes from combining parts
* **Context-dependent**: meaning shifts with context
* **Productive**: infinite new sentences can be formed

### What is grammar?

Grammar is the set of rules that describe how a language is formed and interpreted.

Grammar includes:

* word formation rules
* sentence structure rules
* agreement rules
* ordering constraints

For example:

* “She eats apples” is grammatical in English.
* “She apples eats” is not.

### Grammar in NLP

Computers need grammar because raw text is just a sequence of symbols. Grammar helps determine:

* which sequences are valid,
* how words depend on each other,
* what roles words play in a sentence.

### Why grammar matters for NLP models

A model must often infer:

* who did what to whom,
* what modifies what,
* whether a sentence is a question, command, or statement,
* whether a word is noun/verb/adjective depending on context.

Without grammar, many tasks become ambiguous.

---

## 1.3 Linguistic Essentials

This unit is mainly about **morphology, syntax, and semantics**, but there are other related linguistic ideas that matter for NLP.

### 1. Word forms

A single root word may appear in many forms:

* run, runs, ran, running

A system must know whether these forms share meaning.

### 2. Context

Meaning depends on context:

* “bank” can mean a financial institution or a river side.
* “light” can mean not heavy or illumination.

### 3. Ambiguity

Language is full of ambiguity:

* lexical ambiguity: one word, many meanings
* syntactic ambiguity: one sentence, many parse structures
* semantic ambiguity: one sentence, multiple interpretations
* pragmatic ambiguity: implied meaning depends on situation

### 4. Annotation levels used in NLP

NLP datasets often annotate language at different levels:

* tokenization
* part-of-speech tagging
* named entities
* syntactic parse trees
* semantic roles
* sentiment labels

These annotations help models learn structure.

---

## 1.4 Morphology

Morphology is the study of **word structure**.

### Core idea

Words are made of smaller meaning-bearing units called **morphemes**.

Examples:

* un-happy
* teach-er
* walk-ed

### Types of morphemes

#### A. Free morphemes

Can stand alone as words.

* book
* run
* happy

#### B. Bound morphemes

Cannot stand alone.

* un-
* -ed
* -s

### Inflectional morphology

Inflection changes grammatical form without changing core meaning or part of speech.

Examples:

* cat → cats
* walk → walked
* go → goes

Used to mark:

* tense
* number
* person
* case
* comparative degree

### Derivational morphology

Derivation creates a new word, often changing meaning or part of speech.

Examples:

* teach → teacher
* happy → unhappy
* use → useful

### Why morphology matters in NLP

Morphology helps solve:

* sparsity: “run”, “running”, “ran” are related
* OOV handling: unseen word forms may still be analyzable by roots/affixes
* language modeling: inflected languages have many forms

### Engineering reality

In English, morphology is moderate; in languages like Turkish, Finnish, or Hindi, morphology can be much richer, which makes token-based processing harder.

### Common pitfalls

* Stemming may over-strip and destroy meaning.
* Lemmatization may need POS context.
* Morphological variants can be mistaken for unrelated words.

---

## 1.5 Syntax

Syntax is the study of **sentence structure**.

### Core idea

Syntax explains how words combine into phrases and sentences.

Example:

* The cat sat on the mat.

Here:

* “The cat” is a noun phrase.
* “sat on the mat” is a verb phrase.

### Why syntax matters

Two sentences may use the same words but have different structures and meanings:

* “The dog bit the man.”
* “The man bit the dog.”

Syntax helps identify grammatical relations.

### Constituency syntax

This views a sentence as nested phrases:

* sentence → noun phrase + verb phrase
* noun phrase → determiner + noun

This is often visualized using **parse trees**.

### Dependency syntax

This focuses on direct relationships between words:

* subject of verb
* object of verb
* modifier of noun

For example:

* “cat” is subject of “sat”
* “mat” is object of preposition “on”

### Syntactic ambiguity

A sentence can have multiple valid structures.

Example:

* “I saw the man with a telescope.”

Did I use the telescope, or did the man have it?
Both are possible parses.

### Engineering reality

Syntax is crucial in:

* parsing
* machine translation
* information extraction
* question answering

But syntax alone cannot solve meaning. A sentence can be grammatical and still nonsensical:

* “Colorless green ideas sleep furiously.”

That is syntactically valid but semantically odd.

---

## 1.6 Semantics

Semantics is the study of **meaning**.

### Core idea

Semantics asks:

* what does a word mean?
* what does a sentence mean?
* how do word meanings combine?

### Lexical semantics

Concerns meaning of individual words.
Examples:

* synonymy: big / large
* antonymy: hot / cold
* polysemy: bright = intelligent or full of light
* homonymy: bank = river bank / financial bank

### Compositional semantics

Meaning of a sentence comes from combining meanings of words and structure.

Example:

* “The boy chased the dog.”
  Meaning differs if roles are reversed.

### Sentence meaning vs speaker meaning

A sentence may literally mean one thing while implying another.

* “Can you pass the salt?” is literally a question about ability, but pragmatically a request.

### Why semantics matters

A system must go beyond counting words and understand:

* entailment
* contradiction
* paraphrase
* sentiment
* intent
* factual relations

### Engineering reality

Semantic understanding is hard because:

* the same word changes meaning by context,
* meaning depends on shared world knowledge,
* surface similarity does not imply semantic equivalence.

---

## 1.7 Challenges of NLP

This is one of the most important parts of the unit.

### 1. Ambiguity

Words and sentences often have multiple interpretations.

Examples:

* lexical ambiguity: “bat”
* syntactic ambiguity: attachment ambiguity
* semantic ambiguity: “visiting relatives can be annoying”

### 2. Sparsity

Language has a huge vocabulary space. Most words occur rarely.

Consequences:

* many words appear only a few times,
* sparse features are hard to learn,
* rare forms cause generalization problems.

### 3. Out-of-vocabulary (OOV) words

A model may encounter words not seen during training.

Examples:

* new names
* typos
* slang
* domain-specific terms

### 4. Context dependence

The same word can mean different things in different contexts.

Example:

* “apple” in “Apple released a new device” vs “I ate an apple”.

### 5. Variability in language

Humans do not write uniformly:

* typos
* abbreviations
* emojis
* code-mixing
* informal grammar
* sarcasm
* dialects

### 6. Long-range dependencies

Important information may appear far apart in a sentence or document.

Example:

* “The book that the professor who the student admired recommended was difficult.”
  This creates parsing and modeling difficulty.

### 7. Noisy data

Social media text, OCR output, chat logs, and speech transcripts are messy.

### 8. Multilinguality and code-switching

Real-world NLP often must handle:

* multiple languages
* mixed-language text
* translation issues
* script differences

### 9. World knowledge

Some meaning cannot be inferred from the text alone.

Example:

* “John dropped the glass. It shattered.”
  You need physical common sense to connect the second sentence.

### 10. Annotation limitations

Labeled data is expensive and often limited.
Model performance depends heavily on data quality.

---

## 1.8 Applications of NLP

NLP powers many real systems.

### 1. Text classification

Examples:

* sentiment analysis
* spam detection
* topic classification
* toxicity detection

### 2. Information extraction

Extract structured data from text:

* names
* dates
* organizations
* relations
* events

### 3. Machine translation

Convert text from one language to another.

### 4. Question answering

Answer questions from documents or knowledge bases.

### 5. Summarization

Condense long text into shorter text.

### 6. Chatbots and dialogue systems

Used in customer support, assistants, and conversational agents.

### 7. Search and retrieval

Search engines rely on NLP for:

* query understanding
* ranking
* semantic matching

### 8. Writing assistance

Grammar correction, autocomplete, style suggestions.

### 9. Healthcare and legal NLP

Extract insights from medical reports, case laws, and documents.

### 10. Speech-related NLP

Speech-to-text, text-to-speech, and spoken dialogue systems.

### Engineering reality

The same NLP methods may behave very differently across domains:

* a sentiment model trained on movie reviews may fail on finance text,
* a general tokenizer may struggle with biomedical abbreviations,
* a classifier may overfit to dataset-specific shortcuts.

---

# 2) Text Processing

Text processing is the transformation of raw text into a representation a model can use.

The key principle is:

> **Raw text is not directly understandable by machine learning algorithms.**
> It must first be cleaned, normalized, segmented, and encoded.

---

## 2.1 Tokenization

Tokenization is the process of splitting text into smaller units called **tokens**.

### Types of tokenization

#### A. Sentence tokenization

Splits a paragraph into sentences.

Example:

* “I like NLP. It is powerful.”
  → [“I like NLP.”, “It is powerful.”]

#### B. Word tokenization

Splits a sentence into words.

Example:

* “Deep learning is powerful.”
  → [“Deep”, “learning”, “is”, “powerful”]

#### C. Subword tokenization

Splits words into smaller pieces.

Example:

* “unhappiness” → [“un”, “happi”, “ness”] or other subword pieces

Used heavily in modern NLP because it helps with rare and unseen words.

### Why tokenization matters

Tokenization defines the units the model sees.
It affects:

* vocabulary size
* sequence length
* OOV handling
* representation quality

### Engineering pitfalls

* Punctuation can become separate tokens or remain attached.
* Contractions like “don’t” may split into “do” and “n’t”.
* Languages without spaces need different tokenization approaches.
* Incorrect tokenization can ruin downstream performance.

### Example

Text:

> “NLP is amazing, isn’t it?”

Possible tokens:

* [NLP, is, amazing, ,, isn, 't, it, ?]

Or:

* [NLP, is, amazing, isn’t, it, ?]

Both are valid depending on the tokenizer.

---

## 2.2 Stemming

Stemming reduces a word to a **stem**, often by removing suffixes or prefixes.

### Goal

Collapse different word forms into a simpler form.

Examples:

* connected → connect
* connection → connect
* studying → studi or study depending on stemmer
* fairly → fair

### How stemming works

Usually based on heuristic rules rather than deep linguistic analysis.

Common stemmers:

* Porter Stemmer
* Snowball Stemmer
* Lancaster Stemmer

### Strengths

* Simple
* Fast
* Useful for search and retrieval

### Weaknesses

* Can produce non-words
* Can over-stem or under-stem

Examples:

* “universal” and “university” may be reduced too aggressively by some stemmers.
* “studies” might become “studi”, which is not a real word.

### Engineering reality

Stemming is useful when exact linguistic correctness is not critical, but it can hurt tasks that need precise meaning.

---

## 2.3 Lemmatization

Lemmatization reduces a word to its **dictionary form** or **lemma**.

Examples:

* better → good
* running → run
* cars → car
* was → be

### Difference from stemming

| Stemming                     | Lemmatization                  |
| ---------------------------- | ------------------------------ |
| Rule-based heuristic         | Linguistically informed        |
| May output non-words         | Outputs valid dictionary forms |
| Faster                       | Usually slower                 |
| Less accurate linguistically | More accurate                  |

### Why lemmatization is better linguistically

It considers:

* morphology
* part of speech
* dictionary lookup

For example:

* “saw” as a noun vs “saw” as a verb require different lemmas.

### Engineering reality

Lemmatization is often preferred in tasks where linguistic normalization matters, but it may be slower and more complex than stemming.

---

## 2.4 Stop-word Removal

Stop words are very frequent words that often carry less content by themselves.

Examples:

* the
* is
* a
* an
* of
* to
* and

### Idea

Remove common function words to reduce noise and dimensionality.

### Why it helps

In bag-of-words style models, stop words can dominate counts without helping classification.

Example:

* “the”, “is”, “and” may appear in almost every document.

### When stop-word removal helps

* topic classification
* search indexing
* simple text mining

### When it can hurt

Some tasks need stop words:

* sentiment analysis: “not good” vs “good”
* question answering
* machine translation
* syntax-sensitive tasks

### Engineering reality

Do not blindly remove stop words.
In many modern models, stop words can carry important structure.

---

## 2.5 Punctuation Handling

Punctuation can be:

* removed,
* preserved,
* normalized,
* or turned into features.

### Why punctuation matters

Punctuation can express:

* sentence boundaries
* emphasis
* pauses
* emotion
* structure

Examples:

* “Great!” vs “Great.”
* “Wait… what?”
* “It’s good, not bad.”

### Common strategies

#### A. Remove punctuation

Useful for simple bag-of-words models.

#### B. Keep punctuation as tokens

Useful when punctuation adds meaning.

#### C. Normalize punctuation

Convert repeated punctuation:

* “!!!” → “!”
* “???” → “?”

### Engineering reality

In social media, punctuation is often informative. In formal documents, it may be less critical but still important for sentence splitting.

---

## 2.6 Handling Out-of-Vocabulary (OOV) Words

OOV words are words not present in the training vocabulary.

### Why OOV happens

* new words appear constantly
* typos
* names
* domain-specific terms
* slang
* inflectional variants

### Why OOV is a problem

Traditional models with fixed vocabularies cannot represent unseen words directly.

### Common strategies

#### A. Unknown token

Replace unknown words with a special token like `<UNK>`.

Pros:

* simple
* robust

Cons:

* loses information
* different unknown words collapse together

#### B. Subword tokenization

Break words into smaller pieces:

* WordPiece
* BPE
* unigram-based methods

This helps represent unseen words compositionally.

#### C. Character-level modeling

Use characters instead of whole words.

Helpful for:

* morphology
* typos
* agglutinative languages

#### D. Normalization

Convert variant forms to a canonical one:

* lowercasing
* stemming
* lemmatization
* spelling correction

### Engineering reality

OOV is one reason classical word-level NLP struggles. Subword models dramatically reduce this problem.

---

## 2.7 Normalization

Normalization converts text into a more consistent form.

### Common normalization steps

* lowercasing
* removing extra whitespace
* Unicode normalization
* expanding contractions
* removing URLs/emails or replacing them with placeholders
* standardizing numbers
* spelling correction
* accent normalization

### Example

Raw:

> “I’ve got 2   dogs!!!”

Normalized:

> “i have got <NUM> dogs !”

### Why normalization matters

It reduces unnecessary variability:

* “Apple” and “apple” may map to the same token.
* “co-operate” and “cooperate” may be unified.
* “1000” and “1,000” may be standardized.

### Pitfalls

Normalization can destroy useful distinctions:

* casing helps identify named entities
* numbers may be meaningful
* emojis may encode sentiment
* contractions may matter

### Engineering reality

Normalization is not always “remove everything noisy.”
It is about preserving useful signal while reducing accidental variation.

---

## 2.8 Bag-of-Words (BoW)

Bag-of-Words is a classical text representation method.

### Core intuition

A document is represented by the **counts of words it contains**, ignoring order.

Example:

* “I love NLP”
* “NLP love I”

Under BoW, these may have the same vector if the vocabulary is the same.

### Why it is called “bag”

A bag contains items without order.
BoW ignores syntax and sequence.

### Representation

Suppose vocabulary is:

* [I, love, NLP, deep]

Document: “I love NLP”

BoW vector:

* [1, 1, 1, 0]

### Matrix form

For a corpus with:

* (D) documents
* (V) vocabulary size

BoW becomes a matrix (X \in \mathbb{R}^{D \times V})

Each row = one document
Each column = one term

### Two common variants

#### A. Binary BoW

Use 0/1 to indicate presence/absence.

#### B. Count BoW

Use raw counts.

Example:

* “NLP is fun and NLP is useful”
  Counts:
* NLP = 2
* is = 2
* fun = 1
* and = 1
* useful = 1

### Strengths

* simple
* interpretable
* fast
* effective for many baseline tasks

### Weaknesses

* ignores word order
* ignores context
* high-dimensional and sparse
* cannot understand meaning or synonyms directly

### Engineering reality

BoW is often a strong baseline, but it cannot capture:

* negation properly,
* phrase meaning,
* semantics,
* long-distance dependencies.

---

## 2.9 N-grams

N-grams extend BoW by representing **contiguous sequences of (n) tokens**.

### Definition

An n-gram is a sequence of (n) consecutive tokens.

Examples:

* unigram: 1 word
* bigram: 2 words
* trigram: 3 words

Sentence:

> “deep learning is powerful”

Unigrams:

* deep, learning, is, powerful

Bigrams:

* deep learning
* learning is
* is powerful

Trigrams:

* deep learning is
* learning is powerful

### Why n-grams matter

They capture some local order information missing in BoW.

Example:

* “not good” is very different from “good”
* a unigram model may miss that distinction
* a bigram model can capture it

### Representation

You can build a feature vector over all n-grams in the vocabulary.

### Trade-offs

#### Benefits

* captures short phrase patterns
* useful for classification and language modeling baselines
* improves over simple BoW

#### Costs

* vocabulary grows quickly
* sparse features become worse
* longer n-grams are rarer

### Engineering reality

N-grams help with local context but do not solve long-range syntax or semantics. They are still sparse and combinatorial.

---

## 2.10 TF-IDF

TF-IDF is one of the most important classical text representations.

It measures how important a word is to a document relative to the whole corpus.

### Core intuition

A word is important if:

* it appears often in a given document,
* but not in many documents overall.

Words like “the” appear everywhere, so they are not very informative.

---

### A. Term Frequency (TF)

Measures how often a term appears in a document.

A simple form:
[
TF(t, d) = \frac{\text{count of term } t \text{ in document } d}{\text{total terms in } d}
]

Sometimes raw count is also used.

---

### B. Inverse Document Frequency (IDF)

Measures how rare a term is across the corpus.

A common form:
[
IDF(t) = \log \left(\frac{N}{df(t)}\right)
]
where:

* (N) = total number of documents
* (df(t)) = number of documents containing term (t)

If a term appears in many documents, its IDF is low.

Sometimes smoothing is used:
[
IDF(t) = \log \left(\frac{N + 1}{df(t) + 1}\right) + 1
]

---

### C. TF-IDF score

[
TFIDF(t, d) = TF(t, d) \times IDF(t)
]

This gives high weight to words that are:

* frequent in a document,
* but rare across the corpus.

### Example

Suppose:

* “NLP” appears many times in one specific article but not in others.
* TF is high, IDF is high.
* TF-IDF becomes large.

Suppose:

* “the” appears in almost every document.
* IDF is low.
* TF-IDF becomes small even if TF is high.

---

### Why TF-IDF is useful

It is better than raw counts when you want to emphasize discriminative words.

Used in:

* document classification
* information retrieval
* search ranking
* keyword extraction

### Strengths

* simple and strong baseline
* interpretable
* works well for sparse text classification

### Weaknesses

* ignores word order
* ignores deep semantics
* cannot directly model similarity between synonyms
* still sparse and high-dimensional

### Engineering reality

TF-IDF is excellent for many classical ML pipelines, but it is not a semantic representation. It treats “good” and “great” as unrelated unless they co-occur in similar documents.

---

# 3) Putting It All Together: A Classical NLP Pipeline

A typical classical NLP pipeline looks like this:

1. **Collect text**
2. **Normalize text**
3. **Tokenize**
4. **Optionally remove stop words / punctuation**
5. **Stem or lemmatize**
6. **Extract features**

   * BoW
   * n-grams
   * TF-IDF
7. **Train a model**

   * Naive Bayes
   * Logistic Regression
   * SVM
   * etc.

### Why this matters

This pipeline shows how raw language becomes vectors that machine learning can use.

### Limitation of classical pipelines

They rely heavily on:

* manual preprocessing,
* sparse features,
* weak semantic understanding.

This is why deep learning later became so important.

---

# 4) Summary of Key Conceptual Differences

## Morphology vs Syntax vs Semantics

* **Morphology**: internal structure of words
* **Syntax**: structure of sentences
* **Semantics**: meaning

## Stemming vs Lemmatization

* **Stemming**: crude reduction to stem
* **Lemmatization**: dictionary base form

## BoW vs N-grams vs TF-IDF

* **BoW**: word counts, ignores order
* **N-grams**: captures local order
* **TF-IDF**: weighted BoW emphasizing informative terms

## Tokenization vs Normalization

* **Tokenization**: splitting text into units
* **Normalization**: making text consistent

---

# 5) Engineering Pitfalls You Must Remember

### 1. Removing too much information

Aggressive preprocessing can destroy meaning.

Example:

* removing “not” breaks sentiment understanding.

### 2. Assuming word order does not matter

BoW ignores order, but order often changes meaning.

### 3. Over-trusting stemming

Stemmed forms may be unnatural or misleading.

### 4. Treating TF-IDF as semantics

TF-IDF is about importance, not meaning.

### 5. Ignoring domain dependence

A preprocessing pipeline that works for news may fail for tweets or biomedical text.

### 6. Not handling OOV properly

Without a good OOV strategy, the model breaks on unseen vocabulary.

---

# 6) End-of-Unit NLP Master Challenge

## Problem 1

You are building a spam classifier for short messages. The dataset contains many misspellings, emojis, and repeated punctuation like “FREEEEE!!!”.
Which preprocessing steps would you keep, modify, or avoid, and why?
Explain how tokenization, normalization, punctuation handling, and OOV strategy would affect performance.

## Problem 2

A sentiment model using BoW predicts that:

* “This movie is not good”
  and
* “This movie is good”
  have similar representations.
  Why does this happen mathematically, and how would adding bigrams change the feature space?

## Problem 3

You have two documents:

* Document A uses “purchase”, “refund”, “invoice”
* Document B uses “buy”, “return”, “bill”
  A BoW/TF-IDF model treats them as different.
  Explain why this is a limitation and which later NLP representation methods solve it better.

## Problem 4

A text classifier performs well on training data but fails on real-world user reviews because of slang, typos, and unseen product names.
Analyze the failure in terms of OOV words, tokenization, and normalization.
Propose a classical preprocessing pipeline that would reduce the issue, and explain what it still cannot solve.

---

When you are ready, I can teach **Unit II** with the same depth.
