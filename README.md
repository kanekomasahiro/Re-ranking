# Document level re-ranking

## Key idea
Re-ranking output of neural machine translation system to consider context information using a reranker pre-trained by Quick-Thought and Viterbi algorithm.


## Overview
We propose a re-ranking method to consider context information without parallel corpus of the document-level.
Context information is very important for some NLP tasks.
Because it has an effect on the ambiguity of the input sentence and the consistency of the output.
On the other hand, previous document-level re-ranking methods can't effectively model sentence representation and relationship between each sentences.
Moreover, the best sentences are determined by looking at only the temporal relationship, and the best sentences is not necessarily selected when looking at the whole document.
Therefore, we acquire a more effective reranker by transfer-learning sentence vectors learned by considering relations between sentences.
And we improve total quality in document-level by using Viterbi algorithm to select the sentence in the final selecting process.
We propose the transfer learning without using the document-level parallel corpus.
Therefore, it becomes possible to translate the considering context using sentence-level parallel corpora and document-level monolingual corpora only.
This is efficient because there are fewer document-level parallel corpus than sentence-level parallel corpus.


## Proposed model
- Input: sentences translated by NMT
- Output: score considering the relationship between each sentences

Porposed method has some encoders (maybe 3 encoders) and output layer (maybe RNN).
Each encoder is initialized by the encoder which is trained by Quick-Thought.
I train proposed model to distinguish negative example from positive example, where gold label of negative example is 0 and gold label of positive example is 1.
- negative example: sentences translated by NMT using document-level japanese monolingual corpus
- positive example: sentences in document-level English monolingual corpus

We use Viterbi algorithm to optimally re-rank translated candidates on document-level.
We use translated candidates as nodes and score path with Quick-Thought [1].
We can score path considering a whole document.


## Data (ja-en)
- train
    - open-subtitle 2018 (2M) http://opus.nlpl.eu/OpenSubtitles2018.php
    - ted (0.2M) https://wit3.fbk.eu/

- dev
    - ted (9K)

- test
    - ted (2.6k)


## Experiment (BLEU)
- Train data (ted)
    - Baseline (transfomer): 13.84 (dev), 12.97 (test)
    - Previous method[2]:
    - Proposed method:
- Train data (open-subtitle)
    - Baseline (transfomer): comeing soon!
    - Previous method[2]:
    - Proposed method:

Previous method consider context by window and score by similarity of the bag-of-word vector


## Progresses
- Training NMT model
- Writing Viterbi algorithm.


## Problems
- I am writing code of reranker
    - My mentor teach me some tips!
- N-best outputs and greedy output of NMT are different


# Related Work
1. AN EFFICIENT FRAMEWORK FOR LEARNING.SENTENCE REPRESENTATIONS, Lajanugen Logeswaran et al, ICLR, 2018
2. Document-level Re-ranking
with Soft Lexical and Semantic Features for Statistical Machine Translation, Chenchen Ding et al, AMTA, 2014
