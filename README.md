# Documentlevel re-ranking (DoRe) for NMT

This code is based on Lajanugen Logeswaran's TensorFlow [Quick-Thought Vectors](https://github.com/lajanugen/S2V) code.


## Who am I?
[Masahiro Kaneko](https://sites.google.com/view/masahirokaneko)


## Key idea
Re-ranking the output of neural machine translation (NMT) system by considering the context information on document-level using a reranker (DoRe) pre-trained by Quick-Thought and Viterbi algorithm.


## Overview
<div align="center">
<img src="/images/ambiguity.png" width="80%">
</div>
We propose a re-ranking method which effectively consider the context information by using transfer learning and Viterbi algorithm on low-resource document-level parallel corpus.
Context information is very important for some NLP tasks, because it has an effect on the ambiguity of the input sentence and the consistency of the output sentences.
On the other hand, previous document-level re-ranking methods can't effectively model sentence representation and relationship between each sentences.
Moreover, the best candidates are determined by looking at only the temporal relationship, and the best candidates are not necessarily selected when looking at the whole document.
Therefore, we acquire a more effective reranker by transfer-learning sentence vectors learned by considering relationship between sentences.
And we improve total quality on document-level by using Viterbi algorithm to select the best candidates in the final selecting process.
We propose the transfer learning using low-resource document-level parallel corpus.
Therefore, it becomes possible to translate by considering the context using low-resource document-level parallel corpora and high-resource document-level monolingual corpora of the target side.
This is efficient in cases where there exists fewer document-level parallel corpora than sentence-level parallel corpora.


## Flow of re-ranking using Viterbi algorithm
![vitebi](/images/viterbi.png "viterbi")
<div align="center">
<img src="/images/parts.png" width="50%">
</div>

1. Train the NMT model on sentence-level parallel corpus.
2. Generate N-best candidates using trained NMT model.
2. Score each pass between candidates (forward) using DoRe
3. Find the best combination by tracing the best path from the end (backward)


## Quick-Thought (QT)
![Quick-Thought](/images/QT.png "qt")
- negative example: NMT translation of source side of document-level parallel corpus
- positive example: original target side of document-level parallel corpus--

Our proposed reranker use quick-thoughts (QT) [1] to initialize the encoder of our proposed method.
The advantages of using QT to initialize are below:
- QT needs only document-level monolingual corpora to train, so we can get the benefits of the document-level monolingual corpora that exists on a large scale.
- QT is trained by modeling the relationship of adjacent sentences, so it is effective for re-ranking by relationship between each sentences.


## Architecture of Proposed model (DoRe)
![architecture of model](/images/ma.png "ma")

- Input: sentences translated by NMT
- Output: score considering the relationship between each sentences

## How to train Proposed model (DoRe)
![how to train model](/images/model.png "model")
Proposed method has two encoders and an output layer.
Each encoder is initialized using the parameters of encoder trained by Quick-Thought.
I train proposed model to distinguish negative examples from positive examples, where gold label for negative examples is 0 and gold label for positive examples is 1.
Here,
- negative example: NMT translation of source side of document-level parallel corpus
- positive example: original target side of document-level parallel corpus

We use Viterbi algorithm to optimally re-rank translated candidates on document-level.
We use translated candidates as nodes and score path using reranker.
We can score path considering a whole document.  
We use softmax function for training and sigmoid function for testing.


## Experiment setting
- Data
  - We use document-level [TED corpus](https://wit3.fbk.eu/) (Japanese->English) for experiments.
  - TED is document-level corpus, however I train NMT model on sentence-level to examine the effect of our proposed method.
  - We use TED dev set for transfer learning of DoRe. We split only 420 sentences and used it as dev set for transfer learning.
    - train: 200k
    - dev: 9.3k
    - test: 2.6k

- NMT model
  - transfomer: default setting of tensor2tensor, NMT trained on sentence-level
- Models
  - w/o re-ranking (baseline): 1-best
  - QT 20-best: Reranking the 20-best candidates using QT
  - DoRe 20-best: Reranking the 20-best candidates using proposed approach

## Experiment ([BLEU](https://en.wikipedia.org/wiki/BLEU))
| Model | BLEU |
----|---- 
| w/o re-ranking (baseline) | 12.85 |
| QT 20-best | 13.33 |
| DoRe 20-best | 13.44 |


## Future works
- Comparing with previous work [2]
- Apply the proposed method to other tasks


## Thank you
- Mentor: Hyeongseok Oh (Kakao)
- Supervisor: [Mamoru Komachi](http://cl.sd.tmu.ac.jp/~komachi/index.en.html) (Tokyo Metropolitan University: TMU)
- [Aizhan Imankulova](https://atimankulova.wixsite.com/mysite) (Tokyo Metropolitan University: TMU)
- Wonchang Chung (Element AI)
- Raj Dabre (National Institute of Information and Communications Technology: NICT)
- My appreciation to camp participants and organizers


## Related Work
1. AN EFFICIENT FRAMEWORK FOR LEARNING SENTENCE REPRESENTATIONS, Lajanugen Logeswaran et al, ICLR, 2018
2. Document-level Re-ranking with Soft Lexical and Semantic Features for Statistical Machine Translation, Chenchen Ding et al, AMTA, 2014
