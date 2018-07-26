# Document level re-ranking for NMT

This code is based on Lajanugen Logeswaran's TensorFlow [Quick-Thought Vectors](https://github.com/lajanugen/S2V) code.

## Key idea
Re-ranking output of neural machine translation system to consider context information using a reranker pre-trained by Quick-Thought and Viterbi algorithm.


## Overview
We propose a re-ranking method to consider context information effectively by using transfer learning and viterbi algorithm on few parallel corpus of the document-level.
Context information is very important for some NLP tasks, because it has an effect on the ambiguity of the input sentence and the consistency of the output.
On the other hand, previous document-level re-ranking methods can't effectively model sentence representation and relationship between each sentences.
Moreover, the best sentences are determined by looking at only the temporal relationship, and the best sentences are not necessarily selected when looking at the whole document.
Therefore, we acquire a more effective reranker by transfer-learning sentence vectors learned by considering relations between sentences.
And we improve total quality in document-level by using viterbi algorithm to select the sentence in the final selecting process.
We propose the transfer learning using few parallel corpora in document-level.
Therefore, it becomes possible to translate the considering context using only few document-level parallel corpora and document-level monolingual corpora of target side.
This is efficient because there are fewer document-level parallel corpus than sentence-level 


## Flow of re-ranking by using viterbi algorithm
![vitebi](/images/viterbi.png "viterbi")

1. Training the Machine Translation (MT) model by sentence-level parallel corpus and MT model generates N-best candidates
2. Reranker scores each pass made by candidates (forward)
3. 後ろから最適なパスをたどることでベストな組み合わせを見つける (backward)


## Architecture of Proposed model (Reranker)
![architecture of model](/images/ma.png "ma")

Porposed method has two encoders and output layer.

## How to train Proposed model (Reranker)
![how to train model](/images/model.png "model")


## Proposed model (Reranker)
- Input: sentences translated by NMT
- Output: score considering the relationship between each sentences

Porposed method has two encoders and output layer.
Each encoder is initialized by the encoder which is trained by Quick-Thought.
I train proposed model to distinguish negative example from positive example, where gold label of negative example is 0 and gold label of positive example is 1.
- negative example: sentences translated by NMT using document-level japanese parallel corpus
- positive example: sentences in document-level English parallel corpus

We use Viterbi algorithm to optimally re-rank translated candidates on document-level.
We use translated candidates as nodes and score path with Quick-Thought.
We can score path considering a whole document.


## Experiment setting
- We use TED corpus (ja-en) for experiments. (https://wit3.fbk.eu/)
- TED train corpus is document corpus but I train MT model by sentence-level to check our proposed method effect.
- We use TED dev corpus for transfer learning of reranker. 500文だけ分割してdevとして使う
  - train: 0.2M
  - dev: 9k
  - test: 2.6k

- NMT model
  - transfomer

## Experiment (BLEU)
| Model | BLEU |
----|---- 
| w/o re-ranking (baseline) | 12.85 |
| QT 20-best | 13.33 |
| TQT 20-best | 13.44 |


## Future works
- Comparing previous work
- 他のタスクに適用してみる


## Thank you
- Mentor: 


## Related Work
1. AN EFFICIENT FRAMEWORK FOR LEARNING.SENTENCE REPRESENTATIONS, Lajanugen Logeswaran et al, ICLR, 2018
