# Document level re-ranking for NMT

This code is based on Lajanugen Logeswaran's TensorFlow [Quick-Thought Vectors](https://github.com/lajanugen/S2V) code.


## Who am I
[Masahiro Kaneko](https://sites.google.com/view/masahirokaneko)


## Key idea
Re-ranking the output of neural machine translation (NMT) system by considering the context information using a reranker pre-trained by Quick-Thought and Viterbi algorithm.


## Overview
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
2. Score each pass between candidates (forward) using Reranker
3. 後ろから最適なパスをたどることでベストな組み合わせを見つける (backward)


## Architecture of Proposed model (Reranker)
![architecture of model](/images/ma.png "ma")


## How to train Proposed model (Reranker)
![how to train model](/images/model.png "model")


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
We use softmax function in training and sigmoid function in inference.


## Experiment setting
- We use [TED corpus](https://wit3.fbk.eu/) (Japanese-English) for experiments.
- TED train corpus is document corpus but I train MT model by sentence-level to check our proposed method effect.
- We use TED dev corpus for transfer learning of reranker. 420文だけ分割してdevとして使う
  - train: 0.2M
  - dev: 9.3k
  - test: 2.6k

- NMT model
  - transfomer: default setting of tensor2tensor

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
- Mentor: Hyeongseok Oh (Kakao Brain)
- Superviser: [Mamoru Komachi](http://cl.sd.tmu.ac.jp/~komachi/index.en.html) (Tokyo Metropolitan University: TMU)
- [Mamoru Komachi](https://atimankulova.wixsite.com/mysite) (Tokyo Metropolitan University: TMU)
- Wonchang Chung (Element AI)
- Raj Dabre (National Institute of Information and Communications Technology: NICT)



## Related Work
1. AN EFFICIENT FRAMEWORK FOR LEARNING.SENTENCE REPRESENTATIONS, Lajanugen Logeswaran et al, ICLR, 2018
