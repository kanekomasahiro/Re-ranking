# NMT

## Key idea
Using Reinforcement learning and re-ranking on document-level NMT to consider target side context.

## Overview
Considering the cross-sentence information can help to resolve ambiguities and inconsistencies of translation results.
There are a lot of works to address context of source side so far.
On the other hand, there are few works to address target side context.
It is important to consider context of target side to keep the inconsistencies of the output.
However, a problem of exposure bias exists while addressing target side context.
So we propose a reinforcement learning model to address the exposure bias.
Additionally we propose a re-ranking method to keep inconsistencies of the output on document-level.

## Proposed model
We extend [1] model (Transformer) to get context of target as input.
We add the attention on previous target sentence to decoder of the model.


## Reinforcement learning
We use sentence-BLUE as reward to output sentence, and calculate discounted total reward.
Can we give the same sentence reward to words in a sentence?


## Re-ranking
We use Viterbi algorithm to optimally re-rank translated candidates on document-level. 
We use translated candidates as nodes and score path with quick thoughts [3].
We can score path considering a whole sentence.



## Data
- train
    - open-subtitle 2016 [8] (ge-en)
    - open-subtitle 2016 [2] (en-fr)　http://opus.nlpl.eu/OpenSubtitles2016.php
    - open-subtitle 2018 [1] (en-ru) http://opus.nlpl.eu/OpenSubtitles2018.php, http://data.statmt.org/acl18_contextnmt_data/
    - open-subtitle [4] (zh-en) https://github.com/longyuewangdcu/tvsub
    - LDC [4, 7] (zh-en)
    - IWSLT' 15 [5] (en-de, en-fr, zh-en)
- dev
    - open-subtitle 2016 [8] (ge-en)
    - open-subtitle [4] (zh-en) https://github.com/longyuewangdcu/tvsub
    - ted 10 [4] (zh-en)
    - NIST02 [4] (zh-en)
    - NIST05 [7] (zh-en)
    - IWSLT' 12 [5] (en-de, en-fr)
- test
    - open-subtitle 2016 [8] (ge-en)
    - open-subtitle [4] (zh-en) https://github.com/longyuewangdcu/tvsub
    - discourse test set (en-fr) https://github.com/rbawden/discourse-mt-test-sets
    - ted 10-13 [4] (zh-en)
    - NIST06, NIST08 [7] (zh-en)
    - NIST03-08 (zh-en)
    - IWSLT' 14 [5] (en-de, en-fr)

- other
	- ted https://wit3.fbk.eu/

[2] use OpenSubtitles2016. However, they did not explain how they split data to train, dev and test explicitly. 

## Experiment
We reimplement [1] model and compare it with proposed method.
Corpora settings as in [4], since its target language is English and we can use publicly available quick thoughts model for English.
We can use results from [4] as it is for comparison.

## Analysis
Comparison with the model [1] to verify the effect of considering the target context. 
By comparing the outputs of baseline and proposed models, we show that proposed model outputs' consistency increased. 
With ensemble vs without ensemble   
With reinforcement learning vs without reimbursement learning 




# Related Work of considering cross-sentence
1. Context-Aware Neural Machine Translation Learns Anaphora Resolution, Elena Voita et al, ACL, 2018
2. Evaluating Discourse Phenomena in Neural Machine Translation, Rachel Bawden et al, NAACL, 2018
3. AN EFFICIENT FRAMEWORK FOR LEARNING.SENTENCE REPRESENTATIONS, Lajanugen Logeswaran et al, ICLR, 2018
4. Learning to Remember Translation History with a Continuous Cache, Zhaopeng Tu et al, TACL, 2018
5. Does Neural Machine Translation Benefit from Larger Context?, Sebastien Jean et al, arXiv, 2017
6. Enabling Multi-Source Neural Machine Translation By Concatenating Source Sentences In Multiple Languages, Raj Dabre et al, arXiv, 2017
7. Exploiting Cross-Sentence Context for Neural Machine Translation, Longyue Wang et al, EMNLP, 2017
8. Neural Machine Translation with Extended Context, Jorg Tiedemann et al, DiscoMT, 2017
