<<<<<<< HEAD
# Quick-Thought Vectors

This is a TensorFlow implementation accompanying our paper

Lajanugen Logeswaran, Honglak Lee, 
[An efficient framework for learning sentence representations](https://arxiv.org/pdf/1803.02893.pdf). In ICLR, 2018.

This codebase is based on Chris Shallue's [Tensorflow implementation](https://github.com/tensorflow/models/tree/master/research/skip_thoughts) of the SkipThought model. 
The data preparation, vocabulary expansion and evaluation scripts have been adopted with minor changes.
Other code files have been modified and re-structured with changes specific to our model.

### Contents
* [Model configuration files](#model-configuration-files)
* [Pretrained Models](#pretrained-models)
* [Training a Model](#training-a-model)
* [Evaluating a Model](#evaluating-a-model)

## Model configuration files

We use json configuration files to describe models. These configuration files provide a concise description of a model. They also make it easy to concatenate representations from different models/types of models at evaluation time.

The description of a sentence encoder has the following format.
```
{
        "encoder": "gru",                            # Type of encoder
        "encoder_dim": 1200,                         # Dimensionality of encoder
        "bidir": true,                               # Uni/bi directional
        "checkpoint_path": "",                       # Path to checkpoint
        "vocab_configs": [                           # Configuration of vocabulary/word embeddings
        {
                "mode": "trained",                   # Vocabulary mode: fixed/trained/expand
                "name": "word_embedding",
                "dim": 620,                          # Word embedding size
                "size": 50001,                       # Size of vocabulary
                "vocab_file": "BC_dictionary.txt",   # Dictionary file
                "embs_file": ""                      # Provide external embeddings file
        }
        ]
}
```

Vocabulary mode can be one of *fixed*, *trained* or *expand*. These modes represent the following cases.
* *fixed* - Use fixed, pre-trained embeddings.
* *trained* - Train word embeddings from scratch. 
* *expand* - Use an expanded vocabulary. This mode is only used during evaluation on downstream tasks.

`checkpoint_path` and `vocab_file` have to be specified only for evaluation.

For concatenating representations from multiple sentence encoders at evaluation time, the json file can be a list of multiple encoder specifications. See `model_configs/BC/eval.json` for an example. 


## Pretrained Models
Models trained on the BookCorpus and UMBC datasets can be downloaded from [https://bit.ly/2uttm2j](https://bit.ly/2uttm2j).
These models are the multi-channel variations (MC-QT) discussed in the paper.
If you are interested in evaluating these models or using them in your tasks, jump to [Evaluation on downstream tasks](#evaluation-on-downstream-tasks).


## Training a Model

### Prepare the Training Data

The training script requires data to be in (sharded) TFRecord format. 
`scripts/data_prep.sh` can be used to generate these files.
The script requires a dictionary file and comma-separated paths to files containing tokenized sentences.
* The dictionary file should have a single word in each line. We assume that the first token ("\<unk>") represets OOV words.
* The data files are expected to have a tokenized sentence in each line, in the same order as the source document. 

The following datasets were used for training out models.
* [BookCorpus](http://yknzhu.wixsite.com/mbweb) 
* [UMBC](https://ebiquity.umbc.edu/blogger/2013/05/01/umbc-webbase-corpus-of-3b-english-words)

The dictionary files we used for training our models are available at [https://bit.ly/2G6E14q](https://bit.ly/2G6E14q).

### Run the Training Script

Use the `run.sh` script to train a model. 
The following variables have to be specified.

```
* DATA_DIR      # Path to TFRecord files
* RESULTS_HOME  # Directory to store results
* CFG           # Name of model configuration 
* MDL_CFGS      # Path to model configuration files
* GLOVE_PATH    # Path to GloVe dictionary and embeddings
```

Example configuration files are provided in the model\_configs folder. During training, model files will be stored under a directory named `$RESULTS_HOME/$CFG`.

### Training using pre-trained word embeddings

The implementation supports using fixed pre-trained GloVe word embeddings.
The code expects a numpy array file consisting of the GloVe word embeddings named `glove.840B.300d.npy` in the `$GLOVE_PATH` folder.

## Evaluating a Model

### Expanding the Vocabulary

Once the model is trained, the vocabulary used for training can be optionally expanded to a larger vocabulary using the technique proposed by the SkipThought paper. 
The `voc_exp.sh` script can be used to perform expansion. 
Since Word2Vec embeddings are used for expansion, you will have to download the Word2Vec model. 
You will also need the gensim library to run the script.

### Evaluation on downstream tasks

Use the `eval.sh` script for evaluation. The following variables need to be set.

```
* SKIPTHOUGHTS  # Path to SkipThoughts implementation
* DATA          # Data directory for downstream tasks
* TASK          # Name of the task
* MDLS_PATH     # Path to model files
* MDL_CFGS      # Path to model configuration files
* CFG           # Name of model configuration 
* GLOVE_PATH    # Path to GloVe dictionary and embeddings
```

Evaluation scripts for the downstream tasks from the authors of the SkipThought model are used. These scripts train a linear layer on top of the sentence embeddings for each task. 
You will need to clone or download the [skip-thoughts GitHub repository](https://github.com/ryankiros/skip-thoughts) by [ryankiros](https://github.com/ryankiros).
Set the `DATA` variable to the directory containing data for the downstream tasks. 
See the above repository for further details regarding downloading and setting up the data.

To evaluate the pre-trained models, set the directory variables appropriately.
Set `MDLS_PATH` to the directory of downloaded models.
Set the configuration variable `CFG` to one of 
* `MC-BC` (Multi-channel BookCorpus model) or 
* `MC-UMBC` (Multi-channel BookCorpus + UMBC model)

Set the `TASK` variable to the task of interest.

## Reference

If you found our code useful, please cite us [1](https://arxiv.org/pdf/1803.02893.pdf).

```
@inproceedings{
logeswaran2018an,
  title={An efficient framework for learning sentence representations},
  author={Lajanugen Logeswaran and Honglak Lee},
  booktitle={International Conference on Learning Representations},
  year={2018},
  url={https://openreview.net/forum?id=rJvJXZb0W},
}
```

Contact: [llajan@umich.edu](mailto:llajan@umich.edu)
=======
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
    - Baseline (transfomer): coming soon!
    - Previous method[2]:
    - Proposed method:

Previous method consider context by window and score by similarity of the bag-of-word vector


## Progresses
- Pre-trained Quick-Thought model is open to the public
- Training NMT model
- Writing Viterbi algorithm.


## Problems
- I am writing code of reranker (I'm not familiar with initializing the parts of the model)
- N-best outputs and greedy output of NMT are different


## Related Work
1. AN EFFICIENT FRAMEWORK FOR LEARNING.SENTENCE REPRESENTATIONS, Lajanugen Logeswaran et al, ICLR, 2018
2. Document-level Re-ranking
with Soft Lexical and Semantic Features for Statistical Machine Translation, Chenchen Ding et al, AMTA, 2014
>>>>>>> ba096e0dd072999c9bc8ac38936f0b6f5b04ee51
