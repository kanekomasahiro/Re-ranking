from collections import defaultdict
import numpy as np
import sys
sys.path.append('/home/kaneko/S2V/')
from scoring import Scoring
import pickle

n_best_size = 20
fw = open('re_ranking{}.txt'.format(n_best_size), 'w')


def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))


def viterbi(documents):
  model = Scoring()
  results = []
  num_doc = 0
  for document, original_document in documents:
    best_edge = defaultdict(lambda: ['NULL' for _ in range(n_best_size)])
    best_score = defaultdict(lambda: [0 for _ in range(n_best_size)])

    # forward
    for i, candidates in enumerate(document, 1):
      cur_nbest = candidates[0]
      pre_nbest = candidates[1]
      scores = np.array(model.inference(pre_nbest, cur_nbest))
      scores = -np.log(sigmoid(scores))
      scores += np.broadcast_to(np.array(best_score[i]), (len(best_score[i]), len(best_score[i]))).T.reshape(-1)
      edges = np.argmin(scores.reshape(len(best_score[i]), len(best_score[i])), axis=1)
      scores = np.min(scores.reshape(len(best_score[i]), len(best_score[i])), axis=1)
      best_edge[i] = edges
      best_score[i] = scores
    best_edge = dict(best_edge)
    best_score = dict(best_score)

    # backward
    best_path = []
    next_edge = np.argmin(np.array(best_score[len(document)]))
    best_path += [next_edge]
    for i in range(len(original_document))[:0:-1]:
      next_edge = best_edge[i][next_edge]
      best_path += [next_edge]
    best_path.reverse()
    results += [original_document[i][edge] for i, edge in enumerate(best_path)]
  return results


def main():
  fr = open('vitervi.data', 'rb')
  documents = pickle.load(fr)
  results = viterbi(documents)
  for result in results:
      fw.write(result+'\n')


if __name__=="__main__":
  main()
