from collections import defaultdict
#from nltk import word_tokenize
import numpy as np
import sys
sys.path.append('/home/kaneko/S2V/')
#from scoring import Scoring
import pickle

fw = open('re_ranking.txt', 'w')

document = [['I have a pen .', 'You are a pen .', 'I think your pen .'],
            ['I like this pen .', 'I like this pan .', 'You like this pen .'],
            ['I like this pen .', 'I like this pan .', 'You like this pen .'],
            ['I like this pen .', 'I like this pan .', 'You like this pen .'],
            ['I like this pen .', 'I like this pan .', 'You like this pen .'],
            ['I like this pen .', 'I like this pan .', 'You like this pen .'],
            ['I like this pen .', 'I like this pan .', 'You like this pen .'],
            ['I like this pen .', 'I like this pan .', 'You like this pen .'],
            ['I have a pen .', 'You are a pen .', 'I think your pen .']]


def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))


def viterbi(documents):
  model = Scoring()
  for document in documents:
    best_edge = defaultdict(lambda: ['NULL' for _ in range(len(document[0]))])
    best_score = defaultdict(lambda: [0 for _ in range(len(document[0]))])

    # forward
    for i, candidates in enumerate(document, 1):
      cur_nbest = candidates[0]
      pre_nbest = candidates[1]
      scores = np.array(model.inference(pre_nbest, cur_nbest))
      scores = sigmoid(scores)
      scores += broadcast_to(np.array(best_score[i]), (len(best_score[i]), len(best_score[i]))).T.reshape(-1)
      best_edge[i][j] = np.argmax(scores.reshape(len(best_score[i]), len(best_score[i])), axis=1)
      best_score[i][j] = np.max(scores, axis=1)
  best_edge = dict(best_edge)
  best_score = dict(best_score)

  # backward
  best_path = []
  next_edge = np.argmax(np.array(best_score[len(document)]))
  best_path += [next_edge]
  for i in range(len(document))[:0:-1]:
    next_edge = best_edge[i][next_edge]
    best_path += [next_edge]
  best_path = [edge for edge in best_path[::-1]]
  results = [document[i][edge] for i, edge in enumerate(best_path)]

  return results


def main():
  fr = open('vitervi.data', 'rb')
  documents = pickle.load(fr)
  results = viterbi(documents)
  for result in results:
      fw.write(result+'\n')


if __name__=="__main__":
  main()
