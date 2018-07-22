from collections import defaultdict
#from nltk import word_tokenize
import numpy as np
import sys
sys.path.append('/home/kaneko/S2V/')
#from scoring import Scoring
import pickle

fw = open('vitervi.data', 'wb')


def make_lattice(cur_nbest, pre_nbest):
  cur_l = []
  pre_l = []
  for sentence in cur_nbest:
    cur_l += [sentence for _ in range(len(cur_nbest))]

  pre_l = []
  for _ in range(len(pre_nbest)):
    pre_l += pre_nbest
  return cur_l, pre_l


def main():
  documents = []
  document = []
  for l, r in zip(open('/home/masahirokaneko/TED_data/test.en.50best.translation'), open('/home/masahirokaneko/TED_data/test.en')):
      if r.strip() == '<doc>':
          documents += [document]
          document = []
          continue
      document.append([sentence.strip() for sentence in l.split('\t')])
  documents += [document]
  documents.pop(0)

  data_dict = {}
  data = []
  for i, document in enumerate(documents):
    doc_data = []
    for nbest in document[1:]:
      cur, pre = make_lattice(nbest, document[i])
      doc_data += [(cur, pre)]
    data += [doc_data]
  pickle.dump(data, fw)

if __name__=="__main__":
  main()
