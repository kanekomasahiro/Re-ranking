from collections import defaultdict
#from nltk import word_tokenize
import sys
sys.path.append('/home/kaneko/S2V/')
#from scoring import Scoring
import pickle

fw = open('vitervi20.data', 'wb')


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
  for l, r in zip(open('../test.en.20best.translation'), open('../test.en')):
      if r.strip() == '<doc>':
          documents += [document]
          document = []
          continue
      document.append([unicode(sentence.strip(), 'utf-8') for sentence in l.split('\t')])
  documents += [document]
  documents.pop(0)

  data = []
  for i, document in enumerate(documents):
    #doc_cur_data = []
    #doc_pre_data = []
    doc_data = []
    for nbest in document[1:]:
      cur, pre = make_lattice(nbest, document[i])
      #doc_cur_data += cur
      #doc_pre_data += pre
      doc_data += [(cur, pre)]
    #data += [(doc_cur_data, doc_pre_data, document)]
    data += [(doc_data, document)]
  pickle.dump(data, fw)


if __name__=="__main__":
  main()
