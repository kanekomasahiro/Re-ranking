from collections import defaultdict
#from nltk import word_tokenize
import numpy as np
import sys
sys.path.append('/home/kaneko/S2V/')
from scoring import Scoring

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


def viterbi(document):
    model = Scoring()
    best_edge = defaultdict(lambda: ['NULL' for _ in range(len(document[0]))])
    best_score = defaultdict(lambda: [0 for _ in range(len(document[0]))])

    # forward
    for i, candidates in enumerate(document, 1):
        for j, candidate in enumerate(candidates):
            pre_sentences = document[i-1]
            cur_sentences = [candidate for _ in range(len(document[i-1]))]
            scores = np.array(model.inference(pre_sentences, cur_sentences))
            scores = sigmoid(scores)
            scores += np.array(best_score[i])
            best_edge[i][j] = np.argmax(scores)
            best_score[i][j] = np.max(scores)
            '''
            for k in range(len(candidates)):
                score = best_score[i-1][k] + model.inference([document[i-1][k]], [candidate])
                if score > best_score[i][j]:
                    best_edge[i][j] = k
                    best_score[i][j] = score
            '''
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
    document = []
    for l in open('gec.dev.en.50best.translation'):
        document.append([sentence.strip() for sentence in l.split('\t')])
        #document = [' '.join(word_tokenize(word)) for l in l.split()]
    results = viterbi(document)
    for result in results:
        fw.write(result+'\n')

if __name__=="__main__":
    main()
