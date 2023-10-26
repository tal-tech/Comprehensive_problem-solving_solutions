import numpy as np
from math import log
import time
import ctypes
import sys
import os
sys.path.append(".")

import det_beam_search

def beam_search_decoder(data, k = 5,empty_index = 45):

   sequences = [[list(), 0.0]]

   # walk over each step in sequence

   for row in data:

       all_candidates = list()

       # expand each current candidate

       for i in range(len(sequences)):

           seq, score = sequences[i]

           for j in range(len(row)):
               seq_ = seq.copy()
               if seq and seq[-1] == j and j!=empty_index and seq[-1]!=empty_index:
                   if seq[-1] == empty_index:
                       seq_.remove(empty_index)
                   candidate = [seq_, score + (-log(row[j]))]
               else:
                   if seq and seq[-1] == empty_index:
                       seq_.remove(empty_index)
                   candidate = [seq_ + [j], score + (-log(row[j]))]



               all_candidates.append(candidate)

       # order all candidates by score

       ordered = sorted(all_candidates, key=lambda tup:tup[1])

       # select k best

       # sequences = ordered[:k]

       sequence_list = []
       sequence_temp = []
       for i,sequence in enumerate(ordered):
           if i == 0:
               sequence_list.append(sequence)
               sequence_temp.append(sequence[0])
           else:
               if sequence[0] not in sequence_temp:
                   sequence_list.append(sequence)
                   sequence_temp.append(sequence[0])
           if len(sequence_list) == k:
               break
       sequences = sequence_list

   return sequences


def save_data_file(data):
    f = open('./test.txt', 'w+')
    for image in data:
        s = ''
        for j in range(image.shape[0]):
            s += str(image[j])
            s += ' '

        f.writelines(s)
        f.writelines("\n")


def test_with_python(data):
    for image in data:
        time_start = time.time()
        sequences = beam_search_decoder(image, 5, 45)
        #print(sequences)
        print('cpp run cost', (time.time() - time_start))


def test_with_vec(data):
    for image in data:
        float_vec_vec = det_beam_search.float_vec_vec()
        for row in image:
            float_vec = det_beam_search.float_vec()
            for col in row:
                float_vec.append(float(col))
            float_vec_vec.append(float_vec)
        time_start = time.time()
        det_beam_search.detection1(float_vec_vec, 5, 45)
        print('cpp run cost', (time.time() - time_start))


def test_with_numpy(data):
    image0 = data[0]
    for image in data:
        time_start = time.time()
        result = np.zeros((5, 46))
        det_beam_search.detection(image0, result, 5, 45)
        print('cpp run cost', (time.time() - time_start))


if __name__ == '__main__':
    data = np.load('logits_normal.npy')
    while True:
        test_with_numpy(data)
        # test_with_python(data)
        #test_with_vec(data)


