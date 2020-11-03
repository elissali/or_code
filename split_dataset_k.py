import argparse
import logging
import random
import os
import re
from scipy import stats
from sklearn import mixture

import math
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from utils import mkdir_p
    # mkdir_p(path) creates directory if doesn't already exist


'''
tgrep.id = each sentence's unique id
Mean = mean strength rating of 'or' on a scale from 0-1
sentence_orig = the original sentence
Sentence_BNB = the original sentence with "but not both" inserted
'''

###### DISCRETE DISTRIBUTION ######


def split_k_fold(seed_num, save_path, splits=6):

    # this function doesn't return anything; it just takes the input file
    # and writes it into two separate train/test csv files (it will create those files)
    # and save them to save_path parameter

    logging.info('Splitting data into training/test sets\n========================')
    logging.info(f'Using random seed {seed_num}, file loaded from {input}')

    random.seed(seed_num)

    # 1. read file:
    input_f_1 = open(os.path.join(save_path, "train_db.csv"), "r")
    input_f_2 = open(os.path.join(save_path, "test_db.csv"), "r")

    lines = input_f_1.readlines()
    header_line = lines[0]
    lines = lines[1:]
    lines.extend(input_f_2.readlines()[1:])

    total_num_examples = len(lines)

    print("Length: ", total_num_examples)

    # shuffle
    ids = list(range(0, total_num_examples))
    random.shuffle(ids)
    k_fraction = int(len(ids) / splits)
    for j in range(splits):
     train_ids = ids[0:k_fraction*j] + ids[k_fraction*(j+1):]
     print("Len train ids:", len(train_ids))
     test_ids = ids[k_fraction*j:k_fraction*(j+1)]
     print("Len test ids:", len(test_ids))
     split_save_path = os.path.join(save_path, str(j))
     mkdir_p(split_save_path)
     with open(split_save_path + '/train_db.csv', 'w') as f:
       f.write(header_line)
       for i in train_ids:
           f.write(lines[i])

     with open(split_save_path + '/test_db.csv', 'w') as f:
       f.write(header_line)
       for i in test_ids:
           f.write(lines[i])


    return



def main():
    parser = argparse.ArgumentParser(
        description='Creat data splits ...')
    parser.add_argument('--seed', dest='seed', type=int, default=0)
    parser.add_argument('--path', dest='path', type=str, required=True)
    parser.add_argument('-k', dest='k', type=int, default=6)
    opt = parser.parse_args()
    split_k_fold(opt.seed, opt.path, splits=opt.k)

if __name__ == '__main__':
    main()
