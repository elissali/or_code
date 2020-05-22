import argparse
import logging
import random
import os
import re

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


def get_distrib(array, buckets):
    distrib = np.zeros(buckets)
    tot_ratings = len(array)
    for i in range(buckets):
        bucket = i+1
        distrib[i] = array.count(bucket) / tot_ratings
    return distrib

def get_distrib_dict(ratings_list, buckets):
    distrib_dict = dict()
    for tgrep in ratings_list.keys():
        distrib_dict[tgrep] = get_distrib(ratings_list[tgrep], buckets)
    return distrib_dict



def split_train_test(seed_num, save_path, input = './data_2.csv', buckets = 7, test_pct = 0.3):

    # this function doesn't return anything; it just takes the input file
    # and writes it into two separate train/test csv files (it will create those files)
    # and save them to save_path parameter

    logging.info('Spitting data into training/test sets\n========================')
    logging.info(f'Using random seed {seed_num}, file loaded from {input}')

    random.seed(seed_num)   

    # 1. read file:
    input_df = pd.read_csv(input, sep=',')
    input_df['response_val'] = (input_df['response_val'] * buckets).apply(np.ceil)      # discretize raw ratings
    ratings_list = input_df.groupby('tgrep.id')['response_val'].apply(list)
    dict_sentence_rating = get_distrib_dict(ratings_list, buckets)
        # this is a dict mapping tgrep.id to 7-dim vector distribution of strength ratings
    intermed = input_df.groupby('tgrep.id')['sentence_bnb'].first()
    dict_id_to_sentence = intermed.groupby('tgrep.id').apply(list).to_dict()
        # this maps tgrep.id to sentence string in a list
    assert len(dict_sentence_rating) == len(dict_id_to_sentence)

    big_list = []
    for (key, val) in dict_id_to_sentence.items():
        sentence_str = re.sub(" but not both", "", val[0])      # the sentence string
        values_strength = dict_sentence_rating[key]             # the distribution of ratings (7-dim vec)
        example = key + ',' + format(values_strength).replace('\n', '') + ',' + '"' + format(sentence_str)    + '"'
        big_list.append(example)
            # big_list is a list of strings formatted: 'tgrep.id, distrib, sentence'
            # ['100501:68,[0, 0.33, ... 0.11],blah blah', '100564:48, [0.33, 0.2, ... 0.33],blah blah', ...] 

    # 2. split dataset into test and training

    num_examples = len(big_list)
    num_train = math.ceil((1-test_pct) * num_examples)  # number of examples for training

    ids = list(range(0, num_examples))
    random.shuffle(ids)             # shuffle them! 
    train_ids = ids[:num_train]     # training examples = the first num_train number of examples from big_list, by index in big_list
    test_ids = ids[num_train:]      # testing examples = what's left over, by index in big_list
    

    mkdir_p(save_path)
    head_line = "Item,Distrib,Sentence\n"                   # set the header
    f = open(save_path + '/train_db.csv', 'w')  # creates an empty /train_db.csv file at this path
    f.write(head_line)
    for i in train_ids:
        f.write(big_list[i] + "\n")              # each new line in /train_db.csv will be each key/val pair in big_list
    f.close()

    f = open(save_path + '/test_db.csv', 'w')  # do the same thing for test_db.csv
    f.write(head_line)
    for i in test_ids:
        f.write(big_list[i] + "\n")
    f.close()
    
    return



def split_k_fold(seed_num, save_path, splits=6, input='./data_2.csv', buckets=7):
    logging.info(f'Splitting data into {splits} training/test splits\n========================')
    logging.info(f'Using random seed {seed_num}, file loaded from {input}')

    random.seed(seed_num)

    # 1. read the file (all of this is same as in split_train_test):
    # all of this is the same as in split_train_test
    input_df = pd.read_csv(input, sep=',')
    input_df['response_val'] = (input_df['response_val'] * buckets).apply(np.ceil)      # discretize raw ratings
    ratings_list = input_df.groupby('tgrep.id')['response_val'].apply(list)
    dict_sentence_rating = get_distrib_dict(ratings_list, buckets)
    intermed = input_df.groupby('tgrep.id')['sentence_bnb'].first()
    dict_id_to_sentence = intermed.groupby('tgrep.id').apply(list).to_dict()
    assert len(dict_sentence_rating) == len(dict_id_to_sentence)

    big_list = []
    for (key, val) in dict_id_to_sentence.items():
        sentence_str = re.sub(" but not both", "", val[0])      # the sentence string
        values_strength = dict_sentence_rating[key]             # the distribution of ratings (7-dim vec)
        example = key + ',' + format(values_strength).replace('\n', '') + ',' + '"' + format(sentence_str)    + '"'
        big_list.append(example)

    # 2. split k_fold, and for each k, split into test/train:
    num_examples = len(big_list)
    ids = list(range(0, num_examples))
    random.shuffle(ids)
    k_fraction = int(len(ids) / splits)
    for j in range(splits):
        train_ids = ids[0:k_fraction * j] + ids[k_fraction * (j+1):]
        test_ids = ids[k_fraction * j : k_fraction * (j+1)]
        print("-----------new fold: first 5 test_ids ", test_ids[:5])

        # 3. write out train_db and test_db for each of the splits:
        split_save_path = os.path.join(save_path, str(j))
        mkdir_p(split_save_path)
        head_line = "Item,Distrib,Sentence\n"

        with open(split_save_path + '/train_db.csv', 'w') as f:     # write train_db
            f.write(head_line)
            for i in train_ids:
                f.write(big_list[i]+"\n")
    
        with open(split_save_path + '/test_db.csv', 'w') as f:      # write test_db
            f.write(head_line)
            for i in test_ids:
                f.write(big_list[i]+"\n")
    return


def k_folds_idx(k, num_examples, seed_num):
    all_inds = list(range(num_examples))
    cv = KFold(n_splits = k, shuffle = True, random_state = seed_num)
    return cv.split(all_inds)



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
    