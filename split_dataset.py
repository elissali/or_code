import argparse
import logging
import random
import os
import re
from scipy import stats 

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

    _ = input_df.groupby('tgrep.id')['sentence_bnb'].first()
    dict_id_to_sentence = _.groupby('tgrep.id').apply(list).to_dict()                       # {tgrep.id : sentence_str}
    dict_sentence_mean = input_df.groupby('tgrep.id')['response_val'].mean().to_dict()      # {tgrep.id : mean}
    dict_sentence_var = input_df.groupby('tgrep.id')['response_val'].var().to_dict()        # {tgrep.id : var}
    dict_raw_distrib = input_df.groupby('tgrep.id')['response_val'].apply(list).to_dict()   # {tgrep.id : [raw ratings]}
    dict_beta = input_df.groupby('tgrep.id')['response_val'].apply(lambda x: stats.beta.fit(x, floc=-1e-8)[0:2]).to_dict()  ## {tgrep.id : [alpha, beta]}

    input_df['response_val'] = (input_df['response_val'] * buckets).apply(np.ceil)          # discretize raw ratings
    ratings_list = input_df.groupby('tgrep.id')['response_val'].apply(list)
    dict_discrete_distrib = get_distrib_dict(ratings_list, buckets)                         # {tgrep.id : [7-bucket distribution]}

    assert len(dict_discrete_distrib) == len(dict_id_to_sentence)

    big_list = []
    for (key, val) in dict_id_to_sentence.items():
        if val[0] == 'nan': continue
        else:
            sentence_str = re.sub(" but not both", "", val[0])                          # the sentence string
            raw_distrib = str(dict_raw_distrib[key]).replace(",", " ")
            discrete_distrib = str(dict_discrete_distrib[key]).replace('\n', '')        # the distribution of ratings (7-dim vec)
            mean = str(dict_sentence_mean[key])
            var = str(dict_sentence_var[key])
            alpha, beta = dict_beta[key]
            example = key + ',' + mean + ',' + var + ',' + str(alpha) + ',' + str(beta) + ',' + raw_distrib + ',' + discrete_distrib + ',' + '"' + format(sentence_str) + '"'
            big_list.append(example)
                # big_list is a list of strings formatted: 'tgrep.id, mean, var, alpha, beta, raw_distrib, discrete_distrib, sentence'

    # 2. split dataset into test and training

    num_examples = len(big_list)
    num_train = math.ceil((1-test_pct) * num_examples)  # number of examples for training

    ids = list(range(0, num_examples))
    random.shuffle(ids)             # shuffle them! 
    train_ids = ids[:num_train]     # training examples = the first num_train number of examples from big_list, by index in big_list
    test_ids = ids[num_train:]      # testing examples = what's left over, by index in big_list

    mkdir_p(save_path)
    head_line = "Item,Mean,Var,Alpha,Beta,Raw_Distrib,Discrete_Distrib,Sentence\n"                   # set the header
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
    # split_k_fold(opt.seed, opt.path, splits=opt.k)

if __name__ == '__main__':
    main()
    