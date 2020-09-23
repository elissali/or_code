import argparse
from collections import defaultdict
from datetime import datetime
import logging
import math
import os
import pprint
import random
import re
from statistics import mean
import sys
from scipy import stats

#from allennlp.commands.elmo import ElmoEmbedder # DO_NOTHING
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.distributions import Beta
from tqdm import tqdm
import yaml

from models import split_by_whitespace, RatingModel
from split_dataset import split_train_test, k_folds_idx #, num_examples
from utils import mkdir_p


print("done importing")

cfg = edict()
cfg.SOME_DATABASE = './data_2.csv'      
cfg.CONFIG_NAME = ''
cfg.RESUME_DIR = ''
cfg.SEED = 0
cfg.MODE = 'train'                                  # remember to change to 'test' vs. 'train'
cfg.PREDICTION_TYPE = 'discrete_distrib'            # can be: 'rating', 'discrete_distrib', 'beta_distrib', 'mean_var', 'mixed_gauss'
cfg.SINGLE_SENTENCE = True
cfg.EXPERIMENT_NAME = ''
cfg.OUT_PATH = './'
cfg.GLOVE_DIM = 100                     
cfg.IS_ELMO = True
cfg.IS_BERT = False
cfg.ELMO_LAYER = 2
cfg.BERT_LAYER = 11
cfg.BERT_LARGE = False
cfg.ELMO_MODE = 'concat'
cfg.SAVE_PREDS = False                              # can adjust
cfg.BATCH_ITEM_NUM = 30
cfg.PREDON = 'test'                                 # leave this as 'test'
cfg.CUDA = False
cfg.GPU_NUM = 1                        
cfg.KFOLDS = 5
cfg.CROSS_VALIDATION_FLAG = True                    # this can be changed
cfg.SPLIT_NAME = ""

cfg.LSTM = edict()
cfg.LSTM.FLAG = False
cfg.LSTM.SEQ_LEN = 20
cfg.LSTM.HIDDEN_DIM = 100
cfg.LSTM.DROP_PROB = 0.2
cfg.LSTM.LAYERS = 2
cfg.LSTM.BIDIRECTION = True 
cfg.LSTM.ATTN = False

# Training options
cfg.TRAIN = edict()
cfg.TRAIN.FLAG = True
cfg.TRAIN.BATCH_SIZE = 32
cfg.TRAIN.TOTAL_EPOCH = 200
cfg.TRAIN.INTERVAL = 4
cfg.TRAIN.START_EPOCH = 0
cfg.TRAIN.LR_DECAY_EPOCH = 20
cfg.TRAIN.LR =  0.0001                                        # 5e-2
cfg.TRAIN.COEFF = edict()
cfg.TRAIN.COEFF.BETA_1 = 0.9
cfg.TRAIN.COEFF.BETA_2 = 0.999
cfg.TRAIN.COEFF.EPS = 1e-8
cfg.TRAIN.LR_DECAY_RATE = 0.8
cfg.TRAIN.DROPOUT = edict()
cfg.TRAIN.DROPOUT.FC_1 = 0.75
cfg.TRAIN.DROPOUT.FC_2 = 0.75

# Evaluation options
cfg.EVAL = edict()
cfg.EVAL.FLAG = False
cfg.EVAL.BEST_EPOCH = 100

GLOVE_DIM = 100
NOT_EXIST = torch.FloatTensor(1, GLOVE_DIM).zero_()

#num_train, num_test = num_examples(input=cfg.SOME_DATABASE, test_pct = 0.3)     # added to remove 954 hardcode


def merge_yaml(new_cfg, old_cfg):
    for k, v in new_cfg.items():
        # check type
        old_type = type(old_cfg[k])
        if old_type is not type(v):
            if isinstance(old_cfg[k], np.ndarray):
                v = np.array(v, dtype=old_cfg[k].dtype)
            else:
                raise ValueError(('Type mismatch for config key: {}').format(k))
        # recursively merge dicts
        if type(v) is edict:
            try:
                merge_yaml(new_cfg[k], old_cfg[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            old_cfg[k] = v


def cfg_setup(filename):
    with open(filename, 'r') as f:
        new_cfg = edict(yaml.load(f))
    merge_yaml(new_cfg, cfg)


def load_dataset(input1, t):

    input_df = pd.read_csv(input1, sep=',')
    dict_item_sentence_raw = input_df[['Item', 'Sentence']].drop_duplicates().groupby('Item')['Sentence'].apply(list).to_dict()

    # if t == 'beta_distrib':
    #     dict_params = input_df[['Item', 'Beta_Params']].groupby('Item')['Beta_Params'].apply(list)
    #     dict_alpha = input_df[['Item', 'Alpha']].groupby('Item')['Alpha'].apply(float)
    #     dict_beta = input_df[['Item', 'Beta']].groupby('Item')['Beta'].apply(float)
    #     dict_item_params = dict()
    #     dict_item_sentence = dict()
    #     for (k, v) in dict_params.items():
    #         dict_item_params[k] = [dict_alpha[k], dict_beta[k]]
    #         dict_item_sentence[k] = dict_item_sentence_raw[k]
    #     return dict_item_params, dict_item_sentence
    
    if t == 'mixed_gauss' or t == 'beta_distrib':
        # dict_means = input_df[['Item', 'Mixed_Means']].groupby('Item')['Mixed_Means'].apply(list)
        # dict_stds = input_df[['Item', 'Mixed_Stds']].groupby('Item')['Mixed_Stds'].apply(list)
        dict_raws = input_df[['Item', 'Raw_Distrib']].groupby('Item')['Raw_Distrib'].apply(list)
        dict_item_scores = dict()
        dict_item_sentence = dict()

        def helper(v):
            v = v[0]
            v = v.strip(']')
            v = v.strip('[')
            v = v.split()        
            v = [float(i) for i in v]
            return v

        for (k, raw) in dict_raws.items():
            # means = helper(k, means)
            # stds = helper(k, dict_stds[k])
            # dict_item_params[k] = [means, stds]                         # {tgrep : [[mean1, mean2], [std1, std2]]}
            dict_item_scores[k] = helper(raw)[:9]                         # cut it off at 9, since these are inconsistent lengths
            dict_item_sentence[k] = dict_item_sentence_raw[k]
        return dict_item_scores, dict_item_sentence

    elif t == 'mean_var':
        dict_mean = input_df[['Item', 'Mean']].groupby('Item')['Mean'].apply(float)
        dict_var = input_df[['Item', 'Var']].groupby('Item')['Var'].apply(float)
        dict_item_params = dict()
        dict_item_sentence = dict()
        for (k, v) in dict_mean.items():
            dict_item_params[k] = [dict_mean[k], dict_var[k]]
            dict_item_sentence[k] = dict_item_sentence_raw[k]
        return dict_item_params, dict_item_sentence
    
    elif t == 'discrete_distrib':
        dict_discrete = input_df[['Item', 'Discrete_Distrib']].groupby('Item')['Discrete_Distrib'].apply(list)
        dict_item_distrib = dict()
        dict_item_sentence = dict()
        for (k, v) in dict_discrete.items():
            v = v[0]
            v = v.strip(']')
            v = v.strip('[')
            v = v.split()
            dict_item_distrib[k] = v
            dict_item_sentence[k] = dict_item_sentence_raw[k]
        return dict_item_distrib, dict_item_sentence
    
    elif t == 'rating':
        dict_item_mean_raw = input_df[['Item', 'Mean']].groupby('Item')['Mean'].apply(list).to_dict()
        dict_item_mean = dict()
        dict_item_sentence = dict()
        for (k,v) in dict_item_mean_raw.items():
            dict_item_mean[k] = v[0]
            dict_item_sentence[k] = dict_item_sentence_raw[k]
        return dict_item_mean, dict_item_sentence
    
    elif t == 'mdn_distrib':
        pass


def random_input(num_examples):
    res = []
    for i in range(num_examples):
        lst = []
        for j in range(GLOVE_DIM):
            lst.append(round(random.uniform(-1, 1), 16))
        res.append(lst)
    return torch.Tensor(res)


def main():
    parser = argparse.ArgumentParser(
        description='Run ...')
    parser.add_argument('--conf', dest='config_file', default="unspecified")
    parser.add_argument('--out_path', dest='out_path', default=None)
    opt = parser.parse_args()
    print(opt)

    if opt.config_file is not "unspecified":
        cfg_setup(opt.config_file)
        if not cfg.MODE == 'train':
            cfg.TRAIN.FLAG = False
            cfg.EVAL.FLAG = True
        if opt.out_path is not None:
            cfg.OUT_PATH = opt.out_path
    else:
        print("Using default settings.")

    logging.basicConfig(level=logging.INFO)

    # random seed
    random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(cfg.SEED)

    curr_path = "./datasets/seed_" + str(cfg.SEED)
    if cfg.SPLIT_NAME != "":
      curr_path = os.path.join(curr_path, cfg.SPLIT_NAME)
    if cfg.EXPERIMENT_NAME == "":
        cfg.EXPERIMENT_NAME = datetime.now().strftime('%m_%d_%H_%M')
    log_path = os.path.join(cfg.OUT_PATH, cfg.EXPERIMENT_NAME, "Logging")
    mkdir_p(log_path)
    file_handler = logging.FileHandler(os.path.join(log_path, cfg.MODE + "_log.txt"))
    logging.getLogger().addHandler(file_handler)
    logging.getLogger().setLevel(logging.INFO)
    logging.info('Using configurations:')
    logging.info(pprint.pformat(cfg))
    logging.info(f'Using random seed {cfg.SEED}.')

    if cfg.MODE == 'train':
        load_db = curr_path + "/train_db.csv"
    elif cfg.MODE == 'test':
        load_db = curr_path + "/" + cfg.PREDON + "_db.csv"


    ################################# LOADING WORD EMBEDDINGS ##################################

    sl = []
    word_embs = []
    word_embs_np = None
    word_embs_stack = None

    if not cfg.MODE == 'qual':
        if not os.path.isfile(load_db):
            split_train_test(cfg.SEED, curr_path)
        labels, target_utterances = load_dataset(load_db, cfg.PREDICTION_TYPE)
    else:
        if not os.path.isfile(load_db):
            sys.exit(f'Fail to find the file {load_db} for qualitative evaluation. Exit.')
        with open(load_db, "r") as qual_file:
            sentences = [x.strip() for x in qual_file.readlines()]

    NUMPY_DIR = './datasets/seed_' + str(cfg.SEED)
    # is contextual or not
    if not cfg.SINGLE_SENTENCE:
        NUMPY_DIR += '_contextual'
    # type of pre-trained word embedding
    if cfg.IS_ELMO:
        #NUMPY_DIR += '/elmo_' + "layer_" + str(cfg.ELMO_LAYER) # DO_NOTHING
        NUMPY_DIR += ""
    elif cfg.IS_BERT:
        NUMPY_DIR += '/bert_'
        if cfg.BERT_LARGE:
            NUMPY_DIR += "large"
        NUMPY_DIR += "layer_" + str(cfg.BERT_LAYER)
    else:  # default: GloVe
        NUMPY_DIR += '/glove'
    # Avg/LSTM
    if cfg.LSTM.FLAG:
        NUMPY_DIR += '_lstm'
        NUMPY_PATH = NUMPY_DIR + '/embs_' + cfg.PREDON + '_' + format(cfg.LSTM.SEQ_LEN) + '.npy'
        LENGTH_PATH = NUMPY_DIR + '/len_' + cfg.PREDON + '_' + format(cfg.LSTM.SEQ_LEN) + '.npy'
    else:
        NUMPY_PATH = NUMPY_DIR + '/embs_' + cfg.PREDON + '.npy'
        LENGTH_PATH = NUMPY_DIR + '/len_' + cfg.PREDON + '.npy'
    mkdir_p(NUMPY_DIR)
    print(NUMPY_PATH)
    logging.info(f'Path to the current word embeddings: {NUMPY_PATH}')

    if os.path.isfile(NUMPY_PATH):
        print(NUMPY_PATH)
        word_embs_np = np.load(NUMPY_PATH)
        # print("line 254: ", word_embs_np.shape)
        len_np = np.load(LENGTH_PATH)
        sl = len_np.tolist()
        word_embs_stack = torch.from_numpy(word_embs_np)
    else:
        if cfg.IS_ELMO:
            #ELMO_EMBEDDER = EL
            sl = sl # DO_NOTHING
        if cfg.IS_BERT:
            from pytorch_transformers import BertTokenizer, BertModel
            bert_model = 'bert-large-uncased' if cfg.BERT_LARGE else 'bert-base-uncased'
            bert_tokenizer = BertTokenizer.from_pretrained(bert_model)
            bert_model = BertModel.from_pretrained(bert_model, output_hidden_states=True)
            bert_model.eval()
            if cfg.CUDA:
                bert_model = bert_model.cuda()
        if cfg.MODE != 'qual':
            for (k, v) in tqdm(target_utterances.items(), total=len(target_utterances)):
                #context_v = contexts[k]
                context_v = "a" #DO_NOTHING
                if cfg.SINGLE_SENTENCE:
                    # only including the target utterance
                    input_text = v[0]
                else:
                    # discourse context + target utterance
                    input_text = context_v[0] + v[0]
                if cfg.IS_ELMO:
                    ''' # DO_NOTHING
                    from models import get_sentence_elmo
                    embedder = ELMO_EMBEDDER
                    curr_emb, l = get_sentence_elmo(v[0], context_v[0], embedder=embedder,
                                                    layer=cfg.ELMO_LAYER,
                                                    not_contextual=cfg.SINGLE_SENTENCE,
                                                    LSTM=cfg.LSTM.FLAG,
                                                    seq_len=cfg.LSTM.SEQ_LEN)
                    '''
                    input_text = input_text
                elif cfg.IS_BERT:
                    if cfg.SINGLE_SENTENCE:
                        from models import get_sentence_bert
                        curr_emb, l = get_sentence_bert(input_text,
                                                        bert_tokenizer,
                                                        bert_model,
                                                        layer=cfg.BERT_LAYER,
                                                        GPU=cfg.CUDA,
                                                        LSTM=cfg.LSTM.FLAG,
                                                        max_seq_len=cfg.LSTM.SEQ_LEN,
                                                        is_single=cfg.SINGLE_SENTENCE)
                    else:
                        from models import get_sentence_bert_context
                        curr_emb, l = get_sentence_bert_context(v[0],
                                                                context_v[0],
                                                                bert_tokenizer,
                                                                bert_model,
                                                                layer=cfg.BERT_LAYER,
                                                                GPU=cfg.CUDA,
                                                                LSTM=cfg.LSTM.FLAG,
                                                                max_sentence_len=30,
                                                                max_context_len=120)
                else:
                    from models import get_sentence_glove
                    curr_emb, l = get_sentence_glove(input_text, LSTM=cfg.LSTM.FLAG,
                                                     not_contextual=cfg.SINGLE_SENTENCE,
                                                     seq_len=cfg.LSTM.SEQ_LEN)
                sl.append(l)
                word_embs.append(curr_emb)
        np.save(LENGTH_PATH, np.array(sl))
        word_embs_stack = torch.stack(word_embs)
        np.save(NUMPY_PATH, word_embs_stack.numpy())


    ################################# LOADING Y-LABELS ##################################
    
    normalized_labels = []      # list of arrays
    keys = []                   # list of tgrep ids                                                                            
    if not cfg.MODE == 'qual':
        if cfg.PREDICTION_TYPE == "discrete_distrib" or cfg.PREDICTION_TYPE == "mean_var":
            for (k, v) in labels.items():
                keys.append(k)
                normalized_labels.append(list(map(float, v)))           # (871, 7)
        elif cfg.PREDICTION_TYPE == "rating":
            for (k, v) in labels.items():
                keys.append(k)
                normalized_labels.append(float(v))
        elif cfg.PREDICTION_TYPE == "beta_distrib":
            for (k,v) in labels.items():
                keys.append(k)
                normalized_labels.append(list(map(float, v)))           # (871, 2)
        elif cfg.PREDICTION_TYPE == "mixed_gauss":
            for (k, v) in labels.items():                           
                keys.append(k)
                normalized_labels.append(list(map(float, v)))           # (871, 9)
                # flat_v = [item for sublist in v for item in sublist]
                # normalized_labels.append(flat_v)                      # [mean1, mean2, std1, std2, weight1, weight2]


    ##################################### TRAINING #######################################

    if cfg.TRAIN.FLAG:
        logging.info("Start training\n===============================")
        save_path = cfg.OUT_PATH + cfg.EXPERIMENT_NAME

        X, y, L = dict(), dict(), dict()
        if not cfg.CROSS_VALIDATION_FLAG:
            # no k-fold validation; super simple
            cfg.BATCH_ITEM_NUM = len(normalized_labels)//cfg.TRAIN.BATCH_SIZE
            X["train"], X["val"] = word_embs_stack.float(), None
            y["train"], y["val"] = np.array(normalized_labels), None
            L["train"], L["val"] = sl, None
            r_model = RatingModel(cfg, save_path)
            r_model.train(X, y, L)
        else:
            # train with k folds cross validation
            train_loss_history = np.zeros((cfg.TRAIN.TOTAL_EPOCH, cfg.KFOLDS))
            val_loss_history = np.zeros((cfg.TRAIN.TOTAL_EPOCH, cfg.KFOLDS))
            val_r_history = np.zeros((cfg.TRAIN.TOTAL_EPOCH, cfg.KFOLDS))
            normalized_labels = np.array(normalized_labels)
            sl_np = np.array(sl)
            fold_cnt = 1
            for train_idx, val_idx in k_folds_idx(cfg.KFOLDS, 871, cfg.SEED):                           # manual 871 training size here

                # get training embeddings, y labels for indices in fold/batch:
                logging.info(f'Fold #{fold_cnt}\n- - - - - - - - - - - - -')
                save_sub_path = os.path.join(save_path, format(fold_cnt))
                X_train, X_val = word_embs_stack[train_idx], word_embs_stack[val_idx]
                y_train, y_val = normalized_labels[train_idx], normalized_labels[val_idx]
                L_train, L_val = sl_np[train_idx].tolist(), sl_np[val_idx].tolist()
                X["train"], X["val"] = X_train, X_val
                y["train"], y["val"] = y_train, y_val
                L["train"], L["val"] = L_train, L_val
                cfg.BATCH_ITEM_NUM = len(L_train)//cfg.TRAIN.BATCH_SIZE

                # model load and train
                r_model = RatingModel(cfg, save_sub_path)
                r_model.train(X, y, L)

                # save train and val loss and r history for this fold
                train_loss_history[:, fold_cnt-1] = np.array(r_model.train_loss_history)
                val_loss_history[:, fold_cnt-1] = np.array(r_model.val_loss_history)
                val_r_history[:, fold_cnt-1] = np.array(r_model.val_r_history)
                fold_cnt += 1

            # get total average train/val loss and r over all folds    
            train_loss_mean = np.mean(train_loss_history, axis=1).tolist()
            val_loss_mean = np.mean(val_loss_history, axis=1).tolist()
            val_r_mean = np.mean(val_r_history, axis=1).tolist()
            max_r = max(val_r_mean)
            max_r_idx = 1 + val_r_mean.index(max_r)
            logging.info(f'Highest avg. r={max_r:.4f} achieved at epoch {max_r_idx} (on validation set).')
            logging.info(f'Avg. train loss: {train_loss_mean}')
            logging.info(f'Avg. validation loss: {val_loss_mean}')
            logging.info(f'Avg. validation r: {val_r_mean}')
    

    ##################################### TESTING #######################################

    else:
        eval_path = cfg.OUT_PATH + cfg.EXPERIMENT_NAME
        epoch_lst = [0, 1]
        i = 0
        while i < cfg.TRAIN.TOTAL_EPOCH - cfg.TRAIN.INTERVAL + 1:
            i += cfg.TRAIN.INTERVAL
            epoch_lst.append(i)
        logging.info(f'epochs to test: {epoch_lst}')

        load_path = os.path.join(eval_path, "Model")
        max_epoch_dir = None
        max_value = -1.0
        max_epoch = None
        curr_coeff_lst = []

        # for each epoch: predict scores (means)
        for epoch in epoch_lst:
            cfg.RESUME_DIR = load_path + "/RNet_epoch_" + format(epoch)+ ".pth"
            eval_model = RatingModel(cfg, eval_path)
            preds, attn_weights = eval_model.evaluate(word_embs_stack.float(), sl)      # preds is np array of scores (mean preds)

            # if attention, log the attention weights
            if cfg.LSTM.ATTN:
                attn_path = os.path.join(eval_path, "Attention")
                mkdir_p(attn_path)
                new_file_name = attn_path + '/' + cfg.PREDON + '_attn_epoch' + format(epoch) + '.npy'
                np.save(new_file_name, attn_weights)
                logging.info(f'Write attention weights to {new_file_name}.')
            
            print("Checkpoint: Predicting correlation coefficients... ")
            print(len(normalized_labels))                                   # checkpoint; normalized_labels should be same shape as preds
            print(preds.shape)

            # calculate correlation coefficient between predicted means and actual label means (for this epoch)
            curr_coeff = np.corrcoef(preds, np.array(normalized_labels))[0, 1]
            curr_coeff_lst.append(curr_coeff)

            # if current epoch's coeff is the best one so far, save this coeff and save this epoch
            if max_value < curr_coeff: 
                max_value = curr_coeff
                max_epoch_dir = cfg.RESUME_DIR
                max_epoch = epoch

            if cfg.SAVE_PREDS:
                print("Checkpoint: Saving preds...")
                pred_file_path = eval_path + '/Preds'
                mkdir_p(pred_file_path)
                new_file_name = pred_file_path + '/' + cfg.PREDON + '_preds_rating_epoch' + format(epoch) + '.csv'
                f = open(new_file_name, 'w')
                head_line = "Item_ID\toriginal_mean\tpredicted\n"
                print(f'Start writing predictions to file:\n{new_file_name}\n...')
                f.write(head_line)
                for i in range(len(keys)):
                    k = keys[i]
                    ori = normalized_labels[i]
                    pre = preds[i]
                    curr_line = k + '\t' + format(ori) + '\t' + format(pre)
                    f.write(curr_line+"\n")
                f.close()
        logging.info(f'Max r = {max_value} achieved at epoch {max_epoch}')
        logging.info(f'r by epoch: {curr_coeff_lst}')
    return

if __name__ == "__main__":
    main()