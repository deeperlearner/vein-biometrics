import argparse
import glob
import os
import random

import pandas as pd

parser = argparse.ArgumentParser(description='train valid test generator')

parser.add_argument('--train-val-outdir', default='output_list_train_val.csv', type=str,
                    help='path to the CSV output file including the list of train/validation partitions')
parser.add_argument('--test-outdir', default='output_list_test.csv', type=str,
                    help='path to the CSV output file including the list of test partition')
parser.add_argument('--train-dir', default='train_dis.csv', type=str,
                    help='path to train root dir')
parser.add_argument('--valid-dir', default='val_dis.csv', type=str,
                    help='path to valid root dir')
parser.add_argument('--test-dir', default='test_pairs_dis.csv', type=str,
                    help='path to test root dir')
args = parser.parse_args()


def main():
    df_train_val = pd.read_csv(args.train_val_outdir)
    df_train_val['class'] = df_train_val['class'].astype(int)
    count = len(df_train_val)
    num_train = int(0.8 * count)
    df_sample = df_train_val.sample(frac=1)
    df_train = df_sample.iloc[0:num_train]
    df_valid = df_sample.iloc[num_train:]
    df_train.to_csv(args.train_dir, index=False)
    df_valid.to_csv(args.valid_dir, index=False)

    df_test = pd.read_csv(args.test_outdir)
    pp_dict = dict()
    for index, row in df_test.iterrows():
        cls = row['class']
        idx = row['idx']
        LR = idx[-7]
        if cls not in pp_dict:
            pp_dict[cls] = {'L': [], 'R': []}
        pp_dict[cls][LR].append(idx)

    df_test_pair = pd.DataFrame(columns=['class', 'idx', 'idy'])
    count = 9000
    # impostor
    cls_pool = list(pp_dict.keys())
    LR_pool = ['L', 'R']
    for i in range(count):
        p1, p2 = random.sample(cls_pool, 2)
        LR = random.choice(LR_pool)
        P1_pool = pp_dict[p1][LR]
        idx = random.choice(P1_pool)
        LR = random.choice(LR_pool)
        P2_pool = pp_dict[p2][LR]
        idy = random.choice(P2_pool)
        df_test_pair.loc[i] = [0, idx, idy]
    # genuine
    for i in range(count):
        cls = random.choice(cls_pool)
        LR = random.choice(LR_pool)
        PP_pool = pp_dict[cls][LR]
        idx, idy = random.sample(PP_pool, 2)
        df_test_pair.loc[count+i] = [1, idx, idy]
    df_test_pair.to_csv(args.test_dir, index=False)


if __name__ == '__main__':
    main()
