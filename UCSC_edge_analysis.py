import os
import pdb
import torch
import numpy as np
import pandas as pd


class AccAnalyse():
    def __init__(self):
        pass


class NetAnalyse():
    def __init__(self):
        pass

    def fold_avg(self, fold_n):
        # first
        fold_first_edge_weight_df = pd.read_csv('./UCSC-analysis/fold_' + str(fold_n) +'/first_edge_weight.csv')
        edge_type_list = fold_first_edge_weight_df['EdgeType'].tolist()
        fold_first_edge_weight_df = fold_first_edge_weight_df.drop(['EdgeType'], axis=1)
        # block
        fold_block_edge_weight_df = pd.read_csv('./UCSC-analysis/fold_' + str(fold_n) +'/block_edge_weight.csv')
        fold_block_edge_weight_df = fold_block_edge_weight_df.drop(['EdgeType'], axis=1)
        # last
        fold_last_edge_weight_df = pd.read_csv('./UCSC-analysis/fold_' + str(fold_n) +'/last_edge_weight.csv')
        fold_last_edge_weight_df = fold_last_edge_weight_df.drop(['EdgeType'], axis=1)
        # average
        fold_avg_edge_weight_df = (fold_first_edge_weight_df + fold_block_edge_weight_df + fold_last_edge_weight_df) / 3
        cols = ['src','dest']
        fold_avg_edge_weight_df[cols] = fold_avg_edge_weight_df[cols].astype(int)
        fold_avg_edge_weight_df['EdgeType'] = edge_type_list
        # import pdb; pdb.set_trace()
        # [weight]
        fold_avg_edge_weight_df.to_csv('./UCSC-analysis/fold_' + str(fold_n) +'/fold_avg_edge_weight_df.csv', index=False, header=True)

    def all_avg(self, k):
        # all fold_avg_edge_weight_df
        fold_avg_edge_weight_df_list = []
        for fold_n in range(1, k + 1):
            fold_avg_edge_weight_df = pd.read_csv('./UCSC-analysis/fold_' + str(fold_n) +'/fold_avg_edge_weight_df.csv')
            fold_avg_edge_weight_df_list.append(fold_avg_edge_weight_df)


if __name__ == '__main__':
    fold_n = 1
    k = 5
    NetAnalyse().fold_avg(fold_n)