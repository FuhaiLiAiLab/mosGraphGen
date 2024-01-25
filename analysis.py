import os
import pdb
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.pyplot import figure
from sklearn.metrics import mean_squared_error
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from sklearn.metrics import accuracy_score


class RebuildAcc():
    def __init__(self):
        pass

    def rebuild_acc(self, path, epoch_num):
        test_epoch_acc_list = []
        train_epoch_acc_list = []
        max_test_acc = 0
        max_train_acc = 0
        max_test_id = 0
        for i in range(1, epoch_num + 1):
            # Test ACC
            try:
                # Attempt to read the file
                test_df = pd.read_csv(path + '/TestPred' + str(i) + '.txt', delimiter=',')
                # Process the DataFrame as needed
                print("Test file read successfully!")
            except Exception as e:
                # If an error occurs, this block of code will run
                print("An error occurred:", e)
                break  # Exit the loop
            test_label_list = list(test_df['label'])
            test_pred_list = list(test_df['prediction'])
            test_epoch_acc = accuracy_score(test_label_list, test_pred_list)
            test_epoch_acc_list.append(test_epoch_acc)
            # Train ACC
            try:
                # Attempt to read the file
                train_df = pd.read_csv(path + '/TrainingPred_' + str(i) + '.txt', delimiter=',')
                # Process the DataFrame as needed
                print("Training file read successfully!")
            except Exception as e:
                # If an error occurs, this block of code will run
                print("An error occurred:", e)
                break  # Exit the loop
            train_label_list = list(train_df['label'])
            train_pred_list = list(train_df['prediction'])
            train_epoch_acc = accuracy_score(train_label_list, train_pred_list)
            train_epoch_acc_list.append(train_epoch_acc)
            # Save the max test accuracy id
            if test_epoch_acc > max_test_acc:
                max_test_acc = test_epoch_acc
                max_train_acc = train_epoch_acc
                max_test_id = i
        best_train_df = pd.read_csv(path + '/TrainingPred_' + str(max_test_id) + '.txt', delimiter=',')
        best_train_df.to_csv(path + '/BestTrainingPred.txt', index=False, header=True)
        best_test_df = pd.read_csv(path + '/TestPred' + str(max_test_id) + '.txt', delimiter=',')
        best_test_df.to_csv(path + '/BestTestPred.txt', index=False, header=True)
        print('-------------BEST MODEL ID:' + str(max_test_id) + '-------------')
        print('-------------BEST MODEL ID:' + str(max_test_id) + '-------------')
        print('-------------BEST MODEL ID:' + str(max_test_id) + '-------------')
        print('BEST MODEL PEARSON ACCURACY: ', max_train_acc)
        print('BEST MODEL PEARSON ACCURACY: ', max_test_acc)
        return max_test_id


class AnalyseAcc():
    def __init__(self):
        pass

    def pred_result(self, fold_n, epoch_name, dataset, modelname):
        plot_path = './' + dataset + '-plot' + '/' + modelname
        if os.path.exists(plot_path) == False:
            os.mkdir(plot_path)
        ### TRAIN PRED JOINTPLOT
        train_pred_df = pd.read_csv('./' + dataset + '-result/' + modelname + '/' + epoch_name + '/BestTrainingPred.txt')
        train_label_list = list(train_pred_df['label'])
        train_pred_list = list(train_pred_df['prediction'])
        train_accuracy = accuracy_score(train_label_list, train_pred_list)
        ### TEST PRED JOINTPLOT
        test_pred_df = pd.read_csv('./' + dataset + '-result/' + modelname + '/' + epoch_name + '/BestTestPred.txt')
        test_label_list = list(test_pred_df['label'])
        test_pred_list = list(test_pred_df['prediction'])
        test_accuracy = accuracy_score(test_label_list, test_pred_list)
        print('--- TRAIN ---')
        print('BEST MODEL TRAIN PEARSON ACCURACY: ', train_accuracy)
        print('--- TEST ---')
        print('BEST MODEL TEST PEARSON ACCURACY: ', test_accuracy)
        return train_accuracy, test_accuracy

    def dataset_avg_comparison(self, dataset, gcn_decoder_avg_list, gat_decoder_avg_list, 
                        gformer_decoder_avg_list, gin_decoder_avg_list):
        colors = sns.color_palette("Set2", 4)
        labels = [dataset]
        x = np.arange(len(labels))
        width = 0.01
        print(gcn_decoder_avg_list)
        print(gat_decoder_avg_list)
        print(gformer_decoder_avg_list)
        print(gin_decoder_avg_list)
        sns.set_style(style=None)
        gcn = plt.bar(x - 1.5*width, gcn_decoder_avg_list, width, label='GCN Decoder', color=colors[0])
        gin = plt.bar(x - 0.5*width, gin_decoder_avg_list, width, label='GIN Decoder', color=colors[1])
        gat = plt.bar(x + 0.5*width, gat_decoder_avg_list, width, label='GAT Decoder', color=colors[2])
        gformer = plt.bar(x + 1.5*width, gformer_decoder_avg_list, width, label='GraphFormer Decoder', color=colors[3])
        plt.ylabel('Accuracy')
        if dataset == 'UCSC':
            plt.ylim(0.7, 0.8)
        elif dataset == 'ROSMAP':
            plt.ylim(0.3, 0.5)
        plt.xticks(x, labels=labels)
        plt.legend(loc='upper right', ncol=2)
        plt.savefig('./' + dataset + '-plot/dataset_avg_comparisons.png', dpi=600)
        # plt.show()

def model_result(dataset, modelname, epoch_num, rebuild=True):
    model_test_result_list = []
    for fold_n in np.arange(1, 6):
        fold_num = str(fold_n) + 'th'
        if fold_n == 1:
            epoch_name = 'epoch_' + str(epoch_num)
        else:
            epoch_name = 'epoch_' + str(epoch_num) + '_' + str(fold_n-1)
        # REBUILD BEST ID
        if rebuild == True:
            path = './' + dataset + '-result/' + modelname + '/' + epoch_name
            max_test_id = RebuildAcc().rebuild_acc(path, epoch_num)
        train_path = './' + dataset + '-result/' + modelname + '/' + epoch_name + '/BestTrainingPred.txt'
        test_path = './' + dataset + '-result/' + modelname + '/' + epoch_name + '/BestTestPred.txt'
        train_pearson, test_pearson = AnalyseAcc().pred_result(fold_n=fold_n, epoch_name=epoch_name, dataset=dataset, modelname=modelname)
        model_test_result_list.append(test_pearson)
    average_test_result = sum(model_test_result_list) / len(model_test_result_list)
    model_test_result_list.append(average_test_result)
    print(model_test_result_list)
    return model_test_result_list


if __name__ == "__main__":
    ### DATASET SELECTION
    # dataset = 'UCSC'
    dataset = 'ROSMAP'
    rebuild = False

    if os.path.exists('./'+ dataset + '-plot') == False:
        os.mkdir('./'+ dataset + '-plot')

    # ## MODEL SELECTION
    # gcn_decoder_test_list = model_result(dataset=dataset, modelname='gcn', epoch_num=50, rebuild=rebuild) 
    # gat_decoder_test_list = model_result(dataset=dataset, modelname='gat', epoch_num=50, rebuild=rebuild) 
    # gformer_decoder_test_list = model_result(dataset=dataset, modelname='gformer', epoch_num=50, rebuild=rebuild)
    # gin_decoder_test_list = model_result(dataset=dataset, modelname='gin', epoch_num=50, rebuild=rebuild)

    # print('gcn_decoder_test_list: ', gcn_decoder_test_list)
    # print('gat_decoder_test_list: ', gat_decoder_test_list)
    # print('gformer_decoder_test_list: ', gformer_decoder_test_list)
    # print('gin_decoder_test_list: ', gin_decoder_test_list)
    
    # ### UCSC
    # gcn_decoder_test_list = [0.7520891364902507, 0.754874651810585, 0.7646239554317549, 0.733983286908078, 0.7236111111111111, 0.7458364283503559]
    # gat_decoder_test_list = [0.7361111111111112, 0.7520891364902507, 0.7646239554317549, 0.7409470752089137, 0.7263888888888889, 0.7440320334261838]
    # gformer_decoder_test_list = [0.7646239554317549, 0.7493036211699164, 0.7688022284122563, 0.7506963788300836, 0.7388888888888889, 0.75446301454658]
    # gin_decoder_test_list = [0.766016713091922, 0.7520891364902507, 0.7646239554317549, 0.7381615598885793, 0.7236111111111111, 0.7489004952027235]

    # ### ROSMAP
    # gcn_decoder_test_list = [0.4444444444444444, 0.4074074074074074, 0.4074074074074074, 0.4444444444444444, 0.4666666666666667, 0.4340740740740741]
    # gat_decoder_test_list = [0.48148148148148145, 0.48148148148148145, 0.4444444444444444, 0.3333333333333333, 0.3, 0.40814814814814815]
    # gformer_decoder_test_list = [0.48148148148148145, 0.4444444444444444, 0.4074074074074074, 0.4074074074074074, 0.5, 0.4481481481481481]
    # gin_decoder_test_list = [0.4444444444444444, 0.4444444444444444, 0.4074074074074074, 0.37037037037037035, 0.3, 0.3933333333333333]

    ### DATASET SCORES
    if dataset == 'UCSC':
        gcn_decoder_avg_list = [0.7458364283503559]
        gat_decoder_avg_list = [0.7440320334261838]
        gformer_decoder_avg_list = [0.75446301454658]
        gin_decoder_avg_list = [0.7489004952027235]
    elif dataset == 'ROSMAP':
        gcn_decoder_avg_list = [0.4340740740740741]
        gat_decoder_avg_list = [0.40814814814814815]
        gformer_decoder_avg_list = [0.4481481481481481]
        gin_decoder_avg_list = [0.3933333333333333]

    AnalyseAcc().dataset_avg_comparison(dataset, 
                                    gcn_decoder_avg_list, 
                                    gat_decoder_avg_list, 
                                    gformer_decoder_avg_list, 
                                    gin_decoder_avg_list)
