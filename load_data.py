import os
import torch
import numpy as np
import pandas as pd

from numpy import inf
from scipy import sparse
from sklearn.model_selection import train_test_split

class UCSC_LoadData():
    def __init__(self):
        pass

    def load_batch(self, index, upper_index, place_num, graph_output_folder, processed_dataset):
        # Preload each split dataset
        split_input_df = pd.read_csv(os.path.join(graph_output_folder, 'split-random-survival-label-' + str(place_num + 1) + '.csv'))
        num_feature = 8
        final_annotation_gene_df = pd.read_csv(os.path.join(graph_output_folder, 'map-all-gene.csv'))
        gene_name_list = list(final_annotation_gene_df['Gene_name'])
        num_node = len(gene_name_list)
        print('READING GENE FEATURES FILES ...')
        final_rnaseq_df = pd.read_csv(os.path.join(processed_dataset, 'processed-genotype-gene-expression.csv'))
        final_mut_df = pd.read_csv(os.path.join(processed_dataset, 'processed-genotype-mutation.csv'))
        final_meth_core_promoter_df = pd.read_csv(os.path.join(processed_dataset, 'processed-genotype-methy-Core-Promoter.csv'))
        final_meth_proximal_promoter_df = pd.read_csv(os.path.join(processed_dataset, 'processed-genotype-methy-Proximal-Promoter.csv'))
        final_meth_distal_promoter_df = pd.read_csv(os.path.join(processed_dataset, 'processed-genotype-methy-Distal-Promoter.csv'))
        final_meth_downstream_df = pd.read_csv(os.path.join(processed_dataset, 'processed-genotype-methy-Downstream.csv'))
        final_meth_upstream_df = pd.read_csv(os.path.join(processed_dataset, 'processed-genotype-methy-Upstream.csv'))
        final_proteomics_df = pd.read_csv(os.path.join(processed_dataset, 'processed-genotype-proteomics.csv'))
        num_gene = final_rnaseq_df.shape[0]
        num_gene_meth = final_meth_core_promoter_df.shape[0]
        num_gene_protein = final_proteomics_df.shape[0]
        print('READING FINISHED ...')
        # Combine a batch size as [x_batch, y_batch]
        print('-----' + str(index) + ' to ' + str(upper_index) + '-----')
        tmp_batch_size = 0
        y_input_list = []
        x_batch = np.zeros((1, (num_feature * num_node)))
        for row in split_input_df.iloc[index : upper_index].itertuples():
            tmp_batch_size += 1
            sample_name = row[1]
            y = row[2]
            # Gene features sequence
            gene_rna_list = [float(x) for x in list(final_rnaseq_df[sample_name])]
            gene_mut_list = [float(x) for x in list(final_mut_df[sample_name])]
            gene_meth_core_promoter_list = [float(x) for x in list(final_meth_core_promoter_df[sample_name])]
            gene_meth_proximal_promoter_list = [float(x) for x in list(final_meth_proximal_promoter_df[sample_name])]
            gene_meth_distal_promoter_list = [float(x) for x in list(final_meth_distal_promoter_df[sample_name])]
            gene_meth_downstream_list = [float(x) for x in list(final_meth_downstream_df[sample_name])]
            gene_meth_upstream_list = [float(x) for x in list(final_meth_upstream_df[sample_name])]
            gene_proteomics_list = [float(x) for x in list(final_proteomics_df[sample_name])]
            # Combine [rna, mut, meth, proteomics] as [gene_feature]
            x_input_list = []
            for row in final_annotation_gene_df.itertuples():
                i = row[0]
                node_type = row[3]
                if node_type == 'Gene-TRAN':
                    x_input_list.append(gene_rna_list[i])
                    x_input_list.append(gene_mut_list[i])
                    x_input_list += [0.0] * 6
                elif node_type == 'Gene-METH':
                    x_input_list += [0.0] * 2
                    x_input_list.append(gene_meth_core_promoter_list[i - num_gene])
                    x_input_list.append(gene_meth_proximal_promoter_list[i - num_gene])
                    x_input_list.append(gene_meth_distal_promoter_list[i - num_gene])
                    x_input_list.append(gene_meth_downstream_list[i - num_gene])
                    x_input_list.append(gene_meth_upstream_list[i - num_gene])
                    x_input_list += [0.0] * 1
                elif node_type == 'Gene-PROT':
                    x_input_list += [0.0] * 7
                    x_input_list.append(gene_proteomics_list[i - num_gene - num_gene_meth])
            # import pdb; pdb.set_trace()
            x_input = np.array(x_input_list)
            x_batch = np.vstack((x_batch, x_input))
            # Combine score list
            y_input_list.append(y)
        # import pdb; pdb.set_trace()
        x_batch = np.delete(x_batch, 0, axis = 0)
        y_batch = np.array(y_input_list).reshape(tmp_batch_size, 1)
        print(x_batch.shape)
        print(y_batch.shape)
        return x_batch, y_batch


    def load_all_split(self, batch_size, k, processed_dataset, graph_output_folder):
        form_data_path = './' + graph_output_folder + '/form_data'
        # Load 100 percent data
        print('LOADING ALL SPLIT DATA...')
        # First load each split data
        for place_num in range(k):
            split_input_df = pd.read_csv(os.path.join(graph_output_folder, 'split-random-survival-label-' + str(place_num + 1) + '.csv'))
            input_num, input_dim = split_input_df.shape
            num_feature = 8
            final_annotation_gene_df = pd.read_csv(os.path.join(graph_output_folder, 'map-all-gene.csv'))
            gene_name_list = list(final_annotation_gene_df['Gene_name'])
            num_node = len(gene_name_list)
            x_split = np.zeros((1, num_feature * num_node))
            y_split = np.zeros((1, 1))
            for index in range(0, input_num, batch_size):
                if (index + batch_size) < input_num:
                    upper_index = index + batch_size
                else:
                    upper_index = input_num
                x_batch, y_batch = UCSC_LoadData().load_batch(index, upper_index, place_num, graph_output_folder, processed_dataset)
                x_split = np.vstack((x_split, x_batch))
                y_split = np.vstack((y_split, y_batch))
            x_split = np.delete(x_split, 0, axis = 0)
            y_split = np.delete(y_split, 0, axis = 0)
            print('-------SPLIT DATA SHAPE-------')
            print(x_split.shape)
            print(y_split.shape)
            np.save(form_data_path + '/x_split' + str(place_num + 1) + '.npy', x_split)
            np.save(form_data_path + '/y_split' + str(place_num + 1) + '.npy', y_split)
            

    def load_train_test(self, k, n_fold, graph_output_folder):
        form_data_path = './' + graph_output_folder + '/form_data'
        num_feature = 8
        final_annotation_gene_df = pd.read_csv(os.path.join(graph_output_folder, 'map-all-gene.csv'))
        gene_name_list = list(final_annotation_gene_df['Gene_name'])
        num_node = len(gene_name_list)
        xTr = np.zeros((1, num_feature * num_node))
        yTr = np.zeros((1, 1))
        for i in range(1, k + 1):
            if i == n_fold:
                print('--- LOADING ' + str(i) + '-TH SPLIT TEST DATA ---')
                xTe = np.load(form_data_path + '/x_split' + str(i) + '.npy')
                yTe = np.load(form_data_path + '/y_split' + str(i) + '.npy')
            else:
                print('--- LOADING ' + str(i) + '-TH SPLIT TRAINING DATA ---')
                x_split = np.load(form_data_path + '/x_split' + str(i) + '.npy')
                y_split = np.load(form_data_path + '/y_split' + str(i) + '.npy')
                print('--- COMBINING DATA ... ---')
                xTr = np.vstack((xTr, x_split))
                yTr = np.vstack((yTr, y_split))
        print('--- TRAINING INPUT SHAPE ---')
        xTr = np.delete(xTr, 0, axis = 0)
        yTr = np.delete(yTr, 0, axis = 0)
        print(xTr.shape)
        print(yTr.shape)
        np.save(form_data_path + '/xTr' + str(n_fold) + '.npy', xTr)
        np.save(form_data_path + '/yTr' + str(n_fold) + '.npy', yTr)
        print('--- TEST INPUT SHAPE ---')
        print(xTe.shape)
        print(yTe.shape)
        np.save(form_data_path + '/xTe' + str(n_fold) + '.npy', xTe)
        np.save(form_data_path + '/yTe' + str(n_fold) + '.npy', yTe)

    def combine_whole_dataset(self, k, graph_output_folder):
        form_data_path = './' + graph_output_folder + '/form_data'
        num_feature = 8
        final_annotation_gene_df = pd.read_csv(os.path.join(graph_output_folder, 'map-all-gene.csv'))
        gene_name_list = list(final_annotation_gene_df['Gene_name'])
        num_node = len(gene_name_list)
        xAll = np.zeros((1, num_feature * num_node))
        yAll = np.zeros((1, 1))
        for i in range(1, k + 1):
            print('--- LOADING ' + str(i) + '-TH SPLIT TRAINING DATA ---')
            x_split = np.load(form_data_path + '/x_split' + str(i) + '.npy')
            y_split = np.load(form_data_path + '/y_split' + str(i) + '.npy')
            print('--- COMBINING DATA ... ---')
            xAll = np.vstack((xAll, x_split))
            yAll = np.vstack((yAll, y_split))
        print('--- ALL DATASET INPUT SHAPE ---')
        xAll = np.delete(xAll, 0, axis = 0)
        yAll = np.delete(yAll, 0, axis = 0)
        print(xAll.shape)
        print(yAll.shape)
        np.save(form_data_path + '/xAll.npy', xAll)
        np.save(form_data_path + '/yAll.npy', yAll)
    
    def load_adj_edgeindex(self, graph_output_folder):
        form_data_path = './' + graph_output_folder + '/form_data'
        # Form a whole adjacent matrix
        gene_num_df = pd.read_csv(os.path.join(graph_output_folder, 'all-gene-edge-num.csv'))
        src_gene_list = list(gene_num_df['src'])
        dest_gene_list = list(gene_num_df['dest'])
        final_annotation_gene_df = pd.read_csv(os.path.join(graph_output_folder, 'map-all-gene.csv'))
        gene_name_list = list(final_annotation_gene_df['Gene_name'])
        num_node = len(gene_name_list)
        adj = np.zeros((num_node, num_node))
        # gene-gene adjacent matrix
        for i in range(len(src_gene_list)):
            row_idx = src_gene_list[i]
            col_idx = dest_gene_list[i]
            adj[row_idx, col_idx] = 1
            adj[col_idx, row_idx] = 1 # whether we want ['sym']
        # import pdb; pdb.set_trace()
        # np.save(form_data_path + '/adj.npy', adj)
        adj_sparse = sparse.csr_matrix(adj)
        sparse.save_npz(form_data_path + '/adj_sparse.npz', adj_sparse)
        # [edge_index]
        source_nodes, target_nodes = adj_sparse.nonzero()
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        np.save(form_data_path + '/edge_index.npy', edge_index)


class ROSMAP_LoadData():
    def __init__(self):
        pass

    def load_batch(self, index, upper_index, place_num, graph_output_folder, processed_dataset):
        # Preload each split dataset
        split_input_df = pd.read_csv(os.path.join(graph_output_folder, 'split-random-survival-label-' + str(place_num + 1) + '.csv'))
        num_feature = 10
        final_annotation_gene_df = pd.read_csv(os.path.join(graph_output_folder, 'map-all-gene.csv'))
        gene_name_list = list(final_annotation_gene_df['Gene_name'])
        num_node = len(gene_name_list)
        print('READING GENE FEATURES FILES ...')
        final_rnaseq_df = pd.read_csv(os.path.join(processed_dataset, 'processed-genotype-gene-expression.csv'))
        final_cnv_del_df = pd.read_csv(os.path.join(processed_dataset, 'processed-genotype-cnv_del.csv'))
        final_cnv_dup_df = pd.read_csv(os.path.join(processed_dataset, 'processed-genotype-cnv_dup.csv'))
        final_cnv_mcnv_df = pd.read_csv(os.path.join(processed_dataset, 'processed-genotype-cnv_mcnv.csv'))
        final_meth_core_promoter_df = pd.read_csv(os.path.join(processed_dataset, 'processed-genotype-methy-Core-Promoter.csv'))
        final_meth_proximal_promoter_df = pd.read_csv(os.path.join(processed_dataset, 'processed-genotype-methy-Proximal-Promoter.csv'))
        final_meth_distal_promoter_df = pd.read_csv(os.path.join(processed_dataset, 'processed-genotype-methy-Distal-Promoter.csv'))
        final_meth_downstream_df = pd.read_csv(os.path.join(processed_dataset, 'processed-genotype-methy-Downstream.csv'))
        final_meth_upstream_df = pd.read_csv(os.path.join(processed_dataset, 'processed-genotype-methy-Upstream.csv'))
        final_proteomics_df = pd.read_csv(os.path.join(processed_dataset, 'processed-genotype-proteomics.csv'))
        num_gene = final_rnaseq_df.shape[0]
        num_gene_meth = final_meth_core_promoter_df.shape[0]
        num_gene_protein = final_proteomics_df.shape[0]
        print('READING FINISHED ...')
        # Combine a batch size as [x_batch, y_batch]
        print('-----' + str(index) + ' to ' + str(upper_index) + '-----')
        tmp_batch_size = 0
        y_input_list = []
        x_batch = np.zeros((1, (num_feature * num_node)))
        for row in split_input_df.iloc[index : upper_index].itertuples():
            tmp_batch_size += 1
            sample_name = row[1]
            y = row[2]
            # Gene features sequence
            gene_rna_list = [float(x) for x in list(final_rnaseq_df[sample_name])]
            gene_cnv_del_list = [float(x) for x in list(final_cnv_del_df[sample_name])]
            gene_cnv_dup_list = [float(x) for x in list(final_cnv_dup_df[sample_name])]
            gene_cnv_mcnv_list = [float(x) for x in list(final_cnv_mcnv_df[sample_name])]
            gene_meth_core_promoter_list = [float(x) for x in list(final_meth_core_promoter_df[sample_name])]
            gene_meth_proximal_promoter_list = [float(x) for x in list(final_meth_proximal_promoter_df[sample_name])]
            gene_meth_distal_promoter_list = [float(x) for x in list(final_meth_distal_promoter_df[sample_name])]
            gene_meth_downstream_list = [float(x) for x in list(final_meth_downstream_df[sample_name])]
            gene_meth_upstream_list = [float(x) for x in list(final_meth_upstream_df[sample_name])]
            gene_proteomics_list = [float(x) for x in list(final_proteomics_df[sample_name])]
            # Combine [rna, mut, meth, proteomics] as [gene_feature]
            x_input_list = []
            for row in final_annotation_gene_df.itertuples():
                i = row[0]
                node_type = row[3]
                if node_type == 'Gene-TRAN':
                    x_input_list.append(gene_rna_list[i])
                    x_input_list.append(gene_cnv_del_list[i])
                    x_input_list.append(gene_cnv_dup_list[i])
                    x_input_list.append(gene_cnv_mcnv_list[i])
                    x_input_list += [0.0] * 6
                elif node_type == 'Gene-METH':
                    x_input_list += [0.0] * 4
                    x_input_list.append(gene_meth_core_promoter_list[i - num_gene])
                    x_input_list.append(gene_meth_proximal_promoter_list[i - num_gene])
                    x_input_list.append(gene_meth_distal_promoter_list[i - num_gene])
                    x_input_list.append(gene_meth_downstream_list[i - num_gene])
                    x_input_list.append(gene_meth_upstream_list[i - num_gene])
                    x_input_list += [0.0] * 1
                elif node_type == 'Gene-PROT':
                    x_input_list += [0.0] * 9
                    x_input_list.append(gene_proteomics_list[i - num_gene - num_gene_meth])
            # import pdb; pdb.set_trace()
            x_input = np.array(x_input_list)
            x_batch = np.vstack((x_batch, x_input))
            # Combine score list
            y_input_list.append(y)
        # import pdb; pdb.set_trace()
        x_batch = np.delete(x_batch, 0, axis = 0)
        y_batch = np.array(y_input_list).reshape(tmp_batch_size, 1)
        print(x_batch.shape)
        print(y_batch.shape)
        return x_batch, y_batch


    def load_all_split(self, batch_size, k, processed_dataset, graph_output_folder):
        form_data_path = './' + graph_output_folder + '/form_data'
        # Load 100 percent data
        print('LOADING ALL SPLIT DATA...')
        # First load each split data
        for place_num in range(k):
            split_input_df = pd.read_csv(os.path.join(graph_output_folder, 'split-random-survival-label-' + str(place_num + 1) + '.csv'))
            input_num, input_dim = split_input_df.shape
            num_feature = 10
            final_annotation_gene_df = pd.read_csv(os.path.join(graph_output_folder, 'map-all-gene.csv'))
            gene_name_list = list(final_annotation_gene_df['Gene_name'])
            num_node = len(gene_name_list)
            x_split = np.zeros((1, num_feature * num_node))
            y_split = np.zeros((1, 1))
            for index in range(0, input_num, batch_size):
                if (index + batch_size) < input_num:
                    upper_index = index + batch_size
                else:
                    upper_index = input_num
                x_batch, y_batch = ROSMAP_LoadData().load_batch(index, upper_index, place_num, graph_output_folder, processed_dataset)
                x_split = np.vstack((x_split, x_batch))
                y_split = np.vstack((y_split, y_batch))
            x_split = np.delete(x_split, 0, axis = 0)
            y_split = np.delete(y_split, 0, axis = 0)
            y_split = y_split - 1.0
            print('-------SPLIT DATA SHAPE-------')
            print(x_split.shape)
            print(y_split.shape)
            np.save(form_data_path + '/x_split' + str(place_num + 1) + '.npy', x_split)
            np.save(form_data_path + '/y_split' + str(place_num + 1) + '.npy', y_split)
            

    def load_train_test(self, k, n_fold, graph_output_folder):
        form_data_path = './' + graph_output_folder + '/form_data'
        num_feature = 10
        final_annotation_gene_df = pd.read_csv(os.path.join(graph_output_folder, 'map-all-gene.csv'))
        gene_name_list = list(final_annotation_gene_df['Gene_name'])
        num_node = len(gene_name_list)
        xTr = np.zeros((1, num_feature * num_node))
        yTr = np.zeros((1, 1))
        for i in range(1, k + 1):
            if i == n_fold:
                print('--- LOADING ' + str(i) + '-TH SPLIT TEST DATA ---')
                xTe = np.load(form_data_path + '/x_split' + str(i) + '.npy')
                yTe = np.load(form_data_path + '/y_split' + str(i) + '.npy')
            else:
                print('--- LOADING ' + str(i) + '-TH SPLIT TRAINING DATA ---')
                x_split = np.load(form_data_path + '/x_split' + str(i) + '.npy')
                y_split = np.load(form_data_path + '/y_split' + str(i) + '.npy')
                print('--- COMBINING DATA ... ---')
                xTr = np.vstack((xTr, x_split))
                yTr = np.vstack((yTr, y_split))
        print('--- TRAINING INPUT SHAPE ---')
        xTr = np.delete(xTr, 0, axis = 0)
        yTr = np.delete(yTr, 0, axis = 0)
        print(xTr.shape)
        print(yTr.shape)
        np.save(form_data_path + '/xTr' + str(n_fold) + '.npy', xTr)
        np.save(form_data_path + '/yTr' + str(n_fold) + '.npy', yTr)
        print('--- TEST INPUT SHAPE ---')
        print(xTe.shape)
        print(yTe.shape)
        np.save(form_data_path + '/xTe' + str(n_fold) + '.npy', xTe)
        np.save(form_data_path + '/yTe' + str(n_fold) + '.npy', yTe)

    def combine_whole_dataset(self, k, graph_output_folder):
        form_data_path = './' + graph_output_folder + '/form_data'
        num_feature = 10
        final_annotation_gene_df = pd.read_csv(os.path.join(graph_output_folder, 'map-all-gene.csv'))
        gene_name_list = list(final_annotation_gene_df['Gene_name'])
        num_node = len(gene_name_list)
        xAll = np.zeros((1, num_feature * num_node))
        yAll = np.zeros((1, 1))
        for i in range(1, k + 1):
            print('--- LOADING ' + str(i) + '-TH SPLIT TRAINING DATA ---')
            x_split = np.load(form_data_path + '/x_split' + str(i) + '.npy')
            y_split = np.load(form_data_path + '/y_split' + str(i) + '.npy')
            print('--- COMBINING DATA ... ---')
            xAll = np.vstack((xAll, x_split))
            yAll = np.vstack((yAll, y_split))
        print('--- ALL DATASET INPUT SHAPE ---')
        xAll = np.delete(xAll, 0, axis = 0)
        yAll = np.delete(yAll, 0, axis = 0)
        print(xAll.shape)
        print(yAll.shape)
        np.save(form_data_path + '/xAll.npy', xAll)
        np.save(form_data_path + '/yAll.npy', yAll)
    
    def load_adj_edgeindex(self, graph_output_folder):
        form_data_path = './' + graph_output_folder + '/form_data'
        # Form a whole adjacent matrix
        gene_num_df = pd.read_csv(os.path.join(graph_output_folder, 'all-gene-edge-num.csv'))
        src_gene_list = list(gene_num_df['src'])
        dest_gene_list = list(gene_num_df['dest'])
        final_annotation_gene_df = pd.read_csv(os.path.join(graph_output_folder, 'map-all-gene.csv'))
        gene_name_list = list(final_annotation_gene_df['Gene_name'])
        num_node = len(gene_name_list)
        adj = np.zeros((num_node, num_node))
        # gene-gene adjacent matrix
        for i in range(len(src_gene_list)):
            row_idx = src_gene_list[i]
            col_idx = dest_gene_list[i]
            adj[row_idx, col_idx] = 1
            adj[col_idx, row_idx] = 1 # whether we want ['sym']
        # import pdb; pdb.set_trace()
        # np.save(form_data_path + '/adj.npy', adj)
        adj_sparse = sparse.csr_matrix(adj)
        sparse.save_npz(form_data_path + '/adj_sparse.npz', adj_sparse)
        # [edge_index]
        source_nodes, target_nodes = adj_sparse.nonzero()
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        np.save(form_data_path + '/edge_index.npy', edge_index)


if __name__ == "__main__":
    ### Dataset Selection
    # dataset = 'UCSC'
    dataset = 'ROSMAP'

    # ############# MOUDLE 1 #################
    # ### Load all split data into graph format
    # graph_output_folder = dataset + '-graph-data'
    # processed_dataset = dataset + '-process'
    # if os.path.exists('./' +graph_output_folder + '/form_data') == False:
    #     os.mkdir('./' +graph_output_folder + '/form_data')
    # k = 5
    # batch_size = 64
    # if dataset == 'UCSC':
    #     UCSC_LoadData().load_all_split(batch_size, k, processed_dataset, graph_output_folder)
    # elif dataset == 'ROSMAP':
    #     ROSMAP_LoadData().load_all_split(batch_size, k, processed_dataset, graph_output_folder)

    # ############## MOUDLE 2 #################
    # graph_output_folder = dataset + '-graph-data'
    # processed_dataset = dataset + '-process'
    # if dataset == 'UCSC':
    #     UCSC_LoadData().load_adj_edgeindex(graph_output_folder)
    # elif dataset == 'ROSMAP':
    #     ROSMAP_LoadData().load_adj_edgeindex(graph_output_folder)

    ################ MOUDLE 3 ###############
    # FORM N-TH FOLD TRAINING DATASET
    k = 5
    n_fold = 5
    graph_output_folder = dataset + '-graph-data'
    if dataset == 'UCSC':
        UCSC_LoadData().load_train_test(k, n_fold, graph_output_folder)
    elif dataset == 'ROSMAP':
        ROSMAP_LoadData().load_train_test(k, n_fold, graph_output_folder)

    # Check the default ratio of survival
    n_fold = 5
    graph_output_folder = dataset + '-graph-data'
    form_data_path = './' + graph_output_folder + '/form_data'
    yTr =  np.load(form_data_path + '/yTr' + str(n_fold) + '.npy')
    num_elements = yTr.size
    if dataset == 'UCSC':
        num_ones = np.count_nonzero(yTr) # Dead is one
        num_zeros = num_elements - num_ones # Alive is zero
        ratio_of_ones = num_ones / num_elements
        ratio_of_zeros = num_zeros / num_elements
        print("Ratio of Dead:", ratio_of_ones)
        print("Ratio of Alive:", ratio_of_zeros)
    elif dataset == 'ROSMAP':
        unique_numbers, occurrences = np.unique(yTr, return_counts=True)
        # To see the results
        for number, count in zip(unique_numbers, occurrences):
            print(f"Number {number} occurs {count} times")