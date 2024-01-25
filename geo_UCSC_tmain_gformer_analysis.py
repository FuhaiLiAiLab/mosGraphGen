import os
import pdb
import torch
import argparse
import tensorboardX
import numpy as np
import pandas as pd
import torch.nn as nn
from scipy import sparse
from torch.autograd import Variable

import utils
from geo_loader.read_geograph import read_batch
from geo_loader.geograph_sampler import GeoGraphLoader
from enc_dec.geo_gformer_decoder_analysis import GraphFormerDecoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score

# Parse arguments from command line
def arg_parse():
    parser = argparse.ArgumentParser(description='COEMBED ARGUMENTS.')
    # Add following arguments
    parser.add_argument('--cuda', dest = 'cuda',
                help = 'CUDA.')
    parser.add_argument('--parallel', dest = 'parallel',
                help = 'Parrallel Computing')
    parser.add_argument('--GPU IDs', dest = 'gpu_ids',
                help = 'GPU IDs')
    parser.add_argument('--add-self', dest = 'adj_self',
                help = 'Graph convolution add nodes themselves.')
    parser.add_argument('--model', dest = 'model',
                help = 'Model load.')
    parser.add_argument('--lr', dest = 'lr', type = float,
                help = 'Learning rate.')
    parser.add_argument('--batch-size', dest = 'batch_size', type = int,
                help = 'Batch size.')
    parser.add_argument('--num_workers', dest = 'num_workers', type = int,
                help = 'Number of workers to load data.')
    parser.add_argument('--epochs', dest = 'num_epochs', type = int,
                help = 'Number of epochs to train.')
    parser.add_argument('--input-dim', dest = 'input_dim', type = int,
                help = 'Input feature dimension')
    parser.add_argument('--hidden-dim', dest = 'hidden_dim', type = int,
                help = 'Hidden dimension')
    parser.add_argument('--output-dim', dest = 'output_dim', type = int,
                help = 'Output dimension')
    parser.add_argument('--num-gc-layers', dest = 'num_gc_layers', type = int,
                help = 'Number of graph convolution layers before each pooling')
    parser.add_argument('--dropout', dest = 'dropout', type = float,
                help = 'Dropout rate.')

    # Set default input argument
    parser.set_defaults(cuda = '0',
                        parallel = False,
                        add_self = '0', # 'add'
                        model = '0', # 'load'
                        lr = 0.008,
                        clip = 2.0,
                        batch_size = 64,
                        num_workers = 1,
                        num_epochs = 50,
                        num_head = 2,
                        input_dim = 8,
                        hidden_dim = 24,
                        output_dim = 24,
                        dropout = 0.01)
    return parser.parse_args()

def build_geogformer_model(args, device, graph_output_folder, num_class):
    print('--- BUILDING UP GraphFormer MODEL ... ---')
    # Get parameters
    # [num_gene, (adj)node_num]
    final_annotation_gene_df = pd.read_csv(os.path.join(graph_output_folder, 'map-all-gene.csv'))
    gene_name_list = list(final_annotation_gene_df['Gene_name'])
    node_num = len(gene_name_list)
    # [num_edge]
    gene_num_edge_df = pd.read_csv(os.path.join(graph_output_folder, 'merged-gene-edge-num-all.csv'))
    num_edge = gene_num_edge_df.shape[0]
    # Build up model
    model = GraphFormerDecoder(input_dim=args.input_dim, hidden_dim=args.hidden_dim, embedding_dim=args.output_dim, 
                    node_num=node_num, num_head=args.num_head, device=device, num_class=num_class)
    model = model.to(device)
    return model

def test_geogformer_model(dataset_loader, model, device, args, fold_n):
    batch_loss = 0
    for batch_idx, data in enumerate(dataset_loader):
        x = Variable(data.x, requires_grad=False).to(device)
        edge_index = Variable(data.edge_index, requires_grad=False).to(device)
        label = Variable(data.label, requires_grad=True).to(device)
        # This will use method [def forward()] to make prediction
        output, ypred = model(x, edge_index, fold_n)
        loss = model.loss(output, label)
        batch_loss += loss.item()
    return model, batch_loss, ypred


def test_geogformer(args, fold_n, test_load_path, test_save_path, device, graph_output_folder, num_class):
    # BUILD [GraphFormer, DECODER] MODEL
    model = build_geogformer_model(args, device, graph_output_folder, num_class)
    model.load_state_dict(torch.load(test_load_path, map_location=device))

    # Test model on test dataset
    form_data_path = './' + graph_output_folder + '/form_data'
    xTe = np.load(form_data_path + '/xTe' + str(fold_n) + '.npy')
    yTe = np.load(form_data_path + '/yTe' + str(fold_n) + '.npy')
    edge_index = torch.from_numpy(np.load(form_data_path + '/edge_index.npy') ).long() 

    dl_input_num = xTe.shape[0]
    batch_size = args.batch_size
    # Clean result previous epoch_i_pred files
    path = test_save_path
    # [num_feature, num_node]
    num_feature = 8
    final_annotation_gene_df = pd.read_csv(os.path.join(graph_output_folder, 'map-all-gene.csv'))
    gene_name_list = list(final_annotation_gene_df['Gene_name'])
    num_node = len(gene_name_list)

    # Run test model
    model.eval()
    index = 0
    upper_index = 1
    geo_datalist = read_batch(index, upper_index, xTe, yTe, num_feature, num_node, edge_index, graph_output_folder)
    dataset_loader, node_num, feature_dim = GeoGraphLoader.load_graph(geo_datalist, args)
    print('TEST MODEL...')
    # import pdb; pdb.set_trace()
    model, batch_loss, batch_ypred = test_geogformer_model(dataset_loader, model, device, args, fold_n)
    print('BATCH LOSS: ', batch_loss)


if __name__ == "__main__":
    # Parse argument from terminal or default parameters
    prog_args = arg_parse()

    # Check and allocate resources
    device, prog_args.gpu_ids = utils.get_available_devices()
    # Manual set
    device = torch.device('cuda:0') 
    torch.cuda.set_device(device)
    print('MAIN DEVICE: ', device)
    # Single gpu
    prog_args.gpu_ids = [0]
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    # Dataset Selection
    dataset = 'UCSC'
    
    if os.path.exists('./'+ dataset + '-analysis') == False:
        os.mkdir('./'+ dataset + '-analysis')

    ### Train the model
    # Train [FOLD-1x]
    fold_n = 5
    graph_output_folder = dataset + '-graph-data'
    yTr = np.load('./' + graph_output_folder + '/form_data/yTr' + str(fold_n) + '.npy')
    # yTr = np.load('./' + graph_output_folder + '/form_data/y_split1.npy')
    unique_numbers, occurrences = np.unique(yTr, return_counts=True)
    num_class = len(unique_numbers)

    ### TEST THE MODEL
    test_load_path = './' + dataset + '-result/gformer/fold_' + str(fold_n) + '/best_train_model.pt'
    test_save_path = './' + dataset + '-result/gformer/fold_' + str(fold_n)
    test_geogformer(prog_args, fold_n, test_load_path, test_save_path, device, graph_output_folder, num_class)

