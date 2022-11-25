'''
    $ python infer.py -i <path_to_input(s)> -o {json|html} --lang java
'''
from __future__ import absolute_import
import os
import json
import pickle
import torch
import random
import logging
import argparse
from pathlib import Path

import numpy as np

import networkx as nx

from pyvis import network as pvnet

import torch.nn as nn

import transformers
from transformers import RobertaTokenizer

from model import ProgramDependenceModel


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def write_json(stmts, nodes, max_stmts, file_name):
    '''Writes output into a JSON file.
    '''
    output = {
        'nodes': [],
        'cfg_edges': [],
        'pdg_edges': [],
    }

    for sid, stmt in enumerate(stmts):
        output['nodes'].append(
            {
                'id': f'S{sid + 1}',
                'code': stmt,
            },
        )

    for gtype in ['cfg', 'pdg']:
        graph_nodes = nodes[gtype]
        
        for split_id, split_nodes in enumerate(graph_nodes):
            for node in split_nodes:
                output[f'{gtype}_edges'].append(
                    [
                        f'S{split_id * max_stmts + node[0] + 1}',
                        f'S{split_id * max_stmts + node[1] + 1}'
                    ]
                )

    Path('./outputs').mkdir(exist_ok=True)
    with open(f'./outputs/{file_name}_output.json', 'w') as file_obj:
        json.dump(output, file_obj, indent=4)
    return

  
def draw_graph(stmts, nodes, max_stmts, file_name):
    '''Draws graph into an HTML file.
    '''
    G = nx.MultiDiGraph()
    
    for sid, stmt in enumerate(stmts):
        G.add_node(sid, label=f'S{sid + 1}', title=stmt, size=20)

    for gtype in ['cfg', 'pdg']:
        if gtype == 'cfg':
            color = 'blue'
        else:
            color = 'red'

        graph_nodes = nodes[gtype]
        for split_id, split_nodes in enumerate(graph_nodes):
            for node in split_nodes:
                G.add_edge(
                    split_id * max_stmts + node[0],
                    split_id * max_stmts + node[1],
                    color=color,
                )

    net = pvnet.Network(directed=True, height='600px', width='800px')
    net.from_nx(G)

    Path('./outputs').mkdir(exist_ok=True)
    net.show(f'./outputs/{file_name}_viz.html')
    return


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("-i", default=None, type=str, required=True,
                        help="Path to input file(s).")
    parser.add_argument("-o", default=None, type=str, required=True,
                        choices=['json', 'html'], help="Output Format.")
    parser.add_argument("--lang", default=None, type=str, required=True,
                        choices=['cpp', 'java'], help="Programming language.")

    # Optional parameters
    parser.add_argument("--path_to_tok", type=str, default='./tokenizers/',
                        help="Path to trained RobertaTokenizer.")
    # Model parameters
    parser.add_argument("--max_tokens", default=32, type=int,
                        help="Maximum number of tokens in a statement")
    parser.add_argument("--max_stmts", default=8, type=int,
                        help="Maxmimum number of statements")
    parser.add_argument("--num_layers", default=6, type=int,
                        help="Number of layers for Transformer encoder")
    parser.add_argument("--num_layers_stmt", default=1, type=int,
                        help="Number of layers for Simple Self-Attention Network")
    parser.add_argument("--forward_activation", default="gelu", type=str,
                        help="Non-linear activation function in encoder")
    parser.add_argument("--hidden_size", default=512, type=int,
                        help="Hidden size of decoding MLP")
    parser.add_argument("--intermediate_size", default=2048, type=int,
                        help="Dimensionality of feed-forward layer in Transformer")
    parser.add_argument("--embedding_size", default=512, type=int,
                        help="Dimensionality of encoder layers")
    parser.add_argument("--num_heads", default=4, type=int,
                        help="Number of attention heads")
    parser.add_argument("--vocab_size", default=30522, type=int,
                        help="Vocabulary size")
    parser.add_argument("--use_stmt_types", action='store_true',
                        help="Use statement type information.")
    parser.add_argument("--no_pe", action='store_true',
                        help="Do not use statement-level position encoding.")
    parser.add_argument("--no_ssan", action='store_true',
                        help="Do not use self-attention network for intra-statement CL.")
    parser.add_argument("--no_tr", action='store_true',
                        help="Do not use transformer encoer for inter-statement CL.")
    # Other arguments
    parser.add_argument("--log_interval", default=10, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--dropout_rate", default=0.2, type=float,
                        help="Dropout rate.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for initialization")

    # Print arguments
    args = parser.parse_args()
    logger.info(args)

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.n_gpu = torch.cuda.device_count()

    logger.warning(f"Device: {args.device}, Number of GPU's: {args.n_gpu}")

    # Set seed
    set_seed(args.seed)

    path_to_tok = Path(args.path_to_tok) / f'tokenizer_{args.lang}'

    try:
        tokenizer = RobertaTokenizer(vocab_file=str(path_to_tok / 'vocab.json'),
                                     merges_file=str(path_to_tok / 'merges.txt'))
    except FileNotFoundError:
        logger.error("Trained tokenizer not found in specified path.")

    args.no_se = True
    model = ProgramDependenceModel(args)

    path_to_model = f'./outputs/ablation/{args.lang}_8/Epoch_4/model.ckpt'

    try:
        logger.info(f"Reload model from {path_to_model}")
        model.load_state_dict(torch.load(path_to_model), strict=False)
    except FileNotFoundError:
        logger.error("Trained model not found in specified path.")

    model.to(args.device)
    logger.info(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")

    # Put the model in evaluation mode--the dropout layers behave
    # differently during evaluation.
    model.eval()

    # Load code fragment.
    input_path = Path(args.i)
    if input_path.is_dir():
        input_files = list(input_path.iterdir())
    elif input_path.is_file():
        input_files = [input_path]

    logger.info(f"Loading code fragment(s)")

    for input_file in input_files:
        with open(str(input_file), 'r') as file_obj:
            fragment_lines = file_obj.readlines()

        # Split fragment into chunks containing ``max_stmts`` number of statements.
        fragment_splits = []
        for i in range(0, len(fragment_lines), args.max_stmts):
            fragment_splits.append(fragment_lines[i: i + args.max_stmts])

        nodes = {
            'cfg': [],
            'pdg': [],
        }

        for split_id, fragment_nodes in enumerate(fragment_splits):
            stmts_input_ids, stmts_input_masks = [], []

            for node in fragment_nodes:
                tokens = tokenizer.tokenize(node.strip())
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_ids = input_ids[:args.max_tokens]

                # The mask has 1 for real tokens and 0 for padding tokens.
                # Only real tokens are attended to.
                input_mask = [1] * len(input_ids)

                # Zero-pad up to the sequence length.
                padding_length = args.max_tokens - len(input_ids)
                input_ids = input_ids + ([0] * padding_length)
                input_mask = input_mask + ([0] * padding_length)

                stmts_input_ids.append(input_ids)
                stmts_input_masks.append(input_mask)

            for _ in range(args.max_stmts - len(fragment_nodes)):
                stmts_input_ids.append([0] * args.max_tokens)
                stmts_input_masks.append([0] * args.max_tokens)

            if len(fragment_nodes) < args.max_stmts:
                stmts_masks = [1] * len(fragment_nodes) + \
                              [0] * (args.max_stmts - len(fragment_nodes))
            else:
                stmts_masks = [1] * args.max_stmts

            stmts_input_ids = torch.tensor(stmts_input_ids, dtype=torch.long).to(args.device)
            stmts_input_masks = torch.tensor(stmts_input_masks, dtype=torch.long).to(args.device)
            stmts_masks = torch.tensor(stmts_masks, dtype=torch.long).to(args.device)

            stmts_input_ids = stmts_input_ids[None, :]
            stmts_input_masks = stmts_input_masks[None, :]
            stmts_masks = stmts_masks[None, :]

            # Tell pytorch not to bother with constructing the compute
            # graph during the forward pass, since this is only needed
            # for backpropogation (training).
            with torch.no_grad():
                preds = model(stmts_input_ids, stmts_input_masks, None, stmts_masks)

            for gtype in ['cfg', 'pdg']:
                nodes[gtype].append(((preds[gtype][0] > 0.5) == 1).nonzero(as_tuple=False).tolist())

        if args.o == 'html':
            draw_graph(fragment_lines, nodes, args.max_stmts, input_file.stem)
        elif args.o == 'json':
            write_json(fragment_lines, nodes, args.max_stmts, input_file.stem)


if __name__ == "__main__":
    main()
