'''
Sample Run Command:
===================
1. Intrinsic Evaluation - Java
    $ python run.py --data_dir ./datasets/ --output_dir ./outputs-3/intrinsic/java_8 --lang java --do_train --use_stmt_types --max_stmts 8 --expt_name intrinsic-java-8

2. Intrinsic Evaluation - C++
    $ python run.py --data_dir ./datasets/ --output_dir ./outputs-3/intrinsic/cpp_8 --lang c --do_train --use_stmt_types --max_stmts 8 --expt_name intrinsic-cpp-8

3. Intrinsic Evaluation - C++
    $ python run.py --data_dir ./datasets/ --output_dir ./outputs-3/intrinsic/cpp_16 --lang c --do_train --max_stmts 16 --expt_name intrinsic-cpp-16

4. Ablation Study - No statement types
    $ python run.py --data_dir ./datasets/ --output_dir ./outputs-3/ablation/no_se --lang java --do_train --max_stmts 8 --expt_name ablation-no-se

5. Ablation Study - No position encodings
    $ python run.py --data_dir ./datasets/ --output_dir ./outputs-3/ablation/no_pe --lang java --do_train --use_stmt_types --max_stmts 8 --no_pe --expt_name ablation-no-pe

6. Ablation Study - No statement encodings
    $ python run.py --data_dir ./datasets --output_dir ./outputs-3/ablation/no_ssan --lang java --do_train --use_stmt_types --max_stmts 8 --expt_name ablation-no-ssan --no_ssan

7. Ablation Study - No transformer encoder
    $ python run.py --data_dir ./datasets --output_dir ./outputs-3/ablation/no_tr --lang java --do_train --use_stmt_types --max_stmts 8 --expt_name ablation-no-tr --no_tr

===================
Make Prediction 
===================
1. Java
    $ python run.py --lang java --do_predict --use_stmt_types --max_stmts 8 --load_model_path ./outputs-3/intrinsic/java/Epoch_4/model.ckpt

===================
Test scores 
===================
1. Intrinsic Evaluation - Java
    $ python run.py --data_dir ./datasets/ --output_dir ./no_output --lang java --do_eval --use_stmt_types --max_stmts 8 --load_model_path ./outputs-3/intrinsic/java_8/Epoch_4/model.ckpt

2. Intrinsic Evaluation - C++ (Maximum statements: 8)
    $ python run.py --data_dir ./datasets/ --output_dir ./no_output --lang c --do_eval --use_stmt_types --max_stmts 8 --load_model_path ./outputs-3/intrinsic/cpp_8/Epoch_4/model.ckpt

3. Intrinsic Evaluation - C++ (Maximum statements: 16, No statement types)
    $ python run.py --data_dir ./datasets/ --output_dir ./no_output --lang c --do_eval --max_stmts 16 --load_model_path ./outputs-3/intrinsic/cpp_16/Epoch_4/model.ckpt

4. Ablation Study - No statement types
    $ python run.py --data_dir ./datasets/ --output_dir ./no_output --lang java --do_eval --max_stmts 8 --load_model_path ./outputs-3/ablation/no_se/Epoch_4/model.ckpt

5. Ablation Study - No position encodings
    $ python run.py --data_dir ./datasets/ --output_dir ./no_output --lang java --do_eval --use_stmt_types --max_stmts 8 --no_pe --load_model_path ./outputs-3/ablation/no_pe/Epoch_4/model.ckpt

6. Ablation Study - No statement encodings
    $ python run.py --data_dir ./datasets/ --output_dir ./no_output --lang java --do_eval --use_stmt_types --max_stmts 8 --no_ssan --load_model_path ./outputs-3/ablation/no_ssan/Epoch_4/model.ckpt

7. Ablation Study - No transformer encoder
    $ python run.py --data_dir ./datasets/ --output_dir ./no_output --lang java --do_eval --use_stmt_types --max_stmts 8 --no_tr --load_model_path ./outputs-3/ablation/no_tr/Epoch_4/model.ckpt

===================
Evaluate Top-K
===================
    $ python run.py --data_dir ./datasets/ --output_dir ./no_output --lang java --do_eval_top_k --use_stmt_types --max_stmts 8 --load_model_path ./outputs-3/intrinsic/java_8/Epoch_4/model.ckpt
'''
from __future__ import absolute_import
import os
import sys
import pickle
import torch
import json
import random
import logging
import argparse
from pathlib import Path

import numpy as np

import pandas as pd

import wandb

from tqdm import tqdm, trange

import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.data import (DataLoader, Dataset, SequentialSampler,
                              RandomSampler, TensorDataset)

import transformers
from transformers import (AdamW, get_linear_schedule_with_warmup,
                          RobertaTokenizer)

from utils import (train_hybrid_tokenizer, compute_metrics,
                   compute_metrics_torch, load_w2v_model)
from dataset import DataProcessor, BugDetectionDataProcessor, TopKDataProcessor
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


def make_dataloader(args, examples, tokenizer, lang, stage):
    '''
    '''
    if stage == 'train':
        batch_size = args.train_batch_size
    else:
        batch_size = args.eval_batch_size

    path_to_save = Path(args.data_dir) / f"{lang}_{args.max_stmts}"
    path_to_save.mkdir(exist_ok=True, parents=True)
    path_to_file = path_to_save / f"dataloader_{stage}.pkl"

    if args.no_ssan:
        vocab = load_w2v_model(args.lang, args.max_stmts).wv.key_to_index

    try:
        with open(str(path_to_file), 'rb') as handler:
            dataloader = pickle.load(handler)
    except FileNotFoundError:
        edges = [getattr(example, 'edges') for example in examples]

        with open(f'statement_types_{lang}.json', 'r') as handler:
            stmt_type_dict = json.load(handler)

        (all_input_ids, all_input_masks, all_stmt_masks, all_stmt_types,
         all_cfg_labels, all_pdg_labels) = [], [], [], [], [], []

        for ex_index, example in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info(f"Writing example {ex_index} of {len(examples)}")

            # Truncate to max_stmts.
            nodes = example.nodes[:args.max_stmts]
            stmt_input_ids, stmt_input_masks = [], []
            for node in nodes:
                if args.no_ssan:
                    tokens = node.stmt.split()
                    input_ids = [vocab[token] for token in tokens if token in vocab]
                else:
                    tokens = tokenizer.tokenize(node.stmt)
                    input_ids = tokenizer.convert_tokens_to_ids(tokens)

                input_ids = input_ids[:args.max_tokens]

                # The mask has 1 for real tokens and 0 for padding tokens.
                # Only real tokens are attended to.
                input_mask = [1] * len(input_ids)

                # Zero-pad up to the sequence length.
                padding_length = args.max_tokens - len(input_ids)
                input_ids = input_ids + ([0] * padding_length)
                input_mask = input_mask + ([0] * padding_length)

                stmt_input_ids.append(input_ids)
                stmt_input_masks.append(input_mask)

            for _ in range(args.max_stmts - len(nodes)):
                stmt_input_ids.append([0] * args.max_tokens)
                stmt_input_masks.append([0] * args.max_tokens)

            all_input_ids.append(stmt_input_ids)
            all_input_masks.append(stmt_input_masks)

            stmt_types = [stmt_type_dict[node.tag] for node in example.nodes]

            if len(stmt_types) > args.max_stmts:
                stmt_types = stmt_types[:args.max_stmts]
            else:
                stmt_types += [0] * (args.max_stmts - len(stmt_types))
            all_stmt_types.append(stmt_types)

            if len(example.nodes) > args.max_stmts:
                all_stmt_masks.append([1] * args.max_stmts)
            else:
                stmt_masks = [1] * len(example.nodes) + \
                             [0] * (args.max_stmts - len(example.nodes))
                all_stmt_masks.append(stmt_masks)

            # Retrieving CFG edges.
            nodes_idx = list(str(x) for x in range(args.max_stmts))
            cfg_edges = dict(zip(nodes_idx,
                                 [[] for _ in range(args.max_stmts)]))
            pdg_edges = dict(zip(nodes_idx,
                                 [[] for _ in range(args.max_stmts)]))

            for edge in example.edges['cfg']:
                src_node, dest_node = edge['node_out'], edge['node_in']

                if src_node in nodes_idx and dest_node in nodes_idx:
                    if dest_node not in cfg_edges[src_node]:
                        cfg_edges[src_node].append(dest_node)

            cfg_labels = []
            for node_idx, nodes_out in cfg_edges.items():
                node_cfg_labels = [1 if str(index) in nodes_out else 0 \
                                   for index in range(args.max_stmts)]
                cfg_labels.append(node_cfg_labels)

            all_cfg_labels.append(cfg_labels)

            # Repeat same for PDG edges.
            for edge in example.edges['pdg']:
                src_node, dest_node = edge['node_out'], edge['node_in']

                if src_node in nodes_idx and dest_node in nodes_idx:
                    if dest_node not in pdg_edges[src_node]:
                        pdg_edges[src_node].append(dest_node)

            pdg_labels = []
            for node_idx, nodes_out in pdg_edges.items():
                node_pdg_labels = [1 if str(index) in nodes_out else 0 \
                                   for index in range(args.max_stmts)]
                pdg_labels.append(node_pdg_labels)

            all_pdg_labels.append(pdg_labels)

        dataset = TensorDataset(
                torch.tensor(all_input_ids, dtype=torch.long),
                torch.tensor(all_input_masks, dtype=torch.long),
                torch.tensor(all_stmt_types, dtype=torch.long),
                torch.tensor(all_stmt_masks, dtype=torch.long),
                torch.tensor(all_cfg_labels, dtype=torch.float),
                torch.tensor(all_pdg_labels, dtype=torch.float),
        )

        if stage == 'train':
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)

        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

        with open(str(path_to_file), 'wb') as handler:
            pickle.dump(dataloader, handler)

    return dataloader

def make_dataloader_top_k(args, examples, tokenizer, lang):
    '''
    '''
    batch_size = args.eval_batch_size

    edges = [getattr(example, 'edges') for example in examples]

    with open(f'statement_types_{lang}.json', 'r') as handler:
        stmt_type_dict = json.load(handler)

    (all_input_ids, all_input_masks, all_stmt_masks, all_stmt_types,
     all_cfg_labels, all_pdg_labels) = [], [], [], [], [], []

    for ex_index, example in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info(f"Writing example {ex_index} of {len(examples)}")

        # Truncate to max_stmts.
        nodes = example.nodes[:args.max_stmts]
        stmt_input_ids, stmt_input_masks = [], []
        for node in nodes:
            if args.no_ssan:
                tokens = node.stmt.split()
                input_ids = [vocab[token] for token in tokens if token in vocab]
            else:
                tokens = tokenizer.tokenize(node.stmt)
                input_ids = tokenizer.convert_tokens_to_ids(tokens)

            input_ids = input_ids[:args.max_tokens]

            # The mask has 1 for real tokens and 0 for padding tokens.
            # Only real tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = args.max_tokens - len(input_ids)
            input_ids = input_ids + ([0] * padding_length)
            input_mask = input_mask + ([0] * padding_length)

            stmt_input_ids.append(input_ids)
            stmt_input_masks.append(input_mask)

        for _ in range(args.max_stmts - len(nodes)):
            stmt_input_ids.append([0] * args.max_tokens)
            stmt_input_masks.append([0] * args.max_tokens)

        all_input_ids.append(stmt_input_ids)
        all_input_masks.append(stmt_input_masks)

        stmt_types = [stmt_type_dict[node.tag] for node in example.nodes]

        if len(stmt_types) > args.max_stmts:
            stmt_types = stmt_types[:args.max_stmts]
        else:
            stmt_types += [0] * (args.max_stmts - len(stmt_types))
        all_stmt_types.append(stmt_types)

        if len(example.nodes) > args.max_stmts:
            all_stmt_masks.append([1] * args.max_stmts)
        else:
            stmt_masks = [1] * len(example.nodes) + \
                         [0] * (args.max_stmts - len(example.nodes))
            all_stmt_masks.append(stmt_masks)

        # Retrieving CFG edges.
        nodes_idx = list(str(x) for x in range(args.max_stmts))
        cfg_edges = dict(zip(nodes_idx,
                             [[] for _ in range(args.max_stmts)]))
        pdg_edges = dict(zip(nodes_idx,
                             [[] for _ in range(args.max_stmts)]))

        for edge in example.edges['cfg']:
            src_node, dest_node = edge['node_out'], edge['node_in']

            if src_node in nodes_idx and dest_node in nodes_idx:
                if dest_node not in cfg_edges[src_node]:
                    cfg_edges[src_node].append(dest_node)

        cfg_labels = []
        for node_idx, nodes_out in cfg_edges.items():
            node_cfg_labels = [1 if str(index) in nodes_out else 0 \
                               for index in range(args.max_stmts)]
            cfg_labels.append(node_cfg_labels)

        all_cfg_labels.append(cfg_labels)

        # Repeat same for PDG edges.
        for edge in example.edges['pdg']:
            src_node, dest_node = edge['node_out'], edge['node_in']

            if src_node in nodes_idx and dest_node in nodes_idx:
                if dest_node not in pdg_edges[src_node]:
                    pdg_edges[src_node].append(dest_node)

        pdg_labels = []
        for node_idx, nodes_out in pdg_edges.items():
            node_pdg_labels = [1 if str(index) in nodes_out else 0 \
                               for index in range(args.max_stmts)]
            pdg_labels.append(node_pdg_labels)

        all_pdg_labels.append(pdg_labels)

    dataset = TensorDataset(
            torch.tensor(all_input_ids, dtype=torch.long),
            torch.tensor(all_input_masks, dtype=torch.long),
            torch.tensor(all_stmt_types, dtype=torch.long),
            torch.tensor(all_stmt_masks, dtype=torch.long),
            torch.tensor(all_cfg_labels, dtype=torch.float),
            torch.tensor(all_pdg_labels, dtype=torch.float),
    )
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    return dataloader


def make_dataloader_bug_detection(args, examples, tokenizer, lang, stage):
    '''
    '''
    path_to_save = Path("bug_detection/datasets") / lang
    path_to_file = path_to_save / f"dataloader_{stage}.pkl"

    try:
        with open(str(path_to_file), 'rb') as handler:
            dataloader = pickle.load(handler)
    except FileNotFoundError:
        with open(f'statement_types_{lang}.json', 'r') as handler:
            stmt_type_dict = json.load(handler)

        all_input_ids, all_input_masks, all_stmt_masks, all_stmt_types = [], [], [], []

        for ex_index, example in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info(f"Writing example {ex_index} of {len(examples)}")

            # Truncate to max_stmts.
            nodes = example.nodes[:args.max_stmts]
            stmt_input_ids, stmt_input_masks = [], []
            for node in nodes:
                tokens = tokenizer.tokenize(node.stmt)
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_ids = input_ids[:args.max_tokens]

                # The mask has 1 for real tokens and 0 for padding tokens.
                # Only real tokens are attended to.
                input_mask = [1] * len(input_ids)

                # Zero-pad up to the sequence length.
                padding_length = args.max_tokens - len(input_ids)
                input_ids = input_ids + ([0] * padding_length)
                input_mask = input_mask + ([0] * padding_length)

                stmt_input_ids.append(input_ids)
                stmt_input_masks.append(input_mask)

            for _ in range(args.max_stmts - len(nodes)):
                stmt_input_ids.append([0] * args.max_tokens)
                stmt_input_masks.append([0] * args.max_tokens)

            all_input_ids.append(stmt_input_ids)
            all_input_masks.append(stmt_input_masks)

            stmt_types = [stmt_type_dict[node.tag] for node in example.nodes]

            if len(stmt_types) > args.max_stmts:
                stmt_types = stmt_types[:args.max_stmts]
            else:
                stmt_types += [0] * (args.max_stmts - len(stmt_types))
            all_stmt_types.append(stmt_types)

            if len(example.nodes) > args.max_stmts:
                all_stmt_masks.append([1] * args.max_stmts)
            else:
                stmt_masks = [1] * len(example.nodes) + \
                             [0] * (args.max_stmts - len(example.nodes))
                all_stmt_masks.append(stmt_masks)

        dataset = TensorDataset(
                torch.tensor(all_input_ids, dtype=torch.long),
                torch.tensor(all_input_masks, dtype=torch.long),
                torch.tensor(all_stmt_types, dtype=torch.long),
                torch.tensor(all_stmt_masks, dtype=torch.long),
        )

        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.train_batch_size)

        with open(str(path_to_file), 'wb') as handler:
            pickle.dump(dataloader, handler)

    return dataloader

def evaluate(dataloader, model, args, epoch_stats=None):
    '''
    '''
    # Tracking variables
    total_eval_loss = 0
    # Evaluate data for one epoch
    label_pairs = {
        'cfg': [],
        'pdg': [],
    }

    if not epoch_stats:
        epoch_stats = {}

    for batch in dataloader:
        batch = tuple(t.to(args.device) for t in batch)
        # Tell pytorch not to bother with constructing the compute
        # graph during the forward pass, since this is only needed
        # for backpropogation (training).
        with torch.no_grad():
            input_ids, input_masks, stmt_types, stmt_masks, cfg_labels, pdg_labels = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]

            batch_loss, batch_label_pairs = model(input_ids,
                                                  input_masks,
                                                  stmt_types,
                                                  stmt_masks,
                                                  {
                                                    'cfg': cfg_labels,
                                                    'pdg': pdg_labels,
                                                  })
            # Accumulate the validation loss.
            total_eval_loss += batch_loss.item()

        # Move labels to CPU
        label_pairs['cfg'] += batch_label_pairs['cfg']
        label_pairs['pdg'] += batch_label_pairs['pdg']

    # Calculate the average loss over all of the batches.
    eval_loss = total_eval_loss / len(dataloader)
    eval_metrics = compute_metrics(label_pairs)

    # Record all statistics.
    return {
        **epoch_stats,
        **{'Epoch evaluation loss': eval_loss},
        **eval_metrics,
    }

def predict(dataloader, model, args):
    '''
    '''
    # Predict data for one epoch
    label_preds = {
        'cfg': [],
        'pdg': [],
    }

    for batch in tqdm(dataloader):
        batch = tuple(t.to(args.device) for t in batch)
        # Tell pytorch not to bother with constructing the compute
        # graph during the forward pass, since this is only needed
        # for backpropogation (training).
        with torch.no_grad():
            input_ids, input_masks, stmt_types, stmt_masks = batch[0], batch[1], batch[2], batch[3]

            batch_label_preds = model(input_ids, input_masks, stmt_types, stmt_masks, None)

        # Move labels to CPU
        label_preds['cfg'] += batch_label_preds['cfg']
        label_preds['pdg'] += batch_label_preds['pdg']

    return label_preds


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str,
                        help="Path to datasets directory.")
    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output directory where the model predictions ")
    parser.add_argument("--lang", default=None, type=str, required=True,
                        choices=['c', 'java'], help="Programming language.")

    ## Experiment parameters
    parser.add_argument("--expt_name", default='MAIN', type=str,
                        help="Name of experiment to log in Weights and Biases.")

    ## Model parameters
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
    parser.add_argument("--no_ssan", action='store_true',
                        help="Do not use self-attention network for statement encoding.")
    parser.add_argument("--no_pe", action='store_true',
                        help="Do not use statement-level position encoding.")
    parser.add_argument("--no_tr", action='store_true',
                        help="Do not use transformer encoder.")

    ## Experiment arguments
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_eval_top_k", action='store_true',
                        help="Whether to run eval on the partitioned dev set.")
    parser.add_argument("--do_predict", action='store_true',
                        help="Whether to predict on given dataset.")
    parser.add_argument("--log_interval", default=10, type=int)
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--dropout_rate", default=0.2, type=float,
                        help="Dropout rate.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--num_train_epochs", default=5, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

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

    # Make directory if output_dir does not exist
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        Path(output_dir).mkdir(exist_ok=True, parents=True)

    path_to_tok = Path(f'tokenizer_{args.lang}')

    if not path_to_tok.exists():
        logger.info('Training byte-level BPE tokenizer.')
        train_hybrid_tokenizer(args.vocab_size, args.lang, args.max_stmts)

    tokenizer = RobertaTokenizer(vocab_file=str(path_to_tok / 'vocab.json'),
                                 merges_file=str(path_to_tok / 'merges.txt'))

    if args.use_stmt_types:
        with open(f'statement_types_{args.lang}.json', 'r') as handler:
            stmt_type_dict = json.load(handler)
        args.num_stmt_types = len(stmt_type_dict)

    model = ProgramDependenceModel(args)

    if args.load_model_path is not None:
        logger.info(f"Reload model from {args.load_model_path}")
        model.load_state_dict(torch.load(args.load_model_path), strict=False)

    model.to(args.device)
    print(model)
    print()
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")

    if args.do_train:
        wandb.init(project="program-dependence-learning-4", name=args.expt_name)
        wandb.config.update(args)

        # Prepare training data loader
        dp = DataProcessor(lang=args.lang, data_dir=args.data_dir, max_stmts=args.max_stmts)
        logger.info('Loading training data.')
        train_examples = dp.get_train_examples()
        logger.info('Constructing data loader for training data.')
        train_dataloader = make_dataloader(args, train_examples, tokenizer, args.lang, 'train')

        # Prepare validation data loade.
        logger.info('Loading validation data.')
        eval_examples = dp.get_val_examples()
        logger.info('Constructing data loader for validation data.')
        eval_dataloader = make_dataloader(args, eval_examples, tokenizer, args.lang, 'val')

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() \
                        if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() \
                        if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate,
                          eps=args.adam_epsilon)
        max_steps = len(train_dataloader) * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=max_steps*0.1,
                                                    num_training_steps=max_steps)
        # Start training
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_examples)}")
        logger.info(f"  Batch size = {args.train_batch_size}")
        logger.info(f"  Num epochs = {args.num_train_epochs}")

        training_stats = []
        model.zero_grad()

        for epoch in range(args.num_train_epochs):
            tr_loss = 0
            num_train_steps = 0
            label_pairs = {
                'cfg': [],
                'pdg': [],
            }

            model.train()
            for _, batch in tqdm(enumerate(train_dataloader)):
                batch = tuple(t.to(args.device) for t in batch)
                (input_ids, input_masks, stmt_types, stmt_masks,\
                 cfg_labels, pdg_labels) = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]

                batch_loss, batch_label_pairs = model(input_ids,
                                                      input_masks,
                                                      stmt_types,
                                                      stmt_masks,
                                                      {
                                                        'cfg': cfg_labels,
                                                        'pdg': pdg_labels
                                                      }
                )
                tr_loss += batch_loss.item()
                label_pairs['cfg'] += batch_label_pairs['cfg']
                label_pairs['pdg'] += batch_label_pairs['pdg']
                num_train_steps += 1

                eval_metrics = compute_metrics(label_pairs)

                if _ % 5000 == 0:
                    step_loss = tr_loss / num_train_steps
                    step_cfg_acc = eval_metrics['cfg']['Accuracy']
                    step_pdg_acc = eval_metrics['pdg']['Accuracy']
                    logger.info(f"Epoch {epoch}, Training loss per 5000 steps: {step_loss}")
                    logger.info(f"Epoch {epoch}, Training accuracy per 5000 steps for "
                                f"CFG is {step_cfg_acc}, and for PDG is {step_pdg_acc}")

                batch_loss.backward()
                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # Update parameters
                optimizer.step()
                optimizer.zero_grad()
                # Update the learning rate
                scheduler.step()

            epoch_tr_loss = tr_loss / len(train_dataloader)
            epoch_eval_metrics = compute_metrics(label_pairs)
            epoch_cfg_acc = epoch_eval_metrics['cfg']['Accuracy']
            epoch_pdg_acc = epoch_eval_metrics['pdg']['Accuracy']

            logger.info(f"Epoch {epoch}, Training loss: {epoch_tr_loss}")
            logger.info(f"Epoch {epoch}, Training accuracy for CFG: {epoch_cfg_acc}")
            logger.info(f"Epoch {epoch}, Training accuracy for PDG: {epoch_pdg_acc}")
            
            # After the completion of one training epoch, measure performance
            # on validation set.
            logger.info('Measuring performance on validation set.')
            # Put the model in evaluation mode--the dropout layers behave
            # differently during evaluation.
            model.eval()
            training_stats = evaluate(eval_dataloader, model, args,
                                      epoch_stats={
                                        'Epoch training loss': epoch_tr_loss,
                                        'Epoch CFG accuracy': epoch_cfg_acc,
                                        'Epoch PDG accuracy': epoch_pdg_acc,
                                      }
            )
            print(training_stats)

            wandb.log({"Epoch Training Loss": epoch_tr_loss,
                       "Epoch Validation Loss": training_stats['Epoch evaluation loss'],
                       "Epoch CFG F1-Score": training_stats['cfg']['F1-Score'],
                       "Epoch PDG F1-Score": training_stats['pdg']['F1-Score']},
                      step=epoch)

            epoch_output_dir = Path(output_dir) / f'Epoch_{epoch}'
            epoch_output_dir.mkdir(exist_ok=True, parents=True)
            logger.info(f"Saving model to {epoch_output_dir}")
            torch.save(model.state_dict(), str(epoch_output_dir / 'model.ckpt'))

    if args.do_eval:
        # Put the model in evaluation mode--the dropout layers behave
        # differently during evaluation.
        import time
        model.eval()
        start = time.time()
        # Load test data.
        dp = DataProcessor(lang=args.lang, data_dir=args.data_dir, max_stmts=args.max_stmts)
        logger.info('Loading evaluation data.')
#        test_examples = dp.get_val_examples()
        dl_start = time.time()
        test_examples = dp.get_test_examples()
#        test_dataloader = make_dataloader(args, test_examples, tokenizer, args.lang, 'val')
        test_dataloader = make_dataloader(args, test_examples, tokenizer, args.lang, 'test')
        dl_end = time.time()
        print(f'Time for loading data: {dl_end - dl_start}')

        start_time = time.time()
        stats = evaluate(test_dataloader, model, args)
        print("  Test loss: {0:.4f}".format(stats['Epoch evaluation loss']))
        print("  CFG Accuracy: {0:.4f}".format(stats['cfg']['Accuracy']))
        print("  CFG Precision: {0:.4f}".format(stats['cfg']['Precision']))
        print("  CFG Recall: {0:.4f}".format(stats['cfg']['Recall']))
        print("  CFG F1-Score: {0:.4f}".format(stats['cfg']['F1-Score']))
        print()
        print("  PDG Accuracy: {0:.4f}".format(stats['pdg']['Accuracy']))
        print("  PDG Precision: {0:.4f}".format(stats['pdg']['Precision']))
        print("  PDG Recall: {0:.4f}".format(stats['pdg']['Recall']))
        print("  PDG F1-Score: {0:.4f}".format(stats['pdg']['F1-Score']))
        print()
        print("  Overall Accuracy: {0:.4f}".format(stats['Overall']['Accuracy']))
        print("  Overall Precision: {0:.4f}".format(stats['Overall']['Precision']))
        print("  Overall Recall: {0:.4f}".format(stats['Overall']['Recall']))
        print("  Overall F1-Score: {0:.4f}".format(stats['Overall']['F1-Score']))
        end_time = time.time()
        print(f"  Average time for predictions: {(end_time - start_time)/(len(test_dataloader) * args.eval_batch_size)}")
        end = time.time()
        print(f'Total number of examples: {len(test_dataloader)}')
        print(f'Total time: {end - start}') 

    if args.do_eval_top_k:
        # Put the model in evaluation mode--the dropout layers behave
        # differently during evaluation.
        model.eval()

        # Load test data.
        for k in range(3, args.max_stmts+1):
            dp = TopKDataProcessor(lang=args.lang, data_dir=args.data_dir,
                                   max_stmts=args.max_stmts, k = k)
            logger.info(f'Loading evaluation data for k={k}.')

            eval_examples = dp.get_examples()
            eval_dataloader = make_dataloader_top_k(args, eval_examples, tokenizer, args.lang)
            stats = evaluate(eval_dataloader, model, args)
            print()
            print("Evaluation results for k={}".format(k))
            print("===========================")
            print("  Test loss: {0:.4f}".format(stats['Epoch evaluation loss']))
            print("  CFG Accuracy: {0:.4f}".format(stats['cfg']['Accuracy']))
            print("  CFG Precision: {0:.4f}".format(stats['cfg']['Precision']))
            print("  CFG Recall: {0:.4f}".format(stats['cfg']['Recall']))
            print("  CFG F1-Score: {0:.4f}".format(stats['cfg']['F1-Score']))
            print()
            print("  PDG Accuracy: {0:.4f}".format(stats['pdg']['Accuracy']))
            print("  PDG Precision: {0:.4f}".format(stats['pdg']['Precision']))
            print("  PDG Recall: {0:.4f}".format(stats['pdg']['Recall']))
            print("  PDG F1-Score: {0:.4f}".format(stats['pdg']['F1-Score']))
            print()
            print("  Overall Accuracy: {0:.4f}".format(stats['Overall']['Accuracy']))
            print("  Overall Precision: {0:.4f}".format(stats['Overall']['Precision']))
            print("  Overall Recall: {0:.4f}".format(stats['Overall']['Recall']))
            print("  Overall F1-Score: {0:.4f}".format(stats['Overall']['F1-Score']))

    if args.do_predict:
        # Put the model in evaluation mode--the dropout layers behave
        # differently during evaluation.
        model.eval()

        # Load test data.
        dp = BugDetectionDataProcessor(lang=args.lang, max_stmts=args.max_stmts)
        logger.info('Loading prediction data.')

        for stage in ['train', 'val', 'test']:
            if stage == 'train':
                examples = dp.get_train_examples()
            elif stage == 'val':
                examples = dp.get_val_examples()
            elif stage == 'test':
                examples = dp.get_test_examples()

            dataloader = make_dataloader_bug_detection(args, examples, tokenizer,
                                                       args.lang, stage)

            logger.info(f'Making predictions for {stage} data.')
            preds = predict(dataloader, model, args)
            logger.info(f'Predictions complete for {stage} data.')

            functions = []
            for ex_id, example in tqdm(enumerate(examples)):
                code, nodes = "", {}
                for node_id, node in enumerate(example.nodes):
                    code += f"{node.stmt}\n"
                    nodes[str(node_id + 1)] = {
                        'code': node.stmt,
                        'label': node.tag,
                    }

                cfg_nodes = ((preds['cfg'][ex_id] > 0.5) == 1).nonzero(as_tuple=False).tolist()
                cfg_nodes = [{'node_out': str(node[0] + 1), 'node_in': str(node[1] + 1)} \
                             for node in cfg_nodes]
                pdg_nodes = ((preds['pdg'][ex_id] > 0.5) == 1).nonzero(as_tuple=False).tolist()
                pdg_nodes = [{'node_out': str(node[0] + 1), 'node_in': str(node[1] + 1)} \
                             for node in pdg_nodes]

                function_output = {
                    "func_id": example.fid,
                    "code": code,
                    "nodes": nodes,
                    "cfg_edges": cfg_nodes,
                    "pdg_edges": pdg_nodes,
                    "vul": example.vd_label,
                }
                functions.append(function_output)

            path_to_save = Path("bug_detection/datasets") / args.lang / f'functions_star_{stage}.json'

            with open(str(path_to_save), 'w') as fileobj:
                json.dump(functions, fileobj, indent=4)


#            if stage == 'test':
#                stats = evaluate(dataloader, model, args)
#                print("  Test loss: {0:.4f}".format(stats['Epoch evaluation loss']))
#                print("  CFG Accuracy: {0:.4f}".format(stats['cfg']['Accuracy']))
#                print("  CFG Precision: {0:.4f}".format(stats['cfg']['Precision']))
#                print("  CFG Recall: {0:.4f}".format(stats['cfg']['Recall']))
#                print("  CFG F1-Score: {0:.4f}".format(stats['cfg']['F1-Score']))
#                print()
#                print("  PDG Accuracy: {0:.4f}".format(stats['pdg']['Accuracy']))
#                print("  PDG Precision: {0:.4f}".format(stats['pdg']['Precision']))
#                print("  PDG Recall: {0:.4f}".format(stats['pdg']['Recall']))
#                print("  PDG F1-Score: {0:.4f}".format(stats['pdg']['F1-Score']))
#                print()
#                print("  Overall Accuracy: {0:.4f}".format(stats['Overall']['Accuracy']))
#                print("  Overall Precision: {0:.4f}".format(stats['Overall']['Precision']))
#                print("  Overall Recall: {0:.4f}".format(stats['Overall']['Recall']))
#                print("  Overall F1-Score: {0:.4f}".format(stats['Overall']['F1-Score']))


if __name__ == "__main__":
    main()
