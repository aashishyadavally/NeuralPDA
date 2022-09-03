import json
import shutil
import logging
import argparse
from pathlib import Path

import numpy as np

from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)

from gensim.models import Word2Vec

from tokenizers import ByteLevelBPETokenizer

import torch


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def load_w2v_model(lang, max_stmts):
    model_path = f"{lang}_w2v.model"

    try:
        model = Word2Vec.load(str(model_path))
    except FileNotFoundError:
        print('Word2Vec model not found.')
        train_w2v_model(lang, max_stmts)
        model = Word2Vec.load(str(model_path))
    return model


def train_w2v_model(lang, max_stmts):
    data_path = Path('datasets') / f"{lang}_{max_stmts}" / f"functions_train.json"
    with open(str(data_path), 'r') as file_obj:
        data = json.load(file_obj)

    code_snippets = [data_item['func_code'].split() for data_item in data]
    print(f"Training Word2Vec model for Java.")
    model = Word2Vec(sentences=code_snippets, vector_size=512)
    print(f"Training Word2Vec model complete.")

    model.save(f"{lang}_w2v.model")
    print(f"Saved Word2Vec model.")


def train_hybrid_tokenizer(vocab_size, lang, max_stmts):
    data_path = Path('datasets') / f"{lang}_{max_stmts}" / f"functions_train.json"
    with open(str(data_path), 'r') as file_obj:
        functions = json.load(file_obj)

    code_snippets = []
    for function in functions:
        code_snippets.append(function['func_code'])

    # BPE tokenizer accepts inputs as a set of files.
    tmp_data_path = Path('tmp')
    tmp_data_path.mkdir(exist_ok=True, parents=True)

    files = []
    for _id, code in enumerate(code_snippets):
        tmp_file_path = str(tmp_data_path / f'tmp{_id}.txt')
        with open(tmp_file_path, 'w') as file_obj:
            file_obj.write(code)
        files.append(tmp_file_path)

    path_to_tok = Path(f'tokenizer_{lang}')
    path_to_tok.mkdir(exist_ok=True, parents=True)

    logger.info(f"Training BPE tokenizer for program dependence learning.")
    special_tokens = ["<s>", "</s>", "<unk>", "<mask>", "<pad>"]
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=files,
                    vocab_size=vocab_size,
                    min_frequency=2,
                    special_tokens=special_tokens)
    tokenizer.save_model(str(path_to_tok))
    shutil.rmtree(str(tmp_data_path))
    logger.info(f"Saved BPE tokenizer in /tokenizer_<lang>.")


def compute_metrics(label_pairs):
    metrics = {}

    for edge_type in ['cfg', 'pdg']:
        pairs = label_pairs[edge_type]
        true = [x[0] for x in pairs]
        pred = [x[1] for x in pairs]

        metrics[edge_type] = {
            'Accuracy': accuracy_score(true, pred),
            'Precision': precision_score(true, pred),
            'Recall': recall_score(true, pred),
            'F1-Score': f1_score(true, pred),
        }

    total_pairs = label_pairs['cfg'] + label_pairs['pdg']
    true = [x[0] for x in total_pairs]
    pred = [x[1] for x in total_pairs]

    metrics['Overall'] = {
        'Accuracy': accuracy_score(true, pred),
        'Precision': precision_score(true, pred),
        'Recall': recall_score(true, pred),
        'F1-Score': f1_score(true, pred),
    }

    return metrics


def compute_metrics_torch(label_pairs):
    metrics = {}

    for edge_type in ['cfg', 'pdg']:
        pairs = label_pairs[edge_type]
        true = torch.tensor([x[0] for x in pairs])
        pred = torch.tensor([x[1] for x in pairs])

        confusion_vector = pred / true
        # Element-wise division of the 2 tensors returns a new tensor which
        # holds a unique value for each case:
        #   1     where prediction and truth are 1 (True Positive)
        #   inf   where prediction is 1 and truth is 0 (False Positive)
        #   nan   where prediction and truth are 0 (True Negative)
        #   0     where prediction is 0 and truth is 1 (False Negative)

        TP = torch.sum(confusion_vector == 1).item()
        FP = torch.sum(confusion_vector == float('inf')).item()
        TN = torch.sum(torch.isnan(confusion_vector)).item()
        FN = torch.sum(confusion_vector == 0).item()

        accuracy = (TP + TN) / (TP + FP + TN + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 * precision * recall / (precision + recall)

        metrics[edge_type] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1_score,
        }
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_tok", action='store_true',
                        help="Train byte-level BPE tokenizer")
    parser.add_argument("--lang", default='java', type=str,
                        choices=['c', 'java'], help="Programming language.")
    parser.add_argument("--max_stmts", default=8, type=int,
                        help="Maxmimum number of statements")
    args = parser.parse_args()

    if args.train_tok:
        train_hybrid_tokenizer(vocab_size=30522, lang=args.lang, max_stmts=args.max_stmts)
