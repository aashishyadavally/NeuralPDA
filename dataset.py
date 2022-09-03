import json
import pickle
from pathlib import Path

from tqdm import tqdm


class StatementNode:
    '''Data structure for statement nodes.
    '''
    def __init__(self, stmt, tag):
        self.stmt = stmt
        self.tag = tag


class InputExample:
    '''Data structure for input examples.
    '''
    def __init__(self, fid, nodes, edges=None, vd_label=None):
        '''
        '''
        self.fid = fid
        self.nodes = nodes
        self.edges = edges
        self.vd_label = vd_label


class FragmentExample:
    '''Data structure for code fragment examples.
    '''
    def __init__(self, fid, input_examples):
        '''
        '''
        self.fid = fid
        self.input_examples = input_examples


class DataProcessor:
    '''Processes data and creates examples.
    '''
    def __init__(self, lang, max_stmts, data_dir='./datasets'):
        '''Initialize :class: ``DataProcessor``.
        '''
        self.data_dir = data_dir
        self.max_stmts = max_stmts
        self.lang = lang

    def get_train_examples(self):
        '''Retrieve examples for train partition        
        '''
        examples = []
        try:
            examples = self.load_examples('train', self.lang, self.max_stmts)
        except FileNotFoundError:
            examples = self.create_examples('train', self.lang, self.max_stmts)
        return examples


    def get_val_examples(self):
        '''Retrieve examples for validation partition.
        '''
        try:
            examples = self.load_examples('val', self.lang, self.max_stmts)
        except FileNotFoundError:
            examples = self.create_examples('val', self.lang, self.max_stmts)
        return examples

    def get_test_examples(self):
        '''Retrieve examples for test partition.
        '''
        try:
            examples = self.load_examples('test', self.lang, self.max_stmts)
        except FileNotFoundError:
            examples = self.create_examples('test', self.lang, self.max_stmts)
        return examples

    def create_examples(self, stage, lang, max_stmts):
        '''Create ``InputExample`` objects.
        '''
        data_path = Path(self.data_dir) / f"{lang}_{max_stmts}" / f"functions_{stage}.json"
        with open(str(data_path), 'r') as file_obj:
            functions = json.load(file_obj)

        examples = []

        for function in tqdm(functions):
            fid = f"{stage}-{function['fid']}"
            nodes = [StatementNode(stmt=node['code'], tag=node['label']) \
                     for node in function['nodes'].values()]
            nodes_idx_mapper = dict(zip(function['nodes'].keys(),
                                    [str(i) for i in range(len(nodes))]))

            edges = {'ast': [], 'cfg': [], 'pdg': []}
            for graph in edges.keys():
                for edge in function[f'{graph}_edges']:
                    try:
                        _edge = {
                            'node_out': nodes_idx_mapper[edge['node_out']],
                            'node_in': nodes_idx_mapper[edge['node_in']],
                        }

                        if graph == 'pdg':
                            _edge = {**_edge,
                                     **{'edge_type': edge['edge_type']}
                            }
                        edges[graph].append(_edge)
                    except KeyError:
                        pass

            if 'vul' in function:
                vd_label = function['vul']
            else:
                vd_label = None

            examples.append(
                InputExample(fid=fid,
                             nodes=nodes,
                             edges=edges,
                             vd_label=vd_label)
            )

        self.save(examples, stage, lang, max_stmts)
        return examples

    def load_examples(self, stage, lang, max_stmts):
        '''
        '''
        path_to_file = Path(self.data_dir) / f"{lang}_{max_stmts}" / f"examples_{stage}.pkl"

        with open(str(path_to_file), 'rb') as handler:
            examples = pickle.load(handler)

        return examples

    def save(self, examples, stage, lang, max_stmts):
        '''
        '''
        path_to_save = Path(self.data_dir) / f"{lang}_{max_stmts}"
        path_to_save.mkdir(exist_ok=True, parents=True)
        path_to_file = path_to_save / f"examples_{stage}.pkl"

        with open(str(path_to_file), 'wb') as handler:
            pickle.dump(examples, handler)


class TopKDataProcessor:
    def __init__(self, lang, max_stmts, k, data_dir='./datasets'):
        '''Initialize :class: ``DataProcessor``.
        '''
        self.data_dir = data_dir
        self.max_stmts = max_stmts
        self.lang = lang
        self.k = k

    def get_examples(self):
        '''Retrieve examples for validation partition.
        '''
        data_path = Path(self.data_dir) / f"{self.lang}_{self.max_stmts}" / f"functions_test.json"
        with open(str(data_path), 'r') as file_obj:
            functions = json.load(file_obj)

        examples = []

        for function in tqdm(functions):
            fid = f"test-{self.k}-{function['fid']}"
            nodes = [StatementNode(stmt=node['code'], tag=node['label']) \
                     for node in function['nodes'].values()]
            nodes_idx_mapper = dict(zip(function['nodes'].keys(),
                                    [str(i) for i in range(len(nodes))]))

            edges = {'ast': [], 'cfg': [], 'pdg': []}
            for graph in edges.keys():
                for edge in function[f'{graph}_edges']:
                    try:
                        _edge = {
                            'node_out': nodes_idx_mapper[edge['node_out']],
                            'node_in': nodes_idx_mapper[edge['node_in']],
                        }

                        if graph == 'pdg':
                            _edge = {**_edge,
                                     **{'edge_type': edge['edge_type']}
                            }
                        edges[graph].append(_edge)
                    except KeyError:
                        pass

            nodes = nodes[:self.k]

            _edges = {'ast': [], 'cfg': [], 'pdg': []}
            for key, graph_edges in edges.items():
                for edge in graph_edges:
                    if int(edge['node_out']) < self.k and int(edge['node_in']) < self.k:
                        _edges[key].append(edge)

            examples.append(
                InputExample(fid=fid,
                             nodes=nodes,
                             edges=_edges,
                             vd_label=None)
            )

        return examples


class FragmentDataProcessor:
    def __init__(self, max_stmts, data_dir='./datasets'):
        '''Initialize :class: ``DataProcessor``.
        '''
        self.data_dir = data_dir
        self.max_stmts = max_stmts

    def get_examples(self):
        '''Retrieve examples for validation partition.
        '''
        data_path = Path(self.data_dir) / "fragments"
        data_files = sorted(data_path.iterdir())

        examples = []

        for data_file in tqdm(data_files):
            with open(str(data_file), 'r') as file_obj:
                fragment_lines = file_obj.readlines()

            fid = f"fragment-{data_file.stem}"

            fragment_splits = []
            for i in range(0, len(fragment_lines), self.max_stmts):
                fragment_split = fragment_lines[i: i + self.max_stmts]
                nodes = [StatementNode(stmt=stmt, tag=None) for stmt in fragment_split]
                fragment_splits.append(InputExample(
                    fid=f'{fid}_split{i}',
                    nodes=nodes,
                    edges=None,
                    vd_label=None,
                    )
                )

            examples.append(
                FragmentExample(fid=fid,
                                input_examples=fragment_splits,
                )
            )

        return examples

class BugDetectionDataProcessor(DataProcessor):
    def __init__(self, lang, max_stmts):
        '''Initialize :class: ``DataProcessor``.
        '''
        self.max_stmts = max_stmts
        self.lang = lang

    def create_examples(self, stage, lang, max_stmts):
        '''Create ``InputExample`` objects.
        '''
        data_path = Path("bug_detection/datasets") / lang / f"functions_{stage}.json"
        with open(str(data_path), 'r') as file_obj:
            functions = json.load(file_obj)

        examples = []

        for function in tqdm(functions):
            fid = f"{function['fid']}"
            nodes = [StatementNode(stmt=node['code'], tag=node['label']) \
                     for node in function['nodes'].values()]
            nodes_idx_mapper = dict(zip(function['nodes'].keys(),
                                    [str(i) for i in range(len(nodes))]))

            vd_label = function['vul']

            examples.append(
                InputExample(fid=fid,
                             nodes=nodes,
                             edges=None,
                             vd_label=vd_label)
            )

        self.save(examples, stage, lang, max_stmts)
        return examples

    def load_examples(self, stage, lang, max_stmts):
        '''
        '''
        path_to_file = Path("bug_detection/datasets") / lang / f"examples_{stage}.pkl"

        with open(str(path_to_file), 'rb') as handler:
            examples = pickle.load(handler)

        return examples

    def save(self, examples, stage, lang, max_stmts):
        '''
        '''
        path_to_save = Path("bug_detection/datasets") / lang
        path_to_save.mkdir(exist_ok=True, parents=True)
        path_to_file = path_to_save / f"examples_{stage}.pkl"

        with open(str(path_to_file), 'wb') as handler:
            pickle.dump(examples, handler)
