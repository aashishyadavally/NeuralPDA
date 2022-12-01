import os
import tempfile
import subprocess
import time
import logging
import hashlib
from pathlib import Path
import hashlib
import tempfile
import argparse
import logging
import time
import json
import re
import shutil


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

joern_path = BASE_DIR # change it if you put this script outside of the Joern folder 


def process_tree(code, trees):
    """
    return:
        nodes: []
        cfg_edges: []
        pdg_edges: []
    """
    nodes = {}
    
    statement_types = get_statement_types(trees['AST'])
    for i, line in enumerate(code.splitlines()):
        s_type = ""
        if str(i+1) in statement_types:
            s_type = statement_types[ str(i+1) ]
        nodes[ str(i+1) ] = {
            'code': line.strip(),
            'label': s_type
        }
    # for k, v in nodes.items():
    #     print(k, v)
    
    ast_edges = get_edges(trees['AST'])
    cfg_edges = get_edges(trees['CFG'])
    pdg_edges = get_edges(trees['PDG'])
    rdef_edges = get_edges(trees['REACHING_DEF'])

    pdg_edges_with_type = process_pdg_edges(pdg_edges, rdef_edges)
   
    return nodes, ast_edges, cfg_edges, pdg_edges_with_type


def get_joern_id(line):
    p1 = re.compile(r'joern_id_[(](.*?)[)]', re.S)
    res = re.findall(p1, line)
    if len(res) > 0:
        return res[0]
    return ''

def get_joern_type(line):
    p1 = re.compile(r'joern_type_[(](.*?)[)]', re.S)
    res = re.findall(p1, line)
    if len(res) > 0:
        return res[0]
    return ''

def get_joern_name(line):
    p1 = re.compile(r'joern_name_[(](.*?)[)]', re.S)
    res = re.findall(p1, line)
    if len(res) > 0:
        return res[0]
    return ''

def get_joern_line_no(line):
    p1 = re.compile(r'joern_line_[(](.*?)[)]', re.S)
    res = re.findall(p1, line)
    if len(res) > 0:
        return res[0]
    return ''

def get_edges(tree):
    visited_edges = []
    cfg_edges = []
    for line in tree.splitlines():
        if line.find('" -->> "') > -1:
            a, b = line.split('" -->> "', 1)

            # id1 = get_joern_id(a)
            # id2 = get_joern_id(b)

            l1 = get_joern_line_no(a)
            l2 = get_joern_line_no(b)

            if l1 == '' or l2 == '':
                continue
            if int(l1) >= int(l2) :
                continue


            # t1 = get_joern_type(a)
            # t2 = get_joern_type(b)

            k = l1 + "_" + l2
            if k not in visited_edges:
                visited_edges.append(k)
                cfg_edges.append({
                    "node_out": l1,
                    "node_in": l2
                })
    return cfg_edges

def process_pdg_edges(pdg_edges, rdef_edges):
    pdg_edges_fixed = []

    rdef_keys = []
    for e in rdef_edges:
        k = e['node_out'] + "_" + e['node_in']
        # print("rdef_key:", k)
        if k not in rdef_keys:
            rdef_keys.append(k)
    
    for e in pdg_edges:
        k = e['node_out'] + "_" + e['node_in']
        
        if k in rdef_keys:
            e['edge_type'] = 'data_dependency'
        else:
            e['edge_type'] = 'control_dependency'
        # print("pdg_key:", k, e['edge_type'])
        pdg_edges_fixed.append(e)
    return pdg_edges_fixed
    
def get_statement_types(ast_tree):
    lines_nodes = {}
    statement_types = {}
    for line in ast_tree.splitlines():
        if line.find('" -->> "') > -1:
            a, b = line.split('" -->> "', 1)

            # id1 = get_joern_id(a)
            # id2 = get_joern_id(b)

            l1 = get_joern_line_no(a)
            l2 = get_joern_line_no(b)

            if l1 == '' or l2 == '':
                continue
            if int(l1) > int(l2) :
                # print("???", line)
                continue

            t1 = get_joern_type(a)
            t2 = get_joern_type(b)

            if int(l1) == int(l2) and l1 not in statement_types:
                statement_types[l1] = t1

            if l1 not in lines_nodes:
                lines_nodes[l1] = []

            lines_nodes[l1].append(line)
           
    return statement_types


def generate_prolog(code):
    if code.strip() == "":
        return ""
    tmp_dir = tempfile.TemporaryDirectory()
    md5_v = hashlib.md5(code.encode()).hexdigest()
    short_filename = "func_" + md5_v + ".cpp"
    with open(tmp_dir.name + "/" + short_filename, 'w') as f:
        f.write(code)

    process = subprocess.run([ joern_path + "/joern-parse", tmp_dir.name, "--out", tmp_dir.name + "/cpg.bin.zip"], check=True, capture_output=True, universal_newlines=True) 
    tree = subprocess.check_output(
        "cd " + joern_path + " && ./joern --script joern_cfg_to_dot2.sc --params cpgFile=" + tmp_dir.name + "/cpg.bin.zip",
        shell=True,
        universal_newlines=True,
    )
    full_tree = tree
    pos = tree.find("digraph g {")

    if pos > 0:
        tree = tree[pos:]
    tmp_dir.cleanup()
    return tree, process.stdout


def split_trees(tree):
    res = {}
    tree_type = ""
    tree_body = ""
    for line in tree.splitlines():
        if line.strip() == "":
            continue
        if line[0] == '#':
            if tree_type != "" and tree_body != "":
                res[ tree_type ] = tree_body
                tree_body = ""

            tree_type = line[2:].strip()
            # print(tree_type)
        elif tree_type != "":
            tree_body += line + "\n"
    if tree_body != "":
        res[tree_type] = tree_body
    return res


files = sorted(os.listdir('fragments2/txt_files'))
wrap_files = ['snippet7.txt', 'snippet13.txt', 'snippet57.txt', 'snippet61.txt',
              'snippet65.txt', 'snippet66.txt', 'snippet69.txt', 'snippet74.txt',
              'snippet77.txt', 'snippet78.txt', 'snippet80.txt', 'snippet82.txt',
              'snippet83.txt', 'snippet86.txt', 'snippet89.txt']


if not os.path.isdir('fragments2/snippets'):
    os.mkdir('fragments2/snippets')

for f in files:
    sid = f.split('.')[0]
    # Move code file
    if not os.path.isdir(f"fragments2/snippets/{sid}"):
        os.mkdir(f"fragments2/snippets/{sid}")

    shutil.copyfile(f"fragments2/txt_files/{f}", f"fragments2/snippets/{sid}/code.txt")

    with open(f"fragments2/txt_files/{f}", 'r') as file:
        code = file.read()

    # Wrap incomplete files around method body
    if f in wrap_files:
        code = "method(){\n" + code + "\n}"

    # Generate Joern output.
    op, log = generate_prolog(code)
    with open(f"fragments2/snippets/{sid}/joern.out", 'w') as file:
        file.write(log)

    # Extract human-readable form of dependencies.
    trees = split_trees(op)
    _, _, cfg_edges, pdg_edges_with_type = process_tree(code, trees)
    item = {
        'cfg_edges': cfg_edges,
        'pdg_edges': pdg_edges_with_type
    }
    # Save human-readable form of dependencies.
    with open(f"fragments2/snippets/{sid}/dependencies.json", 'w') as file:
        json.dump(item, file, indent=4)
