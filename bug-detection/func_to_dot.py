import json

def parse_func_to_dot(split, star):
    if star:
        filename = f'functions_star_{split}.json'
        func_key = 'func_id'
        dots_folder = 'dots_star'
    else:
        filename = f'functions_{split}.json'
        func_key = 'fid'
        dots_folder = 'dots'

    with open(filename, 'r') as fileobj:
        functions = json.load(fileobj)

    for function in functions:
        name = function[func_key]
        label = function['vul']
        dot_contents = "digraph FUN1 { \n"

        for node_id, node in function['nodes'].items():
            dot_contents += f"\"{node_id}\" [label = \"{node['code']}\" ]\n"

    for node in function[f'pdg_edges']:
        node_out, node_in = node['node_out'], node['node_in']
        dot_contents += f"  \"{node_out}\" -> \"{node_in}\"  [ label = \"PDG\" ]\n"
        dot_contents += "}\n"

        with open(f'{dots_folder}/{split}/{split}_labels.txt', 'a') as fileobj:
            fileobj.write(f'{name}\t{label}\n')

        with open(f'{dots_folder}/{split}/{name}.dot', 'w') as fileobj:
            fileobj.write(dot_contents)


if __name__ == '__main__':
    for split in ['train', 'val', 'test']:
        parse_func_to_dot(split, star=True)
