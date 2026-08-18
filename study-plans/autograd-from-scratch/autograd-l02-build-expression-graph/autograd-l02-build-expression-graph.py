import numpy as np

def build_expression_graph(leaves, operations):
    nodes = []
    lookup = {}

    for leaf in leaves:
        record = {
            'id': leaf['id'],
            'data': float(np.float64(leaf['data'])),
            'grad': 0.0,
            'op': '',
            'parents': []
        }
        nodes.append(record)
        lookup[record['id']] = record

    final_id = leaves[-1]['id'] if leaves else None

    for op_rec in operations:
        left_node = lookup[op_rec['left']]
        right_node = lookup[op_rec['right']]
        x = np.float64(left_node['data'])
        y = np.float64(right_node['data'])

        if op_rec['op'] == '+':
            z = x + y
        else:
            z = x * y

        record = {
            'id': op_rec['id'],
            'data': float(z),
            'grad': 0.0,
            'op': op_rec['op'],
            'parents': [op_rec['left'], op_rec['right']]
        }
        nodes.append(record)
        lookup[record['id']] = record
        final_id = record['id']

    return (nodes, final_id)