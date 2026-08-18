import numpy as np

def trace_reachable_graph(nodes, output_id):
    lookup = {node['id']: node for node in nodes}

    reachable = set()

    def visit(node_id):
        if node_id in reachable:
            return
        reachable.add(node_id)
        for parent_id in lookup[node_id]['parents']:
            visit(parent_id)

    visit(output_id)

    reachable_ids = [node['id'] for node in nodes if node['id'] in reachable]

    edges = []
    for node in nodes:
        if node['id'] in reachable:
            for parent_id in node['parents']:
                edges.append([parent_id, node['id']])

    return (reachable_ids, edges)