import heapq
import numpy as np

def shared_path_gradient_accumulation(leaves, operations, output_id):
    # Build unified lookup
    lookup = {}
    combined_order = []

    for leaf in leaves:
        lookup[leaf['id']] = {'parents': [], 'is_leaf': True, 'value': np.float64(leaf['value']), 'op': None}
        combined_order.append(leaf['id'])

    for op_rec in operations:
        lookup[op_rec['id']] = {'parents': op_rec['parents'], 'is_leaf': False, 'value': None, 'op': op_rec['op']}
        combined_order.append(op_rec['id'])

    order_index = {node_id: i for i, node_id in enumerate(combined_order)}

    # Step 1: reachable set via backward traversal from output_id (visit each parent once for reachability)
    reachable = set()

    def visit(node_id):
        if node_id in reachable:
            return
        reachable.add(node_id)
        for parent_id in lookup[node_id]['parents']:
            visit(parent_id)

    visit(output_id)

    # Step 2: topological sort restricted to reachable nodes.
    # in_degree counts EACH occurrence of a parent in a child's parents list (edge multiplicity),
    # since each occurrence is a distinct edge/path contribution.
    children = {node_id: [] for node_id in reachable}
    in_degree = {node_id: 0 for node_id in reachable}

    for node_id in reachable:
        for parent_id in lookup[node_id]['parents']:
            children[parent_id].append(node_id)
            in_degree[node_id] += 1

    heap = []
    for node_id in reachable:
        if in_degree[node_id] == 0:
            heapq.heappush(heap, (order_index[node_id], node_id))

    topo_order = []
    # Use a separate "remaining in-degree" tracker; but since multiple edges from same parent
    # to same child exist, we must decrement once per edge, not once per unique child.
    # Standard Kahn's works fine as long as we decrement in_degree once per edge traversal.
    processed_edge_count = {node_id: 0 for node_id in reachable}

    while heap:
        _, node_id = heapq.heappop(heap)
        topo_order.append(node_id)
        for child_id in children[node_id]:
            in_degree[child_id] -= 1
            if in_degree[child_id] == 0:
                heapq.heappush(heap, (order_index[child_id], child_id))

    # Step 3: forward pass in topological order
    values = {}
    for node_id in topo_order:
        node = lookup[node_id]
        if node['is_leaf']:
            values[node_id] = node['value']
        else:
            op = node['op']
            parents = node['parents']
            if op == 'add':
                a_id, b_id = parents
                values[node_id] = values[a_id] + values[b_id]
            elif op == 'mul':
                a_id, b_id = parents
                values[node_id] = values[a_id] * values[b_id]
            elif op == 'tanh':
                (a_id,) = parents
                values[node_id] = np.tanh(values[a_id])

    # Step 4: backward pass, seed output gradient with 1.0, accumulate contributions
    grads = {node_id: np.float64(0.0) for node_id in reachable}
    grads[output_id] = np.float64(1.0)

    for node_id in reversed(topo_order):
        node = lookup[node_id]
        if node['is_leaf']:
            continue
        op = node['op']
        parents = node['parents']
        g = grads[node_id]

        if op == 'add':
            a_id, b_id = parents
            grads[a_id] += g
            grads[b_id] += g
        elif op == 'mul':
            a_id, b_id = parents
            a_val = values[a_id]
            b_val = values[b_id]
            grads[a_id] += g * b_val
            grads[b_id] += g * a_val
        elif op == 'tanh':
            (a_id,) = parents
            t = values[node_id]
            grads[a_id] += g * (1 - t * t)

    # Step 5: build gradients dict for leaves in leaf input order, only reachable
    gradients = {}
    for leaf in leaves:
        if leaf['id'] in reachable:
            gradients[leaf['id']] = float(grads[leaf['id']])

    return gradients