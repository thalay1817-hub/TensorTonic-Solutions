import heapq
import numpy as np

def autodiff_gradient_check(leaves, operations, output_id, h):
  
    lookup = {}
    combined_order = []

    for leaf in leaves:
        lookup[leaf['id']] = {'parents': [], 'is_leaf': True, 'op': None}
        combined_order.append(leaf['id'])

    for op_rec in operations:
        lookup[op_rec['id']] = {'parents': op_rec['parents'], 'is_leaf': False, 'op': op_rec['op']}
        combined_order.append(op_rec['id'])

    order_index = {node_id: i for i, node_id in enumerate(combined_order)}

    # Step 1: reachable set via backward traversal from output_id
    reachable = set()

    def visit(node_id):
        if node_id in reachable:
            return
        reachable.add(node_id)
        for parent_id in lookup[node_id]['parents']:
            visit(parent_id)

    visit(output_id)

    # Step 2: topological sort (Kahn's algorithm) restricted to reachable nodes
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
    while heap:
        _, node_id = heapq.heappop(heap)
        topo_order.append(node_id)
        for child_id in children[node_id]:
            in_degree[child_id] -= 1
            if in_degree[child_id] == 0:
                heapq.heappush(heap, (order_index[child_id], child_id))

    # Helper: forward pass given a dict of leaf values
    def forward(leaf_values):
        values = {}
        for node_id in topo_order:
            node = lookup[node_id]
            if node['is_leaf']:
                values[node_id] = leaf_values[node_id]
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
        return values

    # Base leaf values (float64)
    base_leaf_values = {leaf['id']: np.float64(leaf['value']) for leaf in leaves if leaf['id'] in reachable}

    # Forward pass at baseline
    values = forward(base_leaf_values)
    output_value = values[output_id]

    # Step 3: analytic backward pass
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

    analytic_gradients = {}
    for leaf in leaves:
        if leaf['id'] in reachable:
            analytic_gradients[leaf['id']] = float(grads[leaf['id']])

    # Step 4: numerical gradients via forward finite difference
    h = np.float64(h)
    numerical_gradients = {}
    for leaf in leaves:
        leaf_id = leaf['id']
        if leaf_id not in reachable:
            continue
        perturbed_leaf_values = dict(base_leaf_values)
        perturbed_leaf_values[leaf_id] = perturbed_leaf_values[leaf_id] + h
        perturbed_values = forward(perturbed_leaf_values)
        f_perturbed = perturbed_values[output_id]
        numerical_gradients[leaf_id] = float((f_perturbed - output_value) / h)

    # Step 5: max absolute disagreement
    max_error = 0.0
    for leaf in leaves:
        leaf_id = leaf['id']
        if leaf_id in reachable:
            diff = abs(analytic_gradients[leaf_id] - numerical_gradients[leaf_id])
            if diff > max_error:
                max_error = diff

    return (analytic_gradients, numerical_gradients, float(max_error))