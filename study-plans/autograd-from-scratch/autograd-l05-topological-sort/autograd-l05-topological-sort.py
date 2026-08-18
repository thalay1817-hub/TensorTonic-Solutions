import heapq

def topological_sort(nodes, output_id):
    lookup = {node['id']: node for node in nodes}
    order_index = {node['id']: i for i, node in enumerate(nodes)}

    # Step 1: find reachable set via backward traversal from output_id
    reachable = set()

    def visit(node_id):
        if node_id in reachable:
            return
        reachable.add(node_id)
        for parent_id in lookup[node_id]['parents']:
            visit(parent_id)

    visit(output_id)

    # Step 2: build children adjacency and in-degree (parents count) within reachable subgraph
    children = {node_id: [] for node_id in reachable}
    in_degree = {node_id: 0 for node_id in reachable}

    for node_id in reachable:
        for parent_id in lookup[node_id]['parents']:
            children[parent_id].append(node_id)
            in_degree[node_id] += 1

    # Step 3: Kahn's algorithm with a min-heap keyed by original input order
    heap = []
    for node_id in reachable:
        if in_degree[node_id] == 0:
            heapq.heappush(heap, (order_index[node_id], node_id))

    result = []
    while heap:
        _, node_id = heapq.heappop(heap)
        result.append(node_id)
        for child_id in children[node_id]:
            in_degree[child_id] -= 1
            if in_degree[child_id] == 0:
                heapq.heappush(heap, (order_index[child_id], child_id))

    return result