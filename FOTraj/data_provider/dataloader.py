import ast
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from fsspec.registry import default
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import networkx as nx


def load_data(csv_file):
    data = []
    with open(csv_file, 'r') as file:
        for line in file:
            traj = ast.literal_eval(line.strip())
            data.append(traj)
    return data


def build_graph(traj):
    graph = nx.DiGraph()

    stay_times = defaultdict(int)
    stay_counts = defaultdict(int)

    prev_id = prev_grid_id = prev_time = None
    node_ids = {}
    current_index = 0

    for i, (id, time_tuple) in enumerate(traj):
        time = datetime(*time_tuple)
        grid_id = id
        if id not in node_ids:
            node_ids[id] = current_index
            current_index += 1

        if prev_id is not None and prev_grid_id != grid_id:
            edge_attr = {
                'travel_time': int((time - prev_time).total_seconds() + stay_times[(prev_id, prev_grid_id)]),
            }
            if graph.has_edge(prev_id, id):
                graph[prev_id][id]['travel_time'] += edge_attr['travel_time']
            else:
                graph.add_edge(prev_id, id, **edge_attr)

            stay_counts[(prev_id, prev_grid_id)] = 0
            stay_times[(prev_id, prev_grid_id)] = 0

        stay_counts[(id, grid_id)] += 1
        if prev_id == id:
            stay_times[(id, grid_id)] += (time - prev_time).total_seconds()

        prev_id, prev_grid_id, prev_time = id, grid_id, time

    node_num = len(node_ids)
    adj_matrix = np.zeros((64, 64), dtype=np.float32)
    for u, v in graph.edges():
        if u in node_ids and v in node_ids:
            u_idx = node_ids[u]
            v_idx = node_ids[v]
            if u_idx < 64 and v_idx < 64:
                adj_matrix[u_idx, v_idx] = graph[u][v]['travel_time']

    adj_mask = np.zeros((64, 64), dtype=bool)
    for i in range(min(node_num, 64)):
        adj_mask[i, :node_num] = True
    return graph, adj_matrix, adj_mask


def build_sparse_graph(data, num_workers=4):
    with Pool(num_workers) as pool:
        trajectory_graphs = pool.map(build_graph, data)
    graphs, adj_matrices, adj_mask = zip(*trajectory_graphs)
    return graphs, adj_matrices, adj_mask


class GraphDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.graphs, self.adj_matrices, self.adj_mask = build_sparse_graph(data)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        adj_matrix = self.adj_matrices[idx]
        adj_mask = self.adj_mask[idx]

        node_ids = list(graph.nodes())
        edge_list = list(graph.edges())

        edge_attr = []
        for u, v in edge_list:
            edge_attr.append(graph[u][v].get('travel_time', 0))

        node_tensor = torch.tensor(node_ids, dtype=torch.long)
        edge_indices = torch.tensor(edge_list, dtype=torch.long)
        edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.long)
        adj_tensor = torch.tensor(adj_matrix, dtype=torch.float)
        adj_mask = torch.tensor(adj_mask, dtype=torch.bool)

        return node_tensor, edge_indices, edge_attr_tensor, adj_tensor, adj_mask


def collate_fn(batch):
    nodes = [item[0] for item in batch]
    edge_indices = [item[1] for item in batch]
    edge_attrs = [item[2] for item in batch]
    adj_matrices = [item[3] for item in batch]
    adj_masks = [item[4] for item in batch]

    max_nodes = 64
    nodes_mask = []
    for i in range(len(nodes)):
        if nodes[i].size(0) > max_nodes:
            nodes[i] = nodes[i][:max_nodes]
        padding = max_nodes - nodes[i].size(0)
        if padding > 0:
            nodes[i] = torch.cat([nodes[i], torch.zeros(padding, dtype=torch.long)])
        mask = torch.cat([torch.ones(nodes[i].size(0) - padding, dtype=torch.bool),
                          torch.zeros(padding, dtype=torch.bool)])
        nodes_mask.append(mask)

    max_edges = 64
    edge_mask = []
    edge_attrs_mask = []
    padded_edge_indices = []
    for i in range(len(edge_attrs)):
        edge_attrs[i] = edge_attrs[i].clamp(max=1599)
        if edge_attrs[i].size(0) > max_edges:
            edge_attrs[i] = edge_attrs[i][:max_edges]
        padding = max_edges - edge_attrs[i].size(0)
        if edge_attrs[i].dim() == 1:
            if padding > 0:
                edge_attrs[i] = torch.cat([edge_attrs[i], torch.zeros(padding, dtype=torch.long)])
        else:
            if edge_attrs[i].size(1) > 3:
                edge_attrs[i] = edge_attrs[i][:, :3]
            elif edge_attrs[i].size(1) < 3:
                edge_attrs[i] = torch.cat(
                    [edge_attrs[i], torch.zeros(edge_attrs[i].size(0), 3 - edge_attrs[i].size(1), dtype=torch.long)], dim=1)
            if padding > 0:
                edge_attrs[i] = torch.cat([edge_attrs[i], torch.zeros(padding, edge_attrs[i].size(1), dtype=torch.long)],
                                          dim=0)


        edge_list = edge_indices[i]
        if len(edge_list) > max_edges:
            edge_list = edge_list[:max_edges]
        padding = max_edges - len(edge_list)
        if padding > 0:
            edge_list = torch.cat([edge_list.clone().detach() if isinstance(edge_list, torch.Tensor) else torch.tensor(
                edge_list, dtype=torch.long),
                                   torch.full((padding, 2), 0, dtype=torch.long)], dim=0)

        mask = torch.cat([torch.ones(len(edge_list) - padding, dtype=torch.bool),
                          torch.zeros(padding, dtype=torch.bool)])
        edge_mask.append(mask)

        padded_edge_indices.append(edge_list)
        edge_attrs_mask.append(torch.cat([torch.ones(edge_attrs[i].size(0) - padding, dtype=torch.bool),
                                          torch.zeros(padding, dtype=torch.bool)]))

    nodes_tensor = torch.stack(nodes)
    edge_indices_tensor = torch.stack(padded_edge_indices)
    edge_attrs_tensor = torch.stack(edge_attrs)
    nodes_mask_tensor = torch.stack(nodes_mask)
    edge_mask_tensor = torch.stack(edge_mask)
    edge_attrs_mask_tensor = torch.stack(edge_attrs_mask)
    adj_tensor = torch.stack(adj_matrices)

    adj_masks = torch.stack(adj_masks)

    return nodes_tensor, edge_indices_tensor, edge_attrs_tensor, nodes_mask_tensor, edge_mask_tensor, edge_attrs_mask_tensor, adj_tensor, adj_masks


def static_graph_info(static_graph):
    max_node = 0
    max_edge = 0
    node_lengths = defaultdict(int)
    edge_lengths = defaultdict(int)
    for g in static_graph:
        num_nodes = g.number_of_nodes()
        num_edges = g.number_of_edges()
        node_lengths[num_nodes] += 1
        edge_lengths[num_edges] += 1
        if num_nodes > max_node:
            max_node = num_nodes
        if num_edges > max_edge:
            max_edge = num_edges

    print("\nNode lengths count (sorted by number of nodes):")
    sum_node = 0
    for node_count in sorted(node_lengths.keys()):
        sum_node += node_lengths[node_count]
        print(f"Number of nodes: {node_count}, Count: {node_lengths[node_count]}, Percentage: {sum_node/len(static_graph)}")

    print("\nEdge lengths count (sorted by number of edges):")
    sum_edge = 0
    for edge_count in sorted(edge_lengths.keys()):
        sum_edge += edge_lengths[edge_count]
        print(f"Number of edges: {edge_count}, Count: {edge_lengths[edge_count]}, Percentage: {sum_edge/len(static_graph)}")

