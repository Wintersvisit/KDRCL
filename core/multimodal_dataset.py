import torch
from torch.utils.data import Dataset
import dgl


class MultiModalDataSet(Dataset):  # 将生成的 Metric、Trace、Log 向量以及拓扑关系封装成一个图结构
    def __init__(self):
        self.data = []
        self.raw_events = []

    def add_data(self, metric_Xs, trace_Xs, log_Xs, global_root_id, failure_type_id,
                 local_root, nodes, edges, raw_events=None):
        node_num = len(nodes)

        if raw_events and 'edge_weights' in raw_events:
            # 构建带权重的图
            weighted_edges = raw_events['edge_weights']   # 源节点列表
            src_list = [e[0] for e in weighted_edges]     # 目标节点列表
            dst_list = [e[1] for e in weighted_edges]     # 权重值
            weight_list = [e[2] for e in weighted_edges]
            graph = dgl.graph((src_list, dst_list), num_nodes=node_num)
            graph.edata['weight'] = torch.FloatTensor(weight_list)  # 将权重存入边的属性
        else:  # 构建无权重图
            graph = dgl.graph(edges, num_nodes=node_num)
        # 注入多模态节点特征
        graph.ndata["metric"] = torch.FloatTensor(metric_Xs)
        graph.ndata["trace"] = torch.FloatTensor(trace_Xs)
        graph.ndata["log"] = torch.FloatTensor(log_Xs)
        # 设置根因标签
        root_labels = [0] * len(nodes)
        root_labels[nodes.index(local_root)] = 1   # 只有真正的根因节点标为 1
        graph.ndata["root"] = torch.LongTensor(root_labels)

        # 添加自环处理(如果一个节点没有入边（孤立节点），它就无法接收来自其他节点或自身的信息更新)
        in_degrees = graph.in_degrees()
        zero_indegree_nodes = [i for i in range(len(in_degrees)) if in_degrees[i].item() == 0]
        if zero_indegree_nodes:
            u = zero_indegree_nodes
            v = zero_indegree_nodes
            if 'weight' in graph.edata:
                new_weights = torch.ones(len(zero_indegree_nodes), dtype=graph.edata['weight'].dtype,
                                         device=graph.edata['weight'].device)
                graph.add_edges(u, v, {'weight': new_weights})  # 给入度为0的节点加带权自环
            else:
                graph.add_edges(u, v)

        self.data.append((graph, (global_root_id, failure_type_id)))
        self.raw_events.append(raw_events)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        graph, labels = self.data[idx]
        raw_event = self.raw_events[idx]
        return graph, labels, raw_event