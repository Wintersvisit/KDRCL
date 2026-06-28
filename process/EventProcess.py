import os
from torch.utils.data import DataLoader
from config.exp_config import Config
from core.multimodal_dataset import MultiModalDataSet
from helper import io_util
import json
import pandas as pd
import numpy as np
from process.events.fasttext_w2v import FastTextEncoder
import time


class EventProcess():

    def __init__(self, config: Config, logger):
        self.config = config
        self.logger = logger
        self.dataset = config.dataset

    def process(self, reconstruct=False):
        self.data_path = f"data/{self.dataset}"
        label_path = f"data/{self.dataset}/label.csv"
        metric_path = f"data/{self.dataset}/raw/metrics.json"
        trace_path = f"data/{self.dataset}/raw/traces.json"
        log_path = f"data/{self.dataset}/raw/logs.json"
        edge_path = f"data/{self.dataset}/raw/edges.json"
        node_path = f"data/{self.dataset}/raw/nodes.json"

        self.logger.info(f"Load raw events from {self.dataset} dataset")
        self.labels = pd.read_csv(label_path)
        self.labels['index'] = self.labels['index'].astype(str)
        with open(metric_path, 'r', encoding='utf8') as fp:
            self.metrics = json.load(fp)
        with open(trace_path, 'r', encoding='utf8') as fp:
            self.traces = json.load(fp)
        with open(log_path, 'r', encoding='utf8') as fp:
            self.logs = json.load(fp)
        with open(edge_path, 'r', encoding='utf8') as fp:
            self.edges = json.load(fp)
        with open(node_path, 'r', encoding='utf8') as fp:
            self.nodes = json.load(fp)

        self.types = ['normal'] + self.labels['anomaly_type'].unique().tolist()

        if reconstruct:
            self.build_embedding()

        return self.build_dataset()

    def build_embedding(self):   # FastText 向量化
        self.logger.info(f"Build embedding for raw events")
        data_map = {'metric': self.metrics, 'trace': self.traces, 'log': self.logs}

        for key, data in data_map.items():
            all_nodes = list({item for sublist in self.nodes.values() for item in sublist})    # 把所有故障样本里出现过的服务节点名去重汇总
            encoder = FastTextEncoder(key, all_nodes, self.types, embedding_dim=self.config.alert_embedding_dim,   # 将文本形式的“服务节点名称”转换为数值向量
                                      epochs=5)

            train_idxs = self.labels[self.labels['data_type'] == 'train']['index'].values.tolist()
            train_ins_labels = self.labels[self.labels['data_type'] == 'train']['instance'].values.tolist()        # 根因
            train_type_labels = self.labels[self.labels['data_type'] == 'train']['anomaly_type'].values.tolist()   # 故障类型
            docs = []
            labels = []
            for i, idx in enumerate(train_idxs):  # 对每个训练样本的每个节点，把该节点的事件拼成"文档"
                nodes = self.nodes[str(idx)]
                for node in nodes:
                    if key == 'trace':  #根据配置决定是否包含操作名（op）或异常类型（ab_type）；如果 node 是调用链的起点或终点，则记录该调用关系
                        if self.config.trace_op and self.config.trace_ab_type:
                            doc = ['&'.join(e) for e in data[str(idx)] if (node in e[0] or node in e[1])]
                        elif not self.config.trace_op and self.config.trace_ab_type:
                            doc = ['&'.join([item for i, item in enumerate(e) if i != 2]) for e in data[str(idx)] if
                                   (node in e[0] or node in e[1])]
                        elif self.config.trace_op and not self.config.trace_ab_type:
                            doc = ['&'.join([item for i, item in enumerate(e) if i != 3]) for e in data[str(idx)] if
                                   (node in e[0] or node in e[1])]
                        else:
                            doc = ['&'.join(e[:2]) for e in data[str(idx)] if (node in e[0] or node in e[1])]
                    elif key == 'metric':  # 记录该 node 相关的指标名称和变化趋势
                        if self.config.metric_direction:
                            doc = ['&'.join(e) for e in data[str(idx)] if (node in e[0])]
                        else:
                            doc = ['&'.join([item for i, item in enumerate(e) if i != 3]) for e in data[str(idx)] if
                                   (node in e[0] or node in e[1])]
                    else:                  # 记录该 node 产生的日志模版
                        doc = ['&'.join(e) for e in data[str(idx)] if node in e[0]]
                    docs.append(doc)
                    if node == train_ins_labels[i]:    # 如果这个节点是根因节点；标签构造，让词向量在训练时感知“哪个节点更可能是根因”
                        labels.append(f'__label__{node}{self.types.index(train_type_labels[i])}')
                    else:
                        labels.append(f'__label__{node}0')
            encoder.fit(docs, labels)

            embs = {}
            for idx in self.labels['index']:    # 训练完成后，再次遍历所有样本（包括训练集和测试集），将节点描述转化为向量
                nodes = self.nodes[str(idx)]
                graph_embs = []
                for node in nodes:
                    if key == 'trace':
                        if self.config.trace_op and self.config.trace_ab_type:
                            doc = ['&'.join(e) for e in data[str(idx)] if (node in e[0] or node in e[1])]
                        elif not self.config.trace_op and self.config.trace_ab_type:
                            doc = ['&'.join([item for i, item in enumerate(e) if i != 2]) for e in data[str(idx)] if
                                   (node in e[0] or node in e[1])]
                        elif self.config.trace_op and not self.config.trace_ab_type:
                            doc = ['&'.join([item for i, item in enumerate(e) if i != 3]) for e in data[str(idx)] if
                                   (node in e[0] or node in e[1])]
                        else:
                            doc = ['&'.join(e[:2]) for e in data[str(idx)] if (node in e[0] or node in e[1])]
                    elif key == 'metric':
                        if self.config.metric_direction:
                            doc = ['&'.join(e) for e in data[str(idx)] if (node in e[0])]
                        else:
                            doc = ['&'.join([item for i, item in enumerate(e) if i != 3]) for e in data[str(idx)] if
                                   (node in e[0] or node in e[1])]
                    else:
                        doc = ['&'.join(e) for e in data[str(idx)] if node in e[0]]

                    emb = encoder.get_sentence_embedding(doc)   # 获取该节点在此故障下的句向量
                    graph_embs.append(emb)
                embs[idx] = graph_embs
            tmp_pth = f'data/{self.dataset}/tmp/'
            if not os.path.isdir(tmp_pth):
                os.makedirs(tmp_pth, exist_ok=True)
            io_util.save_pkl(f"{tmp_pth}/{key}.pkl", embs)

    def build_dataset(self):  # 构建图数据集，每个样本是一个调用链图，节点特征是该节点的事件向量，边特征是调用关系权重；标签是根因节点和故障类型
        self.logger.info(f"Build dataset for training")
        metric_embs = io_util.load_pkl(f"data/{self.dataset}/tmp/metric.pkl")
        trace_embs = io_util.load_pkl(f"data/{self.dataset}/tmp/trace.pkl")
        log_embs = io_util.load_pkl(f"data/{self.dataset}/tmp/log.pkl")

        label_dict = {}
        all_nodes = list({item for sublist in self.nodes.values() for item in sublist})
        all_nodes.sort()
        label_dict['instance'], node2idx, idx2root = self.get_root_labels(all_nodes)  # node2idx 将根因服务名（字符串）转成全局节点编号（整数），作为训练标签

        train_data, test_data = MultiModalDataSet(), MultiModalDataSet()

        for _, row in self.labels.iterrows():
            index = str(row['index'])
            data_type = row['data_type']

            raw_events = {
                'nodes': self.nodes[index],
                'edges': self.edges[index],
                'metrics': self.metrics.get(index, []),
                'traces': self.traces.get(index, []),
                'logs': self.logs.get(index, [])
            }

            # 统计调用链异常事件中每条边的出现次数
            trace_counts = {}
            for event in raw_events['traces']:
                # 事件格式：[src, dst, ...] 前两个元素是调用方和被调用方
                if len(event) >= 2:
                    src, dst = event[0], event[1]
                    key = (src, dst)
                    trace_counts[key] = trace_counts.get(key, 0) + 1

            # 将服务名转换为节点索引
            nodes = self.nodes[index]
            node_to_idx = {node: idx for idx, node in enumerate(nodes)}

            # 将字符串键转换为索引键
            edge_counts = {}
            for (src_str, dst_str), cnt in trace_counts.items():
                if src_str in node_to_idx and dst_str in node_to_idx:
                    src_idx = node_to_idx[src_str]
                    dst_idx = node_to_idx[dst_str]
                    edge_counts[(src_idx, dst_idx)] = cnt

            # 为所有边计算权重（归一化到 [0,1]）
            max_count = max(edge_counts.values()) if edge_counts else 1.0
            weighted_edges = []
            for src_idx, dst_idx in self.edges[index]:
                cnt = edge_counts.get((src_idx, dst_idx), 0)
                # 归一化权重，并设置最小权重0.1（避免边完全断开）
                weight = cnt / max_count if max_count > 0 else 0.0
                weight = max(weight, 0.1)
                weighted_edges.append((src_idx, dst_idx, weight))

            raw_events['edge_weights'] = weighted_edges

            metric_Xs, trace_Xs, log_Xs = metric_embs[index], trace_embs[index], log_embs[index]
            global_root_id = node2idx[row['instance']]
            failure_type_id = 0

            if data_type == 'train':
                train_data.add_data(
                    metric_Xs=metric_Xs,
                    trace_Xs=trace_Xs,
                    log_Xs=log_Xs,
                    global_root_id=global_root_id,
                    failure_type_id=failure_type_id,
                    local_root=row['instance'],
                    nodes=nodes,
                    edges=self.edges[index],
                    raw_events=raw_events
                )
            else:
                test_data.add_data(
                    metric_Xs=metric_Xs,
                    trace_Xs=trace_Xs,
                    log_Xs=log_Xs,
                    global_root_id=global_root_id,
                    failure_type_id=failure_type_id,
                    local_root=row['instance'],
                    nodes=nodes,
                    edges=self.edges[index],
                    raw_events=raw_events
                )

        aug_data = []

        return train_data, aug_data, test_data

    def get_root_labels(self, nodes):
        labels2idx = {node: idx for idx, node in enumerate(nodes)}
        idx2label = {idx: label for idx, label in enumerate(nodes)}
        labels = np.array(self.labels['instance'].apply(lambda label_str: labels2idx[label_str]))
        return labels, labels2idx, idx2label

    def get_type_labels(self, types):
        meta_labels = sorted(list(set(types)))
        labels2idx = {label: idx for idx, label in enumerate(meta_labels)}
        idx2label = {idx: label for idx, label in enumerate(meta_labels)}
        labels = np.array(self.labels['anomaly_type'].apply(lambda label_str: labels2idx[label_str]))
        return labels, labels2idx, idx2label
