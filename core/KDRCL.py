import os
import time
import dgl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import random
import json
import re
from openai import OpenAI
import numpy as np
from transformers import BertModel, BertTokenizer, AutoModel, AutoTokenizer

from core.model.MainModel import MainModel
from helper.eval import *
from helper.early_stop import EarlyStopping
from helper.Result import Result
from config.exp_config import Config


class KDRCL(object):
    def __init__(self, config: Config, logger, log_dir: str):
        self.config = config
        self.logger = logger
        os.makedirs(log_dir, exist_ok=True)

        use_gpu = torch.cuda.is_available()
        if use_gpu:
            logger.info("Currently using GPU {}".format(config.gpu_device))
            os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_device
            self.device = 'cuda'
        else:
            logger.info("Currently using CPU (GPU is highly recommended)")
            self.device = 'cpu'

        # 初始化教师模型
        if config.distillation:
            self.teacher_client = OpenAI(
                api_key=config.teacher_api_key,
                base_url=config.teacher_base_url
            )
        else:
            self.teacher_client = None

        # 初始化解释编码器（用于特征对齐）
        if config.feature_alignment and config.explanation_encoder == "bert":
            bert_path = r" "

            self.explanation_tokenizer = BertTokenizer.from_pretrained(bert_path)
            self.explanation_encoder = BertModel.from_pretrained(bert_path)
            self.explanation_encoder.eval()

            if use_gpu:
                self.explanation_encoder.to(self.device)

            self.graph_projection_layer = None

        self.result = Result()
        self.writer = SummaryWriter(log_dir)
        self.printParams()

    def printParams(self):
        self.config.print_configs(self.logger)

    def get_teacher_ratio(self, epoch):
        if not self.config.progressive_distill:
            return self.config.hard_sample_ratio

        progress = min(epoch / self.config.teacher_annealing_epochs, 1.0)
        teacher_ratio = self.config.initial_teacher_ratio - progress * (
                self.config.initial_teacher_ratio - self.config.final_teacher_ratio)

        return teacher_ratio

    @staticmethod
    def calculate_graph_confidences(root_logit, num_nodes_list):
        """Return max root-node probability for every graph in a DGL batch."""
        flat_root_logits = root_logit.reshape(-1)
        graph_confidences = []
        start_idx = 0

        for num_nodes in num_nodes_list:
            node_count = int(num_nodes)
            end_idx = start_idx + node_count
            graph_logits = flat_root_logits[start_idx:end_idx]
            graph_probs = torch.softmax(graph_logits, dim=0)
            graph_confidences.append(graph_probs.max())
            start_idx = end_idx

        if start_idx != flat_root_logits.numel():
            raise ValueError(
                "The sum of graph node counts does not match root_logit."
            )

        return torch.stack(graph_confidences)

    @staticmethod
    def prepare_distillation_distributions(student_logits, teacher_logits,
                                           temperature):
        """Build matching one-dimensional node distributions for one graph."""
        student_logits = student_logits.reshape(-1)
        teacher_logits = teacher_logits.reshape(-1)

        if student_logits.numel() != teacher_logits.numel():
            raise ValueError(
                "Student and teacher logits must contain the same number of "
                "graph nodes: {} != {}.".format(
                    student_logits.numel(), teacher_logits.numel()
                )
            )

        student_log_probs = F.log_softmax(student_logits / temperature, dim=0)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=0)
        return student_log_probs, teacher_probs

    @classmethod
    def calculate_distillation_loss(cls, student_logits, teacher_logits,
                                    temperature):
        student_log_probs, teacher_probs = \
            cls.prepare_distillation_distributions(
                student_logits, teacher_logits, temperature
            )
        return F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='sum'
        ) * (temperature ** 2)

    def identify_hard_samples(self, model, batch_graphs, batch_labels,
                              return_details=False):
        was_training = model.training
        try:
            model.eval()
            with torch.no_grad():
                _, _, root_logit, _ = model(batch_graphs)

                # 获取批次中每个图的节点数量
                num_nodes_list = batch_graphs.batch_num_nodes()
                # 按 DGL batch 中每个图的节点范围切分，并仅在图内节点间归一化。
                root_confidence = self.calculate_graph_confidences(
                    root_logit, num_nodes_list
                )

                # 识别困难样本
                hard_mask = root_confidence < self.config.hard_threshold
                num_threshold_hard = hard_mask.sum().item()

                if hard_mask.sum() == 0:        # 如果没有样本低于阈值
                    k = max(1, int(len(root_confidence) * self.config.hard_sample_ratio))
                    _, indices = torch.topk(root_confidence, k, largest=False)      # 强制取置信度最低的 k 个，保证每个 batch 至少有一些样本会查询老师，避免完全停止蒸馏
                    hard_mask = torch.zeros_like(root_confidence, dtype=torch.bool)
                    hard_mask[indices] = True
        finally:
            model.train(was_training)

        if return_details:
            return hard_mask, root_confidence, num_threshold_hard
        return hard_mask

    def build_soft_label_prompt(self, raw_events):
        nodes = raw_events['nodes']
        edges = raw_events['edges']
        metrics = raw_events['metrics']
        traces = raw_events['traces']
        logs = raw_events['logs']

        system_prompt = """You are a microservice fault diagnosis expert, specializing in root cause localization of anomalies in microservice systems. Given the topology and monitoring events of a microservice system, perform the following tasks:
1. Assign a logit (unnormalized score) to each node indicating its likelihood of being the root cause. These logits can be any real numbers, but for effective knowledge distillation, please keep them within a moderate range, e.g., between -0.5 and 2.5. **Please output logits with exactly 4 decimal places, and try to use a variety of values within the range (e.g., 0.3124, -0.0309, 1.8935) rather than always using the extremes (-0.5000, 0.0000, 2.5000).** A higher logit means higher probability after softmax transformation.
2. Provide a brief overall explanation (2-3 sentences) summarizing the most likely root cause and your reasoning.

To ensure accuracy, reason step-by-step internally:
    - Step 1: Analyze system topology and service dependencies.
    - Step 2: Review monitoring events (metrics, traces, logs) for anomalies.
    - Step 3: Correlate anomalies with the topology to hypothesize potential root causes.
    - Step 4: Consider fault propagation patterns.
    - Step 5: Assign logits based on evidence.

The input will be provided in the user message, including the system node list, service invocation relationships, and monitoring events (metrics, traces, logs). **Output strictly in the following JSON format without any additional formatting, markdown, or explanatory text**:
{
  "logits": {
    "node1": 0.3124,
    "node2": -0.0309,
    "node3": 1.8935,
    ...
  },
  "overall_explanation": "Brief overall explanation summarizing the reasoning... (2-3 sentences)"
}"""

        user_prompt = f"""System node list: {nodes}
        Service invocation relationships: {edges}

        Monitoring Events:
        - Metrics Events: {metrics}
        - Trace Events: {traces}
        - Log Events: {logs}"""

        return system_prompt, user_prompt

    def parse_teacher_logits(self, response, nodes):
        try:
            data = self.parse_teacher_response_json(response)
            logits = data.get("logits", {})

            # 初始化 logit 向量，默认值为 0
            logit_vector = np.zeros(len(nodes), dtype=np.float32)

            for i, node in enumerate(nodes):    # 输出按 nodes 列表顺序排列，与学生模型的 root_logit 索引对齐，便于逐元素做 KL 散度
                if node in logits:
                    logit_vector[i] = logits[node]
                else:
                    logit_vector[i] = 0

            return torch.tensor(logit_vector, dtype=torch.float32, device=self.device)

        except Exception as e:
            self.logger.warning(f"解析教师 logits 失败: {e}")
            return torch.zeros(len(nodes), device=self.device)

    def parse_teacher_explanations(self, response, nodes):
        try:
            data = self.parse_teacher_response_json(response)
            overall_explanation = data.get("overall_explanation", "")

            if not overall_explanation or overall_explanation.strip() == "":    # 兜底：若 LLM 没返回解释
                overall_explanation = f"Based on the provided metrics, traces, and logs, the system is likely experiencing issues. The root cause logits are assigned based on anomaly patterns observed in the monitoring data."

            return overall_explanation
        except Exception as e:
            self.logger.warning(f"解析教师解释失败: {e}")
            return "Unable to generate explanation due to parsing error."

    def parse_teacher_response_json(self, response):
        if isinstance(response, dict):
            return response

        content = str(response).strip()
        parse_errors = []
        try:
            return json.loads(content)
        except json.JSONDecodeError as error:
            parse_errors.append(error)

        fenced_blocks = re.findall(
            r"```(?:json)?\s*(\{.*?\})\s*```", content,
            flags=re.IGNORECASE | re.DOTALL
        )
        for candidate in fenced_blocks:
            try:
                data = json.loads(candidate)
                if isinstance(data, dict) and 'logits' in data:
                    return data
            except json.JSONDecodeError as error:
                parse_errors.append(error)

        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", content):
            try:
                data, _ = decoder.raw_decode(content[match.start():])
                if isinstance(data, dict) and 'logits' in data:
                    return data
            except json.JSONDecodeError as error:
                parse_errors.append(error)

        if parse_errors:
            raise parse_errors[-1]
        raise json.JSONDecodeError("No JSON object found", content, 0)

    def extract_teacher_content(self, response):
        if isinstance(response, str):
            return response

        if isinstance(response, dict):
            if "choices" in response:
                message = response["choices"][0]["message"]
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content
                reasoning_content = message.get("reasoning_content")
                if isinstance(reasoning_content, str) and reasoning_content.strip():
                    return reasoning_content
                return content
            if "content" in response:
                return response["content"]
            if "text" in response:
                return response["text"]
            return json.dumps(response, ensure_ascii=False)

        if hasattr(response, "choices"):
            message = response.choices[0].message
            content = getattr(message, "content", None)
            if isinstance(content, str) and content.strip():
                return content
            reasoning_content = getattr(message, "reasoning_content", None)
            if isinstance(reasoning_content, str) and reasoning_content.strip():
                return reasoning_content
            return content

        if hasattr(response, "content"):
            return response.content

        return str(response)

    def encode_explanations(self, explanation_text, nodes):
        if not self.config.feature_alignment:
            return None

        if not explanation_text or explanation_text.strip() == "":
            return torch.zeros(self.config.explanation_dim, device=self.device)

        # 使用BERT编码解释文本
        inputs = self.explanation_tokenizer(
            explanation_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )

        if self.device == 'cuda':
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():       # BERT 权重不参与梯度更新
            outputs = self.explanation_encoder(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()  # 取[CLS] token 的输出作为整体解释的向量表示
            """BERT 编码后，第 0 个位置的 [CLS] token 的输出向量被训练为整句话的语义摘要"""
        return cls_embedding

    @staticmethod
    def create_teacher_stats():
        return {
            key: 0 for key in (
                'teacher_query_count', 'teacher_api_success_count',
                'teacher_api_failure_count', 'teacher_empty_count',
                'teacher_parse_success_count', 'teacher_parse_failure_count',
                'teacher_logit_valid_count', 'teacher_logit_mismatch_count',
                'teacher_explanation_valid_count', 'distill_update_count',
                'alignment_update_count'
            )
        }

    def parse_and_validate_teacher_response(self, response, nodes, sample_id,
                                            stats):
        try:
            data = self.parse_teacher_response_json(response)
            if not isinstance(data, dict):
                raise TypeError("top-level JSON value must be an object")
        except Exception as error:
            stats['teacher_parse_failure_count'] += 1
            self.logger.warning(
                "[Teacher Warning] JSON parse failure, sample_id=%s, "
                "error_type=%s, error=%s",
                sample_id, type(error).__name__, str(error)[:200]
            )
            return None, None

        stats['teacher_parse_success_count'] += 1
        logits = data.get('logits')
        teacher_logits = None
        if not isinstance(logits, dict):
            stats['teacher_logit_mismatch_count'] += 1
            self.logger.warning(
                "[Teacher Warning] Invalid logits type, sample_id=%s, "
                "expected=dict, received=%s",
                sample_id, type(logits).__name__
            )
        elif set(logits.keys()) != set(nodes):
            stats['teacher_logit_mismatch_count'] += 1
            missing_count = len(set(nodes) - set(logits.keys()))
            extra_count = len(set(logits.keys()) - set(nodes))
            self.logger.warning(
                "[Teacher Warning] Logit node mismatch, sample_id=%s, "
                "expected=%d, received=%d, missing=%d, extra=%d",
                sample_id, len(nodes), len(logits), missing_count, extra_count
            )
        else:
            try:
                values = np.asarray(
                    [float(logits[node]) for node in nodes], dtype=np.float32
                )
                if not np.isfinite(values).all():
                    raise ValueError("logits contain non-finite values")
                teacher_logits = torch.tensor(
                    values, dtype=torch.float32, device=self.device
                )
                stats['teacher_logit_valid_count'] += 1
            except (TypeError, ValueError) as error:
                stats['teacher_logit_mismatch_count'] += 1
                self.logger.warning(
                    "[Teacher Warning] Invalid logit values, sample_id=%s, "
                    "error=%s", sample_id, str(error)[:200]
                )

        explanation = data.get('overall_explanation')
        if isinstance(explanation, str) and explanation.strip():
            explanation = explanation.strip()
            stats['teacher_explanation_valid_count'] += 1
        else:
            explanation = None
            self.logger.warning(
                "[Teacher Warning] Empty explanation, sample_id=%s", sample_id
            )

        return teacher_logits, explanation

    def query_teacher_model(self, system_prompt, user_prompt, sample_id=None,
                            stats=None):      # 调用教师模型 API 获取软标签和解释
        if stats is not None:
            stats['teacher_query_count'] += 1
        try:
            response = self.teacher_client.chat.completions.create(
                model=self.config.teacher_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.config.teacher_max_tokens,
                temperature=self.config.temperature,        # temperature=0：确定性输出，每次相同输入得到相同结果，保证输出可复现
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}}
            )

            if stats is not None:
                stats['teacher_api_success_count'] += 1

            content = self.extract_teacher_content(response) # 如果返回 HTML，应该视为失败
            if not isinstance(content, str) or not content.strip():
                if stats is not None:
                    stats['teacher_empty_count'] += 1
                self.logger.warning(
                    "[Teacher Warning] Empty response, sample_id=%s", sample_id
                )
                return None
            if content.lstrip().lower().startswith("<!doctype html") or content.lstrip().lower().startswith("<html"):
                if stats is not None:
                    stats['teacher_parse_failure_count'] += 1
                self.logger.error("Teacher API returned HTML page. Please check teacher_base_url; it should be an OpenAI-compatible API endpoint, usually ending with /v1.")
                return None
            return content

            # return self.extract_teacher_content(response)
        except Exception as e:
            if stats is not None:
                stats['teacher_api_failure_count'] += 1
            self.logger.warning(
                "[Teacher Warning] API error, sample_id=%s, error_type=%s, "
                "error=%s", sample_id, type(e).__name__, str(e)[:200]
            )
            return None

    def train(self, train_data, aug_data):
        model = MainModel(self.config).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.config.lr,
                               weight_decay=self.config.weight_decay)

        self.logger.info(model)
        self.logger.info(f"Start training for {self.config.epochs} epochs.")

        train_times = []
        earlyStop = EarlyStopping(patience=self.config.patience)

        for epoch in range(self.config.epochs):
            n_iter = 0
            start_time = time.time()
            model.train()
            epoch_loss = 0
            rcl_results = {"HR@1": [], "HR@2": [], "HR@3": [], "HR@4": [], "HR@5": [], "MRR@3": []}
            epoch_confidences = []
            epoch_num_hard_samples = 0
            epoch_num_selected_samples = 0
            epoch_hard_loss_sum = 0.0
            epoch_distill_loss_sum = 0.0
            epoch_alignment_loss_sum = 0.0
            teacher_stats = self.create_teacher_stats()

            current_teacher_ratio = self.get_teacher_ratio(epoch)

            train_dl = DataLoader(train_data, batch_size=self.config.batch_size,
                                  shuffle=True, collate_fn=self.collate)

            for batch_idx, batch in enumerate(train_dl):
                batch_graphs, batch_labels, batch_raw_events = batch
                batch_graphs = batch_graphs.to(self.device)
                batch_labels = batch_labels.to(self.device)

                opt.zero_grad()

                graph_embeddings, node_embeddings, root_logit, _ = model(batch_graphs)

                # 基础监督损失（所有样本）- 硬标签交叉熵损失
                rcl_supervised_loss = self.cal_rcl_loss_supervised(root_logit, batch_graphs)
                supervised_loss = rcl_supervised_loss

                # 知识蒸馏损失（仅困难样本）- 包含特征对齐
                distill_loss = torch.tensor(0.0, device=self.device)
                alignment_loss = torch.tensor(0.0, device=self.device)
                batch_distill_update_count = 0
                batch_alignment_update_count = 0

                if self.config.distillation and self.config.hard_sample_distill and self.teacher_client:
                    # 识别困难样本
                    with torch.no_grad():
                        hard_mask, root_confidence, num_threshold_hard = \
                            self.identify_hard_samples(
                                model, batch_graphs, batch_labels,
                                return_details=True
                            )

                    epoch_confidences.append(root_confidence.detach())
                    epoch_num_hard_samples += num_threshold_hard

                    batch_size = len(batch_graphs.batch_num_nodes())

                    num_hard = hard_mask.sum().item()
                    if num_hard > 0:
                        # 计算实际要使用的困难样本数量
                        target_teacher_samples = int(batch_size * current_teacher_ratio)
                        actual_teacher_samples = min(num_hard, target_teacher_samples)

                        if actual_teacher_samples > 0:
                            # 从困难样本中随机选择一部分
                            hard_indices = torch.where(hard_mask)[0]
                            selected_indices = hard_indices[torch.randperm(len(hard_indices))[:actual_teacher_samples]]
                            epoch_num_selected_samples += len(selected_indices)

                            # 为选中的困难样本查询教师模型
                            for idx in selected_indices:
                                graph_idx = int(idx)
                                raw_events = batch_raw_events[graph_idx]
                                sample_id = raw_events.get(
                                    'index', f"batch={batch_idx},graph={graph_idx}"
                                )
                                system_prompt, user_prompt = self.build_soft_label_prompt(raw_events)
                                teacher_response = self.query_teacher_model(
                                    system_prompt, user_prompt,
                                    sample_id=sample_id, stats=teacher_stats
                                )

                                # self.logger.info(f"Teacher response received: {teacher_response is not None}, type={type(teacher_response)}")
                                # self.logger.info(f"Teacher response preview: {teacher_response[:300] if teacher_response else None}")

                                if teacher_response:
                                    teacher_logits, teacher_explanation = \
                                        self.parse_and_validate_teacher_response(
                                            teacher_response,
                                            raw_events['nodes'],
                                            sample_id,
                                            teacher_stats
                                        )

                                    # 获取当前图的节点数量，即定位当前图在大图中的节点范围
                                    num_nodes = batch_graphs.batch_num_nodes()[graph_idx]
                                    start_idx = sum(batch_graphs.batch_num_nodes()[:graph_idx])
                                    end_idx = start_idx + num_nodes

                                    # 获取学生模型的输出
                                    student_logits = root_logit[start_idx:end_idx]

                                    if teacher_logits is not None:
                                        # 在当前图的节点维度计算 KL(p_teacher || p_student)。
                                        distill_rcl_loss = self.calculate_distillation_loss(
                                            student_logits,
                                            teacher_logits,
                                            self.config.distill_temperature
                                        )
                                        if not torch.isfinite(distill_rcl_loss):
                                            raise ValueError(
                                                f"Non-finite distillation loss for {sample_id}"
                                            )
                                        distill_loss += distill_rcl_loss
                                        batch_distill_update_count += 1
                                        teacher_stats['distill_update_count'] += 1

                                    # 特征对齐损失
                                    if self.config.feature_alignment and teacher_explanation is not None:
                                        explanation_embedding = self.encode_explanations(
                                            teacher_explanation, raw_events['nodes'])
                                        # 获取当前图的图级别特征
                                        graph_embedding_idx = graph_embeddings[graph_idx]

                                        if graph_embedding_idx.shape[0] != explanation_embedding.shape[0]:
                                            if not hasattr(self, 'graph_projection_layer'):
                                                self.graph_projection_layer = torch.nn.Linear(
                                                    graph_embedding_idx.shape[0],
                                                    explanation_embedding.shape[0]
                                                ).to(self.device)
                                            graph_embedding_idx = self.graph_projection_layer(graph_embedding_idx)

                                        # 计算特征对齐损失
                                        if self.config.alignment_loss_type == "cosine":
                                            # 余弦相似度损失
                                            graph_norm = F.normalize(graph_embedding_idx.unsqueeze(0), p=2, dim=1)
                                            explanation_norm = F.normalize(explanation_embedding.unsqueeze(0), p=2,
                                                                           dim=1)
                                            cos_sim = torch.sum(graph_norm * explanation_norm, dim=1)
                                            align_loss = 1 - torch.mean(cos_sim)
                                        else:  # MSE损失
                                            align_loss = F.mse_loss(
                                                graph_embedding_idx.unsqueeze(0),
                                                explanation_embedding.unsqueeze(0)
                                            )

                                        if not torch.isfinite(align_loss):
                                            raise ValueError(
                                                f"Non-finite alignment loss for {sample_id}"
                                            )
                                        alignment_loss += align_loss
                                        batch_alignment_update_count += 1
                                        teacher_stats['alignment_update_count'] += 1

                            # 平均损失
                            if batch_distill_update_count > 0:
                                distill_loss = distill_loss / batch_distill_update_count
                            if batch_alignment_update_count > 0:
                                alignment_loss = alignment_loss / batch_alignment_update_count

                # 总损失 = 监督损失 + 蒸馏损失 + 特征对齐损失
                total_loss = supervised_loss + \
                             self.config.distill_alpha * distill_loss + \
                             self.config.alignment_weight * alignment_loss

                total_loss.backward()
                opt.step()

                epoch_loss += total_loss.detach().item()
                epoch_hard_loss_sum += supervised_loss.detach().item()
                epoch_distill_loss_sum += (
                    distill_loss.detach().item() * batch_distill_update_count
                )
                epoch_alignment_loss_sum += (
                    alignment_loss.detach().item() * batch_alignment_update_count
                )
                n_iter += 1


            mean_epoch_loss = epoch_loss / n_iter if n_iter > 0 else 0
            mean_hard_loss = epoch_hard_loss_sum / n_iter if n_iter > 0 else 0
            mean_distill_loss = (
                epoch_distill_loss_sum / teacher_stats['distill_update_count']
                if teacher_stats['distill_update_count'] > 0 else 0
            )
            mean_alignment_loss = (
                epoch_alignment_loss_sum / teacher_stats['alignment_update_count']
                if teacher_stats['alignment_update_count'] > 0 else 0
            )
            end_time = time.time()
            time_per_epoch = (end_time - start_time)
            train_times.append(time_per_epoch)

            self.logger.info("Epoch {} done. Loss: {:.3f}, Time per epoch: {:.3f}[s], Teacher ratio: {:.2f}"
                             .format(epoch, mean_epoch_loss, time_per_epoch, current_teacher_ratio))

            if epoch_confidences:
                confidence_values = torch.cat(epoch_confidences)
                confidence_min = confidence_values.min().item()
                confidence_mean = confidence_values.mean().item()
                confidence_max = confidence_values.max().item()
                self.logger.info(
                    "Selector epoch {}: confidence_min={:.6f}, "
                    "confidence_mean={:.6f}, confidence_max={:.6f}, "
                    "num_hard_samples={}, num_selected_samples={}, "
                    "teacher_ratio={:.4f}".format(
                        epoch, confidence_min, confidence_mean, confidence_max,
                        epoch_num_hard_samples, epoch_num_selected_samples,
                        current_teacher_ratio
                    )
                )
                self.writer.add_scalar('selector/confidence_min', confidence_min,
                                       global_step=epoch)
                self.writer.add_scalar('selector/confidence_mean', confidence_mean,
                                       global_step=epoch)
                self.writer.add_scalar('selector/confidence_max', confidence_max,
                                       global_step=epoch)
            else:
                self.logger.info(
                    "Selector epoch {}: confidence_min=N/A, "
                    "confidence_mean=N/A, confidence_max=N/A, "
                    "num_hard_samples=0, num_selected_samples=0, "
                    "teacher_ratio={:.4f} (selector not run)".format(
                        epoch, current_teacher_ratio
                    )
                )

            self.writer.add_scalar('selector/num_hard_samples',
                                   epoch_num_hard_samples, global_step=epoch)
            self.writer.add_scalar('selector/num_selected_samples',
                                   epoch_num_selected_samples, global_step=epoch)

            self.logger.info(
                "Teacher pipeline epoch %d: %s",
                epoch,
                ", ".join(
                    f"{key}={value}" for key, value in teacher_stats.items()
                )
            )
            for key, value in teacher_stats.items():
                self.writer.add_scalar(f'teacher/{key}', value,
                                       global_step=epoch)

            # 记录损失分量
            self.writer.add_scalar('loss/total', mean_epoch_loss, global_step=epoch)
            self.writer.add_scalar('loss/supervised', mean_hard_loss,
                                   global_step=epoch)
            self.writer.add_scalar('loss/distill', mean_distill_loss,
                                   global_step=epoch)
            self.writer.add_scalar('loss/alignment', mean_alignment_loss,
                                   global_step=epoch)
            self.writer.add_scalar('teacher/ratio', current_teacher_ratio, global_step=epoch)

            stop = earlyStop.should_stop(mean_epoch_loss, epoch)
            if stop:
                self.logger.info(f"Early stop at epoch {epoch} due to lack of improvement.")
                break

        # 保存模型
        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'opt': opt.state_dict(),
        }
        torch.save(state, os.path.join(self.writer.log_dir, 'KDRCL.pt'))

        self.result.set_train_efficiency(train_times)
        self.logger.info("Training has finished.")

    def cal_rcl_loss_supervised(self, root_logit, batch_graphs):
        num_nodes_list = batch_graphs.batch_num_nodes()
        total_loss = None

        start_idx = 0
        for idx, num_nodes in enumerate(num_nodes_list):
            end_idx = start_idx + num_nodes
            node_logits = root_logit[start_idx: end_idx].reshape(1, -1)
            root = batch_graphs.ndata["root"][start_idx: end_idx].tolist().index(1)     # 标注为 1 的节点索引（真实根因位置）

            # 硬标签交叉熵损失，将问题转化为多分类问题
            loss = F.cross_entropy(node_logits, torch.LongTensor([root]).view(1).to(self.device))

            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss
            start_idx += num_nodes
        # 对 batch 内所有图的损失求平均，作为最终损失
        l_rcl = total_loss / len(num_nodes_list) if total_loss is not None else torch.tensor(0.0, device=self.device)
        return l_rcl

    def collate(self, samples):
        graphs, labels, raw_events = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        batched_labels = torch.tensor(labels)
        return batched_graph, batched_labels, raw_events

    def evaluate(self, test_data, model=None):
        """测试时只使用学生模型"""
        if model is None:
            checkpoint_path = os.path.join(self.writer.log_dir, 'KDRCL.pt')
            if not os.path.exists(checkpoint_path):
                self.logger.error(f"Model checkpoint not found at {checkpoint_path}")
                return self.result

            checkpoint = torch.load(checkpoint_path, map_location=self.device)     # 加载训练好的 checkpoint
            model = MainModel(self.config).to(self.device)
            model.load_state_dict(checkpoint['model'])

        model.eval()
        root_logits = []
        roots = []
        inference_times = []
        num_node_list = []

        for data in test_data:              # 逐样本推理（不是 batch）
            graph, labels, raw_events = data
            idx = raw_events.get('index', None)
            graph = graph.to(self.device)
            roots.append(graph.ndata['root'])
            num_node_list.append(graph.num_nodes())

            start_time = time.time()
            with torch.no_grad():
                _, _, root_logit, _ = model(graph)
                root_logits.append(root_logit.flatten())
            end_time = time.time()
            inference_times.append(end_time - start_time)

        root_logits = torch.hstack(root_logits).cpu()
        roots = torch.hstack(roots)

        rcl_res = RCA_eval(root_logits, num_node_list, roots)
        self.result.set_performance(rcl_res, {})
        self.result.set_inference_efficiency(inference_times)

        avg_3 = np.mean([rcl_res['HR@1'], rcl_res['HR@2'], rcl_res['HR@3']])
        self.logger.info("[Root localization] HR@1: {:.3%}, HR@2: {:.3%}, HR@3: {:.3%}, avg@3: {:.3f}"
                         .format(rcl_res['HR@1'], rcl_res['HR@2'], rcl_res['HR@3'], avg_3))

        return self.result
