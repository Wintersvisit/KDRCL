class Config:
    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.reconstruct = True
        self.log_step = 20
        self.gpu_device = '0'

        self.modalities = ['metric', 'trace', 'log']

        self.metric_direction = True
        self.trace_op = True
        self.trace_ab_type = True

        # 特征对齐配置
        self.feature_alignment = True  # 启用特征对齐
        self.alignment_loss_type = "cosine"  # 对齐损失类型
        self.alignment_weight = 0.1  # 特征对齐损失权重

        # 解释编码器配置
        self.explanation_encoder = "bert"  # 解释编码器类型
        self.explanation_dim = 768  # 解释嵌入维度
        self.explanation_type = "global"  # 解释类型

        # 困难样本知识蒸馏配置
        self.hard_sample_distill = True  # 启用困难样本蒸馏
        self.hard_sample_ratio = 0.5  # 困难样本比例
        self.hard_threshold = 0.3  # 困难样本阈值

        # 知识蒸馏配置
        self.distillation = True
        self.teacher_model = ""
        self.teacher_api_key = ""
        self.teacher_base_url = ""  # 替换为实际的教师模型API地址
        self.distill_temperature = 5.0  # KL散度温度参数
        self.distill_alpha = 0.2  # 蒸馏损失权重

        # 渐进式蒸馏配置
        self.progressive_distill = True  # 启用渐进式蒸馏
        self.initial_teacher_ratio = 0.5  # 初始教师使用比例
        self.final_teacher_ratio = 0.1  # 最终教师使用比例
        self.teacher_annealing_epochs = 10  # 教师使用比例衰减的epoch数

        self.temperature = 0
        self.batch_size = 128
        self.epochs = 100
        self.alert_embedding_dim = 128
        self.graph_hidden_dim = 64
        self.graph_out = 32
        self.graph_layers = 2
        self.linear_hidden = [64]
        self.lr = 0.001
        self.weight_decay = 0.0001

        if self.dataset == 'gaia':
            self.feat_drop = 0
            self.patience = 10
            self.ft_num = 5
            self.aggregator = 'mean'
        elif self.dataset == 'aiops22':
            if not self.trace_op:
                self.lr = 0.01
            self.feat_drop = 0.1
            self.patience =20
            self.ft_num = 9
            self.aggregator = 'mean'
        elif self.dataset == 'sockshop':
            self.feat_drop = 0
            self.patience =10
            self.ft_num = 5
            self.aggregator = 'mean'
        else:
            raise NotImplementedError()
    
    def print_configs(self, logger):
        for attr, value in vars(self).items():
            logger.info(f"{attr}: {value}")