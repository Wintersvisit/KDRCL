import torch
from config.exp_config import Config
from helper.logger import get_logger
from core.KDRCL import KDRCL
from process.EventProcess import EventProcess

def test(dataset, log_dir):
    config = Config(dataset)
    config.reconstruct = False
    config.distillation = False
    config.feature_alignment = True
    config.hard_sample_distill = False

    logger = get_logger(log_dir, f"{dataset}_test")
    processor = EventProcess(config, logger)
    _, _, test_data = processor.process(reconstruct=False)

    model = KDRCL(config, logger, log_dir)
    result = model.evaluate(test_data)

    print(f"HR@1: {result.hr_1:.4f}")
    print(f"HR@2: {result.hr_2:.4f}")
    print(f"HR@3: {result.hr_3:.4f}")
    print(f"AVG@3: {result.avg_3:.4f}")

if __name__ == '__main__':
    test('sockshop', './logs/sockshop')