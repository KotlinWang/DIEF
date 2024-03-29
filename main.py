import yaml
from importlib import import_module
import argparse
from torch.utils.tensorboard import SummaryWriter

from lib.core import function
from lib.dataset.data_load import compile_data
# from lib.dataset.dataloadr_urbansound import compile_data
import torch.backends.cudnn as cudnn
# from lib.utils.draw_plt import test_result
from lib.utils.utils import test_result
from inference import count_params_and_flops


def parse_args():
    parser = argparse.ArgumentParser(description="SpecWaveNeXt")
    parser.add_argument(
        "--config",
        required=True,
        type=str,
    )
    args = parser.parse_args()

    cfg = import_module(args.config)
    return cfg


def main():
    # load configuration
    cfg = parse_args()
    device = cfg.device

    # cudann
    cudnn.benchmark = cfg.cudnn_benchmark
    cudnn.deterministic = cfg.cudnn_deterministic
    cudnn.enabled = cfg.cudnn_enabled
    # writer = SummaryWriter(log_dir=cfg.output_dir + '/log')

    # dataloader
    train_loader, val_loader = compile_data(cfg.dataloader, device)

    # build model
    model = function.build_model(cfg)

    # train model
    # function.train(cfg.train, model, train_loader, val_loader, writer, device)
    # print('Test')
    # function.test(cfg.test, model, val_loader, device)
    function.result_50(cfg.test, model, val_loader, device)
    count_params_and_flops(model, device)


if __name__ == '__main__':
    main()
