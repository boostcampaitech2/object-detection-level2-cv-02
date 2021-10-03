import datetime
import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import transform.transform as module_transform

import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device
from torch.utils.data import DataLoader

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def collate_fn(batch):
    return tuple(zip(*batch))


def main(config):
    data_set = config.init_obj("data_loader", module_data)

    data_loader = DataLoader(data_set, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate_fn)
    model = config.init_obj("arch", module_arch)

    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    criterion = getattr(module_loss, config["loss"])
    metrics = [getattr(module_metric, met) for met in config["metrics"]]

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj("optimizer", torch.optim, trainable_params)
    lr_scheduler = config.init_obj("lr_scheduler", torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(
        model,
        criterion,
        metrics,
        optimizer,
        config=config,
        device=device,
        data_loader=data_loader,
        valid_data_loader=None,
        lr_scheduler=lr_scheduler,
    )

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"),
        CustomArgs(
            ["--ut", "--unique_tag"],
            type=str,
            target="wandb;unique_tag",
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
