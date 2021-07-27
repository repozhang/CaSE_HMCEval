import sys
import os
sys.path.append(os.path.abspath('..'))

import argparse

from shutil import copyfile, rmtree

import torch
#
from confidnet.loaders.get_loader import get_loader
from confidnet.learners.get_learner import get_learner
from confidnet.utils.logger import get_logger
from confidnet.utils.misc import load_yaml
from confidnet.utils.tensorboard_logger import TensorboardLogger


import tensorflow as tf
from pytorch_pretrained_bert.modeling import BertConfig
from transformers.configuration_utils import PretrainedConfig
import random
import numpy as np
LOGGER = get_logger(__name__, level="DEBUG")


# set fixed random seed
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", "-c", type=str, default=None, help="Path for config yaml")
    parser.add_argument(
        "--no_cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--from_scratch",
        "-f",
        action="store_true",
        default=False,
        help="Force training from scratch",
    )
    args = parser.parse_args()

    config_args = load_yaml(args.config_path)

    config=BertConfig(os.path.join(config_args['data']['bert_model_dir'],'bert_config.json')) # ++

    print(isinstance(config, PretrainedConfig)) # ++
    print(config)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(device)

    # random seed check
    # print(np.random.randn())
    setup_seed(42)
    # print(np.random.randn())
    # exit(0)

    # Start from scatch or resume existing model and optim
    if config_args["training"]["output_folder"].exists():
        list_previous_ckpt = sorted(
            [f for f in os.listdir(config_args["training"]["output_folder"]) if "model_bert" in f]
        )
        if args.from_scratch or not list_previous_ckpt:
            start_epoch = 1
        else:
            last_ckpt = list_previous_ckpt[-1]
            checkpoint = torch.load(config_args["training"]["output_folder"] / str(last_ckpt))
            start_epoch = checkpoint["epoch"] + 1
    else:
        LOGGER.info("Starting from scratch")
        os.mkdir(config_args["training"]["output_folder"])
        start_epoch = 1

    # Load dataset
    LOGGER.info(f"Loading dataset {config_args['data']['dataset']}")
    dloader = get_loader(config_args)

    # Make loaders
    dloader.make_loaders()

    # Set learner
    LOGGER.warning(f"Learning type: {config_args['training']['learner']}")
    print('====check point learner 1====')

    learner = get_learner(
        config=config,
        config_args=config_args,
        train_loader=dloader.train_loader,
        val_loader=dloader.val_loader,
        test_loader=dloader.test_loader,
        start_epoch=start_epoch,
        device=device
    )
    print('====check point learner 2====')

    # Resume existing model or from pretrained one
    if start_epoch > 1:
        LOGGER.warning(f"Resuming from last checkpoint: {last_ckpt}")
        learner.model.load_state_dict(checkpoint["model_state_dict"])
        learner.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    elif config_args["model"]["resume"]: # ++
        LOGGER.info(f"Loading pretrained model from {config_args['model']['resume']}")

        pretrained_checkpoint = torch.load(config_args["model"]["resume"]) # --

        print('==load pretrained checkpoint==')
        uncertainty_checkpoint = config_args["model"].get("uncertainty", None)

        if uncertainty_checkpoint:
            LOGGER.warning("Cloning training phase")
            learner.load_checkpoint(
                pretrained_checkpoint["model_state_dict"],
                torch.load(uncertainty_checkpoint)["model_state_dict"],
                strict=False,
            )
        else:
            learner.load_checkpoint(pretrained_checkpoint, strict=False) # ++

    # Log files
    LOGGER.info(f"Using model {config_args['model']['name']}")

    tf.compat.v1.disable_eager_execution() #++
    learner.tb_logger = TensorboardLogger(config_args["training"]["output_folder"])

    copyfile(
        args.config_path, config_args["training"]["output_folder"] / f"config_{start_epoch}.yaml"
    )

    LOGGER.info(f"Saving logs in: {config_args['training']['output_folder']}")


    # Parallelize model
    n_gpu=torch.cuda.device_count()
    if n_gpu > 1:
        LOGGER.info(f"Parallelizing data to {n_gpu} GPUs")
        learner.model = torch.nn.DataParallel(learner.model)

    if n_gpu == 1:
        learner.model = learner.model.cuda()

    # Start training
    for epoch in range(start_epoch, config_args["training"]["nb_epochs"] + 1):
        learner.train(epoch)


if __name__ == "__main__":
    main()
