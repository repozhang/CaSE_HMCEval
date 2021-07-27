import sys
import os
sys.path.append(os.path.abspath('..'))
import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from confidnet.loaders.get_loader import get_loader
from confidnet.learners.get_learner import get_learner
from confidnet.models.get_model import get_model

from confidnet.utils import trust_scores
from confidnet.utils.logger import get_logger
from confidnet.utils.metrics import Metrics,classifiction_metric
from confidnet.utils.misc import load_yaml


LOGGER = get_logger(__name__, level="DEBUG")

MODE_TYPE = ["mcp", "trust_score", "confidnet"]
MAX_NUMBER_TRUSTSCORE_SEG = 3000

from pytorch_pretrained_bert.modeling import BertConfig
import random

# set fixed random seed
def setup_seed(seed): # ++
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", "-c", type=str, default=None, help="Path for config yaml")
    parser.add_argument("--epoch", "-e", type=int, default=None, help="Epoch to analyse")
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="mcp",
        choices=MODE_TYPE,
        help="Type of confidence testing",
    )
    parser.add_argument(
        "--samples", "-s", type=int, default=50, help="Samples in case of MCDropout"
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    args = parser.parse_args()

    config_args = load_yaml(args.config_path)

    # Overwrite for release
    config_args["training"]["output_folder"] = Path(args.config_path).parent

    config_args["training"]["metrics"] = [
        "accuracy",
        "auc",
        "ap_success",
        "ap_errors",
        "fpr_at_95tpr",
        "aurc"
    ]

    # Special case of MC Dropout
    if args.mode == "mc_dropout":
        config_args["training"]["mc_dropout"] = True

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # setting random seed
    setup_seed(42)

    # Load dataset
    LOGGER.info(f"Loading dataset {config_args['data']['dataset']}")
    dloader = get_loader(config_args)

    # Make loaders
    dloader.make_loaders()

    # Set learner
    LOGGER.warning(f"Learning type: {config_args['training']['learner']}")

    config = BertConfig(os.path.join(config_args['data']['bert_model_dir'], 'bert_config.json'))  # ++

    learner = get_learner(
       config, config_args, dloader.train_loader, dloader.val_loader, dloader.test_loader, -1, device
    ) # ++

    # Initialize and load model
    ckpt_path = config_args["training"]["output_folder"] / f"model_bert_{args.epoch:03d}.ckpt" # ++

    checkpoint = torch.load(ckpt_path)
    learner.model.load_state_dict(checkpoint, strict=False)

    # Get scores
    LOGGER.info(f"Inference mode: {args.mode}")

    if args.mode != "trust_score":
        _, scores_test = learner.evaluate(
            learner.test_loader,
            learner.prod_test_len,
            split="test",
            mode=args.mode,
            samples=args.samples,
            verbose=True,
        )

    # Special case TrustScore
    else:
        # Create feature extractor model
        config_args["model"]["name"] = config_args["model"]["name"] + "_extractor"
        print(config_args["model"]["name"])

        features_extractor = get_model(config, config_args).from_pretrained(config_args['data']['bert_model_dir'],config_args).to(device) # ++

        features_extractor.load_state_dict(learner.model.state_dict(), strict=False)

        LOGGER.info(f"Using extractor {config_args['model']['name']}")

        # Get features for KDTree
        LOGGER.info("Get features for KDTree")
        features_extractor.eval()
        metrics = Metrics(
            learner.metrics, learner.prod_test_len, config_args["data"]["num_classes"]
        )
        train_features, train_target = [], []
        with torch.no_grad():
            loop = tqdm(learner.train_loader)
            for step, batch in enumerate(tqdm(loop, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                _, input_ids, input_mask, segment_ids, label_ids = batch
                output = features_extractor(input_ids, segment_ids, input_mask, labels=None)
                output=output.view(output.size(0),-1)
                train_features.append(output)
                train_target.append(label_ids)

        train_features = torch.cat(train_features).detach().cpu().numpy()
        train_target = torch.cat(train_target).detach().cpu().numpy()

        LOGGER.info("Create KDTree")
        trust_model = trust_scores.TrustScore(num_workers=max(config_args["data"]["num_classes"], 2))
        trust_model.fit(train_features, train_target)

        LOGGER.info("Execute on test set")
        test_features, test_pred = [], []
        learner.model.eval()
        with torch.no_grad():
            loop = tqdm(learner.test_loader)
            for step, batch in enumerate(tqdm(loop, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                idx_ids, input_ids, input_mask, segment_ids, label_ids = batch # ++
                output, pooled_output = learner.model(input_ids, segment_ids, input_mask, labels=None)

                confidence, pred = output.max(dim=1, keepdim=True)
                features = features_extractor(input_ids, segment_ids, input_mask, labels=None)

                features = features.view(features.size(0), -1)

                test_features.append(features)
                test_pred.append(pred)
                metrics.update(idx_ids,pred, label_ids, confidence)

        test_features = torch.cat(test_features).detach().to("cpu").numpy()
        test_pred = torch.cat(test_pred).squeeze().detach().to("cpu").numpy()
        proba_pred = trust_model.get_score(test_features, test_pred)
        metrics.update(idx_ids, pred, label_ids, confidence)

        print('test_features', test_features)
        print('idx_ids',idx_ids)
        print('labels', label_ids.detach().cpu())
        print('test_pred', test_pred)
        print('trust_score', proba_pred)

        print('----------------------------------------------------')
        pred_list=[]
        target_list=[]
        confidence_list=[]
        proba_pred_list=[]

        for i, p, t, c in zip(metrics.new_idx, metrics.new_pred, metrics.new_taget, metrics.new_conf):
            print('idx,pred,target,confidence', i, p[0], t, c[0])
            pred_list.append(p[0])
            target_list.append(t)
            confidence_list.append(c[0])

        print('----------------------------------------------------')
        report = classifiction_metric(np.array(pred_list), np.array(target_list),
                                      np.array(config_args['data']['label_list']))
        print(report)
        print('----------------------------------------------------')
        scores_test = metrics.get_scores(split="test")

    LOGGER.info("Results")
    print("----------------------------------------------------------------")

    for st in scores_test:
        print(st)
        print(scores_test[st])
        print("----------------------------------------------------------------")


if __name__ == "__main__":
    main()
