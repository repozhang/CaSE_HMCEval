import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from confidnet.learners.learner import AbstractLeaner
from confidnet.utils import misc
from confidnet.utils.logger import get_logger
from confidnet.utils.metrics import Metrics, classifiction_metric

LOGGER = get_logger(__name__, level="DEBUG")


class SelfConfidLearner(AbstractLeaner):
    def __init__(self, config, config_args, train_loader, val_loader, test_loader, start_epoch, device):
        super().__init__(config, config_args, train_loader, val_loader, test_loader, start_epoch, device)
        self.freeze_layers()
        self.disable_bn(verbose=True)
        if self.config_args["model"].get("uncertainty", None):
            self.disable_dropout(verbose=True)

    def train(self, epoch):
        self.model.train()
        self.disable_bn()
        if self.config_args["model"].get("uncertainty", None):
            self.disable_dropout()
        metrics = Metrics(
            self.metrics, self.prod_train_len, self.num_classes
        )
        loss, confid_loss = 0, 0
        len_steps, len_data = 0, 0

        # Training loop
        loop = tqdm(self.train_loader)

        for step, batch in enumerate(tqdm(loop, desc="Iteration")):
            batch = tuple(t.to(self.device) for t in batch)
            idx_ids, input_ids, input_mask, segment_ids, label_ids = batch

            output = self.model(input_ids, segment_ids, input_mask, labels=None)

            print('output', output[0], output[1], torch.sigmoid(output[1]))

            current_loss = self.criterion(output, label_ids)

            """ loss """
            n_gpu = torch.cuda.device_count()
            if n_gpu > 1:
                current_loss = current_loss.mean()
            if self.config_args['training']['gradient_accumulation_steps'] > 1:
                current_loss = current_loss / self.config_args['training']['gradient_accumulation_steps']

            current_loss.backward()
            loss += current_loss

            len_steps += len(input_ids)
            len_data = len_steps

            # Update metrics
            pred = output[0].argmax(dim=1, keepdim=True)

            confidence = torch.sigmoid(output[1])
            metrics.update(idx_ids, pred, label_ids, confidence)

            pred_detach, label_detach, confidence_detach, idx_detach = pred.detach(), label_ids.detach(), confidence.detach(), idx_ids.detach()

            print('pred', pred_detach.cpu())
            print('label', label_detach.cpu())
            print('idx', idx_detach.cpu())
            print('confidence', confidence_detach.cpu())

            if (step + 1) % self.config_args['training']['gradient_accumulation_steps'] == 0:
                print('optimizer step', step + 1)
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Update the average loss
            loop.set_description(f"Epoch {epoch}/{self.nb_epochs}")
            loop.set_postfix(
                OrderedDict(
                    {
                        "loss_confid": f"{(loss / len_data):05.3e}",
                        "acc": f"{(metrics.accuracy / len_steps):05.2%}",
                    }
                )
            )
            loop.update()

        # Eval on epoch end
        scores = metrics.get_scores(split="train")
        logs_dict = OrderedDict(
            {
                "epoch": {"value": epoch, "string": f"{epoch:03}"},
                "lr": {
                    "value": self.optimizer.param_groups[0]["lr"],
                    "string": f"{self.optimizer.param_groups[0]['lr']:05.1e}",
                },
                "train/loss_confid": {
                    "value": loss / len_data,
                    "string": f"{(loss / len_data):05.4e}",
                },
            }
        )
        for s in scores:
            logs_dict[s] = scores[s]

        # Val scores
        val_losses, scores_val = self.evaluate(self.val_loader, self.prod_val_len, split="val")
        logs_dict["val/loss_confid"] = {
            "value": val_losses["loss_confid"].item() / self.nsamples_val,
            "string": f"{(val_losses['loss_confid'].item() / self.nsamples_val):05.4e}",
        }
        for sv in scores_val:
            logs_dict[sv] = scores_val[sv]

        # Test scores
        test_losses, scores_test = self.evaluate(self.test_loader, self.prod_test_len, split="test")
        logs_dict["test/loss_confid"] = {
            "value": test_losses["loss_confid"].item() / self.nsamples_test,
            "string": f"{(test_losses['loss_confid'].item() / self.nsamples_test):05.4e}",
        }
        for st in scores_test:
            logs_dict[st] = scores_test[st]

        # Print metrics
        misc.print_dict(logs_dict)

        # Save the model checkpoint
        self.save_checkpoint(epoch)

        # CSV logging
        misc.csv_writter(path=self.output_folder / "logs.csv", dic=OrderedDict(logs_dict))

        # Tensorboard logging
        self.save_tb(logs_dict)

        # Scheduler step
        if self.scheduler:
            self.scheduler.step()

    def evaluate(self, dloader, len_dataset, split="test", verbose=False, **args):
        self.model.eval()  # use with torch.no_grad()
        metrics = Metrics(self.metrics, len_dataset, self.num_classes)
        loss = 0

        # Evaluation loop
        loop = tqdm(dloader, disable=not verbose)

        for step, batch in enumerate(tqdm(loop, desc="Iteration")):
            batch = tuple(t.to(self.device) for t in batch)
            idx_ids, input_ids, input_mask, segment_ids, label_ids = batch

            with torch.no_grad():
                output = self.model(input_ids, segment_ids, input_mask, labels=None)
                if self.task == "classification":
                    current_loss = self.criterion(output, label_ids)  # ok
                    loss += current_loss

                # Update metrics
                pred = F.softmax(output[0]).argmax(dim=1, keepdim=True)

                confidence = torch.sigmoid(output[1])

                metrics.update(idx_ids, pred, label_ids, confidence)

                pred_detach, label_detach, confidence_detach, idx_detach = pred.detach(), label_ids.detach(), confidence.detach(), idx_ids.detach()

                print('pred', pred_detach.cpu())
                print('label', label_detach.cpu())
                print('idx', idx_detach.cpu())
                print('confidence', confidence_detach.cpu())

        print('----------------------------------------------------')
        pred_list = []
        target_list = []
        confidence_list = []

        for i, p, t, c in zip(metrics.new_idx, metrics.new_pred, metrics.new_taget, metrics.new_conf):
            print('idx,pred,target,confidence', i, p[0], t, c[0])
            pred_list.append(p[0])
            target_list.append(t)
            confidence_list.append(c[0])

        print('----------------------------------------------------')
        report = classifiction_metric(np.array(pred_list), np.array(target_list),
                                      np.array(self.config_args['data']['label_list']))
        print(report)
        print('----------------------------------------------------')

        scores = metrics.get_scores(split=split)
        losses = {"loss_confid": loss}
        return losses, scores

    def load_checkpoint(self, state_dict, uncertainty_state_dict=None, strict=True):
        if not uncertainty_state_dict:
            print('not uncertainty_state_dict')

            self.model.load_state_dict(state_dict, strict=False)
        else:
            print('have uncertainty_state_dict')

            self.model.pred_network.load_state_dict(state_dict, strict=strict)

            # 1. filter out unnecessary keys
            state_dict = {
                k: v
                for k, v in uncertainty_state_dict.items()
                if k not in ["classifier.weight", "classifier.bias"]

            }
            # 2. overwrite entries in the existing state dict
            self.model.uncertainty_network.state_dict().update(state_dict)
            # 3. load the new state dict
            self.model.uncertainty_network.load_state_dict(state_dict, strict=False)

    def freeze_layers(self):
        LOGGER.info("Freezing every layer except uncertainty")
        for param in self.model.named_parameters():
            print('parameters include', param[0])
            if "uncertainty" in param[0]:
                print(param[0], "kept to training")
                continue
            param[1].requires_grad = False

    def disable_bn(self, verbose=False):
        if verbose:
            LOGGER.info("Keeping original BN parameters")
        for layer in self.model.named_modules():
            if "bert" in layer[0] or "classifier" in layer[0] or 'dropout' in layer[0]:
                if verbose:
                    print(layer[0], "original BN setting")
                layer[1].eval()

    def disable_dropout(self, verbose=False):
        if verbose:
            LOGGER.info("Disable dropout layers to reduce stochasticity")
        for layer in self.model.named_modules():
            if "dropout" in layer[0]:
                if verbose:
                    print(layer[0], "set to eval mode")
                layer[1].eval()
