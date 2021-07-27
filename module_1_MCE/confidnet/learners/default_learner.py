from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from confidnet.learners.learner import AbstractLeaner
from confidnet.utils import misc
from confidnet.utils.logger import get_logger
from confidnet.utils.metrics import Metrics,classifiction_metric


LOGGER = get_logger(__name__, level="DEBUG")


class DefaultLearner(AbstractLeaner):
    def train(self, epoch):
        self.model.train()
        metrics = Metrics(
            self.metrics, self.prod_train_len, self.num_classes
        )
        loss, len_steps, len_data = 0, 0, 0

        loop = tqdm(self.train_loader)


        for step, batch in enumerate(tqdm(loop, desc="Iteration")):

            batch = tuple(t.to(self.device) for t in batch)

            idx_ids, input_ids, input_mask, segment_ids, label_ids = batch

            output,pooled_output = self.model(input_ids, segment_ids, input_mask, labels=None)

            current_loss = self.criterion(output.view(-1, 2), label_ids.view(-1))

            """ loss """
            n_gpu = torch.cuda.device_count()  # ++
            if n_gpu > 1:
                current_loss = current_loss.mean()  #
            if self.config_args['training']['gradient_accumulation_steps'] > 1:
                current_loss = current_loss / self.config_args['training']['gradient_accumulation_steps']

            current_loss.backward()

            loss += current_loss.item()  # loss: epoch loss


            len_steps += len(input_ids)
            len_data = len_steps

            # Update metrics
            confidence, pred = F.softmax(output, dim=1).max(dim=1, keepdim=True)
            metrics.update(idx_ids,pred, label_ids, confidence)


            pred_detach,label_detach,confidence_detach,idx_detach=pred.detach(),label_ids.detach(),confidence.detach(), idx_ids.detach()

            print('pred', pred_detach.cpu())
            print('label', label_detach.cpu())
            print('idx', idx_detach.cpu())
            print('confidence', confidence_detach.cpu())

            """optimizer"""
            if (step + 1) % self.config_args['training']['gradient_accumulation_steps'] == 0:
                print('optimizer step', step+1)
                self.optimizer.step()
                self.optimizer.zero_grad()


            loop.set_description(f"Epoch {epoch}/{self.nb_epochs}")
            loop.set_postfix(
                OrderedDict(
                    {
                        "loss_nll": f"{(loss / len_data):05.4e}",
                        "acc": f"{(metrics.accuracy / len_steps):05.2%}",
                    }
                )
            )
            loop.update()


        scores = metrics.get_scores(split="train")
        logs_dict = OrderedDict(
            {
                "epoch": {"value": epoch, "string": f"{epoch:03}"},
                "lr": {
                    "value": self.optimizer.param_groups[0]["lr"],
                    "string": f"{self.optimizer.param_groups[0]['lr']:05.1e}",
                },
                "train/loss_nll": {
                    "value": loss / len_data,
                    "string": f"{(loss / len_data):05.4e}",
                },
            }
        )
        for s in scores:
            logs_dict[s] = scores[s]

        # Val scores
        if self.val_loader is not None:
            val_losses, scores_val = self.evaluate(self.val_loader, self.prod_val_len, split="val")
            logs_dict["val/loss_nll"] = {
                "value": val_losses["loss_nll"].item() / self.nsamples_val,
                "string": f"{(val_losses['loss_nll'].item() / self.nsamples_val):05.4e}",
            }
            for sv in scores_val:
                logs_dict[sv] = scores_val[sv]

        # Test scores
        test_losses, scores_test = self.evaluate(self.test_loader, self.prod_test_len, split="test")
        logs_dict["test/loss_nll"] = {
            "value": test_losses["loss_nll"].item() / self.nsamples_test,
            "string": f"{(test_losses['loss_nll'].item() / self.nsamples_test):05.4e}",
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


    def evaluate(self, dloader, len_dataset, split="test", mode="mcp", samples=50, verbose=False):
        self.model.eval()
        metrics = Metrics(self.metrics, len_dataset, self.num_classes)
        loss = 0

        # Special case of mc-dropout
        if mode == "mc_dropout":
            self.model.keep_dropout_in_test()
            LOGGER.info(f"Sampling {samples} times")

        # Evaluation loop
        loop = tqdm(dloader, disable=not verbose)
        for step, batch in enumerate(tqdm(loop, desc="Iteration")):
            batch = tuple(t.to(self.device) for t in batch)
            idx_ids, input_ids, input_mask, segment_ids, label_ids = batch
            print(label_ids)

            with torch.no_grad():
                if mode == "mcp":
                    print(True)
                    output, pooled_output = self.model(input_ids, segment_ids, input_mask, labels=None)

                    current_loss = self.criterion(output.view(-1, 2), label_ids.view(-1))
                    loss += current_loss

                    confidence, pred = F.softmax(output, dim=1).max(dim=1, keepdim=True)

                    print(confidence)
                    print(pred)

                elif mode == "tcp":
                    output, pooled_output = self.model(input_ids, segment_ids, input_mask, labels=None)

                    current_loss = self.criterion(output.view(-1, 2), label_ids.view(-1))
                    loss += current_loss

                    probs = F.softmax(output, dim=1)

                    pred = probs.max(dim=1, keepdim=True)[1]

                    labels_hot = misc.one_hot_embedding(label_ids, self.num_classes).to(self.device)

                    confidence, _ = (labels_hot * probs).max(dim=1, keepdim=True)

                elif mode == "mc_dropout":
                    print('---------------input_ids.shape---------------')
                    print(input_ids.shape)
                    outputs = torch.zeros(samples, self.config_args['training']['batch_size'], self.num_classes).to(self.device)

                    for i in range(samples):
                        outputs[i], _ =  self.model(input_ids, segment_ids, input_mask, labels=None)
                    output = outputs.mean(0)

                    loss += self.criterion(output.view(-1, 2), label_ids.view(-1))

                    probs = F.softmax(output, dim=1)
                    confidence = (probs * torch.log(probs + 1e-9)).sum(dim=1)
                    pred = probs.max(dim=1, keepdim=True)[1]



                metrics.update(idx_ids,pred, label_ids, confidence)
                pred_detach, label_detach, confidence_detach, idx_detach = pred.detach(), label_ids.detach(), confidence.detach(), idx_ids.detach()
                print('pred', pred_detach.cpu())
                print('label', label_detach.cpu())
                print('idx', idx_detach.cpu())
                print('confidence', confidence_detach.cpu())


        print('----------------------------------------------------')
        pred_list=[]
        target_list=[]
        confidence_list=[]

        for i,p,t,c in zip(metrics.new_idx,metrics.new_pred,metrics.new_taget,metrics.new_conf):
            print('idx,pred,target,confidence',i,p[0],t,c[0])
            pred_list.append(p[0])
            target_list.append(t)
            confidence_list.append(c[0])

        print('----------------------------------------------------')
        report=classifiction_metric(np.array(pred_list), np.array(target_list), np.array(self.config_args['data']['label_list']))
        print(report)
        print('----------------------------------------------------')


        scores = metrics.get_scores(split=split)
        losses = {"loss_nll": loss}
        return losses, scores
