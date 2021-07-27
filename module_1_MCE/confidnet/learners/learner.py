import os
import torch
import torch.optim as optim
from confidnet.models.get_model import get_model
from confidnet.utils import losses
from confidnet.utils.logger import get_logger
from pytorch_pretrained_bert.optimization import BertAdam

LOGGER = get_logger(__name__, level="DEBUG")


class AbstractLeaner:
    def __init__(self, config, config_args, train_loader, val_loader, test_loader, start_epoch, device):
        self.config = config
        self.config_args = config_args
        self.num_classes = config_args['data']['num_classes']
        self.task = config_args["training"]["task"]
        self.loss_args = config_args["training"]["loss"]
        self.metrics = config_args["training"]["metrics"]
        self.nb_epochs = config_args["training"]["nb_epochs"]
        self.output_folder = config_args["training"]["output_folder"]
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device

        try:
            self.nsamples_train = len(self.train_loader.sampler.indices)
            self.nsamples_val = len(self.val_loader.sampler.indices)
        except:
            self.nsamples_train = len(self.train_loader.dataset)
            self.nsamples_val = len(self.val_loader.dataset)
        self.nsamples_test = len(self.test_loader.dataset)
        if self.task == "classification":
            self.prod_train_len = self.nsamples_train
            self.prod_val_len = self.nsamples_val
            self.prod_test_len = self.nsamples_test

        self.last_epoch = start_epoch - 2
        self.criterion, self.scheduler, self.optimizer, self.tb_logger = None, None, None, None

        # Initialize model
        self.model = get_model(self.config, self.config_args).from_pretrained(
            self.config_args['data']['bert_model_dir'], self.config_args).to(self.device)

        print('============set model=======================')

        # Set optimizer
        self.set_optimizer(config_args["training"]["optimizer"]["name"])
        # Set loss
        self.set_loss()
        # Temperature scaling
        self.temperature = config_args["training"].get("temperature", None)

    def train(self, epoch):
        pass

    def set_loss(self):
        if self.loss_args["name"] in losses.CUSTOM_LOSS:
            self.criterion = losses.CUSTOM_LOSS[self.loss_args["name"]](
                config_args=self.config_args, device=self.device
            )
        elif self.loss_args["name"] in losses.PYTORCH_LOSS:
            self.criterion = losses.PYTORCH_LOSS[self.loss_args["name"]](ignore_index=255)
        else:
            raise Exception(f"Loss {self.loss_args['name']} not implemented")
        LOGGER.info(f"Using loss {self.loss_args['name']}")

    def set_optimizer(self, optimizer_name):
        optimizer_params = {
            k: v for k, v in self.config_args["training"]["optimizer"].items() if k != "name"
        }
        LOGGER.info(f"Using optimizer {optimizer_name}")
        if optimizer_name == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), **optimizer_params)
        elif optimizer_name == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), **optimizer_params)
        elif optimizer_name == "bertadam":
            param_optimizer = list(self.model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            num_train_optimization_steps = int(
                self.prod_train_len / self.config_args['training']['batch_size_train'] / self.config_args['training'][
                    'gradient_accumulation_steps']) * self.config_args['training']['nb_epochs']
            print('num_train_optimization_steps', num_train_optimization_steps)
            self.optimizer = BertAdam(optimizer_grouped_parameters,
                                      lr=self.config_args["training"]["optimizer"]['lr'],
                                      warmup=self.config_args["training"]["optimizer"]['warmup_proportion'],
                                      t_total=num_train_optimization_steps)


        elif optimizer_name == "adadelta":
            self.optimizer = optim.Adadelta(self.model.parameters(), **optimizer_params)
        else:
            raise KeyError("Bad optimizer name or not implemented (sgd, adam, adadelta).")

    """schedule by bertadam warm up, bert adam itself is adaptive"""

    def save_checkpoint(self, epoch):
        output_config_file = os.path.join(self.output_folder, 'bert_config.json')
        output_model_file = os.path.join(self.output_folder, f'model_bert_{epoch:03d}.ckpt')
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        torch.save(model_to_save.state_dict(), output_model_file)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())

    def save_tb(self, logs_dict):
        # ================================================================== #
        #                        Tensorboard Logging                         #
        # ================================================================== #

        epoch = logs_dict["epoch"]["value"]
        del logs_dict["epoch"]

        for tag in logs_dict:
            self.tb_logger.scalar_summary(tag, logs_dict[tag]["value"], epoch)

        for tag, value in self.model.named_parameters():
            tag = tag.replace(".", "/")
            self.tb_logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
            if not value.grad is None:
                self.tb_logger.histo_summary(tag + "/grad", value.grad.data.cpu().numpy(), epoch)
