import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel

from confidnet.utils.logger import get_logger
LOGGER = get_logger(__name__, level="DEBUG")


class ORIGIN(BertPreTrainedModel):
    def __init__(self,config, config_args):
        super().__init__(config)
        self.config_args = config_args
        self.mc_dropout = config_args["model"]["is_dropout"]
        self.num_labels = config_args['data']['num_classes']
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.apply(self.init_bert_weights)


    def forward(self,input_ids, attention_mask=None, token_type_ids=None, labels=None):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        pooled_output = outputs[1] # [cls] token, pooled_output
        out = self.dropout(pooled_output)

        logits = self.classifier(out)
        return logits, pooled_output

    def print_summary(self, input_size):
        summary(self, input_size)

    def keep_dropout_in_test(self):
        if self.mc_dropout:
            LOGGER.warning("Keeping dropout activated during evaluation mode")
            self.training = True
