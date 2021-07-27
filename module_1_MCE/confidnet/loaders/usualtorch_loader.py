import torch
import csv
from confidnet.loaders.Classifier_utils import InputExample, convert_examples_to_features, convert_features_to_tensors
import os
from torchtext.data import Field, TabularDataset, BucketIterator, Iterator
from confidnet.utils.misc import load_yaml
import torch.nn as nn
# Models
from transformers import BertTokenizer, BertForSequenceClassification


class MDRDCLoader:
    def __init__(self,config_args):
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.config_args=config_args

        self.pin_memory = config_args['training']['pin_memory']
        self.num_workers = config_args['training']['num_workers']

    def read_tsv(self,filename):
        with open(filename, "r", encoding='utf-8') as f:
                reader = csv.reader(f, delimiter="\t")
                lines = []
                for line in reader:
                    lines.append(line)
                return lines

    def load_tsv_dataset(self,filename, set_type):
        """
        文件内数据格式: sentence  label
        """
        examples = []
        lines = self.read_tsv(filename)
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = i
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


    def load_data(self,data_dir, tokenizer, max_length, batch_size, data_type, label_list, format_type=0):
        if format_type == 0:
            load_func = self.load_tsv_dataset

        if data_type == "train":
            train_file = os.path.join(data_dir, 'train.tsv')
            examples = load_func(train_file, data_type)
        elif data_type == "dev":
            dev_file = os.path.join(data_dir, 'dev.tsv')
            examples = load_func(dev_file, data_type)
        elif data_type == "test":
            test_file = os.path.join(data_dir, 'test.tsv')
            examples = load_func(test_file, data_type)
        else:
            raise RuntimeError("should be train or dev or test")

        features = convert_examples_to_features(
            examples, label_list, max_length, tokenizer)

        dataloader = convert_features_to_tensors(features, batch_size, data_type)

        examples_len = len(examples)

        return dataloader, examples_len

    def make_loaders(self):
        tokenizer = BertTokenizer.from_pretrained(
            self.config_args['data']['bert_vocab_file'], do_lower_case=True)


        datafile=self.config_args['data']['data_dir']
        max_len = self.config_args['data']['max_len']
        batch_size_train = self.config_args['training']['batch_size_train']
        batch_size_dev = self.config_args['training']['batch_size_dev']
        batch_size_test = self.config_args['training']['batch_size_test']
        label_list=self.config_args['data']['label_list']

        self.train_loader, self.train_examples_len = self.load_data(datafile
            , tokenizer, max_len, batch_size_train, "train",label_list)
        self.val_loader, _ = self.load_data(datafile, tokenizer, max_len, batch_size_dev, "dev",
                                            label_list)
        self.test_loader, _ = self.load_data(datafile, tokenizer, max_len, batch_size_test, "test",
                                             label_list)




if __name__=="__main__":
    train,valid,test = MDRDCLoader().make_loaders()
    for step,batch in enumerate(train):
        print(step)
        _, input_ids, input_mask, segment_ids, label_ids = batch
        print(input_ids,label_ids)

