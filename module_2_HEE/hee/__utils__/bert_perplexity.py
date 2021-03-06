import numpy as np
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import logging

# DEVICE = 'cuda:0'
DEVICE = 'cpu'

logging.basicConfig(level=logging.INFO)


class Perplexity_Checker(object):
    def __init__(self, MODEL_PATH):
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        self.model = BertForMaskedLM.from_pretrained(MODEL_PATH)
        self.model.eval()
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.model.to(DEVICE)

    def sentence_preprocess(self, text):
        # Tokenize input
        tokenized_text = np.array(self.tokenizer.tokenize(text))
        find_sep = np.argwhere(tokenized_text == '[SEP]')
        segments_ids = np.zeros(tokenized_text.shape, dtype=int)
        if find_sep.size == 1:
            start_point = 1
        else:
            start_point = find_sep[0, 0] + 1
            segments_ids[start_point:] = 1

        end_point = tokenized_text.size - 1

        # Mask a token that we will try to predict back with `BertForMaskedLM`
        tokenized_text = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        masked_texts = []
        for masked_index in range(start_point, end_point):
            new_tokenized_text = np.array(tokenized_text, dtype=int)
            new_tokenized_text[masked_index] = self.tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
            masked_texts.append(new_tokenized_text)

        segments_ids = np.tile(segments_ids, (end_point - start_point, 1))

        return masked_texts, segments_ids, start_point, end_point, tokenized_text[start_point:end_point]

    def perplexity(self, text):
        indexed_tokens, segments_ids, start_point, end_point, real_indexs = self.sentence_preprocese(text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor(indexed_tokens)
        segments_tensors = torch.tensor(segments_ids)

        # If you have a GPU, put everything on cuda
        tokens_tensor = tokens_tensor.to(DEVICE)
        segments_tensors = segments_tensors.to(DEVICE)

        # Predict all tokens
        with torch.no_grad():
            outputs = self.model(tokens_tensor, token_type_ids=segments_tensors)[0]
            predictions = torch.softmax(outputs, -1)

        total_perplexity = 0
        for i, step in enumerate(range(start_point, end_point)):
            # predicted_index = torch.argmax(predictions[i, step]).item()
            # predicted_token = self.tokenizer.convert_ids_to_tokens([predicted_index])[0]
            # print(predicted_token)
            total_perplexity += np.log(predictions[i, step, real_indexs[i]].item())

        # total_perplexity = np.exp(-total_perplexity / (end_point - start_point))
        total_perplexity = -total_perplexity / (end_point - start_point)
        return total_perplexity


if __name__ == '__main__':
    MODEL_PATH = 'bert-uncased/'
    text_formatter = lambda x: "[CLS]{} [SEP]".format(x)
    pchecker = Perplexity_Checker(MODEL_PATH)
    ''' for str'''
    mystr1='get away from me'
    mystr2='away get'
    out1=pchecker.perplexity(text_formatter(mystr1))
    out2=pchecker.perplexity(text_formatter(mystr2))
    print(mystr1,out1,mystr2,out2)

    """out:
    get away from me 3.246785984207556 away get 15.858620768486784
    """
