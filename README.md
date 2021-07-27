# General:
The framework include three modules: MCE, HEE, SAE.
The explanation for running the code, requirements and acknowledgements are as follows:

## 1. MCE
Code of MCE is based on the ConfidNet by Charles Corbiere.
name="ConfidNet", version='0.1.0'.

### Data:
Put the data into the data file: /.../module_1_MCE/confidnet/data/text/

### BERT pretrained file:
Download the orginal BERT-Base uncased file from: https://github.com/google-research/bert
and put them into /.../module_1_MCE/confidnet/data/bertfiles/ as follows:

/.../module_1_MCE/confidnet/data/bertfiles/bert-base-uncased/bert_config.json
/.../module_1_MCE/confidnet/data/bertfiles/bert-base-uncased/pytorch_model.bin
/.../module_1_MCE/confidnet/data/bertfiles/bert-base-uncased-vocab.txt

### Running the code:
####Train:
(1) run train_classify.job
(2) run train_confid.job

####Test:
(1) test_classify.job
(2) test_confid.job
(3) test_trustscore.job



## 2. HEE:
### Data:
Put the data into the data file: /.../module_2_HEE/hee/data/

### Running the code:
#### 5-fold validation: 
Change the value of feature, and then run: ` python cross_validation.py` 

#### prediction:
Change the value of feature, and then run: ` python predict.py` 

#### Explanation of feature variable:
feature='f1' means only dialogue related feature;
feature='f2' means both dialogue and worker related feature

#### Test time cost outputfile:
predicted_time_cost_per_dialogue_d.txt
predicted_time_cost_per_dialogue_dw.txt

#### Code of how we calculate bert-based perplexity and readability:
See codes in /.../module_2_HEE/hee/__utils__/


## 3. SAE
### Data:
Put the data into the data file: /.../module_3_SAE/ilp/data/

#### Running the code：
First, choose the inputfile:
    TCP:'/.../module_3_SAE/ilp/data/testc/ilp_data_tcp_f2.tsv'
    MCP:'/.../module_3_SAE/ilp/data/testc/ilp_data_mcp_f2.tsv'
    Trust score:'/.../module_3_SAE/ilp/data/testc/ilp_data_trustscore_f2.tsv'
    Trust score:'/.../module_3_SAE/ilp/data/testc/ilp_data_trustscore_f1.tsv'

Then, run: 
    ` python ilp.py` 

# Requirements
### Pytorch version: 3.7
### Dependency:
Tensorflow 2.0 (for tensorboard, saving model file etc.),
pandas,
numpy,
pyyaml,
torchsummary,
verboselogs,
tqdm,
scikit-learn,
torchtext,
spacy,
mip.

# Acknowledgements
Thanks for the authors for the following resource (Apache 2 License):
BERT: https://github.com/google-research/bert (Google);
Confident: https://github.com/valeoai/ConfidNet (Charles Corbière);
Trust Score: https://github.com/google/TrustScore (Heinrich Jiang);
MIP: https://www.python-mip.com (Túlio A. M. Toffolo, Haroldo G. Santos).

# Please refer to our paper:
@article{zhanghuman,
  title={A Human-machine Collaborative Framework for Evaluating Malevolence in Dialogues},
  author={Zhang, Yangjun and Ren, Pengjie and de Rijke, Maarten and Delhaize, Ahold},
  journal={ACL},
  year={2021},
}



