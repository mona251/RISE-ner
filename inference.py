import argparse
import json
import torch
import os
import evaluate
import warnings
import numpy as np

from tqdm import tqdm
from datasets import load_dataset
from utils.util import preprocess
from utils.eval import compute_scores, compute_inference_metrics
from collections import defaultdict
from itertools import chain

from transformers import AutoTokenizer, AutoModelForTokenClassification

warnings.filterwarnings("ignore")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def inference(system, tokenized_datasets, model, metric, confusion_matrix, label_list):
    
    y_test = []
    y_pred = []
    for inputs in tqdm(tokenized_datasets['test']):
        y_test.append(inputs['labels'])
        with torch.no_grad():
            inputs = {'input_ids': torch.Tensor([inputs['input_ids']]).long().to(device),
                    'attention_mask': torch.Tensor([inputs['attention_mask']]).long().to(device)}
            logits = model(**inputs).logits
            pred = np.argmax(logits.cpu().numpy(), axis = 2)[0]
        y_pred.append(pred)


    y_test = list(chain.from_iterable(y_test))
    y_pred =  list(chain.from_iterable(y_pred))
    
    score, confusion = compute_inference_metrics(y_test, y_pred, metric, confusion_matrix, 
                                                 label_list, label2id)
    
    with open(f'results_system{system}.txt', 'w') as file:
        json.dump(score, file, indent=4)   
        
    
    with open(f'confusion_matrix_system{system}.txt', 'w') as file:
        for row in confusion['confusion_matrix']:
            row_str = '\t'.join(str(val) for val in row)
            file.write(row_str + '\n')
    print("Inference completed! Evaluation scores: ", score, '\n',
         "confusion matrix: ", confusion['confusion_matrix'].tolist())
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-s", "--system", help='Specify the directory of the system you wish to use for inference.',
                       default='distilbert-base-cased-system-A')

    
    f = open('ner_tags.json')
    ner_tags_dict = json.load(f)
    args = parser.parse_args()
    ckpt_dirs = os.listdir(args.system)
    if 'runs' in ckpt_dirs:
        ckpt_dirs.remove('runs')
    ckpt_dirs = sorted(ckpt_dirs, key=lambda x: int(x.split('-')[1]))
    last_ckpt = ckpt_dirs[-1]
    model_name_or_path = args.system+'/'+last_ckpt
    tokenizer_name_or_path = model_name_or_path    
    system = 'A' if args.system=='distilbert-base-cased-system-A' else 'B'
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    
    metric = evaluate.load("seqeval")
    confusion_matrix = evaluate.load("BucketHeadP65/confusion_matrix")

    dataset = load_dataset("Babelscape/multinerd")
    tokenized_datasets, label_list, id2label, label2id = preprocess(dataset, tokenizer, system, ner_tags_dict)
    
    model = AutoModelForTokenClassification.from_pretrained(
    model_name_or_path,
    num_labels = len(label_list),
    id2label=id2label,
    label2id=label2id,
    ).to(device)
    
    inference(system, tokenized_datasets, model, metric, confusion_matrix, label_list)