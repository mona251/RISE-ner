import argparse
import json
import torch
import os
import evaluate
from datasets import load_dataset
from utils.util import preprocess
from utils.eval import compute_scores
from collections import defaultdict

from transformers import AutoTokenizer, AutoModelForTokenClassification

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def inference(system, tokenized_datasets, model, metric, label_list, tokenizer):
    
    scores = compute_scores(tokenized_datasets, model, metric, label_list, device)
    summed_values = defaultdict(int)
    for d in scores:
        for key, value in d.items():
            summed_values[key] += value

    summed_dict = dict(summed_values)

    final_score = {key: score/len(tokenized_datasets['test']['labels']) for key, score in summed_dict.items()}
    
    with open(f'results_system{system}.txt', 'w') as file:
        json.dump(final_score, file, indent=4)   
    
    print("Inference completed! Evaluation scores: ", final_score)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-s", "--system", help='Specify the directory of the system you wish to use for inference.',
                       default='distilbert-base-cased-system-A')

    
    f = open('ner_tags.json')
    ner_tags_dict = json.load(f)
    args = parser.parse_args()
    model_name_or_path = args.system+'/'+os.listdir(args.system)[-1]
    tokenizer_name_or_path = model_name_or_path    
    system = 'A' if args.system=='distilbert-base-cased-system-A' else 'B'
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    
    metric = evaluate.load("seqeval")
    dataset = load_dataset("Babelscape/multinerd")
    tokenized_datasets, label_list, id2label, label2id = preprocess(dataset, tokenizer, system, ner_tags_dict)
    
    model = AutoModelForTokenClassification.from_pretrained(
    model_name_or_path,
    num_labels = len(label_list),
    id2label=id2label,
    label2id=label2id,
    ).to(device)
    
    inference(system, tokenized_datasets, model, metric, label_list, tokenizer)