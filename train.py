from datasets import load_dataset
from transformers import (AutoTokenizer,
                          AutoModelForTokenClassification,
                          Trainer,
                          TrainingArguments,
                          DataCollatorForTokenClassification,
                         )
from utils.util import preprocess
from utils.eval import compute_metrics
from functools import partial

import torch
import evaluate
import argparse
import json

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(tokenized_datasets, model, 
          tokenizer, nr_epochs, label_list, output_dir):
    
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    args = TrainingArguments(
    output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=nr_epochs,
    weight_decay=0.01,
    push_to_hub=False,
    save_total_limit=1
    )
    trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=partial(compute_metrics, metric=metric, label_list=label_list),
    tokenizer=tokenizer,
    )
    
    trainer.train()
    
    print("Fine-tuning completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--system", help='Specify the system you wish to fine-tune.', default='A',
                       type=str)
    parser.add_argument("-m", "--model", help='Specify the model you wish to use to fine-tune.',
                       default='distilbert-base-cased', type=str)
    parser.add_argument("-e", "--epoch", help='number of epochs for fine-tuning.', default=1, type=int)
    
    f = open('ner_tags.json')
    ner_tags_dict = json.load(f)
    
    args = parser.parse_args()
    nr_epochs = args.epoch
    model_name_or_path = args.model
    tokenizer_name_or_path = model_name_or_path
    system = args.system
    output_dir = f"./{args.model}-system-A" if system == 'A' else f"./{args.model}-system-B"
    
    metric = evaluate.load("seqeval")
    dataset = load_dataset("Babelscape/multinerd")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    tokenized_datasets, label_list, id2label, label2id = preprocess(dataset, tokenizer, system, ner_tags_dict)
    model = AutoModelForTokenClassification.from_pretrained(
    model_name_or_path,
    num_labels = len(label_list),
    id2label=id2label,
    label2id=label2id,
    ).to(device)
    
    train(tokenized_datasets, model, 
          tokenizer, nr_epochs, label_list, output_dir)