from functools import partial


def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def preprocess(dataset, tokenizer, system, ner_tags_dict):
    
    #Filter non-English samples
    dataset = dataset.filter(lambda example: example['lang'] == 'en')
    label_list = [key for key in ner_tags_dict.keys()]
    
    if system == 'A':
        id2label = {i: label for i, label in enumerate(label_list)}
        label2id = {v: k for k, v in id2label.items()}
    elif system == 'B':
        allowed_tags = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG',
                        'B-LOC', 'I-LOC', 'B-ANIM', 'I-ANIM', 'B-DIS', 'I-DIS']
        
        allowed_values = [i[1] for i in ner_tags_dict.items() if i[0] in allowed_tags]
        tags_values = {i: j for j, i in enumerate(allowed_values)}
        def replace_values(example):
            feature_values = example['ner_tags']
            #set any values outside of allowed_tags to 0
            replaced_values = [val if val in allowed_values else 0 for val in feature_values]
            replaced_values = [tags_values[i] for i in replaced_values]
            example['ner_tags'] = replaced_values
            return example

        dataset = dataset.map(replace_values)
        label_list = [i for i in allowed_tags]
        id2label = {i: label for i, label in enumerate(label_list)}
        label2id = {v: k for k, v in id2label.items()}
    
    tokenized_datasets = dataset.map(
    partial(tokenize_and_align_labels, tokenizer=tokenizer),
    batched=True,
    remove_columns=dataset["train"].column_names,
    )
    
    return tokenized_datasets, label_list, id2label, label2id