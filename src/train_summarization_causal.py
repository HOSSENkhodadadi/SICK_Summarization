import warnings
warnings.filterwarnings("ignore")
import os
os.environ['WANDB_DISABLED'] = 'true'
import sys
sys.path.append('./')
import numpy as np
import nltk
nltk.download('punkt')
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_metric
from data.dataset import SamsumDataset_total, DialogsumDataset_total
import argparse
from augmentation import augmentation_on_dataset

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

# Define Global Values
model_checkpoint = "gpt2-medium"
tokenizer_checkpoint = "gpt2-medium"
max_len = 1024
vocab_size = 50257

dataset_list = [
    "samsum",
    "dialogsum"
]

import argparse

# Set Argument Parser
parser = argparse.ArgumentParser()

# Training hyperparameters
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--train_batch_size', type=int, default=20)
parser.add_argument('--val_batch_size',type=int, default=4)
parser.add_argument('--test_batch_size',type=int,default=1)

# Model hyperparameters
parser.add_argument('--model_name',type=str, default='openai/gpt2-medium')

# Optimizer hyperparameters
parser.add_argument('--init_lr',type=float, default=3e-6)
parser.add_argument('--warm_up',type=int, default=600)
parser.add_argument('--weight_decay',type=float, default=1e-2)
parser.add_argument('--decay_epoch',type=int, default=0)
parser.add_argument('--adam_beta1',type=float, default=0.9)
parser.add_argument('--adam_beta2',type=float, default=0.999)
parser.add_argument('--adam_eps',type=float, default=1e-12)
parser.add_argument('--dropout_rate',type=float, default=0.1)

# Tokenizer hyperparameters
parser.add_argument('--encoder_max_len', type=int, default=1024)
parser.add_argument('--decoder_max_len', type=int, default=100)
parser.add_argument('--vocab_size',type=int, default=50257)  # Default for GPT-2
parser.add_argument('--eos_idx',type=int, default=50256)     # Default for GPT-2
parser.add_argument('--tokenizer_name',type=str, default='GPT2Tokenizer')

# Checkpoint directory hyperparameters
parser.add_argument('--pretrained_weight_path',type=str, default='pretrained_weights')
parser.add_argument('--finetune_weight_path', type=str, default="./weights_sick_samsum_gpt2")
parser.add_argument('--best_finetune_weight_path',type=str, default='weights_sick_samsum_gpt2_best')

# Dataset hyperparameters
parser.add_argument('--dataset_name',type=str, default='samsum')
parser.add_argument('--use_paracomet',type=bool,default=False)
parser.add_argument('--use_sentence_transformer',type=bool,default=False)
parser.add_argument('--use_roberta',type=bool,default=False)
parser.add_argument('--dataset_directory',type=str, default='./data')
parser.add_argument('--test_output_file_name',type=str, default='samsum_context_trial2.txt')
parser.add_argument('--relation',type=str,default="xReason")
parser.add_argument('--supervision_relation',type=str,default='isAfter')
parser.add_argument('--data_augmentation',type=bool,required=True)

args = parser.parse_args()


# Set GPU
print()
print('***** Setting up GPU *****')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using GPU (cuda) or CPU? ', device)
print('Id of the GPU: ', torch.cuda.current_device())
print('Name of the GPU: ', torch.cuda.get_device_name())
print('Number of available GPUs: ', torch.cuda.device_count())
print()

# Set metric
metric = load_metric("./utils/rouge.py", trust_remote_code=True)

# Load Tokenizer associated to the model
tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)

# Add special token
special_tokens_dict = {'additional_special_tokens':['<I>','</I>']}
tokenizer.add_special_tokens(special_tokens_dict)

# Set dataset
if args.dataset_name == 'samsum':
    total_dataset = SamsumDataset_total(args.encoder_max_len,
                                        args.decoder_max_len,
                                        tokenizer,
                                        extra_context=True,
                                        paracomet=args.use_paracomet,
                                        relation=args.relation,
                                        supervision_relation=args.supervision_relation,
                                        roberta=args.use_roberta,
                                        sentence_transformer=args.use_sentence_transformer)
    
    train_dataset = total_dataset.getTrainData()
    if args.data_augmentation:
        train_dataset = augmentation_on_dataset('samsum', train_dataset)
    eval_dataset = total_dataset.getEvalData()
    test_dataset = total_dataset.getTestData()

elif args.dataset_name == 'dialogsum':
    total_dataset = DialogsumDataset_total(args.encoder_max_len,
                                           args.decoder_max_len,
                                           tokenizer,
                                           extra_context=True,
                                           paracomet=args.use_paracomet,
                                           relation=args.relation,
                                           supervision_relation=args.supervision_relation,
                                           sentence_transformer=args.use_sentence_transformer,
                                           roberta=args.use_roberta)
    
    train_dataset = total_dataset.getTrainData()
    if args.data_augmentation:
        train_dataset = augmentation_on_dataset('dialogsum', train_dataset)
    eval_dataset = total_dataset.getEvalData()
    test_dataset = total_dataset.getTestData()

print('***** Setting up Dataset *****')
print('Training Dataset Size is : ')
print(len(train_dataset))
print('Validation Dataset Size is : ')
print(len(eval_dataset))
print('Test Dataset Size is : ')
print(len(test_dataset))
print()

# Loading checkpoint of model
model = GPT2LMHeadModel.from_pretrained(args.model_name)
model.resize_token_embeddings(len(tokenizer))
model.gradient_checkpointing_enable()
model = model.to(device)
print('***** Setting up pretrained model *****')
print("The number of model's parameters is : ", model.num_parameters())
print()

# Set Training Arguments
training_args = TrainingArguments(
    output_dir=args.finetune_weight_path,
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    evaluation_strategy='epoch',
    logging_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=args.train_batch_size,
    per_device_eval_batch_size=args.val_batch_size,
    learning_rate=args.init_lr,
    weight_decay=args.weight_decay,
    adam_beta1=args.adam_beta1,
    adam_beta2=args.adam_beta2,
    adam_epsilon=args.adam_eps,
    num_train_epochs=args.epoch,
    max_grad_norm=0.1,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    lr_scheduler_type='polynomial',
    warmup_steps=args.warm_up,
    save_total_limit=1,
    fp16=True,
    seed=516,
    load_best_model_at_end=True,
    predict_with_generate=True,
    prediction_loss_only=False,
    generation_max_length=args.decoder_max_len,
    generation_num_beams=5,
    metric_for_best_model='eval_rouge1',
    greater_is_better=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Run Training (Finetuning)
print('Start the training ...')
print()
trainer.train()

# Save final weights
print('Save model in ', args.best_finetune_weight_path)
print()
trainer.save_model(args.best_finetune_weight_path)
