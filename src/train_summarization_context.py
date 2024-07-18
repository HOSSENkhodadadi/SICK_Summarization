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
from transformers import AutoTokenizer
from transformers import AutoConfig, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_metric
from data.dataset import SamsumDataset_total, DialogsumDataset_total
import argparse
from augmentation import augmentation_on_dataset


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}


# Define Global Values
model_checkpoint_list = [
    "facebook/bart-large", 
    "facebook/bart-large-xsum",
    "google/pegasus-large",
    "google/peagsus-xsum",
    "google/t5-large-lm-adapt", 
    "google/t5-v1_1-large",
    'microsoft/prophetnet-large-uncased-cnndm',
    "google-t5/t5-base",
    "microsoft/unilm-large-cased",
    "openai-community/gpt2-medium",
    "allenai/led-large-16384"
]
tokenizer_list = {
    "facebook/bart-large":"RobertaTokenizer",
    "facebook/bart-large-xsum":"RobertaTokenizer",
    "google/pegasus-large":"PegasusTokenizer",
    "google/peagsus-xsum":"PegasusTokenizer",
    "google/t5-large-lm-adapt":"T5Tokenizer", 
    "google/t5-v1_1-large":"T5Tokenizer",
    "google-t5/t5-base": "T5Tokenizer",
    'microsoft/prophetnet-large-uncased-cnndm':'ProphetNetTokenizer',
    "microsoft/unilm-large-cased":"BertTokenizer",
    "openai-community/gpt2-medium": "GPT2Tokenizer",
    "allenai/led-large-16384": "LongformerTokenizer"
}
max_len_list ={
    "facebook/bart-large":1024,
    "facebook/bart-large-xsum":1024,
    "google/pegasus-large":1024,
    "google/peagsus-xsum":512,
    "google/t5-large-lm-adapt":512, 
    "google/t5-v1_1-large":512,
    "google-t5/t5-base":512,
    'microsoft/prophetnet-large-uncased-cnndm':512,
    "microsoft/unilm-large-cased":512,
    "openai-community/gpt2-medium": 1024,
    "allenai/led-large-16384":16384
}
vocab_size_list={
    "facebook/bart-large":50265,
    "facebook/bart-large-xsum":50264,
    "google/pegasus-large":96103,
    "google/peagsus-xsum":96103,
    "google/t5-large-lm-adapt":32128, 
    "google/t5-v1_1-large":32128,
    "google-t5/t5-base":32128,
    'microsoft/prophetnet-large-uncased-cnndm':30522,
    "microsoft/unilm-large-cased": 30522,
    "openai-community/gpt2-medium":50257,
    "allenai/led-large-16384":50256
}
dataset_list = [
    "samsum",
    "dialogsum"
]


# Set Argument Parser
parser = argparse.ArgumentParser()

# Training hyperparameters
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--train_batch_size', type=int, default=20)
parser.add_argument('--val_batch_size',type=int, default=4)
parser.add_argument('--test_batch_size',type=int,default=1)

# Model hyperparameters
parser.add_argument('--model_name',type=str, default='facebook/bart-large')

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
parser.add_argument('--vocab_size',type=int, default=51201)
parser.add_argument('--eos_idx',type=int, default=51200)
parser.add_argument('--tokenizer_name',type=str, default='RobertaTokenizer')

# Checkpoint directory hyperparameters
parser.add_argument('--pretrained_weight_path',type=str, default='pretrained_weights')
parser.add_argument('--finetune_weight_path', type=str, default="./context_BART_weights_Samsum_5epoch")
parser.add_argument('--best_finetune_weight_path',type=str, default='context_final_BART_weights_Samsum_5epoch')

# Dataset hyperparameters
parser.add_argument('--dataset_name',type=str, default='samsum')
parser.add_argument('--use_paracomet',type=bool,default=False)
parser.add_argument('--use_roberta',type=bool,default=False)
parser.add_argument('--use_sentence_transformer',type=bool,default=False)
parser.add_argument('--dataset_directory',type=str, default='./data')
parser.add_argument('--test_output_file_name',type=str, default='samsum_context_trial2.txt')
parser.add_argument('--relation',type=str,default="xReason")
parser.add_argument('--supervision_relation',type=str,default='isAfter')
parser.add_argument('--data_augmentation',type=bool,default=False)
parser.add_argument('--coref',type=bool,default=False)
parser.add_argument('--kw',type=bool,default=False)

args = parser.parse_args()

# Refine arguments based on global values
if args.model_name not in model_checkpoint_list:
    assert "Your Model checkpoint name is not valid"

args.tokenizer_name = tokenizer_list[args.model_name]

args.vocab_size = vocab_size_list[args.model_name]

if args.dataset_name not in dataset_list:
    assert "Your Dataset name is not valid"


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
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

# Add special token 
special_tokens_dict = {'additional_special_tokens':['<I>','</I>']}
tokenizer.add_special_tokens(special_tokens_dict)


# Set dataset
if args.dataset_name=='samsum':
    total_dataset = SamsumDataset_total(args.encoder_max_len,
                                        args.decoder_max_len,
                                        tokenizer,
                                        extra_context=True,
                                        paracomet=args.use_paracomet,
                                        relation=args.relation,
                                        supervision_relation=args.supervision_relation,
                                        roberta=args.use_roberta,
                                        sentence_transformer=args.use_sentence_transformer,
                                        coref=args.coref,
                                        keyword=args.kw
                                        )
    
    train_dataset = total_dataset.getTrainData()
    eval_dataset = total_dataset.getEvalData()
    test_dataset = total_dataset.getTestData()
    
    if args.data_augmentation:
        train_dataset = augmentation_on_dataset('samsum', train_dataset)


elif args.dataset_name=='dialogsum':
    total_dataset = DialogsumDataset_total(args.encoder_max_len,
                                           args.decoder_max_len,
                                           tokenizer,
                                           extra_context=True,
                                           paracomet=args.use_paracomet,
                                           relation=args.relation,
                                           supervision_relation=args.supervision_relation,
                                           sentence_transformer=args.use_sentence_transformer,
                                           roberta=args.use_roberta,
                                           coref=args.coref,
                                           keyword=args.kw
                                           )
    
    train_dataset = total_dataset.getTrainData()
    eval_dataset = total_dataset.getEvalData()
    test_dataset = total_dataset.getTestData()

    if args.data_augmentation:
        train_dataset = augmentation_on_dataset('dialoguesum', train_dataset)
    

print('***** Setting up Dataset *****')
print('Training Dataset Size is : ')
print(len(train_dataset))
print('Validation Dataset Size is : ')
print(len(eval_dataset))
print('Test Dataset Size is : ')
print(len(test_dataset))
print()


# Loading checkpoint of model
config = AutoConfig.from_pretrained(args.model_name)
finetune_model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
# Set extra Configuration for Finetuning on Summarization Dataset
finetune_model.resize_token_embeddings(len(tokenizer))
finetune_model.gradient_checkpointing_enable()
finetune_model = finetune_model.to(device)
print('***** Setting up pretrained model *****')
print("The number of model's parameters is : ",finetune_model.num_parameters())
print()


# Set Training Arguments
finetune_args = Seq2SeqTrainingArguments(
    output_dir = args.finetune_weight_path,
    overwrite_output_dir = True,
    do_train=True,
    do_eval=True,
    do_predict=True,
    evaluation_strategy='epoch',
    logging_strategy="epoch",
    save_strategy= "epoch",
    per_device_train_batch_size = args.train_batch_size,
    per_device_eval_batch_size = args.val_batch_size,
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
    warmup_steps= args.warm_up,
    save_total_limit=1,
    fp16=True,
    seed = 516,
    load_best_model_at_end=True,
    predict_with_generate=True,
    prediction_loss_only=False,
    generation_max_length=100,
    generation_num_beams=5,
    metric_for_best_model='eval_rouge1',
    greater_is_better=True,
)

finetune_trainer = Seq2SeqTrainer(
    model = finetune_model,
    args = finetune_args,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    tokenizer = tokenizer,
    compute_metrics=compute_metrics,
)

# Run Training (Finetuning)
print('Start the training ...')
print()
finetune_trainer.train()

# Save final weights
print('Save model in ', args.best_finetune_weight_path)
print()
finetune_trainer.save_model(args.best_finetune_weight_path)