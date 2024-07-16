import warnings
warnings.filterwarnings("ignore")
import os
os.environ['WANDB_DISABLED'] = 'true'
import sys
sys.path.append('./')
import numpy as np
import nltk
nltk.download('punkt')
from transformers import AutoTokenizer
from data.dataset import SamsumDataset_total, DialogsumDataset_total
import argparse
import spacy
import argparse
from tqdm import tqdm
from data.dataset import SamsumDataset_total, DialogsumDataset_total
import json
from tqdm import tqdm


def resolve_references(doc) -> str:
    """Function for resolving references with the coref ouput
    doc (Doc): The Doc object processed by the coref pipeline
    RETURNS (str): The Doc string with resolved references
    """
    # token.idx : token.text
    token_mention_mapper = {}
    output_string = ""
    clusters = [
        val for key, val in doc.spans.items() if key.startswith("coref_cluster")
    ]

    # Iterate through every found cluster
    for cluster in clusters:
        first_mention = cluster[0]
        # Iterate through every other span in the cluster
        for mention_span in list(cluster)[1:]:
            # Set first_mention as value for the first token in mention_span in the token_mention_mapper
            token_mention_mapper[mention_span[0].idx] = first_mention.text + mention_span[0].whitespace_

            for token in mention_span[1:]:
                # Set empty string for all the other tokens in mention_span
                token_mention_mapper[token.idx] = ""

    # Iterate through every token in the Doc
    for token in doc:
        # Check if token exists in token_mention_mapper
        if token.idx in token_mention_mapper:
            output_string += token_mention_mapper[token.idx]
        # Else add original token text
        else:
            output_string += token.text + token.whitespace_

    return output_string

def resolve_references_in_dialogue(nlp, dataset_name, dataset):
    if dataset_name == 'samsum':
        splitter = '\r\n'
    else:
        splitter = '\n'

    print('======================== Performing Pronoun Resolution ========================')
    for i, dialogue in tqdm(enumerate(dataset.dialogue), total=len(dataset.dialogue)):
        dialogue_without_names = ''
        persons = []

        for sentence in dialogue.split(splitter):
            sentence = sentence.strip()
            if sentence != '':
                if ': ' in sentence:
                    person, sentence = sentence.split(': ', 1)
                    persons.append(person)
                else:
                    persons.append('')

                dialogue_without_names += sentence + '\n'

        doc = nlp(dialogue_without_names)
        text = resolve_references(doc)
        output = ''
        for person, sentence in zip(persons, text.split('\n')):
            output += f'{person}: {sentence}{splitter}'

    dataset.dialogue[i] = output

def convert2json(dataset_name, dataset):
    data = []
    for id_, dialogue, summary in tqdm( zip(dataset.id, dataset.dialogue, data.summary), total=len(dataset.id) ):
        temp = {'id': id_, 'dialogue': dialogue, 'summary': summary}
        data.append(temp)

    with open(f'data/{dataset_name}_with_coref.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)


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
parser.add_argument('--data_augmentation',type=bool,required=True)
parser.add_argument('--coref',type=bool,required=True)

args = parser.parse_args()

args.tokenizer_name = "RobertaTokenizer"
args.vocab_size = 50264
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
special_tokens_dict = {'additional_special_tokens':['<I>','</I>']}
tokenizer.add_special_tokens(special_tokens_dict)

if args.dataset_name=='samsum':
    total_dataset = SamsumDataset_total(args.encoder_max_len,
                                        args.decoder_max_len,
                                        tokenizer,
                                        extra_context=True,
                                        paracomet=args.use_paracomet,
                                        relation=args.relation,
                                        supervision_relation=args.supervision_relation,
                                        roberta=args.use_roberta,
                                        sentence_transformer=args.use_sentence_transformer
                                        )
    
    train_dataset = total_dataset.getTrainData()
    eval_dataset = total_dataset.getEvalData()
    test_dataset = total_dataset.getTestData()

    if args.coref:
        nlp = spacy.load("en_coreference_web_trf")
        train_dataset = resolve_references_in_dialogue(nlp, 'samsum', train_dataset)
        eval_dataset = resolve_references_in_dialogue(nlp, 'samsum', eval_dataset)
        test_dataset = resolve_references_in_dialogue(nlp, 'samsum', test_dataset)

elif args.dataset_name=='dialogsum':
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
    eval_dataset = total_dataset.getEvalData()
    test_dataset = total_dataset.getTestData()
    
    if args.coref:
        nlp = spacy.load("en_coreference_web_trf")
        train_dataset = resolve_references_in_dialogue(nlp, 'dialoguesum', train_dataset)
        eval_dataset = resolve_references_in_dialogue(nlp, 'dialoguesum', eval_dataset)
        test_dataset = resolve_references_in_dialogue(nlp, 'dialoguesum', test_dataset)


convert2json(f'{args.dataset_name}_train', train_dataset)
convert2json(f'{args.dataset_name}_eval', eval_dataset)
convert2json(f'{args.dataset_name}_test', test_dataset)
