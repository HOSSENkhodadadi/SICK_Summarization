import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
import nlpaug.augmenter.word as naw
import emoji
import random

from tqdm import tqdm


def dialogue2sentences(dialogue):
    sentences = []

    for idx, sentence in enumerate(dialogue.split('\r\n')):
        if sentence != '':
            person, sentence = sentence.split(': ', 1)
            sentences.append([idx, person, sentence])

    return sentences

def contains_emoji(text):
    for word in text:
        if word in emoji.EMOJI_DATA:
            return True
    return False

def contextual_word_substitute(text):
    aug = naw.ContextualWordEmbsAug(model_path='roberta-base', action="substitute")
    return aug.augment(text)[0]

def wordnet_word_substitute(text):
    aug = naw.SynonymAug(aug_src='wordnet')
    return aug.augment(text)[0]

def contextual_word_insertion(text):
    aug = naw.ContextualWordEmbsAug(model_path='roberta-base', action="insert")
    return aug.augment(text)[0]

def random_word_swap(text):
    aug = naw.RandomWordAug(action="swap")
    return aug.augment(text)[0]

def random_word_deletion(text):
    aug = naw.RandomWordAug()
    return aug.augment(text)[0]

def word_level_augmentation(dialogue):
    
    sentences = dialogue2sentences(dialogue)
    for i in range(len(sentences)):
        if not contains_emoji(sentences[i][-1]):
            augmentations = i % 5

            if augmentations == 0:
                sentences[i][-1] = wordnet_word_substitute(sentences[i][-1])
            elif augmentations == 1:
                sentences[i][-1] = contextual_word_substitute(sentences[i][-1])
            elif augmentations == 2:
                sentences[i][-1] = contextual_word_insertion(sentences[i][-1])
            elif augmentations == 3:
                sentences[i][-1] = random_word_swap(sentences[i][-1])
            else:
                sentences[i][-1] = random_word_deletion(sentences[i][-1])

    sentences.sort(key=lambda x: x[0]) 
    dialogue = ''
    for sentence in sentences:
        dialogue = dialogue + sentence[1] + ': ' + sentence[2] + '\r\n'

    return dialogue

def translation(dialogue):
    aug = naw.BackTranslationAug(from_model_name='facebook/wmt19-en-de',to_model_name='facebook/wmt19-de-en')
    
    sentences = dialogue2sentences(dialogue)
    for i in range(len(sentences)):
        if not contains_emoji(sentences[i][-1]):
            sentences[i][-1] = aug.augment(sentences[i][-1])[0]

    sentences.sort(key=lambda x: x[0]) 
    dialogue = ''
    for sentence in sentences:
        dialogue = dialogue + sentence[1] + ': ' + sentence[2] + '\r\n'

    return dialogue

def augmentation_on_dataset(dataset):
    print('======================== Performing augmentation on the dataset ========================')
    for i, dialogue in tqdm(enumerate(dataset.data['dialogue'])):
        # print(dialogue)
        if random.random() <= 0.1:
            dataset.data['dialogue'][i] = word_level_augmentation(dialogue)
        else:
            if random.random() <= 0.1:
                dataset.data['dialogue'][i] = translation(dialogue)

    return dataset