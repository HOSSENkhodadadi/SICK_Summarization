## Leveraging Personal Pronoun Resolution, Keywords Extraction, Data Augmentation, and Model Exploration for Enhanced Abstractive Dialogue Summarization
Hey is that too difficult to read the whole README [LET'S CHAT WITH THE REPO!](https://hf.co/chat/assistant/66d7552c06318396b70e04d4)

The repository for the final project of the Deep Natural Language Processing (DNLP) at Politecnico di Torino. (2023/24 academic year)

SICK is a framework for the abstractive dialogue summarization. You can find the links to the paper, YouTube video, and GitHub repository in the following

Paper Link : https://arxiv.org/abs/2209.00930

Youtube Explanation : https://www.youtube.com/watch?v=xHr3Ujlib4Y

GitHub: https://github.com/SeungoneKim/SICK_Summarization

In this project, we tried to improve the performance of SICK using the following methods
- Data Augmentation
- Model Exploration
- Personal Pronoun Resolution (PPR)
- Keyword Extraction

## Dataset Download
For training and evaluating on Samsum, we use dataset provided by [Hugging Face Dataset Library](https://github.com/huggingface/datasets). For Dialogsum, the dataset is not automatically provided, so you can download it from the url below,
```
https://drive.google.com/drive/folders/1CuZaU5Xw0AiIPaBTRrjToFkktS7_6KwG?usp=share_link
```
and put it under the directory of SICK_summarization/data/DialogSum_Data.
```
mkdir data/DialogSum_Data
```

Also, you could download the preprocessed commonsense data from the url below,
```
https://drive.google.com/drive/folders/1z1MXBGJ3pt0lC5dneMfFrQgxXTD8Iqrr?usp=share_link
```
and put it under the directory of SICK_summarization/data/COMET_data.
```
mkdir data/COMET_data
```

## How to train and make inference
Firstly, a vritual environment should be created using the following command and the .yml file
```
conda env create -f SICK_env.yml
```
Then, using the following command, you can train the model in the base situation (without any extention)
```
!python3 src/train_summarization_context.py --finetune_weight_path="a/path" --best_finetune_weight_path="a/path" --dataset_name="dialogsum or samsum" --use_paracomet=True --model_name="facebook/bart-large-xsum" --epoch=5 --use_sentence_transformer True
```
where
- finetune_weight_path: the path used during training to store some files
- best_finetune_weight_path: the path in which the final model is stored
- use_paracomet : If you set to true, it will use paracomet, and if not, it will use comet by default.
- use_sentence_transformer : If you would like to use the commonsense selected with sentence_transformer, you should use this argument

After training, you can make inference using the following command
```
!python3 src/inference.py --dataset_name "dialogsum or samsum" --model_name "facebook/bart-large-xsum" --model_checkpoint="a/path" --test_output_file_name="a/path" --use_paracomet True --num_beams 20 --train_configuration="context" --use_sentence_transformer True
```
where
- model_checkpoint: is the path  in which the final model is saved (the best_finetune_wight_path in the previous command)
- test_output_file_name: the name and path of the file in which test metrics are reported

## Model Exploration
We explored "Microsoft ProphetNet" and "Google T5". To do that, you just need to change the ```--model_name```
- Microsoft ProphetNet -> "microsoft/prophetnet-large-uncased-cnndm"
- Google T5 -> "google-t5/t5-base"

## Data Augmentation
To perform data augmentation, you just need to pass an additional arguemnt ```--data_augmentation True```

## Personal Pronoun Resolution
To perform the Personal Pronoun Resolution (PPR), you just need to pass an additional argument ```--coref True```

We used a specific version of Spacy to perform Personal Pronoun Resolution (PPR). This version has conflicts with the previous virtual environment. To solve the conflicts, firstly, data is sampled from the original datasets by running ```raw_data_to_json.py``` and saved in some JSON files ("raw" appears in their names). Then the virtual environment is deactivated and a new virtual environment is created and the following dependecy is installed

```!pip install https://github.com/explosion/spacy-experimental/releases/download/v0.6.0/en_coreference_web_trf-3.4.0a0-py3-none-any.whl```

Then, ```pronoun_resolution.py``` is run to perform PPR on the data. It again saves the edited data in some new JSON files ("coref" appears in their names). Finally, the command for the training is runn with the additional argument.

## Keyword Extraction
To perform the keyword extraction, you just need to pass an additional argument ```--kw True```

We used ```keybert``` and ```keyphrase_vectorizers``` to extract keywrods. These packages have a similar problem to the one mentioend in the PPR section. The problems is solved with the similar approach. In the new virtual environment, the follosing must be installed
- ```!pip install keyphrase_vectorizers```
- ```!pip install keybert```

Also, instead of ```pronoun_resolution.py```, the script ```extract_keyword.py``` must be run. The corresponding generated files have "with_kws" in their names.
