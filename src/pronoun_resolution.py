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

