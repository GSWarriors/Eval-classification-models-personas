import pandas as pd
import numpy as np
import pandas as pd
import transformers as ppb  #pytorch transformers
#from transformers import BertTokenizer, BertForNextSentencePrediction
import torch
import random as rand



def main():

    my_list = []
    dataframe = pd.read_csv("/Users/arvindpunj/Desktop/Projects/NLP lab research/Extracting-personas-for-text-generation/train_none_original.txt",
    delimiter='\n', header= None, error_bad_lines=False)
    full_doc = dataframe[0]

    #filters our conversation to just the part we want, put into a list
    filtered_convo = filter_conversation(full_doc)
    persona_convo = []
    snippet_convo = []


    #make this a function called filter_persona_and_snippet()

    k = 2
    rand_convo_count = 0
    rand_convo_index = -1
    while rand_convo_count < 1:
        rand_convo_index = rand.randint(0, len(filtered_convo) - 1)
        rand_convo_count += 1


    if len(filtered_convo) > 1:
        if rand_convo_index == len(filtered_convo) - 1:
            snippet_convo = filtered_convo[rand_convo_index - 1:]
            del filtered_convo[rand_convo_index - 1:]
        else:
            snippet_convo = filtered_convo[rand_convo_index: rand_convo_index + 2]
            del filtered_convo[rand_convo_index: rand_convo_index + 2]

    persona_convo = filtered_convo


    persona_convo = add_speaker_tokens(persona_convo)
    #snippet_convo = add_speaker_tokens(snippet_convo)


    #create model, tokenizer and weights for persona and snippets
    #make this a function called tokenize_and_encode()
    """persona_model_class, persona_tokenizer_class, persona_pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
    snippet_model_class, snippet_tokenizer_class, snippet_pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')


    persona_tokenizer = persona_tokenizer_class.from_pretrained(persona_pretrained_weights)
    persona_model = persona_model_class.from_pretrained(persona_pretrained_weights)

    snippet_tokenizer = snippet_tokenizer_class.from_pretrained(snippet_pretrained_weights)
    snippet_model = snippet_model_class.from_pretrained(snippet_pretrained_weights)

    #persona and snippet tokenization
    persona_convo = ' '.join(persona_convo)
    persona_encoding = [persona_tokenizer.encode(persona_convo, add_special_tokens=True)]

    snippet_convo = ' '.join(snippet_convo)
    snippet_encoding = [snippet_tokenizer.encode(snippet_convo, add_special_tokens=True)]



    #bert padding (shorter sentences with 0)
    persona_max_len = 0
    persona_max_len = len(persona_encoding[0])
    padded_persona = np.array([i + [0]*(persona_max_len-len(i)) for i in persona_encoding])

    snippet_max_len = 0
    snippet_max_len = len(snippet_encoding[0])
    padded_snippet = np.array([i + [0]*(snippet_max_len-len(i)) for i in snippet_encoding])


    #processing with BERT, create input tensor
    persona_input_ids = torch.tensor(np.array(padded_persona))
    snippet_input_ids = torch.tensor(np.array(padded_snippet))

    with torch.enable_grad():
        persona_hidden_states = persona_model(persona_input_ids)

    with torch.no_grad():
        snippet_hidden_states = snippet_model(snippet_input_ids)


    #everything in last_hidden_states, now unpack 3-d output tensor.
    #features is 2d array with sentence embeddings of all sentences in dataset.
    #the model treats the entire persona as the "sentence". persona encoding
    print("output tensor of distilbert on persona")
    persona_features = persona_hidden_states[0][:, 0, :].detach().numpy()
    print("persona encoding: " + str(persona_features[0]))

    print("output tensor of distilbert on snippet")
    snippet_features = snippet_hidden_states[0][:, 0, :].numpy()
    print("snippet_encoding: " + str(snippet_features[0]))
    #next step- add special tokens"""


def add_speaker_tokens(convo):

    #for i in range(0, len(convo)):
    #print(convo)
    #print()
    speaker_num = 1
    new_response_str = ""
    tab_count = 0
    token_added = False
    new_convo = []

    for char in convo:

        if char == '\t':
            tab_count += 1
            print("tab")
            if speaker_num == 1:
                speaker_num = 2
                speaker_str = '<speaker-2>'
            else:
                speaker_num = 1
                speaker_str = '<speaker-1>'

            new_response_str = speaker_str + char
            new_convo.append(new_response_str)

        """else:
            if speaker_num == 1:
                speaker_str = '<speaker-1>'
            else:
                speaker_str = '<speaker-2>'

            new_response_str = speaker_str + char
            new_convo.append(new_response_str)"""

    print("new convo for persona is: " + str(new_convo))






def filter_conversation(full_doc):

    count = 0
    convo_list = []
    filtered_convo = []

    for line in range(0, 7):
        convo_list.append(full_doc[line])

    for i in range(0, len(convo_list)):
        filtered_convo.append(filter_responses(convo_list[i]))

    print("keeping tabs: " + repr(filtered_convo))
    print()
    return filtered_convo




def filter_responses(response):

    response_without_number = response[2:]
    #print(response_without_number)
    #print()
    tab_count = 0
    two_speaker_utterances = ""

    for char in response_without_number:
        if tab_count < 2:
            if char == '\t':
                tab_count += 1

            two_speaker_utterances += char

    return two_speaker_utterances



main()
