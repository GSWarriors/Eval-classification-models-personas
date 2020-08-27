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


    k = 2
    number_list = []
    count = 0
    while count < k:
        rand_convo = rand.randint(0, len(filtered_convo) - 1)
        if rand_convo not in number_list:
            number_list.append(rand_convo)
            count += 1


    #add responses in number list to snippet convo, otherwise, persona convo
    for i in range(0, len(filtered_convo)):
        if i in number_list:
            snippet_convo.append(filtered_convo[i])
        else:
            persona_convo.append(filtered_convo[i])

    #action steps now:
    #1. encode the persona using BERT/transfertransfo/dialogGPT

    #tokenizer class
    model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)

    #persona tokenization
    persona_convo = ''.join(persona_convo)
    encoding = [tokenizer.encode(persona_convo, add_special_tokens=True),
    tokenizer.encode(snippet_convo, add_special_tokens=True)]

    #bert padding (shorter sentences with 0)
    max_len = 0
    max_len = max(len(encoding[0]), len(encoding[1]))
    #print("first encoding: " + str(encoding[0]))
    #print("second encoding: " + str(encoding[1]))


    padded = np.array([i + [0]*(max_len-len(i)) for i in encoding])
    #print(padded)
    #print(str(np.array(padded).shape))

    #processing with BERT, create input tensor
    input_ids = torch.tensor(np.array(padded))
    with torch.no_grad():
        last_hidden_states = model(input_ids)


    #everything in last_hidden_states, now unpack 3-d output tensor.
    #features is 2d array with sentence embeddings of all sentences in dataset.
    #the model treats the entire persona as the "sentence". persona encoding
    print("BERT output tensor")
    features = last_hidden_states[0][:, 0, :].numpy()
    print("persona encoding: " + str(features[0]))
    print()
    print("gold snippet encoding: " + str(features[1]))






def filter_conversation(full_doc):

    count = 0
    convo_list = []
    filtered_convo = []

    for line in range(0, 7):
        convo_list.append(full_doc[line])

    for i in range(0, len(convo_list)):
        filtered_convo.append(filter_responses(convo_list[i]))

    #print(filtered_convo)
    return filtered_convo




def filter_responses(response):

    response_without_number = response[2:]
    #print(response_without_number)
    #print()
    tab_count = 0
    response_without_tabs = ""

    for char in response_without_number:
        if tab_count < 2:
            if char == '\t':
                tab_count += 1

            else:
                response_without_tabs += char

    return response_without_tabs



main()
