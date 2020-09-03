import pandas as pd
import numpy as np
import pandas as pd
import transformers as ppb  #pytorch transformers
#from transformers import BertTokenizer, BertForNextSentencePrediction
import torch
import random as rand





def main(df):

    k = 2
    persona_convo = []
    snippet_convo = []
    full_doc = df[0]


    count = 0
    convo_list = []
    filtered_convo = []
    snippet_list = []

    #go through full document, and append to convo list for 8 responses
    #(1 conversation). Then append these to a list, and get the persona and snippet
    #from it. Finally, reset convo_list to just the current line, and filtered_convo

    for line in range(0, len(full_doc)):
        if line > 0 and line % 8 == 0:
            #convo_list = []
            for i in range(0, len(convo_list)):
                filtered_convo.append(filter_for_responses(convo_list[i]))

            persona_convo, snippet_convo = filter_persona_and_snippet(filtered_convo, k)
            snippet_list.append(snippet_convo)
            #function here to add model, tokenizer, padding, and feature extraction

            convo_list = [full_doc[line]]
            filtered_convo = []
        else:
            convo_list.append(full_doc[line])


    """print("persona convo: " + str(persona_convo))
    print()
    print("snippet convo: " + str(snippet_convo))
    print()"""


    tokenization_and_feature_extraction(persona_convo, snippet_convo, snippet_list)




"""This function creates the DistilBertModel, tokenizes persona and snippet input,
pads and encodes it, and extracts feature vectors"""
def tokenization_and_feature_extraction(persona_convo, snippet_convo, snippet_list):
    #create model, tokenizer and weights for persona and snippets
    #make this a function called tokenize_and_encode()
    persona_model_class, persona_tokenizer_class, persona_pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
    snippet_model_class, snippet_tokenizer_class, snippet_pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

    persona_tokenizer = persona_tokenizer_class.from_pretrained(persona_pretrained_weights)
    persona_model = persona_model_class.from_pretrained(persona_pretrained_weights)

    snippet_tokenizer = snippet_tokenizer_class.from_pretrained(snippet_pretrained_weights)
    snippet_model = snippet_model_class.from_pretrained(snippet_pretrained_weights)

    #adding tokenizer for speaker 1 and speaker 2 just for persona and snippet
    special_tokens_dict = {'additional_special_tokens': ['<speaker-1>', '<speaker-2>']}
    num_added_toks = persona_tokenizer.add_special_tokens(special_tokens_dict)
    snippet_tokenizer.add_special_tokens(special_tokens_dict)

    print("number of added tokens: " + str(num_added_toks))
    persona_model.resize_token_embeddings(len(persona_tokenizer))
    snippet_model.resize_token_embeddings(len(snippet_tokenizer))

    #persona and snippet tokenization
    persona_convo = ' '.join(persona_convo)
    persona_encoding = [persona_tokenizer.encode(persona_convo, add_special_tokens=True)]

    snippet_convo = ' '.join(snippet_convo)
    snippet_encoding = []

    for i in range(0, len(snippet_list)):
        snippet_encoding.append(snippet_tokenizer.encode(snippet_list[i]))

    #bert padding (shorter sentences with 0)
    persona_max_len = 0
    persona_max_len = len(persona_encoding[0])
    padded_persona = np.array([i + [0]*(persona_max_len-len(i)) for i in persona_encoding])

    snippet_max_len = 0
    for i in snippet_encoding:
        if len(i) > snippet_max_len:
            snippet_max_len = len(i)

    print("snippet max length is: " + str(snippet_max_len))
    padded_snippet = np.array([i + [0]*(snippet_max_len-len(i)) for i in snippet_encoding])

    #processing with BERT, create input tensor
    #persona_input_ids = torch.tensor(np.array(padded_persona))
    snippet_input_ids = torch.tensor(np.array(padded_snippet))

    #with torch.enable_grad():
    #    persona_hidden_states = persona_model(persona_input_ids)

    with torch.no_grad():
        snippet_hidden_states = snippet_model(snippet_input_ids)


    #everything in last_hidden_states, now unpack 3-d output tensor.
    #features is 2d array with sentence embeddings of all sentences in dataset.
    #the model treats the entire persona as the "sentence". persona encoding
    """print("output tensor of distilbert on persona with special tokens")
    persona_features = persona_hidden_states[0][:, 0, :].detach().numpy()
    print("persona embedding: " + str(persona_features[0]))"""

    print("output tensor of distilbert on snippet with special tokens")
    snippet_features = snippet_hidden_states[0][:, 0, :].numpy()
    print("embedding of all snippets:" + str(snippet_features[0]))




def filter_persona_and_snippet(filtered_convo, snippet_size):


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
            snippet_convo = filtered_convo[rand_convo_index: rand_convo_index + snippet_size]
            del filtered_convo[rand_convo_index: rand_convo_index + snippet_size]

    persona_convo = filtered_convo


    persona_convo = add_speaker_tokens(persona_convo)
    snippet_convo = add_speaker_tokens(snippet_convo)


    return persona_convo, snippet_convo




def add_speaker_tokens(convo):

    speaker_num = 2
    new_response_str = ""
    new_convo = []

    for line in range(0, len(convo)):

        last_speaker_index = 0
        new_response_str = ""
        for char in range(0, len(convo[line])):
            if convo[line][char] == '\t':
                if speaker_num == 1:
                    speaker_num = 2
                    speaker_str = ' <speaker-2> '

                    #add speaker 2 tag from after tab where speaker 1 left off
                    curr_response = convo[line][last_speaker_index + 1: len(convo[line]) - 1]
                    new_response_str += speaker_str + curr_response
                    #print("new response str is : " + new_response_str)

                else:
                    speaker_num = 1
                    speaker_str = '<speaker-1> '

                    #first speaker is 0 up to current char (the tab)
                    curr_response = convo[line][0: char]
                    new_response_str += speaker_str + curr_response
                    #print("new response str is : " + new_response_str)
                    last_speaker_index = char

        new_convo.append(new_response_str)

    return new_convo




#filters 1 back and forth between two speakers and returns the string
def filter_for_responses(response):

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







my_list = []
dataframe = pd.read_csv("/Users/arvindpunj/Desktop/Projects/NLP lab research/Extracting-personas-for-text-generation/train_none_original.txt",
delimiter='\n', header= None, error_bad_lines=False)


main(dataframe)
