import torch
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
import transformers as ppb  #pytorch transformers
import random as rand
import time
import math
import itertools
import json
from matplotlib import pyplot
import nltk
import string

from mymodel import DistilBertTrainingParams
from mymodel import DistilBertandBilinear
from mymodel import create_persona_and_snippet_lists
from mymodel import create_encoding_dict
from mymodel import create_training_file
from mymodel import create_validation_file
from mymodel import add_padding_and_mask

from mymodel_test import create_testing_file


#need 3 models total

def main(test_df):

    #load original distilbert + bilinear saved model and open/modify test set

    training_params = DistilBertTrainingParams()
    training_params.create_tokens_dict()

    saved_model = training_params.convo_classifier
    saved_optimizer = training_params.optimizer

    #mymodel implementation
    saved_model.load_state_dict(torch.load("/Users/arvindpunj/Desktop/Projects/NLP lab research/savedmodels/finaldistilbertmodel.pt", map_location=torch.device('cpu')))

    #print("done loading original distilbert model!")
    test_personas, test_responses = create_persona_and_snippet_lists(test_df)
    encoded_test_dict, smallest_convo_size = create_encoding_dict(training_params, test_responses)
    create_testing_file(test_personas, test_responses)

    #here's the part where I modify the responses in the test set
    modify_responses(test_personas, encoded_test_dict, saved_model, training_params)






def modify_responses(test_personas, encoded_test_dict, saved_model, training_params):
    test_file = open("positive-test-samples.json", "r")
    test_data = json.load(test_file)
    stopwords_list = []

    #opens file with stopwords and puts it into a list
    with open("/Users/arvindpunj/Desktop/Projects/NLP lab research/Extracting-personas-for-text-generation/data/stopwords/english.txt", "r") as stopwords_file:
        for line in stopwords_file:
            line = line.strip('\n')
            stopwords_list.append(line)

    stopwords_list.append('speaker-1')


    for i in range(0, len(test_personas)):
        persona_convo = ' '.join(test_data[i]['text persona'])
        response_convo = test_data[i]['text snippet']
        #persona_encoding = [training_params.tokenizer.encode(persona_convo, add_special_tokens=True)]
        #gold_snippet_encoding = encoded_test_dict[i]
        print("persona convo: " + str(persona_convo))

        filtered_words_list = tag_persona_and_create_dict(persona_convo, stopwords_list)
        print("response convo before: " + str(response_convo))
        print()
        response_convo = create_response_freq_dict(response_convo, filtered_words_list, stopwords_list)
        print("response convo after: " + str(response_convo))



        if i == 0:
            break






def tag_persona_and_create_dict(persona_convo, stopwords_list):

    #tokenize persona into separate words
    persona_convo = nltk.word_tokenize(persona_convo)
    #print("persona after tokenization: " + str(persona_convo))
    #print()

    persona_dict = {}
    final_dict = {}
    persona_words_list = []

    #remove stopwords, get the frequency of words from persona
    for i in range(0, len(persona_convo)):
        curr_elem = persona_convo[i]
        if curr_elem not in stopwords_list and curr_elem not in string.punctuation:
            persona_words_list.append(curr_elem)

            if curr_elem not in persona_dict:
                persona_dict[curr_elem] = 1
            else:
                persona_dict[curr_elem] += 1

    #print("persona after removing punctuation: " + str(persona_words_list))
    #print()
    tagged = nltk.pos_tag(persona_words_list)
    #print("tagged list: " + str(tagged))
    #print()

    noun_tags = ['NN', 'NNS', 'NNP', 'NNPS']

    filtered_words_list = list(filter(lambda x: x[1] in noun_tags, tagged))
    #take out the second tuple now- with the part of speech
    filtered_words_list = [x[0] for x in filtered_words_list]

    #record the frequency of the response dict with only nouns in the persona
    print("filtered persona list: " + str(filtered_words_list))
    print()
    return filtered_words_list






def create_response_freq_dict(response_convo, filtered_words_list, stopwords_list):

    #tokenize response
    response_dict = {}
    for i in range(0, len(response_convo)):
        tokenized_convo = nltk.word_tokenize(response_convo[i])

        #check for stopwords and punctuation. then, add to dict
        for j in range(0, len(tokenized_convo)):
            word = tokenized_convo[j]
            if word in filtered_words_list:
                if word not in response_dict:
                    response_dict[word] = 1
                else:
                    response_dict[word] += 1


    #for removing most frequent word that's in persona and response
    print("response dict: " + str(response_dict))
    print()
    max_key = max(response_dict, key = response_dict.get)

    for elem in range(0, len(response_convo)):
        curr_convo = response_convo[elem]
        curr_convo = curr_convo.replace(max_key, '')
        response_convo[elem] = curr_convo




    #TODO
    #for removing all matching words between persona and response
    #for key, val in response_dict.items():
    #    if key in response_convo:
    #        response_convo.remove(key)


    return response_convo







test_dataframe = pd.read_csv("/Users/arvindpunj/Desktop/Projects/NLP lab research/Extracting-personas-for-text-generation/data/test_other_original.txt", delimiter='\n', header= None, error_bad_lines=False)

main(test_dataframe)
