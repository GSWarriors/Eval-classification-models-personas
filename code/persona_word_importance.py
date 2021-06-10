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

    print("model parameters initialized, and tokens dict created")
    saved_model = training_params.convo_classifier
    saved_optimizer = training_params.optimizer

    #mymodel implementation
    saved_model.load_state_dict(torch.load("/Users/arvindpunj/Desktop/Projects/NLP lab research/savedmodels/finaldistilbertmodel.pt", map_location=torch.device('cpu')))

    print("done loading original distilbert model!")


    test_personas, test_responses = create_persona_and_snippet_lists(test_df)
    encoded_test_dict, smallest_convo_size = create_encoding_dict(training_params, test_responses)
    create_testing_file(test_personas, test_responses)
    print("created test file")

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



    for i in range(0, len(test_personas)):
        persona_convo = ' '.join(test_data[i]['text persona'])
        response_convo = test_data[i]['text snippet']
        #persona_encoding = [training_params.tokenizer.encode(persona_convo, add_special_tokens=True)]
        #gold_snippet_encoding = encoded_test_dict[i]
        #print("persona before split: " + str(persona_convo))
        #print()


        filtered_words_list = tag_persona_and_create_dict(persona_convo, stopwords_list)
        create_response_freq_dict(response_convo, filtered_words_list, stopwords_list)

        #combine persona and response dicts

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

    #print("frequency dict: " + str(persona_dict))
    print("persona after removing punctuation: " + str(persona_words_list))
    print()
    tagged = nltk.pos_tag(persona_words_list)

    print("tagged list: " + str(tagged))
    print()

    noun_tags = ['NN', 'NNS', 'NNP', 'NNPS']

    filtered_words_list = list(filter(lambda x: x[1] in noun_tags, tagged))

    #record the frequency of the response dict with only nouns in the persona
    print("filtered persona list: " + str(filtered_words_list))
    print()
    return filtered_words_list




def create_response_freq_dict(response_convo, filtered_words_list, stopwords_list):

    #tokenize response

    response_dict = {}
    for i in range(0, len(response_convo)):
        response_convo[i] = nltk.word_tokenize(response_convo[i])
        curr_response = response_convo[i]

        #check for stopwords and punctuation. then, add to dict
        for j in range(0, len(curr_response)):
            if (curr_response[j] not in stopwords_list) and curr_response[j] not in string.punctuation:
                if curr_response[j] not in response_dict:
                    response_dict[curr_response[j]] = 1
                else:
                    response_dict[curr_response[j]] += 1

    print("the response dict: " + str(response_dict))




#code to combine persona and response- should be in prev function (modify responses)

"""print("persona frequency dict: " + str(persona_dict))
    print()
    print("response frequency dict: " + str(response_dict))
    print()

    #add values from both persona and response into a final dict
    for key, val in response_dict.items():
        if key not in final_dict:
            final_dict[key] = persona_dict[key] + val

    print("the final dict: " + str(final_dict))"""





test_dataframe = pd.read_csv("/Users/arvindpunj/Desktop/Projects/NLP lab research/Extracting-personas-for-text-generation/data/test_other_original.txt", delimiter='\n', header= None, error_bad_lines=False)

main(test_dataframe)
