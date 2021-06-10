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

    #print("stopwords list: " + str(stopwords_list))




    for i in range(0, len(test_personas)):
        persona_convo = ' '.join(test_data[i]['text persona'])
        response_convo = test_data[i]['text snippet']
        #persona_encoding = [training_params.tokenizer.encode(persona_convo, add_special_tokens=True)]
        #gold_snippet_encoding = encoded_test_dict[i]
        print("persona before split: " + str(persona_convo))
        print()


        persona_convo = persona_convo.split()
        #persona_convo = nltk.word_tokenize(persona_convo)
        print("persona after split: " + str(persona_convo))


        freq_dict = {}

        for j in range(0, len(persona_convo)):
            curr_word = persona_convo[j]
            if curr_word not in stopwords_list:
                if curr_word not in freq_dict:
                    freq_dict[curr_word] = 1
                else:
                    freq_dict[curr_word] += 1

        print("frequency dict: " + str(freq_dict))
        #print("response: " + str(response_convo))


        if i == 0:
            break








test_dataframe = pd.read_csv("/Users/arvindpunj/Desktop/Projects/NLP lab research/Extracting-personas-for-text-generation/data/test_other_original.txt", delimiter='\n', header= None, error_bad_lines=False)

main(test_dataframe)
