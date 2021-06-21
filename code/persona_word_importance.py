import torch
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from nltk.stem.snowball import SnowballStemmer
import transformers as ppb  #pytorch transformers
import random as rand
import time
import math
import itertools
import json
from matplotlib import pyplot
import nltk
import string
import copy


from mymodel import DistilBertTrainingParams
from mymodel import DistilBertandBilinear
from mymodel import create_persona_and_snippet_lists
from mymodel import create_encoding_dict
from mymodel import create_training_file
from mymodel import create_validation_file
from mymodel import add_padding_and_mask

from mymodel_test import create_testing_file
from mymodel_test import test_model


#need 3 models total. bilinear + sigmoid, logistic, tf-idf

def main(test_df):

    #load original distilbert + bilinear saved model and open/modify test set

    training_params = DistilBertTrainingParams()
    training_params.create_tokens_dict()

    saved_model = training_params.convo_classifier
    saved_optimizer = training_params.optimizer

    #mymodel implementation
    saved_model.load_state_dict(torch.load("/Users/arvindpunj/Desktop/Projects/NLP lab research/savedmodels/finaldistilbertmodel.pt", map_location=torch.device('cpu')))

    #print("done loading original distilbert model!")
    #first testing file create (with all response words)
    test_personas, test_responses = create_persona_and_snippet_lists(test_df)
    create_testing_file(test_personas, test_responses)


    #here's the part where I modify the responses in the test set- remove words that have most freq stem
    test_responses = modify_responses(test_personas, saved_model, training_params)
    encoded_test_dict, smallest_convo_size = create_encoding_dict(training_params, test_responses)

    #recreate testing file
    create_testing_file(test_personas, test_responses)
    test_model(test_personas, encoded_test_dict, saved_model, training_params)







def modify_responses(test_personas, saved_model, training_params):
    test_file = open("positive-test-samples.json", "r")
    test_data = json.load(test_file)
    stopwords_list = []

    #opens file with stopwords and puts it into a list
    with open("/Users/arvindpunj/Desktop/Projects/NLP lab research/Extracting-personas-for-text-generation/data/stopwords/english.txt", "r") as stopwords_file:
        for line in stopwords_file:
            line = line.strip('\n')
            stopwords_list.append(line)

    stopwords_list.append('speaker-1')


    new_test_responses = []
    for i in range(0, len(test_personas)):

        persona_convo = ' '.join(test_data[i]['text persona'])
        response_convo = test_data[i]['text snippet']

        filtered_words_list = tag_persona_and_create_dict(persona_convo, stopwords_list)
        create_response_freq_dict(response_convo, filtered_words_list, stopwords_list)
        new_test_responses.append(response_convo)


    print("length of new test responses: " + str(len(new_test_responses)))
    print()
    print("new test responses: " + str(new_test_responses))

    return new_test_responses





def tag_persona_and_create_dict(persona_convo, stopwords_list):

    #tokenize persona into separate words
    persona_convo = nltk.word_tokenize(persona_convo)

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
    #print("filtered persona list: " + str(filtered_words_list))
    #print()
    return filtered_words_list



def remove_response_words(response_dict, response_convo):

    #dictionary that stores the most frequent nouns/noun phrases that come from the same stem
    response_stem_dict = {}
    remove_list = []
    stemmer = SnowballStemmer("english")

    for key in response_dict.keys():
        stem_key = stemmer.stem(key)
        if stem_key not in response_stem_dict:
            response_stem_dict[stem_key] = 1
        else:
            response_stem_dict[stem_key] += 1


    #max key only exists if response stem dict >= 1
    if len(response_stem_dict) >= 1:
        max_key = max(response_stem_dict, key = response_stem_dict.get)

        for key in response_dict.keys():
            stem_key = stemmer.stem(key)

            if stem_key == max_key:
                remove_list.append(key)

    #print("remove list: " + str(remove_list))

    #remove all words from response convo that are in remove list
    for i in range(0, len(response_convo)):
        curr_convo = response_convo[i]

        for j in range(0, len(remove_list)):
            curr_convo = curr_convo.replace(remove_list[j], '')

        response_convo[i] = curr_convo

    #print()
    #print("response after removing words: " + str(response_convo))
    return response_stem_dict



def remove_all_response_words(response_dict, response_convo):

    #dictionary that stores the most frequent nouns/noun phrases that come from the same stem
    response_stem_dict = {}
    remove_list = []
    stemmer = SnowballStemmer("english")

    for key in response_dict.keys():
        stem_key = stemmer.stem(key)
        if stem_key not in response_stem_dict:
            response_stem_dict[stem_key] = 1
        else:
            response_stem_dict[stem_key] += 1


    #remove all words from response that match stems in response dict
    for i in range(0, len(response_convo)):
        curr_convo = response_convo[i]

        curr_convo_list = curr_convo.split()
        curr_convo_list_copy = curr_convo_list.copy()

        for j in range(0, len(curr_convo_list_copy)):
            curr_word = curr_convo_list_copy[j]
            curr_word_stem = stemmer.stem(curr_word)

            if curr_word_stem in response_stem_dict.keys():
                curr_convo_list.remove(curr_word)

        response_convo[i] = ' '.join(curr_convo_list)

    return response_stem_dict







def create_response_freq_dict(response_convo, filtered_words_list, stopwords_list):
    #tokenize response
    response_dict = {}
    for i in range(0, len(response_convo)):
        tokenized_convo = nltk.word_tokenize(response_convo[i])

        #create new tokenized convo without stopwords and punctuation
        new_tokenized_convo = []
        for token in tokenized_convo:
            if token not in string.punctuation and token not in stopwords_list:
                new_tokenized_convo.append(token)

        #first checks if the word we have is directly in the filtered words list
        #if it is, we add it, otherwise we check whether its a substring of one of those
        #words

        for j in range(0, len(new_tokenized_convo)):
            word = new_tokenized_convo[j]


            if word in filtered_words_list:
                if word not in response_dict:
                    response_dict[word] = 1
                else:
                    response_dict[word] += 1

            else:
                for filtered_word in filtered_words_list:

                    stemmer = SnowballStemmer("english")
                    filtered_word_stem = stemmer.stem(filtered_word)
                    word_stem = stemmer.stem(word)

                    if word_stem == filtered_word_stem:
                        if word not in response_dict:
                            #print("special case. the word added is: " + str(word))
                            response_dict[word] = 1
                        else:
                            response_dict[word] += 1

    #for removing most frequent word that's in persona and response
    #response_stem_dict = remove_response_words(response_dict, response_convo)

    #for removing all persona nouns from response
    other_response_stem_dict = remove_all_response_words(response_dict, response_convo)
    #print("response stem dict for all nouns: " + str(other_response_stem_dict))
    #print()






test_dataframe = pd.read_csv("/Users/arvindpunj/Desktop/Projects/NLP lab research/Extracting-personas-for-text-generation/data/test_other_original.txt", delimiter='\n', header= None, error_bad_lines=False)

main(test_dataframe)
