"""
Here we compare how the model performs when we analyze only the longest
four responses vs the shortest four. This is to see which models
we've trained perform better with more context based sentence, and which perform worse.

Train all models this way, bilinear + sigmoid, logistic and TF-IDF + cosin sim
"""
import torch
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
import transformers as ppb  #pytorch transformers
import random as rand
import time
import math
import itertools
import json
import copy

from mymodel import create_persona_and_snippet_lists
from mymodel import create_encoding_dict


def main(train_df, valid_df):

    #largest 4 for both training and validation

    training_personas, training_snippets = create_persona_and_snippet_lists(train_df)
    validation_personas, validation_snippets = create_persona_and_snippet_lists(valid_df)

    parse_long_responses(training_snippets, validation_snippets)


def parse_long_responses(train_responses, validation_responses):


    train_responses_first = train_responses[0]
    validation_responses_first = validation_responses[0]
    largest_list = []

    train_responses_first.sort(key=len)

    print("sorted list: " + str(train_responses_first))
    print()

    largest_list = train_responses_first[len(train_responses_first) - 4:]

    for i in range(0, len(largest_list)):
        print("response: " + str(largest_list[i]))
        print("length of the response: " + str(len(largest_list[i])))
        print()



    """
    for i in range(0, len(train_responses_first)):
        print("length of the response: " + str(len(train_responses_first[i])))
        print(str(train_responses_first[i]))
        print()

    for j in range(0, len(validation_responses_first)):
        print("length of the response: " + str(len(validation_responses_first[j])))
        print(str(validation_responses_first[j]))
        print()"""




train_dataframe = pd.read_csv("/Users/arvindpunj/Desktop/Projects/NLP lab research/Extracting-personas-for-text-generation/data/train_other_original.txt",
delimiter='\n', header= None, error_bad_lines=False)
validation_dataframe = pd.read_csv("/Users/arvindpunj/Desktop/Projects/NLP lab research/Extracting-personas-for-text-generation/data/valid_other_original.txt",
delimiter='\n', header= None, error_bad_lines=False)



main(train_dataframe, validation_dataframe)

