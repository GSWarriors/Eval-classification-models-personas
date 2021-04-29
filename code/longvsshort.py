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

from mymodel import create_persona_and_snippet_lists
from mymodel import create_encoding_dict


def main(train_df, valid_df):

    #largest 4 for both training and validation

    training_personas, training_snippets = create_persona_and_snippet_lists(train_df)
    validation_personas, validation_snippets = create_persona_and_snippet_lists(valid_df)

    print("validation persona 1: " + str(validation_personas[0]))
    print()
    print("train response: " + str(validation_snippets[0]))

    train_responses_first = training_snippets[0]
    validation_responses_first = validation_snippets[0]

    """for i in range(0, len(responses_first)):
        print("length of the response: " + str(len(train_responses_first[i])))
        print(str(train_responses_first[i]))
        print()"""


    for j in range(0, len(validation_responses_first)):
        print("length of the response: " + str(len(validation_responses_first[j])))
        print(str(validation_responses_first[j]))
        print()





train_dataframe = pd.read_csv("/Users/arvindpunj/Desktop/Projects/NLP lab research/Extracting-personas-for-text-generation/data/train_other_original.txt",
delimiter='\n', header= None, error_bad_lines=False)
validation_dataframe = pd.read_csv("/Users/arvindpunj/Desktop/Projects/NLP lab research/Extracting-personas-for-text-generation/data/valid_other_original.txt",
delimiter='\n', header= None, error_bad_lines=False)



main(train_dataframe, validation_dataframe)
