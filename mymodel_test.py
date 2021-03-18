"""file for test set"""
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
from mymodel import DistilbertTrainingParams
from mymodel import DistilBertandBilinear
from mymodel import create_persona_and_snippet_lists
from mymodel import create_encoding_dict
from mymodel import create_training_file
from mymodel import create_validation_file




def main(train_df, valid_df):

    #start_time = time.perf_counter()
    #end_time = time.perf_counter()
    #end - start to calc time

    training_params = DistilbertTrainingParams()
    training_params.create_tokens_dict()

    print("model parameters initialized, and tokens dict created")
    #convo classifier is already created by running putting distilbert training params
    saved_model = training_params.convo_classifier
    saved_optimizer = training_params.optimizer
    epoch = 0
    prev_loss = 0

    checkpoint = torch.load('savedmodels/finalmodel.pt')
    saved_model.load_state_dict(checkpoint['model_state_dict'])
    saved_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    prev_loss = checkpoint['prev_loss']

    print("model: " + str(saved_model))
    print("optimizer: " + str(saved_optimizer))
    print("last epoch was:" + str(epoch))
    print("the prev loss saved was: " + str(prev_loss))
    print()

    epoch = epoch + 1

    training_personas, training_snippets = create_persona_and_snippet_lists(train_df)
    validation_personas, validation_snippets = create_persona_and_snippet_lists(valid_df)

    encoded_training_dict, smallest_convo_size = create_encoding_dict(training_params, training_snippets)
    encoded_validation_dict, smallest_convo_size = create_encoding_dict(training_params, validation_snippets)

    create_training_file(training_personas, training_snippets)
    create_validation_file(validation_personas, validation_snippets)
    print("continuing training")

    training_params.prev_loss = prev_loss
    training_params.train_model(training_personas, validation_personas, encoded_training_dict, encoded_validation_dict, epoch)


    #next- pass on the max loss to the next model
    #check if this done correctly- run fully on sunday



train_dataframe = pd.read_csv("data/train_other_original.txt",delimiter='\n', header= None, error_bad_lines=False)
validation_dataframe = pd.read_csv("data/valid_other_original.txt", delimiter='\n', header= None, error_bad_lines=False)
#test_dataframe = pd.read_csv("data/test_other_original.txt", delimiter='\n', header= None, error_bad_lines=False)



main(train_dataframe, validation_dataframe)
