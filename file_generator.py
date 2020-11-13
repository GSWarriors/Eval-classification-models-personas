import pandas as pd
import numpy as np
import transformers as ppb  #pytorch transformers
#from transformers import BertTokenizer, BertForNextSentencePrediction
import torch
import random as rand
import time
import math
import itertools


"""This script should be able to parse the first 10 conversations from any "non-original.txt" type
of file from personachat dataset"""


def main(df):

    k = 2
    persona_convo = []
    snippet_convo = []
    full_doc = df[0]

    filtered_convo = []
    persona_list = []
    snippet_list = []



    #going through full text file, but only saving to personas at the moment.
    #conversations can be variable length, usually 6-7 lines.
    train_count = 0
    f = open("test.txt", "w")

    for line in range(0, len(full_doc)):

        #print("the current line is: " + str(full_doc[line]))
        #print()

        first_char = full_doc[line][0]
        second_char = full_doc[line][1]

        if first_char == '1' and not (ord(second_char) >= 48 and ord(second_char) <= 57):
            train_count += 1

            if train_count == 12:
                break

        f.write(full_doc[line] + "\n")


    f.close()







dataframe = pd.read_csv("/Users/arvindpunj/Desktop/Projects/NLP lab research/Extracting-personas-for-text-generation/test_none_original.txt",
delimiter='\n', header= None, error_bad_lines=False)

main(dataframe)
