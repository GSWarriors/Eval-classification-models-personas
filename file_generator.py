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
    #print("shape is: " + str(df.shape))

    first_convo = df[0][:11]
    saved_persona = []
    new_saved_persona = []
    #go through and only print lines with "your persona"

    for i in range(0, len(first_convo)):
        if "partner's persona: " in first_convo[i]:
            saved_persona.extend([first_convo[i][2:]])

    #further filtering
    for i in range(0, len(saved_persona)):
        new_persona = saved_persona[i].replace("partner's persona: ", "")
        new_saved_persona.extend([new_persona])

    print("persona list: " + str(new_saved_persona))
    new_str_persona = ' '.join(new_saved_persona)
    print("string version: " + str(new_str_persona))





    """k = 2
    persona_convo = []
    snippet_convo = []
    full_doc = df[0]

    filtered_convo = []
    persona_list = []
    snippet_list = []

    print("10 lines: " + str(df[0][10]))

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


    f.close()"""





#use train_self_original.txt for training file

dataframe = pd.read_csv("train_other_original.txt",
delimiter='\n', header= None, error_bad_lines=False)

main(dataframe)
