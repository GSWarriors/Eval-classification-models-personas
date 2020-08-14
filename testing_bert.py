import pandas as pd
import numpy as np
import pandas as pd
import transformers as ppb  #pytorch transformers
from transformers import BertTokenizer, BertForNextSentencePrediction
import torch




def main():

    my_list = []
    dataframe = pd.read_csv("/Users/arvindpunj/Desktop/Projects/NLP lab research/Extracting-personas-for-text-generation/train_self_original.txt",
    delimiter='\n', header= None, error_bad_lines=False)
    #print(dataframe)
    first_convo = dataframe[0]
    convo_string = filter_conversation(first_convo)
    #my_list.append(convo_string)
    first_sentence, second_sentence = filter_by_sentence(convo_string)
    print("First sentence: " + first_sentence)
    print("Second sentence: " + second_sentence)


    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
    #encoding = tokenizer()



def filter_conversation(first_convo):
    #print(first_convo)

    convo_string = ""
    for line in range(5, 11):
        #print(first_convo[line])
        current_convo = first_convo[line]
        convo_without_number = current_convo[2:]
        #print(convo_without_number)
        tab_count = 0
        for char in convo_without_number:
            if tab_count < 2:
                if char == '\t':
                    tab_count += 1

                convo_string += char

    #print(convo_string)
    return convo_string


def filter_by_sentence(convo_string):

    #filter convo for 2 tabs
    #first tab is first convo, second tab is second
    first_sentence = ""
    second_sentence = ""
    first_end_index = 0
    tab_count = 0
    print("convo string: " + convo_string)

    for char in range(0, len(convo_string)):

        if convo_string[char] == '\t' and tab_count < 2:
            print("this is a tab")

            if tab_count == 0:
                first_sentence += convo_string[0:char]
                first_end_index = char
            else:
                if tab_count == 1:
                    second_sentence += convo_string[first_end_index + 1:char]

            tab_count += 1

    return first_sentence, second_sentence



main()
