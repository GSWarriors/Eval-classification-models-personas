import pandas as pd
import numpy as np
import pandas as pd
import transformers as ppb  #pytorch transformers
from transformers import BertTokenizer, BertForNextSentencePrediction
import torch
import random as rand



def main():

    my_list = []
    dataframe = pd.read_csv("/Users/arvindpunj/Desktop/Projects/NLP lab research/Extracting-personas-for-text-generation/train_none_original.txt",
    delimiter='\n', header= None, error_bad_lines=False)
    full_doc = dataframe[0]

    #filters our conversation to just the part we want, put into a list
    filtered_convo = filter_conversation(full_doc)
    print("original convo: " + str(filtered_convo))
    print()

    k = 2
    number_list = []
    count = 0
    while count < k:
        rand_convo = rand.randint(0, len(filtered_convo) - 1)
        if rand_convo not in number_list:
            number_list.append(rand_convo)
            count += 1


    #print(number_list)
    for i in range(0, len(number_list)):
        filtered_convo.pop(number_list[i])
        print("removing conversation " + str(number_list[i]))

    print(filtered_convo)
    print(len(filtered_convo))



    """tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
    encoding = tokenizer(first_chat, marked_second_chat, return_tensors='pt')

    #print(encoding)
    loss, logits = model(**encoding, next_sentence_label=torch.LongTensor([1]))"""







def filter_conversation(full_doc):

    count = 0
    convo_list = []
    filtered_convo = []

    for line in range(0, 7):
        convo_list.append(full_doc[line])

    for i in range(0, len(convo_list)):
        filtered_convo.append(filter_responses(convo_list[i]))

    #print(filtered_convo)
    return filtered_convo




def filter_responses(response):


    response_without_number = response[2:]
    #print(response_without_number)
    #print()
    tab_count = 0
    response_without_tabs = ""

    for char in response_without_number:
        if tab_count < 2:
            if char == '\t':
                tab_count += 1

            else:
                response_without_tabs += char

    return response_without_tabs



main()
