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
    first_convo = dataframe[0]
    convo_string = filter_conversation(first_convo)

    #my_list.append(convo_string)
    first_chat, second_chat = filter_by_sentence(convo_string)
    print("First person: " + first_chat)
    print("Second person: " + second_chat)

    #bert tokenization [CLS], [SEP]. can put this into a function
    marked_first_chat = ""
    marked_second_chat = ""

    for char in range(0, len(first_chat)):
        if first_chat[char] == '.' or first_chat[char] == '?' or first_chat[char] == '!':
            #print("end of sentence")
            marked_first_chat += first_chat[char] + "[SEP]"
        else:
            marked_first_chat += first_chat[char]

    for char in range(0, len(second_chat)):
        if second_chat[char] == '.' or second_chat[char] == '?' or second_chat[char] == '!':
            #print("end of sentence")
            marked_second_chat += second_chat[char] + "[SEP]"
        else:
            marked_second_chat += second_chat[char]

    marked_first_chat = "[CLS]" + marked_first_chat
    marked_second_chat = "[CLS]" + marked_second_chat
    print(marked_first_chat)
    print(marked_second_chat)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
    encoding = tokenizer(first_chat, second_chat, return_tensors='pt')

    #print(encoding)
    loss, logits = model(**encoding, next_sentence_label=torch.LongTensor([1]))



    if logits[0, 0] < logits[0, 1]:
        print("same conversation")
    else:
        print("different conversation")

    print(str(logits[0, 0]))
    print(str(logits[0, 1]))




def filter_conversation(first_convo):
    #print(first_convo)

    convo_string = ""
    for line in range(8, 11):
        print(first_convo[line])
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

    for char in range(0, len(convo_string)):

        if convo_string[char] == '\t' and tab_count < 2:
            if tab_count == 0:
                first_sentence += convo_string[0:char]
                first_end_index = char
            else:
                if tab_count == 1:
                    second_sentence += convo_string[first_end_index + 1:char]

            tab_count += 1

    return first_sentence, second_sentence



main()


"""tokenized_first = tokenizer.tokenize(marked_first_chat)
tokenized_second = tokenizer.tokenize(marked_second_chat)
print(tokenized_first)
print(tokenized_second)
#print(encoding)"""
