
#a lot of the below can be replaced later by "from imports"
import torch
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
import transformers as ppb  #pytorch transformers
import random as rand
import math
import json
import copy
    # Building a TF IDF matrix out of the corpus of reviews
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



def main(train_df):

    training_personas, training_snippets = create_persona_and_snippet_lists(train_df)
    documents = []
    snippet_set_size = 4
    training_size = len(training_personas)
    all_batch_sum = 0
    acc_avg = 0


    #loop through all training personas and calculate accuracy in batches
    #handle edge cases for starting and ending responses

    for i in range(0, len(training_personas)):
        curr_doc = ' '.join(training_personas[i])
        documents = [curr_doc]
        distractor_set = []
        full_set = []

        print("on persona: " + str(i))

        if i + (snippet_set_size/2) >= training_size and i - snippet_set_size >= 0:
            #take the preceding 4 snippets as distractors
            for elem in range(i - snippet_set_size, i):
                distractor_set.append(training_snippets[elem][1])

        elif i - (snippet_set_size/2) < 0 and i + snippet_set_size < training_size:
            #take the proceeding 4 snippets as distractors
            for elem in range(i + 1, i + snippet_set_size + 1):
                distractor_set.append(training_snippets[elem][1])

        else:
            distractor_set = [training_snippets[i - 2][1], training_snippets[i - 1][1],
            training_snippets[i + 1][1], training_snippets[i + 2][1]]

        #right now documents contains the first document, followed by 2 negative and 2 positive responses samples
        responses = [training_snippets[i][1], training_snippets[i][2], training_snippets[i][3], training_snippets[i][4]]
        full_set = distractor_set + responses

        documents = documents + full_set
        #print("all documents (persona and responses): " + str(documents))
        #print()



        #learns vocabulary and idf, returns document term matrix
        vectorizer = TfidfVectorizer(min_df= 1, stop_words="english")
        doc_term_matrix = vectorizer.fit_transform(documents).toarray()
        features_list = vectorizer.get_feature_names()

        #matrix multiply the transpose and the doc term matrix together by using
        #dot product
        similarity_arr = doc_term_matrix.dot(np.transpose(doc_term_matrix))

        #mask the ones that we get. get highest similarity using nanargmax
        #np.nanargmax gets us the highest values in the axis we have, ignoring nans
        np.fill_diagonal(similarity_arr, np.nan)
        input_doc = documents[0]
        input_idx = documents.index(input_doc)

        most_similar_indices = []
        input_doc_row = similarity_arr[input_idx]
        copy_arr = input_doc_row.copy()

        for j in range(0, len(input_doc_row) - 1):
            largest_index = np.nanargmax(input_doc_row)
            input_doc_row[largest_index] = 'nan'
            most_similar_indices.append(largest_index)

        print("all indices: " + str(most_similar_indices))


        #put below in separate function
        correct_preds = 0
        is_zero = False

        for k in range(0, len(most_similar_indices)):


            if k < len(most_similar_indices)/2:
                if copy_arr[k] == 0:
                    is_zero = True

                if copy_arr[k] != 0 and (most_similar_indices[k] >= 5 and most_similar_indices[k] <= 8):
                    correct_preds += 1
            else:

                if most_similar_indices[k] <= 4 and most_similar_indices[k] >= 1:
                    correct_preds += 1


        if is_zero:
            print("some positive values ended up being 0")

        all_batch_sum += correct_preds
        accuracy = (correct_preds/(len(most_similar_indices)))*100
        print("current accuracy: " + str(accuracy))
        print()



    acc_avg = all_batch_sum/((snippet_set_size*2)*(training_size + 1))*100
    print("the average accuracy of all: " + str(acc_avg))







    #result_idx = np.nanargmax(similarity_arr[input_idx])
    #print("index of result doc: " + str(result_idx))
    #print("the most similar document: " + str(documents[result_idx]))"""




def add_speaker_tokens(convo):

    speaker_num = 2
    new_response_str = ""
    new_convo = []

    for line in range(0, len(convo)):

        last_speaker_index = 0
        new_response_str = ""
        for char in range(0, len(convo[line])):
            if convo[line][char] == '\t':
                if speaker_num == 1:
                    speaker_num = 2
                    #speaker_str = ' <speaker-2> '
                    #add speaker 2 tag from after tab where speaker 1 left off
                    #curr_response = convo[line][last_speaker_index + 1: len(convo[line]) - 1]
                    #new_response_str += speaker_str + curr_response

                else:
                    speaker_num = 1
                    speaker_str = '<speaker-1> '

                    #first speaker is 0 up to current char (the tab)
                    curr_response = convo[line][0: char]
                    new_response_str += speaker_str + curr_response
                    #print("new response str is : " + new_response_str)
                    last_speaker_index = char

        new_convo.append(new_response_str)

    return new_convo


#filters 1 back and forth between two speakers and returns the string
def filter_for_responses(response):
    #print("number is: " + str(response[0]))
    first_char = response[0]
    second_char = response[1]
    tab_count = 0
    two_speaker_utterances = ""

    #we need to remove the line number depending on whether its 1 or 2 digits
    if not (ord(second_char) >= 48 and ord(second_char) <= 57):
        response = response[2:]
    else:
        response = response[3:]

    for char in response:
        if tab_count < 2:
            if char == '\t':
                tab_count += 1

            two_speaker_utterances += char

    return two_speaker_utterances





def create_persona_and_snippet_lists(df):

    k = 2
    full_doc = df[0]
    persona_list = []
    saved_persona = []

    saved_snippets = []
    snippet_list = []
    checking_persona = True

    #go through and only print lines with "your persona"
    for line in range(0, len(full_doc)):

        #we add saved_snippets list to snippet_list if our list of saved_snippets is not empty
        if "partner's persona: " in full_doc[line]:
            checking_persona = True
            saved_persona.extend([full_doc[line][2:]])

        else:
            #add the created persona to persona list
            #reset saved persona and the saved snippets
            if checking_persona:
                persona_list.append(saved_persona)

                saved_snippets = add_speaker_tokens(saved_snippets)

                if saved_snippets:
                    snippet_list.append(saved_snippets)

                saved_persona = []
                saved_snippets = []
                checking_persona = False


            #if we're moving into the non-persona lines, then we add to saved_snippets list
            if not checking_persona:
                filtered_line = filter_for_responses(full_doc[line])
                saved_snippets.append(filtered_line)


    for i in range(0, len(persona_list)):
        for j in range(0, len(persona_list[i])):
            persona_list[i][j] = persona_list[i][j].replace("partner's persona: ", "")


    #should add this portion somewhere in the loop later

    saved_snippets = add_speaker_tokens(saved_snippets)
    snippet_list.append(saved_snippets)
    return persona_list, snippet_list












#can edit this to valid.txt and test.txt in order to run on different files

train_dataframe = pd.read_csv("/Users/arvindpunj/Desktop/Projects/NLP lab research/data/train_other_original.txt",delimiter='\n', header= None, error_bad_lines=False)



main(train_dataframe)
