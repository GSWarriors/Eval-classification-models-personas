
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
from persona_word_importance import modify_responses

    # Building a TF IDF matrix out of the corpus of reviews
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from matplotlib import pyplot



"""TF-IDF with cosine similarity implemented on Personachat dataset as a non-neural baseline.
Negative samples are the first 4 samples in the sampled list, and positive samples are the last 4.

In order to determine which documents (responses) have the most direct match with the personas,
we take the 4 with the highest percentage in the similarity array for the persona.
Then, we store their indices in the most_similar_indices list.

We take the similarity arr value for the most similar indices and store them as the last four values in
the model output list. Similarly, we store the four least similar values as the first four values in the model
output list. This is in order to correspond with the actual output list, which starts with negative samples
and ends with positive samples.

However, we don't count any of the values that happen to be zero for the best matches, since
then there is no similarity to begin with.

"""



def main(train_df):


    train_personas, train_responses = create_persona_and_snippet_lists(train_df)
    train_responses = modify_responses(train_personas)

    documents = []
    response_set_size = 4
    training_size = len(train_personas)
    actual_output = ([0]*response_set_size + [1]*response_set_size)*training_size
    output = []
    predicted_output = []
    all_batch_sum = 0
    acc_avg = 0


    #loop through all training personas and calculate accuracy in batches
    #handle edge cases for starting and ending responses
    print()

    for i in range(0, len(train_personas)):
        curr_doc = ' '.join(train_personas[i])
        documents = [curr_doc]
        distractor_set = []
        full_set = []

        print("for tf-idf, on persona: " + str(i))

        if i + (response_set_size/2) >= training_size and i - response_set_size >= 0:
            #take the preceding 4 snippets as distractors
            for elem in range(i - response_set_size, i):
                distractor_set.append(train_responses[elem][1])

        elif i - (response_set_size/2) < 0 and i + response_set_size < training_size:
            #take the proceeding 4 snippets as distractors
            for elem in range(i + 1, i + response_set_size + 1):
                distractor_set.append(train_responses[elem][1])

        else:
            distractor_set = [train_responses[i - 2][1], train_responses[i - 1][1],
            train_responses[i + 1][1], train_responses[i + 2][1]]

        #right now documents contains the first document, followed by 2 negative and 2 positive responses samples
        responses = [train_responses[i][1], train_responses[i][2], train_responses[i][3], train_responses[i][4]]
        full_set = distractor_set + responses

        documents = documents + full_set

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
        #print("input doc row: " + str(input_doc_row))

        copy_arr = input_doc_row.copy()

        for j in range(0, len(input_doc_row) - 1):
            largest_index = np.nanargmax(input_doc_row)
            input_doc_row[largest_index] = 'nan'
            most_similar_indices.append(largest_index)

        #print("all indices: " + str(most_similar_indices))


        model_output = [0]*len(most_similar_indices)
        mid = int(len(model_output)/2)

        for k in range(0, len(most_similar_indices)):

            if k < len(most_similar_indices)/2:
                model_output[mid + k] = copy_arr[most_similar_indices[k]]

            #the second statement is incorrect, above is fine
            else:
                model_output[k - mid] = copy_arr[most_similar_indices[k]]

        output += model_output

        rounded_output, correct_preds = round_and_calc_acc(model_output)
        predicted_output += rounded_output
        all_batch_sum += correct_preds

        accuracy = (correct_preds/(len(model_output)))*100



    acc_avg = all_batch_sum/((response_set_size*2)*(training_size + 1))*100
    print("the average accuracy of all: " + str(acc_avg))
    print()
    calculate_prc_and_f1(actual_output, predicted_output, output)




def round_and_calc_acc(model_output):

    #just need to calc accuracy to complete
    rounded_output = [0]*len(model_output)
    correct_preds = 0

    for i in range(0, len(model_output)):
        if i >= len(model_output)/2:
            if model_output[i] >= 0.5:
                rounded_output[i] = 1

    for j in range(0, len(rounded_output)):

        if j < len(rounded_output)/2:
            if rounded_output[j] == 0:
                correct_preds += 1
        else:
            if rounded_output[j] == 1:
                correct_preds += 1


    return rounded_output, correct_preds





def calculate_prc_and_f1(actual_output, predicted_output, output):

    #Note: predicted output is the model output rounded, output is the model output
    #not rounded. actual output is the test set output
    #print("rounded output: " + str(predicted_output))
    #print()
    #print("actual output: " + str(actual_output))

    np_actual_output = np.asarray(actual_output)
    np_output = np.asarray(output)
    np_predicted_output = np.asarray(predicted_output)
    f1 = 0

    precision, recall, thresholds = precision_recall_curve(np_actual_output, np_output)
    f1 = f1_score(np_actual_output, np_predicted_output)
    print("f1 score: " + str(f1))
    print()
    print("recall: " + str(recall[5000:5500]))
    print()
    print("precision: " + str(precision[5000:5500]))
    print()
    print(thresholds[5000:5500])
    print()


    # plot the precision-recall curves
    no_skill = len(np_actual_output[np_actual_output==1]) / len(np_actual_output)
    pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    pyplot.plot(recall, precision, marker='.', label='TF-IDF cosine sim')

    # axis labels
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()





#delete later
def normalize_output(input_doc_row):

    min_val = np.nanmin(input_doc_row)
    max_val = np.nanmax(input_doc_row)

    for i in range(0, len(input_doc_row)):
        if not (min_val == 0 and max_val == 0):
            input_doc_row[i] = (input_doc_row[i] - min_val)/(max_val - min_val)


    return input_doc_row


    #output from model in this case is the percentages we get from the similarity arr (input doc row)
    #get input_doc_row[most_similar_indices[1 to 4]]





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

train_dataframe = pd.read_csv("/Users/arvindpunj/Desktop/Projects/NLP lab research/Extracting-personas-for-text-generation/data/train_other_original.txt",delimiter='\n', header= None, error_bad_lines=False)
#test_dataframe = pd.read_csv("/Users/arvindpunj/Desktop/Projects/NLP lab research/Extracting-personas-for-text-generation/data/test_other_original.txt",delimiter='\n', header= None, error_bad_lines=False)



main(train_dataframe)
