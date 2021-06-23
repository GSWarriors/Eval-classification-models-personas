"""file for test set"""
import torch
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
import transformers as ppb  #pytorch transformers
import random as rand
import time
import math
import itertools
import json
from matplotlib import pyplot


"""from mymodel import DistilBertTrainingParams
from mymodel import DistilBertandBilinear
from mymodel import create_persona_and_snippet_lists
from mymodel import create_encoding_dict
from mymodel import create_training_file
from mymodel import create_validation_file
from mymodel import add_padding_and_mask"""


from baseline_model import BertTrainingParams
from baseline_model import BertandBilinear
from baseline_model import create_persona_and_snippet_lists
from baseline_model import create_encoding_dict
from baseline_model import create_training_file
from baseline_model import create_validation_file
from baseline_model import add_padding_and_mask




def main(train_df, valid_df, test_df):

    #this code is only for resuming training, not testing
    """training_params = DistilbertTrainingParams()
    training_params.create_tokens_dict()

    #convo classifier is already created by running putting distilbert training params
    saved_model = training_params.convo_classifier
    saved_optimizer = training_params.optimizer
    epoch = 0
    prev_loss = 0

    checkpoint = torch.load('savedmodels/resumemodel.pt')
    saved_model.load_state_dict(checkpoint['model_state_dict'])
    saved_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    prev_loss = checkpoint['prev_loss']

    print("model: " + str(saved_model))
    print("optimizer: " + str(saved_optimizer))
    print("last epoch was:" + str(epoch))
    print("the prev loss saved was: " + str(prev_loss))
    print()

    resume_training(train_df, valid_df, training_params, epoch, prev_loss)"""



    """#this code is for testing
    training_params = BertTrainingParams()
    training_params.create_tokens_dict()

    print("model parameters initialized, and tokens dict created")
    saved_model = training_params.convo_classifier
    saved_optimizer = training_params.optimizer

    #mymodel implementation
    #saved_model.load_state_dict(torch.load("/Users/arvindpunj/Desktop/Projects/NLP lab research/savedmodels/finaldistilbertmodel.pt", map_location=torch.device('cpu')))

    #baseline logistic regression (BERT)
    saved_model.load_state_dict(torch.load("/Users/arvindpunj/Desktop/Projects/NLP lab research/savedmodels/finalbertbaseline.pt", map_location=torch.device('cpu')))

    test_personas, test_snippets = create_persona_and_snippet_lists(test_df)
    encoded_test_dict, smallest_convo_size = create_encoding_dict(training_params, test_snippets)

    create_testing_file(test_personas, test_snippets)
    print("created test file")
    print("smallest convo size: " + str(smallest_convo_size))

    #test below- maybe changed saved model back to training params
    test_model(test_personas, encoded_test_dict, saved_model, training_params)"""





def test_model(test_personas, encoded_test_dict, saved_model, training_params):

    snippet_set_size = 4
    test_size = len(test_personas)
    test_loss = 0
    acc_avg = 0
    all_batch_sum = 0

    actual_output = ([0]*snippet_set_size + [1]*snippet_set_size)*test_size
    output = []
    predicted_output = []

    test_file = open("positive-test-samples.json", "r")
    test_data = json.load(test_file)
    test_loop_losses = []

    with torch.no_grad():
        saved_model.model.eval()

        for i in range(0, len(test_personas)):
            persona_convo = ' '.join(test_data[i]['text persona'])
            snippet_convo = test_data[i]['text snippet']
            persona_encoding = [training_params.tokenizer.encode(persona_convo, add_special_tokens=True)]
            gold_snippet_encoding = encoded_test_dict[i]

            encoded_snippet_set = []
            print("starting test iteration: " + str(i))

            if i + (snippet_set_size/2) >= test_size and i - snippet_set_size >= 0:
                #take the preceding 4 snippets as distractors
                for elem in range(i - snippet_set_size, i):
                    #print("on distractor snippet: " + str(elem))
                    encoded_snippet_set.append(encoded_test_dict[elem][1])

            elif i - (snippet_set_size/2) < 0 and i + snippet_set_size < test_size:
                #take the proceeding 4 snippets as distractors
                for elem in range(i + 1, i + snippet_set_size + 1):
                    #print("on distractor snippet: " + str(elem))
                    encoded_snippet_set.append(encoded_test_dict[elem][1])

            else:
                encoded_snippet_set = [encoded_test_dict[i - 2][1], encoded_test_dict[i - 1][1],
                encoded_test_dict[i + 1][1], encoded_test_dict[i + 2][1]]

            pos_snippet_encodings = [gold_snippet_encoding[1], gold_snippet_encoding[2],
            gold_snippet_encoding[3], gold_snippet_encoding[4]]
            full_encoded_snippet_set = encoded_snippet_set + pos_snippet_encodings

            labels_list = [0]*snippet_set_size
            gold_labels = [1, 1, 1, 1]
            labels_list = labels_list + gold_labels

            #mymodel labels
            #labels = torch.tensor(labels_list, requires_grad=False, dtype=torch.float, device=training_params.device)

            #baseline model labels
            labels = torch.tensor(labels_list, requires_grad=False, dtype=torch.long, device=training_params.device)

            padded_snippet, snippet_attention_mask = add_padding_and_mask(full_encoded_snippet_set)
            snippet_input_ids = torch.from_numpy(padded_snippet).type(torch.long).to(training_params.device)
            #send input to distilbert
            with torch.no_grad():
                snippet_hidden_states = training_params.model(snippet_input_ids)

            snippet_set_features = snippet_hidden_states[0][:, 0, :].to(training_params.device)
            torch_snippet_features = snippet_set_features.clone().detach().requires_grad_(False)

            #my model forward and loss and accuracy
            #model_output = saved_model.forward(persona_encoding, len(full_encoded_snippet_set), torch_snippet_features)
            #curr_loss, correct_preds, predictions = training_params.calc_loss_and_accuracy(model_output, labels)
            #output += model_output
            #predicted_output += predictions

            #baseline loss and accuracy
            softmax_output, model_output = saved_model.forward(persona_encoding, len(full_encoded_snippet_set), torch_snippet_features)
            curr_loss, correct_preds, predictions = training_params.calc_loss_and_accuracy(model_output, softmax_output, labels)
            #print("output: (without threshold) " + str(softmax_output))
            #print("predictions (rounded output): " + str(predictions))
            #print()
            output += softmax_output
            predicted_output += predictions


            test_loss += curr_loss
            all_batch_sum += correct_preds
            snippet_set_size = 4
            test_loop_losses.append(test_loss.item())


        acc_avg = ((all_batch_sum)/((snippet_set_size*2)*(test_size + 1)))*100
        print("the avg test accuracy for epoch: " + str(acc_avg))
        print()
        test_loop_losses = sum(test_loop_losses)
        print("total test loss: " + str(test_loop_losses))


    #my model
    #for j in range(0, len(output)):
    #    output[j] = output[j].item()

    #baseline
    #only take second elements of each tensor to keep probabilities of only positive outcome
    for j in range(0, len(output)):
        output[j] = output[j][1].item()

    for k in range(0, len(predicted_output)):
        predicted_output[k] = predicted_output[k][1].item()

    calculate_prc_and_f1(actual_output, predicted_output, output)




def calculate_prc_and_f1(actual_output, predicted_output, output):

    #Note: predicted output is the model output rounded, output is the model output
    #not rounded. actual output is the test set output
    print("output: " + str(output))
    print()
    print("predicted output: " + str(predicted_output))
    print()

    np_actual_output = np.asarray(actual_output)
    np_output = np.asarray(output)
    np_predicted_output = np.asarray(predicted_output)
    f1 = 0


    precision, recall, thresholds = precision_recall_curve(np_actual_output, np_output)
    f1 = f1_score(np_actual_output, np_predicted_output)

    count = 0

    for i in range(0, len(thresholds)):

        if i < 100:
            print("thresholds: " + str(thresholds[i]))
            print("precision: " + str(precision[i]))
            print("recall: " + str(recall[i]))
            print()

        elif abs(0.5 - thresholds[i]) <= 0.02:
            print("thresholds: " + str(thresholds[i]))
            print("precision: " + str(precision[i]))
            print("recall: " + str(recall[i]))
            print()

        else:
            if len(thresholds) - 100 <= i:
                print("thresholds: " + str(thresholds[i]))
                print("precision: " + str(precision[i]))
                print("recall: " + str(recall[i]))
                print()


    print("f1 score: " + str(f1))

    # plot the precision-recall curves
    no_skill = len(np_actual_output[np_actual_output==1]) / len(np_actual_output)
    pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    pyplot.plot(recall, precision, marker='.', label='Logistic regression')

    # axis labels
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()



def create_testing_file(test_personas, test_snippets):

    persona_dict = {}
    snippet_dict = {}

    for i in range(0, len(test_personas)):
        persona_dict[i] = test_personas[i]
        snippet_dict[i] = test_snippets[i]

    #create new json file
    pos_list = []

    with open("positive-test-samples.json", "w+") as pos_file:
        for i in range(0, len(test_personas)):
            data = {
                "ID": i,
                "text persona": persona_dict[i],
                "text snippet": snippet_dict[i],
                "class": 1
            }

            pos_list.append(data)

        json.dump(pos_list, pos_file, indent=4)
    pos_file.close()


    return persona_dict, snippet_dict




def resume_training(train_df, valid_df, training_params, epoch, prev_loss):
    #start from next epoch
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




train_dataframe = pd.read_csv("/Users/arvindpunj/Desktop/Projects/NLP lab research/Extracting-personas-for-text-generation/data/train_other_original.txt",delimiter='\n', header= None, error_bad_lines=False)
validation_dataframe = pd.read_csv("/Users/arvindpunj/Desktop/Projects/NLP lab research/Extracting-personas-for-text-generation/data/valid_other_original.txt", delimiter='\n', header= None, error_bad_lines=False)
test_dataframe = pd.read_csv("/Users/arvindpunj/Desktop/Projects/NLP lab research/Extracting-personas-for-text-generation/data/test_other_original.txt", delimiter='\n', header= None, error_bad_lines=False)



main(train_dataframe, validation_dataframe, test_dataframe)
