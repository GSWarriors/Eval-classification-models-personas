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
from mymodel import add_padding_and_mask



"""from baseline_model import DistilbertTrainingParams
from baseline_model import DistilBertandBilinear
from baseline_model import create_persona_and_snippet_lists
from baseline_model import create_encoding_dict
from baseline_model import create_training_file
from baseline_model import create_validation_file
from baseline_model import add_padding_and_mask"""


#run test set here today
#start_time = time.perf_counter()
#end_time = time.perf_counter()
#end - start to calc time



def main(train_df, valid_df, test_df):

    """
    this code is only for resuming training, not testing

    training_params = DistilbertTrainingParams()
    training_params.create_tokens_dict()

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

    resume_training(train_df, valid_df, training_params, epoch)"""



    #this code is for testing
    training_params = DistilbertTrainingParams()
    training_params.create_tokens_dict()

    print("model parameters initialized, and tokens dict created")
    saved_model = training_params.convo_classifier
    saved_optimizer = training_params.optimizer
    saved_model.load_state_dict(torch.load('savedmodels/practicemodel.pt'))

    test_personas, test_snippets = create_persona_and_snippet_lists(test_df)
    encoded_test_dict, smallest_convo_size = create_encoding_dict(training_params, test_snippets)

    create_testing_file(test_personas, test_snippets)
    print("created test file")
    print("smallest convo size: " + str(smallest_convo_size))

    #test below- maybe changed saved model back to training params
    test_model(test_personas, encoded_test_dict, saved_model, training_params)



def test_model(test_personas, encoded_test_dict, saved_model, training_params):

    #try with training params, then try with saved model

    #writer = SummaryWriter('runs/bert_classifier')
    snippet_set_size = 4
    test_size = len(test_personas)
    test_loss = 0
    acc_avg = 0
    all_batch_sum = 0

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
                #print("nearing the end ")

                #take the preceding 4 snippets as distractors
                for elem in range(i - snippet_set_size, i):
                    #print("on distractor snippet: " + str(elem))
                    encoded_snippet_set.append(encoded_test_dict[elem][1])

            elif i - (snippet_set_size/2) < 0 and i + snippet_set_size < test_size:

                #print("starting out ")
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

            #this size of this is 1 except for last set
            labels_list = [0]*snippet_set_size
            gold_labels = [1, 1, 1, 1]
            labels_list = labels_list + gold_labels
            labels = torch.tensor(labels_list, requires_grad=False, dtype=torch.float, device=training_params.device)

            padded_snippet, snippet_attention_mask = add_padding_and_mask(full_encoded_snippet_set)
            snippet_input_ids = torch.from_numpy(padded_snippet).type(torch.long).to(training_params.device)
            #send input to distilbert
            with torch.no_grad():
                snippet_hidden_states = training_params.model(snippet_input_ids)

            snippet_set_features = snippet_hidden_states[0][:, 0, :].to(training_params.device)
            torch_snippet_features = snippet_set_features.clone().detach().requires_grad_(False)
            model_output = saved_model.forward(persona_encoding, len(full_encoded_snippet_set), torch_snippet_features)

            curr_loss, correct_preds = training_params.calc_loss_and_accuracy(model_output, labels)
            test_loss += curr_loss
            all_batch_sum += correct_preds

            snippet_set_size = 4
            test_loop_losses.append(test_loss.item())

            if i == 20:
                break

        acc_avg = ((all_batch_sum)/((snippet_set_size*2)*(test_size + 1)))*100
        print("the avg test accuracy for epoch: " + str(acc_avg))
        print()
        test_loop_losses = sum(test_loop_losses)
        print("total test loss: " + str(test_loop_losses))

        #writer.add_scalar("loss/test", test_loop_losses, epoch)
        #writer.add_scalar("accuracy/test", acc_avg, epoch)

        #writer.flush()
        #writer.close()







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




def resume_training(train_df, valid_df, training_params, epoch):
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




train_dataframe = pd.read_csv("/Users/arvindpunj/Desktop/Projects/NLP lab research/Extracting-personas-for-text-generation/data/train_other_original.txt",
delimiter='\n', header= None, error_bad_lines=False)
validation_dataframe = pd.read_csv("/Users/arvindpunj/Desktop/Projects/NLP lab research/Extracting-personas-for-text-generation/data/valid_other_original.txt",
delimiter='\n', header= None, error_bad_lines=False)
test_dataframe = pd.read_csv("/Users/arvindpunj/Desktop/Projects/NLP lab research/Extracting-personas-for-text-generation/data/test_other_original.txt",
delimiter='\n', header= None, error_bad_lines=False)



main(train_dataframe, validation_dataframe, test_dataframe)
