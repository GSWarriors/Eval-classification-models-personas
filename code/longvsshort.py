"""
Here we compare how the model performs when we analyze only the longest
four responses vs the shortest four. This is to see which models
we've trained perform better with more context based sentence, and which perform worse.

Train all models this way, bilinear + sigmoid, logistic and TF-IDF + cosin sim
"""
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
import copy

from mymodel import create_persona_and_snippet_lists
from mymodel import create_encoding_dict
from mymodel import DistilBertTrainingParams
from mymodel import create_training_file
from mymodel import create_validation_file
from mymodel import add_padding_and_mask


class longvsshort(DistilBertTrainingParams):


    def validate_model(self, validation_personas, encoded_validation_dict, epoch, first_iter, writer):
        snippet_set_size = 4
        validation_size = 8
        validation_loss = 0
        acc_avg = 0
        all_batch_sum = 0

        val_file = open("positive-validation-samples.json", "r")
        val_data = json.load(val_file)
        validation_loop_losses = []

        with torch.no_grad():
            self.convo_classifier.model.eval()

            for i in range(0, len(validation_personas)):

                persona_convo = ' '.join(val_data[i]['text persona'])
                snippet_convo = val_data[i]['text snippet']
                persona_encoding = [self.tokenizer.encode(persona_convo, add_special_tokens=True)]
                gold_snippet_encoding = encoded_validation_dict[i]

                encoded_snippet_set = []
                print("starting validation iteration: " + str(i))

                if i + (snippet_set_size/2) >= validation_size and i - snippet_set_size >= 0:
                    #print("nearing the end ")

                    #take the preceding 4 snippets as distractors
                    for elem in range(i - snippet_set_size, i):
                        #print("on distractor snippet: " + str(elem))
                        encoded_snippet_set.append(encoded_validation_dict[elem][1])

                elif i - (snippet_set_size/2) < 0 and i + snippet_set_size < validation_size:

                    #print("starting out ")
                    #take the proceeding 4 snippets as distractors
                    for elem in range(i + 1, i + snippet_set_size + 1):
                        #print("on distractor snippet: " + str(elem))
                        encoded_snippet_set.append(encoded_validation_dict[elem][1])

                else:
                    encoded_snippet_set = [encoded_validation_dict[i - 2][1], encoded_validation_dict[i - 1][1],
                    encoded_validation_dict[i + 1][1], encoded_validation_dict[i + 2][1]]

                pos_snippet_encodings = pos_snippet_encodings = gold_snippet_encoding
                full_encoded_snippet_set = encoded_snippet_set + pos_snippet_encodings

                #this size of this is 1 except for last set
                labels_list = [0]*snippet_set_size
                gold_labels = [1, 1, 1, 1]
                labels_list = labels_list + gold_labels
                labels = torch.tensor(labels_list, requires_grad=False, dtype=torch.float, device=self.device)

                padded_snippet, snippet_attention_mask = add_padding_and_mask(full_encoded_snippet_set)
                snippet_input_ids = torch.from_numpy(padded_snippet).type(torch.long).to(self.device)
                #send input to distilbert
                with torch.no_grad():
                    snippet_hidden_states = self.model(snippet_input_ids)

                snippet_set_features = snippet_hidden_states[0][:, 0, :].to(self.device)
                torch_snippet_features = snippet_set_features.clone().detach().requires_grad_(False)
                model_output = self.convo_classifier.forward(persona_encoding, len(full_encoded_snippet_set), torch_snippet_features)

                curr_loss, correct_preds, predictions = self.calc_loss_and_accuracy(model_output, labels)
                validation_loss += curr_loss
                all_batch_sum += correct_preds

                snippet_set_size = 4
                validation_loop_losses.append(validation_loss.item())

                if i == 8:
                    break

            acc_avg = ((all_batch_sum)/((snippet_set_size*2)*(validation_size + 1)))*100
            print("the avg validation accuracy for epoch: " + str(acc_avg))
            print()
            validation_loop_losses = sum(validation_loop_losses)


            if not first_iter and validation_loop_losses > self.prev_loss:
                print("we have exceeded the validation loss from last time, breaking from validation")
                print("the loss that exceeded: " + str(validation_loop_losses))
                return True

            self.prev_loss = validation_loop_losses
            print("current loss is: " + str(validation_loop_losses))
            print("the prev loss is saved as: " + str(self.prev_loss))

            writer.add_scalar("loss/validation", validation_loop_losses, epoch)
            writer.add_scalar("accuracy/validation", acc_avg, epoch)






    def train_model(self, training_personas, validation_personas, encoded_training_dict, encoded_validation_dict, epoch):

        #run model on test set after training longer.
        writer = SummaryWriter('/Users/arvindpunj/Desktop/Projects/NLP lab research/Extracting-personas-for-text-generation/runs/bert_classifier')
        train = True
        first_iter = True
        snippet_set_size = 4
        acc_avg = 0

        training_size = 20
        pos_file = open("positive-training-samples.json", "r")
        pos_data = json.load(pos_file)

        exceeded_loss = False

        while not exceeded_loss:
            if epoch > 0:
                first_iter = False

            self.convo_classifier.model.train()
            training_loop_losses = []
            all_batch_sum = 0

            for i in range(0, len(training_personas)):

                persona_convo = ' '.join(pos_data[i]['text persona'])
                snippet_convo = pos_data[i]['text snippet']
                persona_encoding = [self.tokenizer.encode(persona_convo, add_special_tokens=True)]
                gold_snippet_encoding = encoded_training_dict[i]

                training_loss = 0
                encoded_snippet_set = []
                print("starting training iteration: " + str(i))

                if i + (snippet_set_size/2) >= training_size and i - snippet_set_size >= 0:
                    #take the preceding 4 snippets as distractors
                    for elem in range(i - snippet_set_size, i):
                        #print("on distractor snippet: " + str(elem))
                        encoded_snippet_set.append(encoded_training_dict[elem][1])

                elif i - (snippet_set_size/2) < 0 and i + snippet_set_size < training_size:
                    #take the proceeding 4 snippets as distractors
                    for elem in range(i + 1, i + snippet_set_size + 1):
                        #print("on distractor snippet: " + str(elem))
                        encoded_snippet_set.append(encoded_training_dict[elem][1])

                else:
                    encoded_snippet_set = [encoded_training_dict[i - 2][1], encoded_training_dict[i - 1][1],
                    encoded_training_dict[i + 1][1], encoded_training_dict[i + 2][1]]


                pos_snippet_encodings = gold_snippet_encoding
                full_encoded_snippet_set = encoded_snippet_set + pos_snippet_encodings

                #this size of this is 4 except for last set
                labels_list = [0]*snippet_set_size
                gold_labels = [1, 1, 1, 1]
                labels_list = labels_list + gold_labels
                labels = torch.tensor(labels_list, requires_grad=False, dtype=torch.float, device=self.device)
                padded_snippet, snippet_attention_mask = add_padding_and_mask(full_encoded_snippet_set)
                snippet_input_ids = torch.from_numpy(padded_snippet).type(torch.long).to(self.device)

                #send input to distilbert
                with torch.no_grad():
                    snippet_hidden_states = self.model(snippet_input_ids)

                snippet_set_features = snippet_hidden_states[0][:, 0, :].to(self.device)
                torch_snippet_features = snippet_set_features.clone().detach().requires_grad_(False)

                model_output = self.convo_classifier.forward(persona_encoding, len(full_encoded_snippet_set), torch_snippet_features)
                print("model output (without threshold): " + str(model_output))

                curr_loss, correct_preds, predictions = self.calc_loss_and_accuracy(model_output, labels)
                training_loss += curr_loss
                all_batch_sum += correct_preds

                snippet_set_size = 4
                training_loop_losses.append(training_loss.item())
                training_loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                if i == 20:
                    break


            #adding up correct predictions from each batch. Then adding up all batches
            #finally, taking average of correct predictions for all samples over the entire snippet sets
            acc_avg = ((all_batch_sum)/((snippet_set_size*2)*(training_size + 1)))*100
            print("the avg training accuracy for epoch: " + str(acc_avg))
            print()

            training_loop_losses = sum(training_loop_losses)
            writer.add_scalar("loss/train", training_loop_losses, epoch)
            writer.add_scalar("accuracy/train", acc_avg, epoch)
            #validation loop here
            exceeded_loss = self.validate_model(validation_personas, encoded_validation_dict, epoch, first_iter, writer)

            if epoch == 6:
                torch.save(
                    {'epoch': epoch,
                    'model_state_dict': self.convo_classifier.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'prev_loss': self.prev_loss
                    }, "/Users/arvindpunj/Desktop/Projects/NLP lab research/Extracting-personas-for-text-generation/savedmodels/resumemodel.pt")
                print("checkpointing model on epoch: " + str(epoch))
                break

            epoch += 1


        writer.flush()
        writer.close()
        torch.save(self.convo_classifier.state_dict(), "/Users/arvindpunj/Desktop/Projects/NLP lab research/Extracting-personas-for-text-generation/savedmodels/practicemodel.pt")



    def parse_long_responses(self, responses):

        all_longest = []

        for i in range(0, len(responses)):
            #one list of response for current persona
            curr_longest = []
            curr_response = responses[i]
            responses[i].sort(key=len)

            curr_longest = responses[i][len(responses[i]) - 4:]
            all_longest.append(curr_longest)

            """for j in range(0, len(curr_longest)):
                print("response: " + str(curr_longest[j]))
                print("length of the response: " + str(len(curr_longest[j])))"""

        #print("the longest 4 responses for all personas: " + str(all_longest))
        #print("length: " + str(len(all_longest)))
        return all_longest



    def parse_short_responses(self, responses):


        all_shortest = []

        for i in range(0, len(responses)):
            #one list of response for current persona
            curr_shortest = []
            curr_response = responses[i]
            responses[i].sort(key=len)

            curr_shortest = curr_response[0: 4]
            all_shortest.append(curr_shortest)

            #debugging to check last shortest response list
            """if i == 999:
                for j in range(0, len(curr_response)):
                    print("response: " + str(curr_response[j]))
                    print("length of the response: " + str(len(curr_response[j])))
                    print()
                print()"""

        return all_shortest









"""Note: can replace parse long and short responses to get different responses
and different results"""
def main(train_df, valid_df):

    """test_one = longvsshort()

    #largest 4 for both training and validation
    training_personas, training_responses = create_persona_and_snippet_lists(train_df)
    validation_personas, validation_responses = create_persona_and_snippet_lists(valid_df)

    #longest personas and responses
    longest_train_responses = test_one.parse_long_responses(training_responses)
    longest_validation_responses = test_one.parse_long_responses(validation_responses)
    #print("longest responses from training set: " + str(longest_train_responses))

    #shortest personas and responses
    #shortest_train_responses = test_one.parse_short_responses(training_responses)
    #shortest_validation_responses = test_one.parse_short_responses(validation_responses)
    #print("shortest validation responses: " + str(shortest_validation_responses[999]))
    #print("shortest training responses: " + str(shortest_train_responses))


    test_one.create_tokens_dict()

    encoded_training_dict, smallest_convo_size = create_encoding_dict(test_one, longest_train_responses)
    encoded_validation_dict, smallest_convo_size = create_encoding_dict(test_one, longest_validation_responses)

    train_persona_dict, train_response_dict = create_training_file(training_personas, longest_train_responses)
    valid_persona_dict, valid_response_dict = create_validation_file(validation_personas, longest_validation_responses)

    epoch = 0

    #test_one version of training model function
    #the encoded training and validation dictionaries include the shortest/longest responses
    test_one.train_model(training_personas, validation_personas, encoded_training_dict, encoded_validation_dict, epoch)"""

    print("in test mode for longvsshort")






train_dataframe = pd.read_csv("/Users/arvindpunj/Desktop/Projects/NLP lab research/Extracting-personas-for-text-generation/data/train_other_original.txt",
delimiter='\n', header= None, error_bad_lines=False)
validation_dataframe = pd.read_csv("/Users/arvindpunj/Desktop/Projects/NLP lab research/Extracting-personas-for-text-generation/data/valid_other_original.txt",
delimiter='\n', header= None, error_bad_lines=False)



main(train_dataframe, validation_dataframe)
