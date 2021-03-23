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




"""
Main separates dataset:
1. into personas and snippets
2. Initializes distilbert-bilinear model and creates dictionary with speaker tokens
3. encodes each line of the personas/snippets and puts them into a dictionary
4. adds file positive-training-samples.json
5. trains the model

"""



def main(train_df, valid_df):

    """training_personas, training_snippets = create_persona_and_snippet_lists(train_df)
    validation_personas, validation_snippets = create_persona_and_snippet_lists(valid_df)

    init_params = DistilbertTrainingParams()
    init_params.create_tokens_dict()

    encoded_training_dict, smallest_convo_size = create_encoding_dict(init_params, training_snippets)
    encoded_validation_dict, smallest_convo_size = create_encoding_dict(init_params, validation_snippets)

    train_persona_dict, train_snippet_dict = create_training_file(training_personas, training_snippets)
    valid_persona_dict, valid_snippet_dict = create_validation_file(validation_personas, validation_snippets)
    epoch = 0

    init_params.train_model(training_personas, validation_personas, encoded_training_dict, encoded_validation_dict, epoch)"""
    print("running main")





def partition_snippets(first_snippet_list, k):

    for i in range(0, len(first_snippet_list), k):
        yield first_snippet_list[i: i + k]


def create_encoding_dict(init_params, snippets):


    encoded_dict = {}

    for i in range(0, len(snippets)):

        partitioned_gold_snippet = partition_snippets(snippets[i], 1)
        partitioned_gold_snippet = list(partitioned_gold_snippet)

        encoded_gold_snippets = init_params.encode_snippets(partitioned_gold_snippet)
        encoded_dict[i] = encoded_gold_snippets


    smallest_convo_size = 10
    for key, val in encoded_dict.items():
        list_size = len(val)
        smallest_convo_size = min(smallest_convo_size, list_size)

    print("the smallest convo size is: " + str(smallest_convo_size))

    return encoded_dict, smallest_convo_size




def create_training_file(training_personas, training_snippets):

    persona_dict = {}
    snippet_dict = {}

    for i in range(0, len(training_personas)):
        persona_dict[i] = training_personas[i]
        snippet_dict[i] = training_snippets[i]

    #create new json file
    pos_list = []

    with open("positive-training-samples.json", "w+") as pos_file:
        for i in range(0, len(training_personas)):
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



def create_validation_file(validation_personas, validation_snippets):
    persona_dict = {}
    snippet_dict = {}

    for i in range(0, len(validation_personas)):
        persona_dict[i] = validation_personas[i]
        snippet_dict[i] = validation_snippets[i]

    #create new json file
    pos_list = []

    with open("positive-validation-samples.json", "w+") as pos_file:
        for i in range(0, len(validation_personas)):
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





"""This class initializes parameters needed for using distilbert as well as the parameters
needed for fine-tuning it for personachat"""
class DistilbertTrainingParams:

    #create model, tokenizer and weights for persona and snippets
    #make this a function called tokenize_and_encode()
    def __init__(self):
        distilbert_size = 768

        self.model_class, self.tokenizer_class, self.pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
        self.tokenizer = self.tokenizer_class.from_pretrained('./model/')
        self.model = self.model_class.from_pretrained('./model/')
        print("model created")

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.binary_loss = torch.nn.BCELoss()
        self.bi_layer = torch.nn.Bilinear(distilbert_size, distilbert_size, 1)
        self.convo_classifier = DistilBertandBilinear(self.model, self.bi_layer).to(self.device)
        self.optimizer = torch.optim.AdamW(self.convo_classifier.parameters(), lr=1e-6)
        self.prev_loss = 0


    def create_tokens_dict(self):
        #adding tokenizer for speaker 1 and speaker 2 just for persona and snippet
        special_tokens_dict = {'additional_special_tokens': ['<speaker-1>', '<speaker-2>']}
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
        self.model.resize_token_embeddings(len(self.tokenizer))


    def encode_snippets(self, snippet_list):
        encoded_snippets = []

        for i in range(0, len(snippet_list)):
            curr_snippet = ' '.join(snippet_list[i])
            snippet_encoding = self.tokenizer.encode(curr_snippet, add_special_tokens=True)
            encoded_snippets.extend([snippet_encoding])
        return encoded_snippets

    def calc_loss_and_accuracy(self, model_output, labels):

        curr_loss = self.binary_loss(model_output, labels)

        #need to move this below tensor to cuda. the tensor ones and zeroes
        rounded_output = torch.where(model_output >= 0.5, torch.tensor(1), torch.tensor(0))
        print("model output: " + str(model_output))
        print("rounded output: " + str(rounded_output))

        predictions = rounded_output.numpy()
        correct_preds = 0

        for i in range(0, len(predictions)):

            if i < len(predictions)/2:
                #gold = 0 for distractors
                if predictions[i] == 0:
                    correct_preds += 1
            else:
                #gold = 1 for gold
                if predictions[i] == 1:
                    correct_preds += 1

        return curr_loss, correct_preds




    def validate_model(self, validation_personas, encoded_validation_dict, epoch, first_iter, writer):

        snippet_set_size = 4
        validation_size = 10
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

                pos_snippet_encodings = [gold_snippet_encoding[1], gold_snippet_encoding[2],
                gold_snippet_encoding[3], gold_snippet_encoding[4]]
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

                curr_loss, correct_preds = self.calc_loss_and_accuracy(model_output, labels)
                validation_loss += curr_loss
                all_batch_sum += correct_preds

                snippet_set_size = 4
                validation_loop_losses.append(validation_loss.item())

                if i == 10:
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






    """This function does the actual training over the personas.
    optimizer adjusts distilbertandbilinear model by subtracting lr*model.parameters().grad
    and lr*bilinear_layer.parameters.grad(). After that, we zero the gradients"""
    def train_model(self, training_personas, validation_personas, encoded_training_dict, encoded_validation_dict, epoch):

        #run model on test set after training longer.

        writer = SummaryWriter('runs/bert_classifier')
        train = True
        first_iter = True
        snippet_set_size = 4
        acc_avg = 0

        training_size = 20
        start_time = 0
        end_time = 0
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

                pos_snippet_encodings = [gold_snippet_encoding[1], gold_snippet_encoding[2],
                gold_snippet_encoding[3], gold_snippet_encoding[4]]
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

                curr_loss, correct_preds = self.calc_loss_and_accuracy(model_output, labels)
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

            """if epoch % 10 == 0:
                torch.save(
                    {'epoch': epoch,
                    'model_state_dict': self.convo_classifier.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'prev_loss': self.prev_loss
                    }, 'savedmodels/resumemodel.pt')
                print("checkpointing model on epoch: " + str(epoch))"""

            if epoch == 5:
                break


            epoch += 1

        writer.flush()
        writer.close()

        torch.save(self.convo_classifier.state_dict(), 'savedmodels/practicemodel.pt')



#note: take model that I have on remote and replace here




"""This is the class for passing in the distilbert and bilinear function for the model we've created
Hidden states: everything in last_hidden_states, now unpack 3-d output tensor.
        #features is 2d array with sentence embeddings of all sentences in dataset.
        #the model treats the entire persona as one "sentence"""
class DistilBertandBilinear(torch.nn.Module):

    def __init__(self, model, bilinear_layer):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model
        self.bilinear_layer = bilinear_layer


    #can modify to find hidden states without detaching to numpy? (requires more computation)
    def forward(self, persona_encoding, snippet_set_len, torch_snippet_features):

        sigmoid = torch.nn.Sigmoid()
        padded_persona, persona_attention_mask = add_padding_and_mask(persona_encoding)
        persona_input_ids = torch.from_numpy(padded_persona).type(torch.long).to(self.device)

        with torch.enable_grad():
            persona_hidden_states = self.model(persona_input_ids)

        #output for distilbert CLS token for each row- gets features for persona embedding. then replicate over snippet set
        persona_features = persona_hidden_states[0][:, 0, :].to(self.device)
        repl_persona_features = persona_features.repeat(snippet_set_len, 1)
        torch_persona_features = repl_persona_features.clone().detach().requires_grad_(True)

        output = self.bilinear_layer(torch_persona_features, torch_snippet_features)

        squeezed_output = torch.squeeze(output, 1)
        model_output = sigmoid(squeezed_output)
        #print("predicted normalized output between 1 and 0: " + str(model_output))
        return model_output








"""Function to add padding to input ids, as well as a mask"""
def add_padding_and_mask(input_ids_list):
    max_input_len = max([len(ids) for ids in input_ids_list])
    padded_arr = np.array([i + [0]*(max_input_len-len(i)) for i in input_ids_list])
    #padded_tensor_arr = torch.tensor([i + [0]*(max_input_len-len(i)) for i in input_ids_list])
    #masking- create another variable to mask the padding we've created for persona and snippets

    attention_mask = np.where(padded_arr != 0, 1, 0)

    return padded_arr, attention_mask





def filter_persona_and_snippet(filtered_convo, snippet_size):

    rand_convo_count = 0
    rand_convo_index = -1
    snippet_convo = ""
    while rand_convo_count < 1:
        rand_convo_index = rand.randint(0, len(filtered_convo) - 1)
        rand_convo_count += 1


    if len(filtered_convo) > 1:
        if rand_convo_index == len(filtered_convo) - 1:
            snippet_convo = filtered_convo[rand_convo_index - 1:]
            del filtered_convo[rand_convo_index - 1:]
        else:
            snippet_convo = filtered_convo[rand_convo_index: rand_convo_index + snippet_size]
            del filtered_convo[rand_convo_index: rand_convo_index + snippet_size]

    persona_convo = filtered_convo

    persona_convo = add_speaker_tokens(persona_convo)
    snippet_convo = add_speaker_tokens(snippet_convo)
    return persona_convo, snippet_convo





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





#can edit this to valid.txt and test.txt in order to run on different files

train_dataframe = pd.read_csv("data/train_other_original.txt",delimiter='\n', header= None, error_bad_lines=False)
validation_dataframe = pd.read_csv("data/valid_other_original.txt", delimiter='\n', header= None, error_bad_lines=False)



main(train_dataframe, validation_dataframe)
