
import torch
from torch.utils.tensorboard import SummaryWriter

import pandas as pd
import numpy as np
import transformers as ppb  #pytorch transformers
#from transformers import BertTokenizer, BertForNextSentencePrediction
import random as rand
import time
import math
import itertools
import json



def main(train_df, valid_df):

    #separate snippets into training and validation sets.
    training_personas, training_snippets = create_persona_and_snippet_lists(train_df)
    #validation_personas, validation_snippets = create_persona_and_snippet_lists(valid_df)

    #store the encoded first lines of snippets into this variable
    init_params = DistilbertTrainingParams()
    init_params.create_tokens_dict()

    #creates encoding dictionary of each line in the snippets.
    #after that we create a file storing the persona and snippets

    encoded_snippets_dict, smallest_convo_size = create_encoding_dict(init_params, training_snippets)
    persona_dict, snippet_dict = create_persona_and_snippet_dict(training_personas, training_snippets)
    init_params.train_model(training_personas, training_snippets, encoded_snippets_dict)



def partition_snippets(first_snippet_list, k):

    for i in range(0, len(first_snippet_list), k):
        yield first_snippet_list[i: i + k]


def create_encoding_dict(init_params, training_snippets):


    encoded_dict = {}

    for i in range(0, len(training_snippets)):
        partitioned_gold_snippet = partition_snippets(training_snippets[i], 1)
        partitioned_gold_snippet = list(partitioned_gold_snippet)

        encoded_gold_snippets = init_params.encode_snippets(partitioned_gold_snippet)
        encoded_dict[i] = encoded_gold_snippets

    #print()

    smallest_convo_size = 10
    for key, val in encoded_dict.items():
        list_size = len(val)
        smallest_convo_size = min(smallest_convo_size, list_size)

    #print("the smallest convo size is: " + str(smallest_convo_size))

    return encoded_dict, smallest_convo_size






def create_negative_samples(training_personas, training_snippets, persona_dict, snippet_dict):

    #this function creates negative samples, and checks whether or not the negative samples id are
    #equal to i. However, doesn't check whether an id has already been assigned (can use same one twice).
    #can change this later
    neg_list = []
    NUM_SAMPLES = 10
    neg_persona_dict = {}
    neg_snippet_dict = {}


    with open("negative-training-samples.json", "w+") as neg_file:
        for i in range(0, len(persona_dict)):
            sample_assigned = False
            sampled_ids = rand.sample(range(0, len(snippet_dict)), NUM_SAMPLES)
            #print("the sampled ids are: " + str(sampled_ids))


            for j in range(0, len(sampled_ids)):
                if i != sampled_ids[j]:
                    data = {
                        "ID": i,
                        "text persona": persona_dict[i],
                        "text snippet": snippet_dict[sampled_ids[j]],
                        "class": 0,
                        "snippet ID": sampled_ids[j]
                    }

                    neg_list.append(data)
                    break


        json.dump(neg_list, neg_file, indent=4)
        neg_file.close()







def create_persona_and_snippet_dict(training_personas, training_snippets):

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

        self.persona_model_class, self.persona_tokenizer_class, self.persona_pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
        self.snippet_model_class, self.snippet_tokenizer_class, self.snippet_pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
        self.persona_tokenizer = self.persona_tokenizer_class.from_pretrained('./model/')

        self.persona_model = self.persona_model_class.from_pretrained('./model/')
        print("persona model created")


        self.snippet_tokenizer = self.snippet_tokenizer_class.from_pretrained('./model')
        self.snippet_model = self.snippet_model_class.from_pretrained('./model')

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.binary_loss = torch.nn.BCELoss()
        self.bi_layer = torch.nn.Bilinear(distilbert_size, distilbert_size, 1)
        self.convo_classifier = DistilBertandBilinear(self.persona_model, self.bi_layer).to(self.device)
        self.optimizer = torch.optim.AdamW(self.convo_classifier.parameters(), lr=1e-6)
        self.max_loss = 0


    def create_tokens_dict(self):
        #adding tokenizer for speaker 1 and speaker 2 just for persona and snippet
        special_tokens_dict = {'additional_special_tokens': ['<speaker-1>', '<speaker-2>']}
        num_added_toks = self.persona_tokenizer.add_special_tokens(special_tokens_dict)
        self.snippet_tokenizer.add_special_tokens(special_tokens_dict)
        self.persona_model.resize_token_embeddings(len(self.persona_tokenizer))
        self.snippet_model.resize_token_embeddings(len(self.snippet_tokenizer))



    def encode_snippets(self, snippet_list):
        encoded_snippets = []

        for i in range(0, len(snippet_list)):
            curr_snippet = ' '.join(snippet_list[i])
            snippet_encoding = self.snippet_tokenizer.encode(curr_snippet, add_special_tokens=True)
            encoded_snippets.extend([snippet_encoding])
        return encoded_snippets


    def verify_distractors(self, i, j, smaller_set, snippet_set_size, training_size, encoded_snippets_dict):

        #need to modify to make sure i is not the index of the persona at the time.
        #take only 4 positive and negative samples

        if not smaller_set and (j + snippet_set_size < training_size):
            if j == i:
                encoded_snippet_set = [encoded_snippets_dict[j + 1][0], encoded_snippets_dict[j + 2][0],
                encoded_snippets_dict[j + 3][0], encoded_snippets_dict[j + 4][0]]
            elif j + 1 == i:
                encoded_snippet_set = [encoded_snippets_dict[j][0], encoded_snippets_dict[j + 2][0],
                encoded_snippets_dict[j + 3][0], encoded_snippets_dict[j + 4][0]]
            elif j + 2 == i:
                    encoded_snippet_set = [encoded_snippets_dict[j][0], encoded_snippets_dict[j + 1][0],
                    encoded_snippets_dict[j + 3][0], encoded_snippets_dict[j + 4][0]]
            elif j + 3 == i:
                    encoded_snippet_set = [encoded_snippets_dict[j][0], encoded_snippets_dict[j + 1][0],
                    encoded_snippets_dict[j + 2][0], encoded_snippets_dict[j + 4][0]]

            else:
                encoded_snippet_set = [encoded_snippets_dict[j][0], encoded_snippets_dict[j + 1][0],
                encoded_snippets_dict[j + 2][0], encoded_snippets_dict[j + 3][0]]

        return encoded_snippet_set





    """doing validation using snippet random sampling (size 7) with gold snippets at end of snippet set"""
    def validate_model(self, validation_personas, encoded_val_snippets, epoch, first_iter, writer):

        validation_loss = 0
        snippet_set_size = 7
        validation_size = len(validation_personas)

        #go through entire persona list and randomly sample snippets. Then, calculate forward
        #on the randomly sampled set and persona.

        with torch.no_grad():
            self.convo_classifier.persona_distilbert.eval()

            i = 0
            while i < validation_size:
                #last set of snippets if at end of training
                if i + snippet_set_size > validation_size:
                    snippet_set_size = validation_size - i

                persona_convo = ' '.join(validation_personas[i])
                persona_encoding = [self.persona_tokenizer.encode(persona_convo, add_special_tokens=True)]
                gold_snippet_encoding = encoded_val_snippets[i]

                encoded_snippet_set = []
                encoded_snippet_set = encoded_val_snippets[i: i + snippet_set_size]
                encoded_snippet_set.extend([gold_snippet_encoding])
                #print("the encoded snippet set: " + str(encoded_snippet_set))

                labels_list = [0]*snippet_set_size
                gold_label = [1]
                labels_list = labels_list + gold_label
                labels = torch.tensor(labels_list, requires_grad=False, dtype=torch.float, device=self.device)

                padded_snippet, snippet_attention_mask = add_padding_and_mask(encoded_snippet_set)
                snippet_input_ids = torch.from_numpy(padded_snippet).type(torch.long).to(self.device)

                with torch.no_grad():
                    snippet_hidden_states = self.snippet_model(snippet_input_ids)

                #output for distilbert CLS token for each row- gets features for persona embedding. then replicate over snippet set.
                #afterwards, normalize the output with sigmoid function
                snippet_set_features = snippet_hidden_states[0][:, 0, :].to(self.device)
                torch_snippet_features = snippet_set_features.clone().detach().requires_grad_(False)

                model_output = self.convo_classifier.forward(persona_encoding, len(encoded_snippet_set), torch_snippet_features)
                curr_loss = self.binary_loss(model_output, labels)

                print("validating snippet number: " + str(i))

                #show loss after going through 20 validation personas
                if i == 70:
                    break

                i += snippet_set_size
                validation_loss += curr_loss.item()

                """if not first_iter and total_loss > self.max_loss:
                    print("we have exceeded the validation loss from last time, breaking from validation")
                    print("the loss that exceeded: " + str(total_loss))
                    writer.add_scalar("loss/validation", total_loss, epoch)
                    return True"""


        print("the loss for this epoch is: " + str(validation_loss))
        #writer.add_scalar("loss/validation", validation_loss, epoch)
        #self.max_loss = max(self.max_loss, validation_loss)
        #print("the max loss is saved as: " + str(self.max_loss))





    """This function does the actual training over the personas."""
    def train_model(self, training_personas, training_snippets, encoded_snippets_dict):


        #note on inner loop:
        #usually don't have to worry about add backward, pytorch returns the last
        #operation performed on the tensor
        writer = SummaryWriter('runs/bert_classifier')
        num_epochs = 5
        train = True
        first_iter = True
        snippet_set_size = 4
        training_size = 4
        start_time = 0
        end_time = 0

        pos_file = open("positive-training-samples.json", "r")
        pos_data = json.load(pos_file)

        for epoch in range(0, num_epochs):
            if epoch > 0:
                first_iter = False

            self.convo_classifier.persona_distilbert.train()
            self.snippet_model.eval()

            training_loop_losses = []
            start_time = time.perf_counter()

            for i in range(0, len(training_personas)):

                persona_convo = ' '.join(pos_data[i]['text persona'])
                snippet_convo = pos_data[i]['text snippet']
                persona_encoding = [self.persona_tokenizer.encode(persona_convo, add_special_tokens=True)]
                gold_snippet_encoding = encoded_snippets_dict[i]
                #print("persona convo: " + str(persona_convo))
                #print("snippet convo: " + str(snippet_convo))

                training_loss = 0
                print("starting training iteration: " + str(i))

                for j in range(0, training_size, snippet_set_size):

                    encoded_snippet_set = []

                    if j + snippet_set_size > training_size:
                        print("in the last set ")
                        snippet_set_size = training_size - j
                        for elem in range(j, j + snippet_set_size):
                            print("on distractor snippet: " + str(elem))
                            encoded_snippet_set.append(encoded_snippets_dict[elem][0])
                    else:

                        #encoded_snippet_set = self.verify_distractors(i, j, smaller_set, snippet_set_size, training_size, encoded_snippets_dict)
                        encoded_snippet_set = [encoded_snippets_dict[j][0], encoded_snippets_dict[j + 1][0],
                        encoded_snippets_dict[j + 2][0], encoded_snippets_dict[j + 3][0]]

                    pos_snippet_encodings = [gold_snippet_encoding[0], gold_snippet_encoding[1],
                    gold_snippet_encoding[2], gold_snippet_encoding[3]]
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
                        snippet_hidden_states = self.snippet_model(snippet_input_ids)


                    snippet_set_features = snippet_hidden_states[0][:, 0, :].to(self.device)
                    torch_snippet_features = snippet_set_features.clone().detach().requires_grad_(False)
                    model_output = self.convo_classifier.forward(persona_encoding, len(full_encoded_snippet_set), torch_snippet_features)
                    #print("the model output is: " + str(model_output))
                    #print()

                    curr_loss = self.binary_loss(model_output, labels)
                    training_loss += curr_loss

                snippet_set_size = 4
                training_loop_losses.append(training_loss.item())
                print("the total loss for this iteration: " + str(training_loss))
                training_loss.backward()

                #optimizer adjusts distilbertandbilinear model by subtracting lr*persona_distilbert.parameters().grad
                #and lr*bilinear_layer.parameters.grad(). After that, we zero the gradients
                self.optimizer.step()
                self.optimizer.zero_grad()


            training_loop_losses = sum(training_loop_losses)
            print("the total loss for epoch " + str(epoch) +  ": " + str(training_loop_losses))
            writer.add_scalar("loss/train", training_loop_losses, epoch)

            end_time = time.perf_counter()
            print("total time it took for this epoch: " + str(end_time - start_time))
            print()


        writer.flush()
        writer.close()


        """
        #print("moving to validation:")

        #self.validate_model(validation_personas, encoded_val_snippets, epoch, first_iter, writer)
        #if exceeded_loss:
        #    break
        writer.flush()
        writer.close()

        #save the model
        #torch.save(self.convo_classifier, 'mysavedmodels/model.pt')"""






"""This is the class for passing in the distilbert and bilinear function for the model we've created
Hidden states: everything in last_hidden_states, now unpack 3-d output tensor.
        #features is 2d array with sentence embeddings of all sentences in dataset.
        #the model treats the entire persona as one "sentence"""
class DistilBertandBilinear(torch.nn.Module):

    def __init__(self, persona_distilbert, bilinear_layer):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.persona_distilbert = persona_distilbert
        self.bilinear_layer = bilinear_layer


    #can modify to find hidden states without detaching to numpy? (requires more computation)
    def forward(self, persona_encoding, snippet_set_len, torch_snippet_features):

        sigmoid = torch.nn.Sigmoid()
        padded_persona, persona_attention_mask = add_padding_and_mask(persona_encoding)
        persona_input_ids = torch.from_numpy(padded_persona).type(torch.long).to(self.device)

        with torch.enable_grad():
            persona_hidden_states = self.persona_distilbert(persona_input_ids)

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

train_dataframe = pd.read_csv("train_other_original.txt",delimiter='\n', header= None, error_bad_lines=False)
validation_dataframe = pd.read_csv("valid_other_original.txt", delimiter='\n', header= None, error_bad_lines=False)



main(train_dataframe, validation_dataframe)
