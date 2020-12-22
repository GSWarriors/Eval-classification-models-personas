
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



def main(train_df, valid_df):

    #separate snippets into training and validation sets.

    create_persona_and_snippet_lists(train_df)
    #validation_personas, validation_snippets = create_persona_and_snippet_lists(valid_df)

    """init_params = DistilbertTrainingParams()
    init_params.create_tokens_dict()

    encoded_train_snippets = init_params.encode_snippets(training_snippets)
    encoded_val_snippets = init_params.encode_snippets(validation_snippets)

    init_params.train_model(training_personas, validation_personas, training_snippets, validation_snippets, encoded_train_snippets, encoded_val_snippets)"""



    """snippet_set_size = 7
    second_persona_encoding, snippet_set_len, torch_snippet_features, train = init_params.predict_second_persona(training_personas, encoded_train_snippets, snippet_set_size)
    my_classifier = DistilBertandBilinear(init_params.persona_model, init_params.bi_layer)
    output = my_classifier.forward(second_persona_encoding, snippet_set_len, torch_snippet_features)
    print("the predicted output is: " + str(output))"""




def create_persona_and_snippet_lists(df):

        k = 2
        full_doc = df[0]
        persona_list = []
        saved_persona = []
        checking_persona = True

        #go through and only print lines with "your persona"

        for line in range(0, len(full_doc)):
            if "partner's persona: " in full_doc[line]:
                checking_persona = True
                saved_persona.extend([full_doc[line][2:]])
                #print(str(full_doc[line]))

            else:
                #add the created persona to persona list
                #reset saved persona
                if checking_persona:
                    persona_list.append(saved_persona)
                    saved_persona = []
                    checking_persona = False

                if not checking_persona:
                    filtered_line = filter_for_responses(full_doc[line])
                    #print("the non-persona line: "  + str(filtered_line))
                    #print()


            if line == 100:
                print("persona list: " + str(persona_list))
                break



        """#further filtering
        for i in range(0, len(saved_persona)):
            new_persona = saved_persona[i].replace("partner's persona: ", "")
            new_saved_persona.extend([new_persona])

        print("persona list: " + str(new_saved_persona))
        new_str_persona = ' '.join(new_saved_persona)
        print("string version: " + str(new_str_persona))"""






"""This class initializes parameters needed for using distilbert as well as the parameters
needed for fine-tuning it for personachat"""
class DistilbertTrainingParams:


    #create model, tokenizer and weights for persona and snippets
    #make this a function called tokenize_and_encode()
    def __init__(self):
        distilbert_size = 768

        self.persona_model_class, self.persona_tokenizer_class, self.persona_pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
        self.snippet_model_class, self.snippet_tokenizer_class, self.snippet_pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

        #print("persona and snippet model class created")
        self.persona_tokenizer = self.persona_tokenizer_class.from_pretrained('./model/')
        #print("persona tokenizer created")

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
            #print("the current snippet is: " + str(curr_snippet))
            snippet_encoding = self.snippet_tokenizer.encode(curr_snippet, add_special_tokens=True)
            encoded_snippets.extend([snippet_encoding])
        return encoded_snippets




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
        writer.add_scalar("loss/validation", validation_loss, epoch)
        #self.max_loss = max(self.max_loss, validation_loss)
        #print("the max loss is saved as: " + str(self.max_loss))





    """This function does the actual training over the personas. Need to add including a new random persona
    every time. Will get from a persona list that I pass in as a parameter."""
    def train_model(self, training_personas, validation_personas, training_snippets, validation_snippets, encoded_train_snippets, encoded_val_snippets):
        #for name, param in self.convo_classifier.named_parameters():
        #    if param.requires_grad and name == 'bilinear_layer.weight':
        #        print(str(name) + ", " + str(param.data))

        writer = SummaryWriter('runs/bert_classifier')
        num_epochs = 1
        train = True
        first_iter = True
        training_size = len(training_personas)

        for epoch in range(0, num_epochs):

            training_loss = 0

            if epoch > 0:
                first_iter = False

            snippet_set_size = 7
            #randomly select a persona here
            persona_convo = ' '.join(training_personas[epoch])
            persona_encoding = [self.persona_tokenizer.encode(persona_convo, add_special_tokens=True)]
            gold_snippet_encoding = encoded_train_snippets[epoch]
            self.convo_classifier.persona_distilbert.train()
            self.snippet_model.eval()

            #distractor set creation, going through training set
            #need to add another loop here to go through all personas
            print("training persona is: " + str(training_personas[epoch]))

            i = 0
            while i < training_size:
                #last set of snippets if at end of training
                if i + snippet_set_size > training_size:
                    snippet_set_size = training_size - i

                print("training snippet set starting with snippet: " + str(i))
                full_set = training_snippets[i: i + snippet_set_size]
                print("the gold snippet here: " + str(training_snippets[epoch]))


                #get the encoded snippets of snippet set size, then extend the gold snippet
                encoded_snippet_set = []
                #encoded_snippet_set = rand.sample(encoded_train_snippets, snippet_set_size)
                encoded_snippet_set = encoded_train_snippets[i: i + snippet_set_size]
                encoded_snippet_set.extend([gold_snippet_encoding])

                #the last snippet is the matching one
                labels_list = [0]*snippet_set_size
                gold_label = [1]
                labels_list = labels_list + gold_label
                labels = torch.tensor(labels_list, requires_grad=False, dtype=torch.float, device=self.device)
                padded_snippet, snippet_attention_mask = add_padding_and_mask(encoded_snippet_set)
                snippet_input_ids = torch.from_numpy(padded_snippet).type(torch.long).to(self.device)


                #output for distilbert CLS token for each row- gets features for persona embedding. then replicate over snippet set.
                #afterwards, normalize the output with sigmoid function
                with torch.no_grad():
                    snippet_hidden_states = self.snippet_model(snippet_input_ids)

                snippet_set_features = snippet_hidden_states[0][:, 0, :].to(self.device)
                torch_snippet_features = snippet_set_features.clone().detach().requires_grad_(False)

                print(str("snippet set features:" + str(snippet_set_features)))
                print()
                model_output = self.convo_classifier.forward(persona_encoding, len(encoded_snippet_set), torch_snippet_features)
                curr_loss = self.binary_loss(model_output, labels)
                curr_loss.backward()
                #optimizer adjusts distilbertandbilinear model by subtracting lr*persona_distilbert.parameters().grad
                #and lr*bilinear_layer.parameters.grad(). After that, we zero the gradients
                self.optimizer.step()
                self.optimizer.zero_grad()

                if i == 140:
                    break

                i += snippet_set_size
                training_loss += curr_loss.item()

            #print("training loss:" + str(training_loss) + " , epoch: " + str(epoch))
            #writer.add_scalar("loss/train", training_loss, epoch)
            #validation loop
            #print("moving to validation:")

            #self.validate_model(validation_personas, encoded_val_snippets, epoch, first_iter, writer)
            #if exceeded_loss:
            #    break
        writer.flush()
        writer.close()

        #save the model
        #torch.save(self.convo_classifier, 'mysavedmodels/model.pt')






    def predict_second_persona(self, training_personas, encoded_train_snippets, snippet_set_len):
        #function to determine whether second persona returns a one hot encoded vector of [0, 1, 0, 0, 0, 0, 0]
        print()
        print("seeing model output when feeding in second persona: ")
        train = False
        second_persona = training_personas[1]
        second_persona_convo = ' '.join(second_persona)
        second_persona_encoding = [self.persona_tokenizer.encode(second_persona_convo, add_special_tokens=True)]
        gold_snippet_encoding = encoded_train_snippets[1]
        print("the second persona is: " + str(second_persona_convo))

        encoded_snippet_set = []
        encoded_snippet_set = rand.sample(encoded_train_snippets, snippet_set_len - 1)
        #encoded_snippet_set = encoded_train_snippets[0: snippet_set_len - 1]
        encoded_snippet_set.extend([gold_snippet_encoding])

        #pad the snippet set
        padded_snippet, snippet_attention_mask = add_padding_and_mask(encoded_snippet_set)
        snippet_input_ids = torch.from_numpy(padded_snippet).type(torch.long).to(self.device)
        with torch.no_grad():
            snippet_hidden_states = self.snippet_model(snippet_input_ids)
        #output for distilbert CLS token for each row- gets features for persona embedding. then replicate over snippet set.
        #afterwards, normalize the output with sigmoid function
        snippet_set_features = snippet_hidden_states[0][:, 0, :].to(self.device)
        torch_snippet_features = snippet_set_features.clone().detach().requires_grad_(False)
        print("the snippet features for this set are: " + str(torch_snippet_features))
        return second_persona_encoding, snippet_set_len, torch_snippet_features, train









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
                    speaker_str = ' <speaker-2> '

                    #add speaker 2 tag from after tab where speaker 1 left off
                    curr_response = convo[line][last_speaker_index + 1: len(convo[line]) - 1]
                    new_response_str += speaker_str + curr_response
                    #print("new response str is : " + new_response_str)

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
    number = response[0]
    tab_count = 0
    two_speaker_utterances = ""

    if number != '1':
        response = response[2:]

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


"""1. Report the number of unique questions (Q_ID) in this dataset
 2. Find the number of samples (all the examples: including same question based examples)"""
