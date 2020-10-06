import pandas as pd
import numpy as np
import pandas as pd
import transformers as ppb  #pytorch transformers
#from transformers import BertTokenizer, BertForNextSentencePrediction
import torch
import random as rand
import time
import math
import itertools



def main(df):

    k = 2
    persona_convo = []
    snippet_convo = []
    full_doc = df[0]

    filtered_convo = []
    persona_list = []
    snippet_list = []



    #going through full text file, but only saving to personas at the moment.
    #conversations can be variable length, usually 6-7 lines.

    for line in range(0, len(full_doc)):

        filtered_line = filter_for_responses(full_doc[line])
        first_char = filtered_line[0]
        if line == 0:
            first_response = filter_for_responses(full_doc[line])
            filtered_convo = [first_response[2:]]
        #break
        elif first_char != '1' and line > 0:
            response = filter_for_responses(full_doc[line])
            filtered_convo.extend([response])

        else:
            #print convo, and persona. then, reset the filtered convo to just the current line (convo ended)
            if first_char == '1' and line > 0:
                persona_convo, snippet_convo = filter_persona_and_snippet(filtered_convo, k)
                first_response = filter_for_responses(full_doc[line])
                filtered_convo = [first_response[2:]]

                if len(persona_list) < 2:
                    persona_list.extend([persona_convo])
                snippet_list.extend([snippet_convo])


    #print("snippet list is: " + str(snippet_list))
    #print()
    #print("persona list is: " + str(persona_list))

    #separate snippets into training and validation sets.
    training_size = math.floor(0.8*len(snippet_list))
    validation_size = len(snippet_list) - training_size
    snippet_set_size = 7

    init_params = DistilbertTrainingParams()
    init_params.create_tokens_dict()

    print("optimizer: " + str(init_params.optimizer))
    print("my model parameters: " + str(init_params.convo_classifier.parameters()))




    #into training function, pass in all the parameters needed

    #tokenization_and_feature_extraction(persona_tokenizer, snippet_tokenizer, persona_list, snippet_list)



"""This function creates the DistilBertModel, tokenizes persona and snippet input,
pads and encodes it, and extracts feature vectors"""
"""def tokenization_and_feature_extraction(persona_tokenizer, snippet_tokenizer, persona_list, snippet_list):


    for epoch in range(0, num_epochs):
        persona_convo = ' '.join(persona_convo)
        persona_num = 0

        persona_encoding = [persona_tokenizer.encode(persona_convo, add_special_tokens=True)]
        gold_snippet_encoding = snippet_tokenizer.encode(snippet_convo, add_special_tokens=True)
        convo_classifier.persona_distilbert.train()
        snippet_model.eval()

        #distractor set creation, going through training set
        for i in range(0, training_size, snippet_set_size):
            if i + snippet_set_size > training_size:
                snippet_set_size = training_size - i
                snippet_set = snippet_list[i: i+snippet_set_size]

            elif i + snippet_set_size <= training_size and (persona_num < i or persona_num > i + snippet_set_size):
                snippet_set = snippet_list[i: i+snippet_set_size]
            else:
                if i + snippet_set_size + 1 <= training_size and (persona_num >= i and persona_num <= i + snippet_set_size):
                    snippet_set = snippet_list[i:persona_num] + snippet_list[persona_num + 1: i+snippet_set_size + 1]

            #join, add encoding, and add to a list. also add the gold snippet
            encoded_snippet_set = []
            for j in range(0, len(snippet_set)):
                curr_snippet = ' '.join(snippet_set[j])
                snippet_encoding = snippet_tokenizer.encode(curr_snippet, add_special_tokens=True)
                encoded_snippet_set.extend([snippet_encoding])
            encoded_snippet_set.extend([gold_snippet_encoding])

            #the last snippet is the matching one
            labels_list = [0]*snippet_set_size
            gold_label = [1]
            labels_list = labels_list + gold_label

            labels = torch.tensor(labels_list, requires_grad=False, dtype=torch.float, device=device)

            padded_snippet, snippet_attention_mask = add_padding_and_mask(encoded_snippet_set)
            snippet_input_ids = torch.from_numpy(padded_snippet).type(torch.long).to(device)

            with torch.no_grad():
                snippet_hidden_states = snippet_model(snippet_input_ids)

            #output for distilbert CLS token for each row- gets features for persona embedding. then replicate over snippet set.
            #afterwards, normalize the output with sigmoid function
            snippet_set_features = snippet_hidden_states[0][:, 0, :].to(device)
            torch_snippet_features = snippet_set_features.clone().detach().requires_grad_(False)

            model_output = convo_classifier.forward(persona_encoding, len(encoded_snippet_set), torch_snippet_features)
            curr_loss = binary_loss(model_output, labels)
            #print("binary loss is now: " + str(curr_loss.item()))
            print("snippet number: " + str(i))
            #print()
            curr_loss.backward()
            #optimizer adjusts distilbertandbilinear model by subtracting lr*persona_distilbert.parameters().grad
            #and lr*bilinear_layer.parameters.grad(). After that, we zero the gradients
            optimizer.step()
            optimizer.zero_grad()"""




class DistilbertTrainingParams:


    #create model, tokenizer and weights for persona and snippets
    #make this a function called tokenize_and_encode()
    def __init__(self):
        distilbert_size = 768
        num_epochs = 1

        self.persona_model_class, self.persona_tokenizer_class, self.persona_pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
        self.snippet_model_class, self.snippet_tokenizer_class, self.snippet_pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

        self.persona_tokenizer = self.persona_tokenizer_class.from_pretrained(self.persona_pretrained_weights)
        self.persona_model = self.persona_model_class.from_pretrained(self.persona_pretrained_weights)

        self.snippet_tokenizer = self.snippet_tokenizer_class.from_pretrained(self.snippet_pretrained_weights)
        self.snippet_model = self.snippet_model_class.from_pretrained(self.snippet_pretrained_weights)


        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.binary_loss = torch.nn.BCELoss()
        self.bi_layer = torch.nn.Bilinear(distilbert_size, distilbert_size, 1)
        self.convo_classifier = DistilBertandBilinear(self.persona_model, self.bi_layer).to(self.device)
        self.optimizer = torch.optim.AdamW(self.convo_classifier.parameters(), lr=0.001)



    def create_tokens_dict(self):
        #adding tokenizer for speaker 1 and speaker 2 just for persona and snippet
        special_tokens_dict = {'additional_special_tokens': ['<speaker-1>', '<speaker-2>']}
        num_added_toks = self.persona_tokenizer.add_special_tokens(special_tokens_dict)
        self.snippet_tokenizer.add_special_tokens(special_tokens_dict)
        self.persona_model.resize_token_embeddings(len(self.persona_tokenizer))
        self.snippet_model.resize_token_embeddings(len(self.snippet_tokenizer))











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

        #print("torch repeated is: " + str(repl_persona_features))
        output = self.bilinear_layer(torch_persona_features, torch_snippet_features)
        squeezed_output = torch.squeeze(output, 1)
        model_output = sigmoid(squeezed_output)
        #print("normalized output between 1 and 0: " + str(model_output))

        return model_output




"""Function to add padding to input ids, as well as a mask"""
def add_padding_and_mask(input_ids_list):
    #max_input_len = 0
    max_input_len = max([len(ids) for ids in input_ids_list])
    padded_arr = np.array([i + [0]*(max_input_len-len(i)) for i in input_ids_list])
    #padded_tensor_arr = torch.tensor([i + [0]*(max_input_len-len(i)) for i in input_ids_list])
    #masking- create another variable to mask the padding we've created for persona and snippets
    print("padded arr is: " + str(padded_arr))

    attention_mask = np.where(padded_arr != 0, 1, 0)
    #print("attention mask is: " + str(attention_mask))

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







my_list = []
dataframe = pd.read_csv("/Users/arvindpunj/Desktop/Projects/NLP lab research/Extracting-personas-for-text-generation/train_none_original.txt",
delimiter='\n', header= None, error_bad_lines=False)


main(dataframe)
