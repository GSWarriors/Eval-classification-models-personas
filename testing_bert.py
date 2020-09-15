import pandas as pd
import numpy as np
import pandas as pd
import transformers as ppb  #pytorch transformers
#from transformers import BertTokenizer, BertForNextSentencePrediction
import torch
import random as rand
import time
import math




def main(df):

    k = 2
    persona_convo = []
    snippet_convo = []
    full_doc = df[0]


    count = 0
    convo_list = []
    filtered_convo = []
    snippet_list = []

    first_persona = []
    first_snippet_convo = []

    #go through full document, and append to convo list for 8 responses
    #(1 conversation). Then append these to a list, and get the persona and snippet
    #from it. Finally, reset convo_list to just the current line, and filtered_convo

    for line in range(0, len(full_doc)):
        if line > 0 and line % 8 == 0:
            #convo_list = []
            for i in range(0, len(convo_list)):
                filtered_convo.append(filter_for_responses(convo_list[i]))

            persona_convo, snippet_convo = filter_persona_and_snippet(filtered_convo, k)
            if len(first_persona) == 0:
                first_persona = persona_convo

            if len(first_snippet_convo) == 0:
                first_snippet_convo = snippet_convo

            snippet_list.append(snippet_convo)
            #function here to add model, tokenizer, padding, and feature extraction

            convo_list = [full_doc[line]]
            filtered_convo = []
        else:
            convo_list.append(full_doc[line])


    print("persona convo: " + str(first_persona))
    print()
    print("snippet convo: " + str(first_snippet_convo))
    print()


    tokenization_and_feature_extraction(persona_convo, snippet_convo, snippet_list)



"""This function creates the DistilBertModel, tokenizes persona and snippet input,
pads and encodes it, and extracts feature vectors"""
def tokenization_and_feature_extraction(persona_convo, snippet_convo, snippet_list):
    #create model, tokenizer and weights for persona and snippets
    #make this a function called tokenize_and_encode()
    persona_model_class, persona_tokenizer_class, persona_pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
    snippet_model_class, snippet_tokenizer_class, snippet_pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

    persona_tokenizer = persona_tokenizer_class.from_pretrained(persona_pretrained_weights)
    persona_model = persona_model_class.from_pretrained(persona_pretrained_weights)

    snippet_tokenizer = snippet_tokenizer_class.from_pretrained(snippet_pretrained_weights)
    snippet_model = snippet_model_class.from_pretrained(snippet_pretrained_weights)

    #adding tokenizer for speaker 1 and speaker 2 just for persona and snippet
    special_tokens_dict = {'additional_special_tokens': ['<speaker-1>', '<speaker-2>']}
    num_added_toks = persona_tokenizer.add_special_tokens(special_tokens_dict)
    snippet_tokenizer.add_special_tokens(special_tokens_dict)

    #print("number of added tokens: " + str(num_added_toks))
    persona_model.resize_token_embeddings(len(persona_tokenizer))
    snippet_model.resize_token_embeddings(len(snippet_tokenizer))


    #separate snippets into training and validation sets
    training_size = math.floor(0.8*len(snippet_list))
    validation_size = len(snippet_list) - training_size


    num_epochs = 1
    #batch_size = 32
    distilbert_size = 768
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    #training snippets for this particular persona, not in batches
    #only compare one snippet and the current persona at a time so that can calculate bilinear loss for
    #each
    print("length of training set: " + str(training_size))

    for epoch in range(0, num_epochs):
        persona_convo = ' '.join(persona_convo)

        persona_encoding = [persona_tokenizer.encode(persona_convo, add_special_tokens=True)]
        for i in range(0, training_size):
            #handling the last batch (if less than 32)

            curr_snippet = ' '.join(snippet_list[i])
            snippet_encoding = [snippet_tokenizer.encode(curr_snippet, add_special_tokens=True)]
            #snippet_input_ids.extend(snippet_encoding)

            padded_persona, persona_attention_mask = add_padding_and_mask(persona_encoding)
            padded_snippet, snippet_attention_mask = add_padding_and_mask(snippet_encoding)

            persona_input_ids = torch.from_numpy(padded_persona).type(torch.LongTensor).to(device)
            snippet_input_ids = torch.from_numpy(padded_snippet).type(torch.LongTensor).to(device)
            snippet_attention_mask = torch.tensor(snippet_attention_mask).to(device)


            #we want a scalar output after passing a persona and snippet embedding, both of length 768
            m = torch.nn.Bilinear(distilbert_size, distilbert_size, 1)
            convo_classifier = DistilBertandBilinear(persona_model, snippet_model, persona_input_ids, snippet_input_ids, m)
            convo_classifier.forward()
            break





class DistilBertandBilinear:

    #for distilbert, we need persona_input_ids and snippet_input_ids, as well as the distilbert model
    #for bilinear, we just need the feature vector output from distilbert- the embedding for persona and snippet, as well
    #as the bilinear model


    def __init__(self, persona_distilbert, snippet_distilbert, persona_input_ids, snippet_input_ids, bilinear_layer):
        self.persona_distilbert = persona_distilbert
        self.snippet_distilbert = snippet_distilbert
        self.persona_input_ids = persona_input_ids
        self.snippet_input_ids = snippet_input_ids
        self.bilinear_layer = bilinear_layer


    def forward(self):
        #pass snippet into snippet distilbert, and persona into persona distilbert

        self.persona_distilbert.train()
        self.snippet_distilbert.eval()

        with torch.enable_grad():
            persona_hidden_states = self.persona_distilbert(self.persona_input_ids)

        with torch.no_grad():
            snippet_hidden_states = self.snippet_distilbert(self.snippet_input_ids)


        #everything in last_hidden_states, now unpack 3-d output tensor.
        #features is 2d array with sentence embeddings of all sentences in dataset.
        #the model treats the entire persona as one "sentence"
        persona_features = persona_hidden_states[0][:, 0, :].detach().numpy()
        snippet_features = snippet_hidden_states[0][:, 0, :].detach().numpy()

        print("persona feature vector from distilbert: " + str(persona_features))
        print()
        print("snippet feature vector from distilbert: " + str(snippet_features))

        torch_persona_features = torch.from_numpy(persona_features[0])
        torch_snippet_features = torch.from_numpy(snippet_features[0])

        combined_embedding = self.bilinear_layer(torch_persona_features, torch_snippet_features)
        print("output from bilinear: " + str(combined_embedding))


    """


    #combine using bilinear layer, then use crossentropy loss.



    distilbert_size = persona_features.shape[1]
    torch_persona_features = torch.from_numpy(persona_features[0])
    torch_snippet_features = torch.from_numpy(snippet_features)
    print("shape of persona: " + str(persona_features.shape))
    print("shape of snippet: " + str(snippet_features.shape))

    #(1, 768) persona and a (500, 768) list of lists of snippets
    print("persona: " + str(torch_persona_features))

    #go through all snippets and do bilinear with persona
    count = 0
    for snippet in range(0, len(torch_snippet_features)):
        m = torch.nn.Bilinear(distilbert_size, distilbert_size, distilbert_size)
        #print("current snippet: " + str(torch_snippet_features[snippet]))
        output = m(torch_persona_features, torch_snippet_features[snippet])
        #print("output from bilinear: " + str(output))
        #print()
        count += 1
        print("count is now " + str(count))"""





"""Function to add padding to input ids, as well as a mask"""
def add_padding_and_mask(input_ids_list):
    max_input_len = 0
    for ids in input_ids_list:
        if len(ids) > max_input_len:
            max_input_len = len(ids)


    padded_arr = np.array([i + [0]*(max_input_len-len(i)) for i in input_ids_list])

    #masking- create another variable to mask the padding we've created for persona and snippets
    attention_mask = np.where(padded_arr != 0, 1, 0)

    return padded_arr, attention_mask





def filter_persona_and_snippet(filtered_convo, snippet_size):


    rand_convo_count = 0
    rand_convo_index = -1
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

    response_without_number = response[2:]
    #print(response_without_number)
    #print()
    tab_count = 0
    two_speaker_utterances = ""

    for char in response_without_number:
        if tab_count < 2:
            if char == '\t':
                tab_count += 1

            two_speaker_utterances += char

    return two_speaker_utterances







my_list = []
dataframe = pd.read_csv("/Users/arvindpunj/Desktop/Projects/NLP lab research/Extracting-personas-for-text-generation/train_none_original.txt",
delimiter='\n', header= None, error_bad_lines=False)


main(dataframe)
