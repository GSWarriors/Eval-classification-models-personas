import pandas as pd

def main():

    #feed in data through pandas
    #with open('ParlAI/data/Persona-Chat/personachat/test_none_original.txt', 'r') as f:
    #    contents =
    dataframe = pd.read_csv("/Users/arvindpunj/Desktop/Projects/NLP lab research/Extracting-personas-for-text-generation/train_self_original.txt",
    delimiter='\n', header= None, error_bad_lines=False)
    #print(dataframe)
    first_convo = dataframe[0]
    #print("first line: " + dataframe[0][0:11])
    filter_conversation(first_convo)




def filter_conversation(first_convo):
    #print(first_convo)
    for line in range(5, 11):
        print(first_convo[line])
        #print()




main()
