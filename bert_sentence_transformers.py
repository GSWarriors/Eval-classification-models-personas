import pandas as pd
import numpy as np
import pandas as pd
import transformers as ppb  #pytorch transformers
from transformers import BertTokenizer, BertForNextSentencePrediction
import torch
import logging
from sentence_transformers import SentenceTransformer, LoggingHandler



model = SentenceTransformer('bert-base-nli-mean-tokens')
corpus = ['A man is eating food.',
          'A man is eating a piece of bread.',
          'The girl is carrying a baby.',
          'A man is riding a horse.',
          'A woman is playing violin.',
          'Two men pushed carts through the woods.',
          'A man is riding a white horse on an enclosed ground.',
          'A monkey is playing drums.',
          'A cheetah is running behind its prey.'
          ]

corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)


for sentence, embedding in zip(sentences, sentence_embeddings):
    print("sentence:", sentence)
    print("Embedding:", embedding)
    print("")
