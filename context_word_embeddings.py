'''
Contextual Word Embeddings implementation
LewisErick
'''
import os
import shutil
import tarfile 
import wget

import numpy as np
import torch

from transformers import BertForMaskedLM, BertTokenizer
from scipy.spatial import distance

class ContextWordEmbeddings:
    '''
    Utility class to calculate the contextual word embedding distance
    between two texts, an approach to find semantically similar sentences in a document.
    Reference:
    - https://towardsdatascience.com/nlp-extract-contextualized-word-embeddings-from-bert-keras-tf-67ef29f60a7b
    - https://becominghuman.ai/extract-a-feature-vector-for-any-image-with-pytorch-9717561d1d4c
    '''
    def __init__(self):
        if os.path.isdir('pytorch/') == False:
            self._download_model()
        self._tokenizer = BertTokenizer.from_pretrained("pytorch/", do_lower_case=False)
        self._model = BertForMaskedLM.from_pretrained("pytorch/")

    def _download_model(self):
        '''
        Downloads the BETO model's weights, vocabulary and configuration.
        '''
        weights = wget.download('https://users.dcc.uchile.cl/~jperez/beto/cased_2M/pytorch_weights.tar.gz')
        vocab =   wget.download('https://users.dcc.uchile.cl/~jperez/beto/cased_2M/vocab.txt')
        config =  wget.download('https://users.dcc.uchile.cl/~jperez/beto/cased_2M/config.json')

        with tarfile.open("pytorch_weights.tar.gz") as f:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(f)
        
        os.makedirs('pytorch', exist_ok=True)
        shutil.move('config.json', 'pytorch/config.json')
        shutil.move('vocab.txt', 'pytorch/vocab.txt')
    
    def _get_tokens_tensor(self, text):
        '''
        Given a text, convert it to BETO's required tokens
        '''
        tokens = self._tokenizer.tokenize(text)
        indexed_tokens = self._tokenizer.convert_tokens_to_ids(tokens)
        tokens_tensor = torch.tensor([indexed_tokens])
        return tokens_tensor
    
    def get_contextualized_word_embedding(self, text):
        '''
        Using BETO's last four layers, get the contextual embedding of the text.
        1. Get the embedding of each token
        2. Avg pool the token tensor (1,N,768) to a tensor of (1,1,768)
        3. Sum the embeddings from the four layers.
        '''
        # Get the last 4 layers of the encoder.
        context_layers = [self._model._modules.get('bert').encoder.layer[-(4-i)] for i in range(4)]
        context_embeddings = []
        for layer in context_layers:
            tokens = self._get_tokens_tensor(text)
            # Initialize embeddings as zero
            context_embedding = torch.zeros(1,
                                            tokens.shape[1],
                                            768)

            # Define hook to copy embedding after layer activation with example.
            def copy_data(m, i, o):
                context_embedding.copy_(o[0])
        
            # Register the hook after the forward operation in that layer
            h = layer.register_forward_hook(copy_data)

            # Run the model with the text.
            self._model(tokens)

            # Remove hook
            h.remove()

            context_embedding_numpy = np.copy(context_embedding.detach().numpy()[0][0])
            avg_context_embedding = np.mean(context_embedding.detach().numpy(), axis=1)
            # Add layer embedding to array to sum.
            context_embeddings.append(avg_context_embedding)
        return sum(context_embeddings)

    def get_similarity(self, text1, text2):
        '''
        Given two texts, calculate the cosine similarity between their contextualized word embeddings.
        '''
        text1_embedding = self.get_contextualized_word_embedding(text1)
        text2_embedding = self.get_contextualized_word_embedding(text2)
        return distance.cosine(text1_embedding, text2_embedding)