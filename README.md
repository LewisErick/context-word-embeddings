# context-word-embeddings
Extract contextual word embeddings using NLP and BETO

## Use in terminal

To get the embedding of a single word:
```
python demo.py presidente`
```

To get the cosine similarity between the embeddings of two words:
```
python demo.py presidente democracia
```

## Use in a Python file

```
from context_word_embeddings import ContextWordEmbeddings

text1 = 'presidente'
text2 = 'democracia'

text1_embedding = ContextWordEmbeddings().get_contextualized_word_embedding(text1)
text2_embedding = ContextWordEmbeddings().get_contextualized_word_embedding(text2)

similarity = ContextWordEmbeddings().get_similarity(text1, text2)
```

## References
- https://towardsdatascience.com/nlp-extract-contextualized-word-embeddings-from-bert-keras-tf-67ef29f60a7b
- https://becominghuman.ai/extract-a-feature-vector-for-any-image-with-pytorch-9717561d1d4c
