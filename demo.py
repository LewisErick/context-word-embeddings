import sys

from context_word_embeddings import ContextWordEmbeddings

if __name__ == '__main__':
    if len(sys.argv) == 3:
        print('Similarity:', ContextWordEmbeddings().get_similarity(sys.argv[1], sys.argv[2]))
    elif len(sys.argv) == 2:
        print('Embedding:', ContextWordEmbeddings().get_contextualized_word_embedding(sys.argv[1]))
    else:
        raise Exception('Please specify one or two strings')