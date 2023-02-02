from torch import nn
from torchtext.vocab import GloVe   
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import WordSimilarityDataset, UnsupervisedDataset

# Word Similarity Task
# We use a GloVe embedding layer to get the embeddings for the input words
# Then, we use cosine similarity to calculate word similary and get baseline

DEVICE = "cpu"
BATCH_SIZE = 16

class WordSimilarityBaseline():
    """
    Class to generate baseline for Word Similarity task using GloVe embeddings
    """
    def __init__(self, vocab, glove_dim=100, glove_name="6B"):
        self.glove_dim = glove_dim
        self.cos = nn.CosineSimilarity()    # default dim=1, epsilon=1e-08
        self.glove_embeddings = GloVe(name=glove_name, dim=glove_dim) # get glove embeddings
        self.vocab_embeddings = self.glove_embeddings.get_vecs_by_tokens(vocab.get_itos()).to(DEVICE)
        self.embedding_layer = nn.Embedding.from_pretrained(self.vocab_embeddings).requires_grad_(False).to(DEVICE)

    def getSimilarity(self, batch):
        """
        Function to get the embeddings of a dataset by creating the DataLoader
        """
        embeddings1 = self.embedding_layer(batch[0])
        embeddings2 = self.embedding_layer(batch[1])
        similarity = self.cos(embeddings1, embeddings2)
        return similarity

class SentenceSimilarityBaseline():
    """
    Class to generate baseline for Sentence Similarity task by averaging GloVe embeddings
    """
    def __init__(self, vocab, glove_dim=100, glove_name="6B"):
        self.glove_dim = glove_dim
        self.cos = nn.CosineSimilarity()    # default dim=1, epsilon=1e-08
        self.glove_embeddings = GloVe(name=glove_name, dim=glove_dim) # get glove embeddings
        self.vocab_embeddings = self.glove_embeddings.get_vecs_by_tokens(vocab.get_itos()).to(DEVICE)
        self.embedding_layer = nn.Embedding.from_pretrained(self.vocab_embeddings).requires_grad_(False).to(DEVICE)

    def getSimilarity(self, batch):
        """
        Function to get the embeddings of a dataset by creating the DataLoader
        """
        embeddings1 = self.embedding_layer(batch[0]).mean(dim=1)
        embeddings2 = self.embedding_layer(batch[1]).mean(dim=1)
        similarity = self.cos(embeddings1, embeddings2)
        return similarity
