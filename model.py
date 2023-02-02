import torch
from torch import nn
from torchtext import vocab
from torchtext.vocab import GloVe   
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import UnsupervisedDataset, WordSimilarityDataset, STSDataset
from kmeans_pytorch import kmeans
import ot
from torchmetrics.functional import pairwise_cosine_similarity

DEVICE = "cpu"
BATCH_SIZE = 10

class Histograms():
    def __init__(self, data : UnsupervisedDataset, glove_dim : int = 100, glove_name : str = '6B', k : int = 100):
        super().__init__()
        self.glove_dim = glove_dim
        self.data = data
        self.k = k

        self.glove_embeddings = GloVe(name=glove_name, dim=glove_dim) # get glove embeddings
        self.vocab_embeddings = self.glove_embeddings.get_vecs_by_tokens(data.vocab.get_itos()).to(DEVICE)

        self.get_clusters()
    
    def get_clusters(self):
        clusters, self.means = kmeans(self.vocab_embeddings, self.k, device = DEVICE)
        # [vsize] [k, dim]

        self.clustered_coocc = torch.zeros(len(self.data.vocab), self.k)
        vocabsize = len(self.data.vocab)

        for token in tqdm(range(vocabsize), desc="Summing within clusters"):
            for cluster in range(self.k):
                self.clustered_coocc[token, cluster] = ((clusters == cluster).float().mul(self.data.sppmi[token])).sum()
                # self.clustered_coocc[token, cluster] = \
                #     sum(self.data.sppmi[token, i] for i in range(vocabsize) \
                #                                   if clusters[i] == cluster)

        
        
        self.histogram = torch.zeros(vocabsize, self.k)
        for i, row in tqdm(enumerate(self.clustered_coocc), desc="Normalising"):
            if row.sum() > 0: self.histogram[i] = row/row.sum()
            else: self.histogram[i, clusters[i]] = 1.

        # for token in range(vocabsize):
        #     denom = self.clustered_coocc[token].sum()
        #     self.histogram[token] = self.clustered_coocc[token] / denom

        #self.histogram = torch.stack([row/row.sum() if row.sum() > 0 else row for row in tqdm(self.clustered_coocc, desc="Normalising")]).to(DEVICE)

class OptimalTransport():
    def __init__(self, hist : Histograms):
        self.histograms = hist
        self.ground_cost = torch.cdist(hist.means, hist.means).to(DEVICE)
        # self.ground_cost = pairwise_cosine_similarity(hist.means)

class WordOptimalTransport(OptimalTransport):
    def __init__(self, hist: Histograms):
        super().__init__(hist)
    
    def getSimilarity(self, batch):
        h1 = torch.index_select(self.histograms.histogram, 0, batch[0])
        h2 = torch.index_select(self.histograms.histogram, 0, batch[1])
        similarities = []
        for h, h_ in zip(h1,h2):
            wass_dist = ot.emd(h, h_, self.ground_cost).to(DEVICE)
            wass_dist = wass_dist.mul(self.ground_cost).sum()
            #wass_dist = ot.emd2(h, h_, self.ground_cost)
            similarities.append(wass_dist)

        return torch.tensor(similarities).to(DEVICE)

class SentenceOptimalTransport(OptimalTransport):
    def __init__(self, hist: Histograms):
        super().__init__(hist)
    
    def getSimilarity(self, batch):

        bz, _ = batch[0].shape

        hists1 = torch.stack([torch.index_select(self.histograms.histogram, 0, batch[0][i]) for i in range(bz)])
        barys1 = torch.stack([ot.barycenter(hist.transpose(0,1), self.ground_cost, reg=0.2) for hist in hists1])

        hists2 = torch.stack([torch.index_select(self.histograms.histogram, 0, batch[1][i]) for i in range(bz)])
        barys2 = torch.stack([ot.barycenter(hist.transpose(0,1), self.ground_cost, reg=0.2) for hist in hists2])

        similarities = []
        for b1, b2 in zip(barys1, barys2):
            wass_dist = ot.emd(b1, b2, self.ground_cost).to(DEVICE)
            wass_dist = wass_dist.mul(self.ground_cost).sum()
            #wass_dist = ot.emd2(b1, b2, self.ground_cost)
            similarities.append(wass_dist)

        return torch.tensor(similarities).to(DEVICE)
