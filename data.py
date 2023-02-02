from torch.utils.data import Dataset
from torchtext import vocab
from torchtext.vocab import GloVe
from tqdm import tqdm
from icecream import ic
from nltk.tokenize import word_tokenize
import pandas as pd
import torch
from random import random
from scipy.sparse import lil_matrix,csr_matrix
from sppmi import get_sppmi

DEVICE = "cpu"

class UnsupervisedDataset(Dataset):
    def __init__(self, split : str = 'train', window_size : int = 2, min_freq : int = 4, max_sentence_length : int = 300, smoothing_parameter = 0.75, k_shift = 1.0):
        self.filename = "./wikitext-103/wiki." + split + ".tokens"
        self.window_size = window_size
        self.min_freq = min_freq
        self.coocc = None
        self.max_sentence_length = max_sentence_length

        self.get_vocab()

        self.get_data()

        self.sppmi = get_sppmi(self.coocc, smoothing_parameter, k_shift)

    def get_vocab(self):
        tokens = []
        with open(self.filename, "r") as file: nrows = sum(1 for _ in file)
        with open(self.filename, "r") as file:
            for line in tqdm(file, desc="Vocabulary", total=nrows):
                if (len(line) <= 1 or line[0] == '='): continue
                if random() > 0.5: continue
                tokens += [[word.lower()] for word in line.split()]

        self.vocab = vocab.build_vocab_from_iterator(tokens,
                                                    min_freq = self.min_freq,
                                                    specials = ['<UNK>', '<PAD>'])
        self.vocab.set_default_index(self.vocab['<UNK>'])
        self.coocc = lil_matrix((len(self.vocab), len(self.vocab)))
    
    def get_data(self):
        self.max_length = 0
        self.sentences = []
        with open(self.filename, "r") as file: nrows = sum(1 for _ in file)
        with open(self.filename, "r") as file:
            for line in tqdm(file, desc="Sentences", total=nrows):
                if (len(line) <= 2 or line[1] == '='): continue
                tokens = line.split()
                indices = [self.vocab[word.lower()] for word in tokens]
                if len(indices) > self.max_sentence_length:
                    continue
                self.max_length = max(self.max_length, len(indices))
                self.sentences.append(indices)

            for i, token in enumerate(indices[:-self.window_size]):
                for j in range(1, self.window_size+1):
                    count = 1/j
                    self.coocc[token, indices[i+j]] += count
                    self.coocc[indices[i+j], token] += count
            for i, token in list(enumerate(indices))[-self.window_size:]:
                for j in range(i+1, len(indices)):
                    count = 1/(j-i)
                    self.coocc[token, indices[j]] += count
                    self.coocc[indices[j], token] += count

        self.coocc = csr_matrix(self.coocc)

    def get_counts(self):
        pass

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        return self.sentences[index]

class WordSimilarityDataset(Dataset):
    def __init__(self, vocab : vocab, filename : str = 'SimLex-999/SimLex-999.txt'):
        self.filename = filename
        self.vocab = vocab

        self.get_data()
    
    def get_data(self):
        df = pd.read_csv(self.filename, sep='\t')

        self.word1s = []
        self.word2s = []
        self.sims = []
        for row in df.iterrows():
            if (row[1]['word1'] not in self.vocab or row[1]['word2'] not in self.vocab): continue
            self.word1s.append(self.vocab[row[1]['word1']])
            self.word2s.append(self.vocab[row[1]['word2']])
            self.sims.append(row[1]['SimLex999'])

        self.word1s = torch.tensor(self.word1s)
        self.word2s = torch.tensor(self.word2s)
        self.sims = torch.tensor(self.sims)
    
    def __len__(self):
        return len(self.word1s)
    
    def __getitem__(self, index):
        return (self.word1s[index], self.word2s[index], self.sims[index])

class HypernymyDataset(Dataset):
    def __init__(self, vocab: vocab, dataset : str = 'lex', split : str = 'train'):
        self.filename = 'HypeNet/dataset_' + dataset + '/' + split + '.tsv'
        self.vocab = vocab

        self.get_data()
    
    def get_data(self):
        df = pd.read_csv(self.filename, sep='\t', header=0)

        self.hyponyms = []
        self.hypernyms = []
        self.isHyp = []
        for row in df.iterrows():
            if (row[1][0] not in self.vocab or row[1][1] not in self.vocab): continue
            self.hyponyms.append(self.vocab[row[1][0]])
            self.hypernyms.append(self.vocab[row[1][1]])
            self.isHyp.append(float(row[1][2]))
        
        self.hyponyms = torch.tensor(self.hyponyms)
        self.hypernyms = torch.tensor(self.hypernyms)
        self.isHyp = torch.tensor(self.isHyp)
    
    def __len__(self):
        return len(self.hyponyms)
    
    def __getitem__(self, index):
        return (self.hyponyms[index], self.hypernyms[index], self.isHyp[index])

class STSDataset(Dataset):
    def __init__(self, vocab : vocab, split : str = 'train'):
        self.filename = 'stsbenchmark/sts-' + split + '.csv'
        self.vocab = vocab

        self.get_data()
    
    def get_data(self):
        #df = pd.read_csv(self.filename, sep='\t')

        self.sent1s = []
        maxlength1 = 0
        self.sent2s = []
        maxlength2 = 0
        self.sims = []

        with open(self.filename, 'r') as file:
            for line in file:
                sent1 = [word.lower() for word in word_tokenize(line.split('\t')[5])]
                sent1 = [self.vocab[word] for word in sent1]
                self.sent1s.append(sent1)
                maxlength1 = max(maxlength1, len(sent1))

                sent2 = [word.lower() for word in word_tokenize(line.split('\t')[6])]
                sent2 = [self.vocab[word] for word in sent2]
                self.sent2s.append(sent2)
                maxlength2 = max(maxlength2, len(sent2))

                sim = float(line.split('\t')[4])
                self.sims.append(sim)
        
        PAD_TOKEN = self.vocab['<PAD>']
        self.sent1s = [sent + [PAD_TOKEN]*(maxlength1 - len(sent)) for sent in self.sent1s]
        self.sent2s = [sent + [PAD_TOKEN]*(maxlength2 - len(sent)) for sent in self.sent2s]

        self.sent1s = torch.tensor(self.sent1s).to(DEVICE)
        self.sent2s = torch.tensor(self.sent2s).to(DEVICE)
        self.sims = torch.tensor(self.sims).to(DEVICE)
    
    def __len__(self):
        return len(self.sent1s)
    
    def __getitem__(self, index):
        return (self.sent1s[index], self.sent2s[index], self.sims[index])
        