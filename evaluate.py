import torch
from baseline import BATCH_SIZE, DEVICE, SentenceSimilarityBaseline, WordSimilarityBaseline
from data import STSDataset, WordSimilarityDataset, UnsupervisedDataset
from tqdm import tqdm
from torchmetrics import SpearmanCorrCoef, PearsonCorrCoef
from torch.utils.data import DataLoader

from model import Histograms, WordOptimalTransport, SentenceOptimalTransport

def normalise(xs, reverse):
    M = torch.max(xs)
    m = torch.min(xs)
    xs = (M-xs)/(M-m) if reverse else (xs-m)/(M-m)
    return xs

def evaluate(dataloader, model, metric, task="", reverse=False):
        """
        Function to get the word similarity baseline for a dataset by creating the DataLoader
        Input:
            dataloader of the dataset
            model is an instance of any class which has a getSimilarity method (eg:WordSimilarityBaseline)
        Output:
            The spearman/pearson correlation coefficient of the ground truth with the model's output
        """
        outputs = []
        truth = []
        for batch in tqdm(dataloader, desc=f"Evaluate {task}"):
            similarity_baseline = model.getSimilarity(batch)
            outputs.append(similarity_baseline)
            truth.append(batch[2])
        outputs = torch.concat(outputs, dim=0)
        truth = torch.concat(truth, dim=0)

        outputs = normalise(outputs, reverse)
        truth = normalise(truth, reverse)

        score = metric().to(DEVICE)
        return outputs, truth, score(outputs, truth)

usd = UnsupervisedDataset(split='valid', min_freq=5)

wsd = WordSimilarityDataset(vocab=usd.vocab)
wsd_dl = DataLoader(wsd,BATCH_SIZE)
wsb = WordSimilarityBaseline(vocab=usd.vocab, glove_dim=50)
gow, gtw, glove_performance_wsd = evaluate(wsd_dl, wsb, SpearmanCorrCoef,"glove word")

ssd = STSDataset(vocab=usd.vocab)
ssd_dl = DataLoader(ssd, BATCH_SIZE)
ssb = SentenceSimilarityBaseline(vocab=usd.vocab, glove_dim=50)
gos, gts, glove_performance_ssd = evaluate(ssd_dl, ssb, PearsonCorrCoef,"glove sentence")

hg = Histograms(usd, glove_dim=50, k=200)

wot = WordOptimalTransport(hg)
sot = SentenceOptimalTransport(hg)

mow, mtw, model_performance_wsd = evaluate(wsd_dl, wot, SpearmanCorrCoef, "model word", reverse=True)
mos, mts, model_performance_ssd = evaluate(ssd_dl, sot, PearsonCorrCoef, "model sentence", reverse=True)

print(model_performance_wsd)
print(model_performance_ssd)
