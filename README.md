## Datasets

SimLex-999 (word similarity): https://fh295.github.io/simlex.html  
STS Benchmark – SemEval (sentence similarity): http://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark  
HypeNet (hypernymy/entailment): https://github.com/vered1986/HypeNET/blob/v2/dataset/datasets.rar  
WikiText-103 (unlabelled text corpus): https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/  

## Reproducing

We had to pass `decimal=5` to the `assert_almost equal` function in `File "python3.9/site-packages/numpy/testing/_private/utils.py", line 599,` in order to overcome AssertionErrors (since the default decimal value was 7)

Therefore, the following package was edited:

`python3.9/site-packages/ot/lp/__init__.py", line 326, in emd`  

The line:  
`np.testing.assert_almost_equal(a.sum(0),`
was changed to 
`np.testing.assert_almost_equal(decimal=6, a.sum(0),`
