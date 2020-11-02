from word2gm_loader import Word2GM

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import argparse

def func(x, a, b):
    return b/np.power(x,a)


def read_vocab(vocab_path):
    
    vocab = dict()
    freq_vocab = dict()
    word_ids = []
    
    with open(vocab_path) as f:
        next(f)
        for line_no, line in enumerate(f):
            word = line.split()[0]
            vocab[word[:]] = line_no
            freq_vocab[word[:]] = int(line.split()[1])
            word_ids.append(line_no)
            

    return vocab, word_ids, freq_vocab
    

def get_sorted_variances(model, vocab, words_ids, mixtures):
    
    num_mixtures = mixtures.shape[1]    
    var_idx, var_pair = model.sort_low_var(list(range(0,(len(vocab)+1)*num_mixtures)))
    var_pair = sorted(var_pair, key=lambda item: item[0])
    var_pair = var_pair[num_mixtures:]
    
    avg_var = []
    for i in words_ids:
        var = 0
        
        for mix in range(num_mixtures):
            var += mixtures[i][mix]*np.exp(var_pair[num_mixtures*i+mix][1])
        
        avg_var.append(var)
    
    return avg_var


def calculate_correlations(variances, freq_vocab):
    
    total_word_count = 0
    for key in freq_vocab:
        total_word_count += freq_vocab[key]

    word_freq = [freq_vocab[key]/total_word_count for key in freq_vocab.keys()]        
    
    spearman = stats.spearmanr(variances,word_freq)
    pearson = stats.pearsonr(variances,word_freq) 
    kendall = stats.kendalltau(variances,word_freq)
    
    return spearman, pearson, kendall

def plot_results(variances, word_ids, swa=False):
     
    plt.figure()
    popt, pcov = curve_fit(func, (np.asarray(word_ids)+1), np.asarray(variances))
    fig = plt.hexbin((np.asarray(word_ids)+1), np.asarray(variances), cmap='summer', mincnt=3, gridsize=75, edgecolors='black', bins=100)
    plt.plot((np.asarray(word_ids)+1), func((np.asarray(word_ids)+1), *popt), 'r', label=r"a = " + str(round(popt[0],2)) + ", b = " + str(round(popt[1],2)))
    cb = plt.colorbar(fig)
    cb.set_label('Word Counts falling into the bin')
    plt.legend()
    
    plt.ylim(ymax = max(variances))
    plt.ylabel(r'Variance', fontsize=14)
    plt.xlabel(r'Rank of Words', fontsize=14)
    plt.title('Average Variances of Multimodals of Given Model', fontsize=12)
    plt.tight_layout()
    
    plt.show()

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', required=True, type=str)
args = parser.parse_args()


model =  Word2GM(args.model_path)
mixtures = model.mixture[1:][:]

vocab, word_ids, freq_vocab = read_vocab(args.model_path+"/vocab.txt")

variances = get_sorted_variances(model, vocab, word_ids, mixtures)

spearman, pearson, kendall = calculate_correlations(variances, freq_vocab)

print("Sperman: r = "+str(spearman[0])+", p = "+str(spearman[1]))
print("Pearson: rho = "+str(pearson[0])+", p = "+str(pearson[1]))
print("Kendall: tau = "+str(kendall[0])+", p = "+str(kendall[1]))

plot_results(variances, word_ids)