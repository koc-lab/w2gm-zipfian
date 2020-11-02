from word2gm_loader import Word2GM

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import argparse

def func(x, a, b):
    return b/np.power(x,a)


def read_vocabs(model,vocab_path, separated_vocab_path, swadesh_path, numbers_path):
    
    trained_vocab = dict()
    trained_vocab_lower = dict()
    separated_vocab_words = []
    numbers = []
    swadesh_words = dict()
    
    with open(vocab_path) as f:
        for line_no, line in enumerate(f):
            word = line.split()[0]
            trained_vocab[word[:]] = line_no
            if word.lower() not in trained_vocab_lower:
                trained_vocab_lower[word[:].lower()] = line_no
                
    with open(separated_vocab_path) as f:
        for line in f:
            word = line.split()[0]
            separated_vocab_words.append(word[:])  
    
    separated_word_idx = [trained_vocab_lower[word] for word in separated_vocab_words if word in trained_vocab_lower]
    
    with open(swadesh_path) as f:
        for line in f:
            word = line.split()[0]
            swadesh_words[word] = separated_vocab_words.index(word)
        
    with open(numbers_path) as f:
        for line in f:
            word = line.split()[0]
            numbers.append(word[:])
    
    swadesh_sorted_words = [key for (key,value) in sorted(swadesh_words.items(), key=lambda x: x[1])]
    swadesh_word_idx = [trained_vocab_lower[word] for word in swadesh_sorted_words if word in trained_vocab_lower]

    numbers_idx = model.words_to_idxs(numbers)

    return trained_vocab, trained_vocab_lower, separated_word_idx, swadesh_word_idx, numbers_idx
    

def get_sorted_variances(model, trained_vocab, separated_word_idx, swadesh_word_idx, numbers_idx, mixtures):
    
    num_mixtures = mixtures.shape[1]    
    var_idx, var_pair = model.sort_low_var(list(range(0,(len(trained_vocab))*num_mixtures)))
    var_pair = sorted(var_pair, key=lambda item: item[0])
    
    avg_var = []
    swa_var = []
    num_var = []
    
    for i in separated_word_idx:
        var = 0
        
        for mix in range(num_mixtures):
            var += mixtures[i][mix]*np.exp(var_pair[num_mixtures*i+mix][1])
        
        avg_var.append(var)
    
    for i in swadesh_word_idx:
        var = 0
        
        for mix in range(num_mixtures):
            var += mixtures[i][mix]*np.exp(var_pair[num_mixtures*i+mix][1])
        
        swa_var.append(var) 
    
    for i in numbers_idx:
        var = 0
        
        for mix in range(num_mixtures):
            var += mixtures[i][mix]*np.exp(var_pair[num_mixtures*i+mix][1])
        
        num_var.append(var) 
    
    return avg_var, swa_var, num_var



def plot_results(variances, swa_var, num_var, word_ids, swadesh_word_idx, numbers_idx):
     
    
    ind = np.arange(len(separated_word_idx))

    plt.figure(0)
    popt, pcov = curve_fit(func, ind[:75000]+1, np.asarray(variances)[:75000])
    fig = plt.hexbin(ind[:75000]+1, np.asarray(variances)[:75000], cmap='summer', mincnt=3, gridsize=75, edgecolors='black', bins=100)
    plt.plot(ind[:75000]+1, func(ind[:75000]+1, *popt), 'r', label=r"a = " + str(round(popt[0],2)) + ", b = " + str(round(popt[1],2)))
    cb = plt.colorbar(fig)
    cb.set_label('Word Counts falling into the bin')
    plt.legend()
    
    plt.ylim(ymax = max(variances))
    plt.ylabel(r'Variance', fontsize=14)
    plt.xlabel(r'Rank of Words', fontsize=14)
    plt.title('Average Variances of Multimodals of Given Model', fontsize=12)
    plt.tight_layout()
    
    plt.figure(1)
    swa_popt, swa_pcov = curve_fit(func, np.asarray(swadesh_word_idx), np.asarray(swa_var))
    fig = plt.hexbin(np.asarray(swadesh_word_idx), np.asarray(swa_var), xscale='log', yscale='log', cmap='summer', mincnt=1, gridsize=50, edgecolors='black', bins=10)
    plt.loglog(np.asarray(swadesh_word_idx), func(np.asarray(swadesh_word_idx), *swa_popt), 'r', label=r"a = " + str(round(swa_popt[0],2)) + ", b = " + str(round(swa_popt[1],2)))
    cb = plt.colorbar(fig)
    cb.set_label('Word Counts falling into the bin')
    plt.legend()
    
    plt.ylabel(r'Variance', fontsize=14)
    plt.xlabel(r'Rank of Words', fontsize=14)
    plt.title(r'Average Variances of Swadesh Words', fontsize=16)
    plt.tight_layout()
    
    plt.figure(2)
    num_popt, num_pcov = curve_fit(func, np.asarray([1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19]), np.asarray(num_var))
    fig = plt.hexbin(np.asarray([1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19]), np.asarray(num_var), xscale='log', yscale='log', cmap='summer', mincnt=1, gridsize=50, edgecolors='black')
    plt.loglog(np.asarray([1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19]), func(np.asarray([1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19]), *num_popt), 'r', label=r"a = " + str(round(num_popt[0],2)) + ", b = " + str(round(num_popt[1],2)))
    cb = plt.colorbar(fig)
    cb.set_label('Word Counts falling into the bin')
    plt.legend()
    
    plt.ylim(ymax=max(num_var))
    plt.ylabel(r'Variance', fontsize=14)
    plt.xlabel(r'Cardinality of Numbers', fontsize=14)
    plt.title(r'Average Variances of Number Words', fontsize=16)
    plt.tight_layout()
    
    plt.show()

parser = argparse.ArgumentParser()
parser.add_argument('--model_path',required=True, type=str)
args = parser.parse_args()


model =  Word2GM(args.model_path)
mixtures = model.mixture

trained_vocab, trained_lower_vocab, separated_word_idx, swadesh_word_idx, numbers_idx = read_vocabs(model,
                                                                     args.model_path+"/vocab.txt", 
                                                                     args.model_path+"/separate_vocab.txt",
                                                                     args.model_path+"/swadesh_eng.txt",
                                                                     args.model_path+"/numbers.txt")

variances, swa_var, num_var = get_sorted_variances(model, trained_vocab, separated_word_idx, swadesh_word_idx, numbers_idx, mixtures)

plot_results(variances, swa_var, num_var, separated_word_idx, swadesh_word_idx, numbers_idx)