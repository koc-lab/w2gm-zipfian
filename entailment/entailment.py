import numpy as np
import re
import argparse

def cosine_sim(x, y):
    return float(np.sum(x*y))/float(np.sqrt(np.sum(x**2)*np.sum(y**2)))

def get_cos_scores(vectors, vocab, filename, ignore_words=None):
    total = 0
    seen = 0
    res = []
    words = []
    
    total_word_count = 0
    for key in vocab:
        total_word_count += vocab[key]
        
    with open(filename) as f:
        for sentence in f:
            f_word, s_word, entail = re.split(r'\t+', sentence)
            entail = entail.strip()
            entail = 1 if entail == "True" else 0
            if f_word in vectors and s_word in vectors:
                # skip the words which are not of the interest
                if ignore_words is not None and f_word not in ignore_words:
                    continue
                seen += 1
                # compute scores for each observed pair
                f_vec = vectors[f_word]
                s_vec = vectors[s_word]
                freq_score = vocab[s_word]/total_word_count*100 + np.log10(vocab[s_word]/vocab[f_word])/40
                score = cosine_sim(f_vec, s_vec) + freq_score
                res.append([score, entail])
                words.append([f_word, s_word])

            total += 1
        res = np.array(res)
    return res, seen, total, np.array(words)

def read_vectors_to_dict(vectors_file, vocab_file, vocab=None, header=False):
    vector_dict = {}
    with open(vectors_file) as f:
            for i, sentence in enumerate(f):
                if header and i==0:
                    continue

                parts = sentence.strip().split(" ")
                word = parts[0]

                vector = np.array(parts[1:], dtype="float32")

                vector_dict[word] = vector
    vocab_dict = {}
    with open(vocab_file) as f:
            for i, sentence in enumerate(f):
                if header and i==0:
                    continue

                parts = sentence.strip().split(" ")
                word = parts[0]

                freq = int(parts[1])

                vocab_dict[word] = freq
                
    return vector_dict, vocab_dict

def test_entailment(vectors_path, vocab_path, test_path, plot=True):
    # read test files
    vectors, vocab = read_vectors_to_dict(vectors_path, vocab_path)
    filenames = [test_path]  # that means there is only one file
    
    for filename in filenames:
            scores_and_ent, seen, total, words = get_cos_scores(vectors, vocab, filename)


            opt_th, pred, PR, R, F1 = __fit_th(scores_and_ent)
            print(" ---------------------------------------------------- ")
            print(" ------------- NON-DIRECTIONAL ENTAILMENT TEST -------------")
            print("------------- %s -------------" % filename)
            print("pos/neg ratio: %f" % (float(np.sum(scores_and_ent[:, 1] == 1))/float(np.sum(scores_and_ent[:,1]==0))))
            print("PR is: %f, R is: %f, F1 is: %f " %(PR, R, F1))
            print("seen %f %% " % ((float(seen)/float(total)) * 100.0))
            print("optimal th is %f " % opt_th)
    

# fits the optimal threshold based on F1 score
def __fit_th(scores_and_ent, max_iter=100000):
    step = (max(scores_and_ent[:, 0]) - min(scores_and_ent[:, 0]))/float(max_iter)
    th = min(scores_and_ent[:, 0])
    ths = []
    f1s = []
    for i in range(max_iter):
        pred = scores_and_ent[:, 0] > th
        _, _ , f1 = __compute_PR_R_F1(scores_and_ent, pred)
        ths.append(th)
        f1s.append(f1)
        th += step
    max_f1_idx = f1s.index(max(f1s))
    opt_th = ths[max_f1_idx]
    # TODO avoid extra call
    pred = scores_and_ent[:, 0] > opt_th
    PR, R, F1 = __compute_PR_R_F1(scores_and_ent, pred)
    return opt_th, pred, PR, R, F1

def __compute_PR_R_F1(scores_and_ent, pred):
    TP = np.sum((pred == 1) * (scores_and_ent[:, 1] == 1))
    TN = np.sum((pred == 0) * (scores_and_ent[:, 1] == 0))
    FP = np.sum((pred == 1) * (scores_and_ent[:, 1] == 0))
    FN = np.sum((pred == 0) * (scores_and_ent[:, 1] == 1))
    if TP == 0:
        return 0, 0, 0
    PR = float(TP)/float(TP+FP)
    R = float(TP) / float(TP + FN)
    F1 = (2*PR*R)/(PR+R)
    return PR, R, F1

parser = argparse.ArgumentParser()
parser.add_argument('--data',  required=True, type=str)
parser.add_argument('--vectors',  required=True, type=str)
parser.add_argument('--vocab',  required=True, type=str)
args = parser.parse_args()


test_entailment(args.vectors,args.vocab, args.data)
