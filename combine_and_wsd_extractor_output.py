import codecs
import time
import os
import nltk
from nltk.wsd import lesk
from nltk.corpus import wordnet
import re

def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None

wordnet_tag_dict = {"J":wordnet.ADJ, "V":wordnet.VERB, "N":wordnet.NOUN, "R":wordnet.ADV}

out_file = codecs.open("path_to_output","w","UTF-8")

source_path = "path_to_source"
folders = sorted(os.listdir(source_path))
file_count = 0

t = time.time()

token_count = 0
skip = False
space = False
for folder_no, folder in enumerate(folders):
    files = sorted(os.listdir(source_path + "/" + folder))
    for file in files:
        with codecs.open(source_path + "/" + folder + "/" + file) as f:
            lines = f.read().splitlines()
            for line in lines:
                length = len(line)
                if skip: # skips only a single line afer setting skip = True
                    skip = False
                elif line[0:4] == "<doc":
                    skip = True  
                elif not (line == "</doc>" or line == ""):
                    sentences = line.split(". ")
                    for sent in sentences:
                        
                        # lesk algorithm
                        to_be_wsd = re.sub(r'\W+', ' ', sent).lower()
                        nltk_tagged = nltk.pos_tag(nltk.word_tokenize(to_be_wsd))
                        for tup in nltk_tagged:
                            try:
                                syn = lesk(to_be_wsd.split(), tup[0], wordnet_tag_dict[tup[1][0]])
                            except:
                                syn = lesk(to_be_wsd.split(), tup[0])
                                
                            if syn is not None:
                                out_file.write(syn.name().replace(".","_"))
                                out_file.write(" ")
                                
                            else:
                                out_file.write(tup[0]+"_x_01")
                                out_file.write(" ")
                                
                            token_count += 1
                            
                   
                            
    elapsed = time.time() - t
    print(str(folder_no+1)) Folder: " + folder + " Tokens Processed: " + str(token_count) + " Elapsed Time: " + str(int(elapsed/3600)) + "h " + str(int((elapsed % 3600)/60)) + "m")

out_file.close()
elapsed = time.time() - t
print("\nAll extractor outputs are processed and combined in " + str(int(elapsed/3600)) + "h " + str(int((elapsed % 3600)/60)) + "m")
print("Token Count: " + str(token_count))
