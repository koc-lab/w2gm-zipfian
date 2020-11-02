# Zipfian Regularities in “Non-point” Word Representations

This repository is to observe Zipfian regularities existing in the Gaussian mixture embeddings [1].

## Observing the Word Variances

We split the Gaussian models into original and additional models. As an original model we utilized the model released original implementation [1]. To observe Zipfian regularities on the original multimodal implementation:

```
python analyze_word_variances_original.py --model_path original_models/w2gm-k2-d50
```
At the end of the running, average variances of words will be displayed. In addition behaviors of the Swadesh and numbers words will be also demonstrated.
For addtional experiments (e.g. experiment results on sense controlled corpus or different language corpus), please run:

```
python analyze_word_variances_additional.py --model_path additional_models/model_folder_name_to_be_used
```
Again, at the end of the implementation, behavior of variances will be shown with correlation results. 

In order to train from scratch, one can utilize the implementation from [here.](https://github.com/benathi/word2gm)

## Requirements for Evaluation
* tensorflow (We used tensorflow 1.15, but tensowflow 1.0 and above would be compatible.)
* numpy
* scipy
* matplotlib

## References

[1] Athiwaratkun, B., & Wilson, A. (2017). Multimodal  word  distributions. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long  Papers)(pp. 1645–1656).
