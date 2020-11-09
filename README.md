# Zipfian Regularities in “Non-point” Word Representations

This repository is to observe Zipfian regularities existing in the Gaussian mixture embeddings [1].

## Observing the Word Variances

We split the Gaussian models into original and additional models. As an original model we utilized the model released original implementation [1]. Model file can be downloaded [here](https://drive.google.com/file/d/1BlNxIp8yQ-vG4zxWo_kjrAN2om-1gW4a/view?usp=sharing). To observe Zipfian regularities on the original multimodal implementation:

```
python analyze_word_variances_original.py --model_path original_models/w2gm-k2-d50
```
At the end of the running, average variances of words will be displayed. In addition behaviors of the Swadesh and numbers words will be also demonstrated.
For addtional experiments (e.g. experiment results on sense controlled corpus or different language corpus), please find the model files [here](https://drive.google.com/file/d/1d0HiEJWXCKwu5tv_hbVq-hnocMzPzBY2/view?usp=sharing) and run:

```
python analyze_word_variances_additional.py --model_path additional_models/model_folder_name_to_be_used
```
Again, at the end of the implementation, behavior of variances will be shown with correlation results. 

In order to train from scratch, one can utilize the implementation from [here](https://github.com/benathi/word2gm) or you can run word2gm_trainer.py 

```
python word2gm_trainer.py --num_mixtures 2 --train_data your_data_file.txt --spherical --embedding_size 50 --epochs_to_train 5 --var_scale 0.05 --save_path your_save_path --learning_rate 0.05  --subsample 1e-5 --wout --adagrad --min_count 100 --batch_size 128 --max_to_keep 5 --checkpoint_interval 1000 --window_size 10
```
If necessary you can re-build word2vec_ops.so file by


```
chmod +x compile_skipgram_ops.sh
./compile_skipgram_ops.sh
```

We also added lesk-corpus algorithm. It will easily run when the proper environment is provided.

## Entailment

For entailment task, we utilized the entailment repository from [2]. To run entailment task, plase download [word vectors](https://drive.google.com/file/d/19uKgj4Uc3HiG2GFfNpsxjRYZi1nNLQix/view?usp=sharing) and place them into entailment folder, then run:

```
python entailment.py --data bless_2011_data.tsv --vectors word2vec.txt --vocab vocab.txt
```

## Requirements for Evaluation
* tensorflow (We used tensorflow 1.15, but tensowflow 1.0 and above would be compatible.)
* numpy
* scipy
* matplotlib

## References

[1] Athiwaratkun, B., & Wilson, A. (2017). Multimodal  word  distributions. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long  Papers)(pp. 1645–1656).

[2] Brazinskas, A., Havrylov, S., & Titov, I. (2018). Embedding words as distributions with a Bayesian skip-gram model. In Proceedings of the 27th International Conference on Computational Linguistics (pp. 1775{1789). Santa Fe, New Mexico, USA: Association for Computational Linguistics.
