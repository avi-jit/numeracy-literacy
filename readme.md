# Numeracy Enhances the Literacy of Language Models
Accepted to EMNLP 2021.

# Wiki-Convert

A novel dataset of Wikipedia sentences annotated with numbers. Get the data at [this link](https://drive.google.com/drive/folders/1FINtp5yC8J-ObLZ8p1Q0Oij9ttav1w91?usp=sharing).

**train.py:** model description and training
```
nice python train.py --batch-size 256 --gpus 0, --tsamples 100_000 --dsamples 10_000 --max_epochs 10 --enc exp --hidden 200 --accumulate_grad_batches 4 --seed 0 --dataset WC
```
**eval.py:** reports perplexity and hit@k scores
```
nice python eval.py --limit 10_000 --ckpt checkpoints/read-WC-def-adj-noun/epoch=9.ckpt --maxtoks 150 --batch-size 128 --device 0
```
**dataset.py:** tokenized dataset description

**valids/common...txt:** list of sentence indices to evaluate
