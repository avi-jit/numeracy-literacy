# Numeracy enhances the Literacy of Language Models

## Wiki-Convert

A novel dataset of Wikipedia sentences annotated with numbers. Get the data at [this link](https://drive.google.com/drive/folders/1FINtp5yC8J-ObLZ8p1Q0Oij9ttav1w91?usp=sharing).

Example:
| Sentence | Number | Unit |
| --- | --- | --- |
| U-559 had a displacement of `NUM` `UNIT` while submerged | 871.0 | tonne |

## Code

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

## Citation

This paper has been accepted to EMNLP 2021 - we will share the citation and link to Anthology soon. In the meantime, [here's a pdf version](https://github.com/avi-jit/numeracy-literacy/blob/main/camera_ready.pdf) of the same.

