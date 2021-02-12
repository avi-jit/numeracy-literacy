# Numeracy Enhances the Literacy of Language Models
Anonymous ACL-IJCNLP 2021 Submission

**dataset.py:** tokenized dataset description
**train.py:** model description and training
**eval.py:** reports perplexity and hit@k scores
**common...txt:** list of sentence indices to evaluate

```
nice python train.py --batch-size 256 --gpus 0, --tsamples 100_000 --dsamples 10_000 --max_epochs 10 --enc exp --hidden 200 --accumulate_grad_batches 4 --seed 0 --dataset WC
```

```
nice python eval.py --limit 10_000 --ckpt checkpoints/read-WC-def-adj-noun/epoch=9.ckpt --maxtoks 150 --batch-size 128 --device 0
```