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

[Anthology](https://aclanthology.org/2021.emnlp-main.557/) | [PDF](https://aclanthology.org/2021.emnlp-main.557.pdf) | [Slides](https://drive.google.com/file/d/1-GIUOTRLavVzA_ynQ0HqTR_RMq2GezOI/view?usp=sharing) | [Video](https://drive.google.com/file/d/1QluCr79hAHkA_oCwD6JHUBQAQ81rMste/view?usp=sharing) | [Poster](https://drive.google.com/file/d/1DntS8pRlpsRnO3UpYZeo3wzAOJiHLfY1/view?usp=sharing) | [Thread](https://twitter.com/thawani_avijit/status/1434168008046301185) | [Code](https://github.com/avi-jit/numeracy-literacy) | [ACL21 Reviews](https://drive.google.com/file/d/1IUv9Rk3VqxceP58NyrEENAcr30P0etis/view?usp=sharing) 

Here's how to cite us for the results or the Wiki-Convert dataset:
```
@inproceedings{thawani-etal-2021-numeracy,
    title = "Numeracy enhances the Literacy of Language Models",
    author = "Thawani, Avijit  and
      Pujara, Jay  and
      Ilievski, Filip",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.557",
    pages = "6960--6967",
    abstract = "Specialized number representations in NLP have shown improvements on numerical reasoning tasks like arithmetic word problems and masked number prediction. But humans also use numeracy to make better sense of world concepts, e.g., you can seat 5 people in your {`}room{'} but not 500. Does a better grasp of numbers improve a model{'}s understanding of other concepts and words? This paper studies the effect of using six different number encoders on the task of masked word prediction (MWP), as a proxy for evaluating literacy. To support this investigation, we develop Wiki-Convert, a 900,000 sentence dataset annotated with numbers and units, to avoid conflating nominal and ordinal number occurrences. We find a significant improvement in MWP for sentences containing numbers, that exponent embeddings are the best number encoders, yielding over 2 points jump in prediction accuracy over a BERT baseline, and that these enhanced literacy skills also generalize to contexts without annotated numbers. We release all code at https://git.io/JuZXn.",
}
```
