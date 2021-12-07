# Numeracy enhances the Literacy of Language Models

This repository holds the code and data (Wiki-Convert) for our EMNLP 2021 short paper. We show that magnitude-aware number encoders help language models predict words better, and the results transfer to non-numeric contexts as well. Here are some links to better understand our work:

[Anthology](https://aclanthology.org/2021.emnlp-main.557/) | [PDF](https://aclanthology.org/2021.emnlp-main.557.pdf) | [Slides](https://drive.google.com/file/d/1-GIUOTRLavVzA_ynQ0HqTR_RMq2GezOI/view?usp=sharing) | [Video](https://drive.google.com/file/d/1QluCr79hAHkA_oCwD6JHUBQAQ81rMste/view?usp=sharing) | [Poster](https://drive.google.com/file/d/1DntS8pRlpsRnO3UpYZeo3wzAOJiHLfY1/view?usp=sharing) | [Twitter thread](https://twitter.com/thawani_avijit/status/1434168008046301185) | [ACL21 Reviews](https://drive.google.com/file/d/1IUv9Rk3VqxceP58NyrEENAcr30P0etis/view?usp=sharing) 

Please reach out to me at `thawani@usc.edu` in case you face any issues or just to chat!

## Dataset

**Wiki-Convert**: A novel dataset of Wikipedia sentences annotated with numbers. The easiest way to get the data is via [Huggingface Datasets](https://huggingface.co/docs/datasets/) library. Simply install the datasets library and run: 
```python3
from datasets import load_dataset
ds = load_dataset("usc-isi/WikiConvert")
```

Example:
| id | comment | offset | length | number |
| :--- | --- | :---: | :---: | :---: |
| 0 | With a total of 1500 miles of inland waterways, Alabama has among the most of any state. |16 | 4 |  1500 |

Here, the Wikipedia sentence is provided under the key `comment` and the annotated `number` is provided via its character `offset` and `length`, i.e., `comment[offset:offset+length] = number`. You will find additional keys `UNIQUE_STORY_INDEX` and `magnitude` which are irrelevant and were simply added for consistency with the format of the [Numeracy600K](https://github.com/aistairc/Numeracy-600K) dataset.

Note that when loading from the Datasets library, numbers larger than `sys.maxsize` will be capped to avoid an overflow in PyArrow. For the uncapped version, you may download the json files directly for the [train](https://huggingface.co/datasets/usc-isi/WikiConvert/resolve/main/train_wiki.json), [dev](https://huggingface.co/datasets/usc-isi/WikiConvert/resolve/main/train_wiki.json), and [test](https://huggingface.co/datasets/usc-isi/WikiConvert/resolve/main/train_wiki.json) splits.

The dataset sizes are as follows:

|  | Train | Dev | Test |
| --- | ---: | ---: | ---: |
| # examples | 739583 | 92447 | 92449 |
| file size (MBs) | 169 | 20.9 | 20.5 |

If you prefer the `NUM` and `UNIT` annotations as described in the paper, [here](https://drive.google.com/file/d/1Jqtv70cwk6yZsMosHjKTbIm0snHTovkD/view?usp=sharing) is a 233 MB json file. You may also retrieve a larger, unprocessed version of the data at [this link](https://drive.google.com/drive/folders/1FINtp5yC8J-ObLZ8p1Q0Oij9ttav1w91?usp=sharing).

## Code

**train.py:** model description and training
```bash
nice python train.py --batch-size 256 --gpus 0, --tsamples 100_000 --dsamples 10_000 --max_epochs 10 --enc exp --hidden 200 --accumulate_grad_batches 4 --seed 0 --dataset WC
```
**eval.py:** reports perplexity and hit@k scores
```bash
nice python eval.py --limit 10_000 --ckpt checkpoints/read-WC-def-adj-noun/epoch=9.ckpt --maxtoks 150 --batch-size 128 --device 0
```
**dataset.py:** tokenized dataset description

**valids/common...txt:** list of sentence indices to evaluate

## Citation

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
