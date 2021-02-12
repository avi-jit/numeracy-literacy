import dataset
import train

import transformers
import numpy as np
import torch
import pytorch_lightning as pl
from randw import words
from tqdm.auto import tqdm

import importlib
import csv
from collections import Counter, defaultdict
import argparse
import os
import json
from time import time
import random
import logging

torch.manual_seed(42)
pl.seed_everything(42)
np.random.seed(int(time()%1e5))
random.seed(int(time()%1e5))

class EvalDataset(torch.utils.data.Dataset): # similar to dataset.ReadDataset
    def __init__(self, dataset='../../pycharm/data/wiki.json', maxtoks=30, maxlen=150, minlen=15, numlen=2, range_=(1e-4,1e+6), sample=None, tok='bert-base-uncased', offset=0,typ='json', numtok='def', filter_range=True, valid_idxs=[], debug=False):
        self.tokenizer, self.mask_token = self._get_tokenizer(tok)
        self.mask_token = '[NUM]'
        self.tokenizer.add_special_tokens({'additional_special_tokens':["[NUM]"]})
        self.numlen = numlen
        self.numtok = numtok
        if '600' in dataset:
            add1 = 1
            add_ = ' '
        else:
            add1 = 0
            add_ = ''
        raw = json.load(open(dataset))[:sample]
        temp = []
        for row in raw:
            if (range_[1] > float(row['number'])> range_[0]) and (maxlen > len(row['comment']) > minlen):
                temp.append(row)
        raw = temp
        
        if self.numtok == 'def':
            _number_encoder = self.get_string
        elif self.numtok in ['val','lval','bin','lbin','num']: # keep numbers, numpos, replace by mask.
            _number_encoder = lambda x: self.mask_token + ' '
        elif self.numtok == 'none':
            _number_encoder = lambda x: '' # remove numbers entirely
        
        texts = []
        for r in raw:
            _number_encoded = _number_encoder(float(r['number']))
            texts.append((r['comment'][:r['offset']] + _number_encoded + add_ + r['comment'][r['offset'] + add1 + r['length']:]).replace("   "," ").replace("  "," "))
            
        encs = self.tokenizer.batch_encode_plus(texts, padding='max_length', 
                                                truncation=True, max_length=maxtoks)
        self.data = []
        print("raw:",len(raw))
        print("valids:",len(valid_idxs))
        if not debug:
            tqdm = lambda x: x
        for j,(text,row) in tqdm(enumerate(zip(texts, raw))):
            if row['id'] not in valid_idxs:
                continue
            if self.numtok == 'none':
                numpos = 0
            else:
                numpos = encs.char_to_token(j, row['offset'])
                if not numpos or numpos == -1: # None if space / out of range
                    continue
            NUM = float(row['number'])
            if self.numtok in ['val','lval','lbin']:
                if encs['input_ids'][j].index(30522) != numpos:
                    continue
            
            # now masking one token at a time
            might_be_num = True
            i = encs['input_ids'][j]
            p = numpos
            if self.numtok in ['lbin','lval','bin','val','num']:
                i[p] = self.tokenizer.mask_token_id 
            a = encs['attention_mask'][j]
            for idx,(i_,a_) in enumerate(list(zip(i,a))[1:sum(a)-1]):
                if idx+1 == p and self.numtok not in ['none']: # ignoring all after numpos!
                    continue
                token = self.tokenizer.decode([i_])
                if token == '.':
                    continue
                try:
                    int(token.strip('#'))
                    continue
                except:
                    pass
                
                i_mask = i.copy()
                i_mask[idx+1] = self.tokenizer.mask_token_id
                i_true = [-100]*len(i)
                i_true[idx+1] = i[idx+1]
                
                self.data.append((row['id'], i_mask, a, NUM, numpos, i_true, idx+1))
            
    def __getitem__(self, idx):
        idx_, i_mask, a, n, p, i, pos = self.data[idx]
        return torch.tensor(idx_), torch.tensor(i_mask), torch.tensor(a), torch.tensor(n), torch.tensor(p), torch.tensor(i), torch.tensor(pos)
        
    def __len__(self):
        return len(self.data)
    
    def _get_tokenizer(self, model_name):
        if model_name[:5] == 'bert-':
            return transformers.BertTokenizerFast.from_pretrained(model_name), '[MASK]'
        elif model_name[:8] == 'roberta-':
            return transformers.RobertaTokenizerFast.from_pretrained(model_name), '<MASK>'
        else:
            print("Tokenizer not recognized")
            raise NotImplementedError

    def get_string(self, num): # num is a float
        if num > 1.0 and round(num, self.numlen) == int(num):
            num = int(num)
        else:
            num = round(num, self.numlen)
        return str(num)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', help='checkpoint to load', default='', type=str)
    parser.add_argument('--hits', type=int, nargs='*', default=[1,5,20,100])
    parser.add_argument('--maxtoks', type=int, help='max number of tokens', default=30)
    parser.add_argument('--debug', default=False, action='store_true', help='Debug?')
    parser.add_argument('--limit', type=int, help='limit eval (10)', default=10)
    parser.add_argument('--device', help='cpu* or 0/1/2/3', default='cpu', type=str)
    parser.add_argument('--batch-size', type=int, help='batch size', default=64)
    parser.add_argument('--nworks', type=int, help='Number of dataloader workers (default:1)', default=50)
    args = parser.parse_args()
    
    typ = 'json'
    if '600' in args.ckpt:
        testfile = 'data/test600.json'
        valids = [int(l.strip('\n')) for l in open('common600_8223.txt').readlines()]
    else:
        testfile = 'data/test_wiki.json'
        valids = [int(l.strip('\n')) for l in open('commonWikiConvert_8600.txt').readlines()]
    # we provide text files of common sentence indices to evaluate models on, for comparable results.
    
    if args.device in ['0','1','2','3']:
        device = 'cuda:' + args.device
    else:
        device = 'cpu'
    net = train.Reader.load_from_checkpoint(checkpoint_path=args.ckpt)
    net = net.eval()
    net = net.to(device)

    edata = EvalDataset(numtok=net.hparams.enc, sample=args.limit, tok=net.hparams.base, dataset=testfile, typ=typ, range_=[1e-4,1e6], maxtoks=args.maxtoks, maxlen=150, minlen=15, valid_idxs=valids, debug=args.debug)
    if args.debug:
        print(edata[0])
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    eloader = torch.utils.data.DataLoader(edata, batch_size=args.batch_size, num_workers=args.nworks)
    if args.debug:
        print(len(edata), len(eloader))
    hits = {k:0 for k in args.hits}
    nlls = []
    if not args.debug:
        tqdm = lambda x: x
    for idx, i_mask, a, n, p, i, pos in tqdm(iter(eloader)):
        output = net(i_mask.to(device), a.to(device), p.to(device), n.to(device), i.to(device))
        l = output.loss.item()
        nlls.append(l)
        i_ = i[torch.arange(i.shape[0]), pos].to(device) # (B)
        topk = torch.topk(output.logits[torch.arange(i.shape[0]), pos], dim=-1, k=max(hits.keys())).indices
        for k in hits.keys():
            anys = torch.any(i_.unsqueeze(-1) == topk[torch.arange(i.shape[0]), :k],dim=-1)
            hits[k] += torch.mean(anys.float()).item() 
        if args.debug:
            print(round(l,3), anys)
        
    ppl = round(2 ** (sum(nlls) / len(nlls)), 3)
    hits = {k:round(v*100/len(nlls),3) for k,v in hits.items()}
    print(str(args.maxtoks)+' '+args.ckpt+'\t'+str(len(edata))+'\t'+str(ppl)+'\t'+str(hits)+'\n')
    
if __name__ == "__main__":
    main()
    
# nice python eval.py --limit 10_000 --ckpt checkpoints/read-WC-def-adj-noun/epoch=9.ckpt --maxtoks 150 --batch-size 128 --device 0
