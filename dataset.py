import transformers
import torch
from tqdm import tqdm
import numpy as np
import json
from time import time
import csv

class ReadDataset(torch.utils.data.Dataset):
    def __init__(self, dataset='data/train_wiki.json', maxtoks=50, maxlen=150, minlen=15, numlen=2, range_=(1e-4,1e+6), seed=42, sample=None, tok='bert-base-uncased', offset=0, typ='json', numtok='def', filter_range=True, valids=None):
        np.random.seed(seed)
        self.tokenizer, self.mask_token = self._get_tokenizer(tok)
        self.mask_token = '[NUM]'
        self.tokenizer.add_special_tokens({'additional_special_tokens':["[NUM]"]})
        self.numlen = numlen
        self.maxtoks = maxtoks
        self.maxlen = maxlen
        self.minlen = minlen
        self.numtok = numtok # method: def, val, lval, exp, none, num
        self.range = range_
        self.valids = valids
        if '600' in dataset: # accounting for mismatch in how dataset files handle offsets
            self.add1 = 1
            self.add_ = ' '
        else:
            self.add1 = 0
            self.add_ = ''
        
        if typ == 'json':
            self.raw = json.load(open(dataset))[:sample]
        else:
            raise NotImplementedError
                
        self.opts_to_idx = self.tokenizer.get_vocab()
        if filter_range:
            self._filter_range()
        self._init_dataset()

    def __getitem__(self, idx):
        idx, text_new, true_ids, mask_attn, num, numpos = self.data[idx]
        return torch.tensor(idx), torch.tensor(true_ids), torch.tensor(mask_attn), torch.tensor(num), torch.tensor(numpos), text_new
        
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
    
    def _filter_range(self):
        temp = []
        for row in tqdm(self.raw):
            if (self.range[1] > float(row['number']) > self.range[0]) and (self.maxlen > len(row['comment']) > self.minlen):
                temp.append(row)
        self.raw = temp

    def _init_dataset(self):
        # Setting _number_encoder (how number appears in text)
        if self.numtok == 'def':
            _number_encoder = self.get_string
        elif self.numtok in ['val','lval','exp','num']: # keep numbers, numpos, replace by mask.
            _number_encoder = lambda x: self.mask_token + ' '
        elif self.numtok == 'none':
            _number_encoder = lambda x: '' # remove numbers entirely
        
        texts = []
        for r in self.raw:
            _number_encoded = _number_encoder(float(r['number']))
            texts.append((r['comment'][:r['offset']] + _number_encoded + self.add_ + r['comment'][r['offset'] + self.add1 + r['length']:]).replace("   "," ").replace("  "," "))
            
        encs = self.tokenizer.batch_encode_plus(texts, padding='max_length', truncation=True, max_length=self.maxtoks)
        
        self.data = []
        for i,(text,row) in tqdm(enumerate(zip(texts, self.raw))):
            if self.numtok == 'none':
                if row['id'] not in self.valids:
                    continue
                numpos = 0
            else:
                numpos = encs.char_to_token(i, row['offset']) # numpos is single idx
                if not numpos or numpos == -1: # None if space / out of range
                    continue
            NUM = float(row['number']) # may be used by Reader
            if self.numtok in ['val','lval','exp','num']:
                if encs['input_ids'][i].index(30522) != numpos:
                    continue
            self.data.append((row['id'], text, encs['input_ids'][i], encs['attention_mask'][i], NUM, numpos))
            
