from dataset import ReadDataset

import transformers
import numpy as np
import torch
import pytorch_lightning as pl
from randw import words
from tqdm.auto import tqdm

import csv
import argparse
import os
from time import time
import random

np.random.seed(int(time()%1e5))
random.seed(int(time()%1e5)) # used by randw

class Reader(pl.LightningModule):
    def __init__(self, base='bert-base-uncased', unfreeze=False, hidden=75, lr=1e-3, range_=[1e-4,1e+6], enc='def', mask_token_id=103, name='', topk=[1,5,20,100], maskprob=0.85, invalids=None, *args, **kwargs):
        super().__init__()
        try:
            self.invalids = torch.tensor(invalids).unsqueeze(0).unsqueeze(0)
        except:
            self.invalids = None # will throw an error if we retrain
        invalids = None
        self.save_hyperparameters()
        print('init:', self.hparams.enc)
        self.bertmaskedlm = transformers.BertForMaskedLM.from_pretrained(base)
        if not unfreeze:
            for param in self.bertmaskedlm.base_model.parameters():
                param.requires_grad = False
        self.lr = lr
        if enc in ['def','none']:
            self.encoder = self._default_encoder
        elif enc[-3:] == 'val':
            self._init_value_encoder()
        elif enc == 'exp':
            self._init_exp_encoder()
        elif enc == 'num':
            self._init_num_encoder()
        else:
            raise NotImplementedError
    
    def _default_encoder(self, x, *args, **kwargs):
        return x
    
    def _init_num_encoder(self):
        self.fc0 = torch.nn.Linear(1, self.bertmaskedlm.config.hidden_size, bias=False) # lookup
        self.encoder = self._forward_num_encoder

    def _forward_num_encoder(self, embeds, pos_nums, nums):
        nums = nums.unsqueeze(-1).float()
        nums = torch.ones(nums.shape).type_as(nums)
        vec = self.fc0(nums)
        embeds[torch.arange(embeds.shape[0]).type_as(embeds).long(), pos_nums] = vec
        return embeds
    
    def _init_exp_encoder(self):
        self.fc0 = torch.nn.Linear(self.hparams.hidden, self.bertmaskedlm.config.hidden_size)
        self.hparams.range_ = torch.tensor(self.hparams.range_)
        self.log = torch.log10
        gap = (self.log(self.hparams.range_[1]) - self.log(self.hparams.range_[0])) / self.hparams.hidden
        self.bins = self.log(self.hparams.range_[0]) + (torch.arange(0, self.hparams.hidden)+0.5)*gap # log scale
        self.encoder = self._forward_exp_encoder
        
    def _forward_exp_encoder(self, embeds, pos_nums, nums):
        bincoded = torch.argmin(torch.abs(self.bins.type_as(nums) - self.log(nums).unsqueeze(1)), dim=1)
        one_hot = torch.nn.functional.one_hot(bincoded, num_classes=self.hparams.hidden) # (B,H)
        vec = self.fc0(one_hot.float()) # (B,D)
        embeds[torch.arange(embeds.shape[0]).type_as(embeds).long(), pos_nums] = vec
        return embeds
    
    def _init_value_encoder(self):
        if self.hparams.enc[0] == 'l':
            self.loglayer = torch.log10
        else:
            self.loglayer = torch.nn.Identity()
        self.fc0 = torch.nn.Linear(1, self.hparams.hidden)
        self.drop0 = torch.nn.Dropout(p=0.2)  # prob of element to be zeroed, default=0.5
        self.relu0 = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(self.hparams.hidden, self.bertmaskedlm.config.hidden_size)
        self.relu1 = torch.nn.ReLU()
        self.encoder = self._forward_value_encoder

    def _forward_value_encoder(self, embeds, pos_nums, nums):
        nums = nums.unsqueeze(-1)
        nums = self.loglayer(nums.float())
        vec = self.fc0(nums)
        vec = self.drop0(vec)
        vec = self.relu0(vec)
        vec = self.fc1(vec)
        vec = self.relu1(vec)
        embeds[torch.arange(embeds.shape[0]).type_as(embeds).long(), pos_nums] = vec
        return embeds
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
        
    def forward(self, input_ids, attention_mask, pos_nums, nums, labels=None): # (B,L), (B,L), (B), (B)
        embeds = self.bertmaskedlm.base_model.embeddings(input_ids=input_ids) # (B,L,D)
        embeds = self.encoder(embeds, pos_nums, nums)
        return self.bertmaskedlm(inputs_embeds=embeds, attention_mask=attention_mask, labels=labels)
        
    def common_step(self, batch):
        idx, i, a, n, p, t = batch
        valids = a * (1 - torch.sum(self.invalids.type_as(i) == i.unsqueeze(-1), dim=-1)) # (B,L)
        maskpos = ((torch.rand(i.shape) > self.hparams.maskprob).type_as(i) * valids).long() # (B,L) 1s 0s
        if self.hparams.enc not in ['def','proto','none']:
            i[torch.arange(i.shape[0]).type_as(i), p] = self.hparams.mask_token_id # NUM out of bound, so MASK
        i_ = maskpos * self.hparams.mask_token_id + (1-maskpos) * i
        i_true = -100 * (1-maskpos) + maskpos * i
        return maskpos, self(i_, a, p, n, i_true)
    
    def training_step(self, batch, batch_nb):
        _, output = self.common_step(batch)
        result = pl.TrainResult(minimize=output.loss)
        result.log('train_loss', output.loss, prog_bar=True, on_epoch=True, reduce_fx=torch.mean)
        return result
    
    def validation_step(self, batch, batch_nb):
        maskpos, output = self.common_step(batch)
        result = pl.EvalResult(checkpoint_on=output.loss)
        result.log('dev_loss', output.loss, prog_bar=True, on_epoch=True, reduce_fx=torch.mean)
        
        maskpos = maskpos.reshape(-1)
        idx, i, a, n, p, t = batch
        logits = output.logits # (B,L,V)
        B,L,V = logits.shape
        batch_range = torch.arange(B).repeat_interleave(L) # 0,0, ... 1,1, ...
        seq_range = torch.arange(L).repeat(B) # 0,1, .. 0,1, ...
        for k in self.hparams.topk:
            topk = torch.topk(logits, dim=-1, k=k).indices # (B,L,k)
            z = torch.zeros((B,L,V), dtype=torch.long).type_as(topk)
            z.scatter_(2, topk, 1)
            targets = z[batch_range, seq_range, i.reshape(-1)] # select B*L items, gives (B,L) 1s 0s
            targets *= maskpos # only those that were masked
            acc = torch.sum(targets) / torch.sum(maskpos) # what if none were masked?
            result.log(f'dev_hit@{k}', acc, prog_bar=True, on_epoch=True, reduce_fx=torch.mean)
        return result
    
    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        return tqdm_dict
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', help='wandb and checkpoints (default:)', default='', type=str)
    parser.add_argument('--tok', help='tokenizer', default='bert-base-uncased', type=str)
    parser.add_argument('--enc', help='[val/lval/drnn/dcnn/digit/dice/exp/def*]', default='def', type=str)
    parser.add_argument('--dataset', type=str, help='40 or 600 or WC (default: WC)', default='WC')
    parser.add_argument('--tsamples', type=int, help='Number of samples (default: 100)', default=100)
    parser.add_argument('--dsamples', type=int, help='Number of samples (default: 100)', default=100)
    parser.add_argument('--maxtoks', type=int, help='Maximum number of tokens per sentence (default: 30)', default=30)
    parser.add_argument('--maxlen', type=int, help='Maximum number of chars per sentence (default: 150)', default=150)
    parser.add_argument('--minlen', type=int, help='Minimum number of chars per sentence (default: 15)', default=15)
    parser.add_argument('--split', type=float, nargs=3, help='Train:Dev:Test (def 0.8:0.1:0.1)', default=[.8,.1,.1])
    parser.add_argument('--range', type=float, nargs=2, help='Range (def [1e-4,1e+6])',default=[1e-4, 1e+6])
    parser.add_argument('--dont-train', default=False, action='store_true', help='Don\'t Train?')
    parser.add_argument('--seed', type=int, help='Seed for pytorch (default: 42)', default=42)
    parser.add_argument('--hidden', type=int, help='Hidden neurons or # bins (default: 200)', default=200)
    parser.add_argument('--maskprob', type=float, help='1 - masking probability (default: 0.85)', default=0.85)
    parser.add_argument('--batch-size', type=int, help='Training Batch Size (default: 256)', default=256)
    parser.add_argument('--lr', type=float, help='Learning Rate (default: 1e-3)', default=1e-3)
    parser.add_argument('--debug', default=False, action='store_true', help='Debug i.e. no wandb logging?')
    parser.add_argument('--unfreeze', default=False, action='store_true', help='Unfreeze BERT weights')
    parser.add_argument('--nworks', type=int, help='Number of dataloader workers (default:50)', default=50)
    
    parser = pl.Trainer.add_argparse_args(parser)
    # max_epochs, gpus, val_check_interval, accumulate_grad_batches, gradient_clip_val, distributed_backend, 
    
    args = parser.parse_args()
    if args.name == '':
        if args.enc not in ['def','none','num']:
            temp = str(args.hidden)
        else:
            temp=''
        args.name = 'read-' + str(args.seed) + '-' + args.dataset + '-' + args.enc + temp
        if args.unfreeze:
            args.name = '*'+args.name
        
    os.environ["TOKENIZERS_PARALLELISM"]="false"
    typ = 'json'
    if args.dataset == 'WC': # Wiki-Convert
        tfile = 'data/train_wiki.json' 
        # we've provided .tsv version of Wiki-Convert due to softconf upload limits
        dfile = 'data/dev_wiki.json'
    elif args.dataset == '40': # Wiki40b
        tfile = 'wiki40b/train100k_1k.tsv'
        dfile = 'wiki40b/val1k_1k.tsv'
        typ = 'tsv'
        raise NotImplementedError
    elif args.dataset == '600': # Numeracy600k
        tfile = 'train600.json'
        dfile = 'data/dev600.json'
    
    torch.manual_seed(args.seed)
    pl.seed_everything(args.seed)
    
    if args.enc == 'none': # none gets a larger dataset, hence must be restricted to use the same as others.
        ttemp = ReadDataset(numtok='exp', sample=args.tsamples, dataset=tfile, tok=args.tok, range_=args.range, 
                        maxtoks=args.maxtoks, maxlen=args.maxlen, minlen=args.minlen, typ=typ)
        tvalids = [i[0].item() for i in ttemp]
        dtemp = ReadDataset(numtok='exp', sample=args.tsamples, dataset=dfile, tok=args.tok, range_=args.range, 
                        maxtoks=args.maxtoks, maxlen=args.maxlen, minlen=args.minlen, typ=typ)
        dvalids = [i[0].item() for i in dtemp]
    else:
        tvalids = None
        dvalids = None
    tdata = ReadDataset(numtok=args.enc, sample=args.tsamples, dataset=tfile, tok=args.tok, range_=args.range, 
                        maxtoks=args.maxtoks, maxlen=args.maxlen, minlen=args.minlen, typ=typ, valids=tvalids)
    ddata = ReadDataset(numtok=args.enc, sample=args.dsamples, dataset=dfile, tok=args.tok, range_=args.range,
                        maxtoks=args.maxtoks, maxlen=args.maxlen, minlen=args.minlen, typ=typ, valids=dvalids)
    
    print(tdata[0]) # logging
    
    # filter out numbers to avoid masking them
    invalids = [len(tdata.tokenizer.vocab)]
    for k,v in tdata.tokenizer.vocab.items():
        try:
            int(k.strip('#'))
            invalids.append(v)
        except:
            if k in ['[CLS]','[SEP]','.']:
                invalids.append(v)
    
    print(f"datasets prepared: {len(tdata)}, {len(ddata)}")
    tloader = torch.utils.data.DataLoader(tdata, batch_size=args.batch_size, drop_last=True, num_workers=args.nworks)
    dloader = torch.utils.data.DataLoader(ddata, batch_size=args.batch_size, drop_last=True, num_workers=args.nworks)
    print(f'dataloaders prepared: {len(tloader)}, {len(dloader)}')
    
    if args.dont_train:
        net = Reader.load_from_checkpoint(checkpoint_path = 'checkpoints/'+args.name+".ckpt")
    else:
        net = Reader(enc=args.enc, name=args.name, lr=args.lr, maskprob=args.maskprob,
                    hidden=args.hidden, invalids=invalids, unfreeze=args.unfreeze)
    print("model loaded")
    
    wandb_logger = None
    if not args.debug:
        wandb_logger = pl.loggers.WandbLogger(project='num2vec', name=args.name)
        wandb_logger.experiment.config.config = net.hparams
        wandb_logger.experiment.config.batch_size = args.batch_size
        wandb_logger.experiment.config.samples = len(tdata)
        wandb_logger.experiment.config.accumulate_grad_batches = args.accumulate_grad_batches
        wandb_logger.experiment.config.max_epochs = args.max_epochs
        wandb_logger.experiment.config.gradient_clip_val = args.gradient_clip_val
        wandb_logger.experiment.config.gpus = args.gpus
        
    ckpt_cb = pl.callbacks.ModelCheckpoint(save_top_k=2, verbose=True, filepath='checkpoints/'+args.name+'/{epoch}')
    trainer = pl.Trainer.from_argparse_args(args, progress_bar_refresh_rate=1, logger=wandb_logger, profiler=True,
                    checkpoint_callback=ckpt_cb, weights_save_path='checkpoints/'+args.name)
    
    if not args.dont_train:
        trainer.fit(net, tloader, val_dataloaders=[dloader])


if __name__ == "__main__":
    main()

# nice python train.py --batch-size 256 --gpus 0, --tsamples 100_000 --dsamples 10_000 --max_epochs 10 --enc exp --hidden 200 --accumulate_grad_batches 4 --seed 0 --dataset WC