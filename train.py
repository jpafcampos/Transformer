import math
import argparse
from typing import Iterable, List
from timeit import default_timer as timer

# Torch and numpy
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Transformer
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from Models import get_model

# Dataset
from TorchPtEnDataset import TorchPtEnDataset

# Tokenizer hugging face library
from tokenizers import Tokenizer

UNK_IDX, BOS_IDX, EOS_IDX, PAD_IDX = 0, 1, 2, 3
special_symbols = ['[UNK]', '[CLS]', '[SEP]', '[PAD]']

'''
FUNCTION TO RETURN TOKENIZERS (IN A DICT)
'''
def create_tokenizers(src_lang='pt', tgt_lang='en'):
    token_transform = {}
    token_transform[src_lang] = Tokenizer.from_file(
        "huggingface-tokenizer-pt.json")
    token_transform[tgt_lang] = Tokenizer.from_file(
        "huggingface-tokenizer-en.json")

    return token_transform

'''
FUNCTION TO RETURN DATASETS
'''
def create_datasets(data_path):

    # Change DATASET_PATH to path containing the ptbr-en dataset.
    DATASET_PATH = '/home/joao/Desktop/SHOWCASE/data/br-en/'
    SRC_LANGUAGE = 'pt'
    TGT_LANGUAGE = 'en'

    # Load dataset
    #train_iter = TorchPtEnDataset(path=DATASET_PATH, split='train')
    train_iter = TorchPtEnDataset(path=DATASET_PATH, split='train-small')
    #val_iter = TorchPtEnDataset(path=DATASET_PATH, split='test')
    val_iter = TorchPtEnDataset(path=DATASET_PATH, split='test-small')

    return train_iter, val_iter


def tokenize_pairs(pt, en, token_transform):
    pt = token_transform['pt'].encode(pt)
    en = token_transform['en'].encode(en)
    return torch.tensor(pt.ids), torch.tensor(en.ids)

'''
# Function to collate data samples into batch tesors
# Called by DataLoader over every batch of data sampled.
'''
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_tokenized, tgt_tokenized = tokenize_pairs(src_sample, tgt_sample, token_transform)
        src_batch.append(src_tokenized)
        tgt_batch.append(tgt_tokenized)

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch


def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float(
        '-inf')).masked_fill(mask == 1, float(0.0))
    return mask

'''
GENERATE MASKS
'''
def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, DEVICE)
    src_mask = torch.zeros((src_seq_len, src_seq_len),
                           device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def nopeak_mask(size, opt):
    np_mask = np.triu(np.ones((1, size, size)),
                      k=1).astype('uint8')
    np_mask = (torch.from_numpy(np_mask) == 0)

    return np_mask


def create_masks(src, trg, opt):

    src_mask = (src != PAD_IDX).unsqueeze(-2)

    if trg is not None:
        tgt_mask = (trg != PAD_IDX).unsqueeze(-2)
        size = trg.size(1)  # get seq_len for matrix
        np_mask = nopeak_mask(size, opt)

        tgt_mask = tgt_mask & np_mask

    else:
        tgt_mask = None

    return src_mask, tgt_mask


'''
Custom Learning Rate policy
This has shown to make a major difference in training performance
Without it, this torch model was significantly worse than its tensorflow counterpart
'''
class LRPolicy(object):
    def __init__(self, d_model, warmup_steps=4000):
        self.d_model = torch.tensor(d_model, dtype=torch.float32)
        self.warmup_steps = torch.tensor(warmup_steps, dtype=torch.float32)
        self.steps = torch.tensor(1, dtype=torch.float32)

    def getLR(self):
        lr1 = torch.rsqrt(self.steps)
        lr2 = self.steps * (self.warmup_steps ** -1.5)
        lr = torch.rsqrt(self.d_model) * torch.minimum(lr1, lr2)
        #self.steps += 1
        return lr

    def step(self):
        self.steps += 1

'''
TRAIN ONE EPOCH
'''
def train_epoch(model, opt, scaler):
    model.train()
    losses = 0
    train_dataloader = DataLoader(train_iter, batch_size=opt.batch_size, collate_fn=collate_fn, shuffle=True)
    batch_count = 0
    for src, tgt in train_dataloader:

        batch_count = batch_count+1
        if batch_count % 50 == 0:
            print(f'Batch {batch_count} Loss {loss:.4f}')

        opt.optimizer.zero_grad()

        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)
        tgt_input = tgt[:, :-1]
        src_mask, tgt_mask = create_masks(src, tgt_input, opt)
        # Send tensors to GPU
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        src_mask = src_mask.to(DEVICE)
        tgt_mask = tgt_mask.to(DEVICE)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(src, tgt_input, src_mask, tgt_mask)
                tgt_out = tgt[1:, :]
                loss = opt.loss_fn(
                    logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        
            scaler.scale(loss).backward()

            scaler.step(opt.optimizer)
            scaler.update()

        else:
            logits = model(src, tgt_input, src_mask, tgt_mask)
            tgt_out = tgt[1:, :]
            loss = opt.loss_fn(
                logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss.backward()
            opt.optimize.step()

        losses += loss.item()

        # Update learning rate after batch
        for g in opt.optimizer.param_groups:
            g['lr'] = lr_policy.getLR()
        lr_policy.step()

    return losses / len(list(train_dataloader))

'''
EVALUATE
'''
def evaluate(model, opt):
    model.eval()
    losses = 0
    val_dataloader = DataLoader(val_iter, batch_size=opt.batch_size, collate_fn=collate_fn, shuffle=True)
    for src, tgt in val_dataloader:

        # Send tensors to gpu
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        # Compute loss over batch
        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input)
        logits = model(src, tgt_input, src_mask, tgt_mask)
        tgt_out = tgt[1:, :]
        loss = opt.loss_fn(
            logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(list(val_dataloader))


if __name__ == "__main__":

    # Enable cuda for pytorch
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    IDX_dict = {
        'UNK_IDX': 0,
        'BOS_IDX': 1,
        'EOS_IDX': 2,
        'PAD_IDX': 3
    }

    print("Using device:")
    print(DEVICE)

    parser = argparse.ArgumentParser()
    parser.add_argument('-src_lang', type=str, default='pt')
    parser.add_argument('-tgt_lang', type=str, default='en')
    parser.add_argument('-dataset_path', type=str, default='/home/joao/Desktop/SHOWCASE/data/br-en/')
    parser.add_argument('-SGDR', action='store_true')
    parser.add_argument('-scaler', action='store_true')
    parser.add_argument('-epochs', type=int, default=2)
    parser.add_argument('-d_model', type=int, default=128)
    parser.add_argument('-num_enc_layers', type=int, default=4)
    parser.add_argument('-num_dec_layers', type=int, default=4)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-printevery', type=int, default=100)
    parser.add_argument('-lr', type=int, default=0.0001)
    parser.add_argument('-load_weights')
    parser.add_argument('-create_valset', action='store_true')
    parser.add_argument('-max_strlen', type=int, default=80)
    parser.add_argument('-floyd', action='store_true')
    parser.add_argument('-checkpoint', type=int, default=0)
    opt = parser.parse_args()
    opt.device = 0 if torch.cuda.is_available() is False else -1


    train_iter, val_iter = create_datasets(opt.dataset_path)
    print(len(val_iter))

    token_transform = create_tokenizers(opt.src_lang, opt.tgt_lang)
    print(token_transform)

    SRC_VOCAB_SIZE = token_transform[opt.src_lang].get_vocab_size()
    TGT_VOCAB_SIZE = token_transform[opt.tgt_lang].get_vocab_size()

    model=get_model(opt, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE)
    model = model.to(DEVICE)

    opt.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # Define LR policy and optimizer. Learning rate is updated every batch
    # Check train_epoch.
    lr_policy = LRPolicy(opt.d_model, 4000)
    opt.optimizer = torch.optim.Adam(model.parameters(), lr=lr_policy.getLR(), betas=(0.9, 0.98), eps=1e-9)
    scaler = torch.cuda.amp.GradScaler() if opt.scaler else None
    lr = lr_policy.getLR()
    #print(lr)
    #print(opt)

    for epoch in range(1, opt.epochs+1):
        start_time = timer()
        train_loss = train_epoch(model, opt, scaler)
        end_time = timer()
        val_loss = evaluate(model, opt)
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))

    #train_dataloader = DataLoader(train_iter, batch_size=opt.batch_size, collate_fn=collate_fn, shuffle=True)
    #for src, tgt in train_dataloader:
    #    print("NOSSA VERSAO ORIGINAL")
    #    print("src shape")
    #    print(src.shape)
    #    tgt_input = tgt[:-1, :]
    #    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
    #    print("src mask shape")
    #    print(src_mask.shape)
    #    print("tgt mask shape")
    #    print(tgt_mask.shape)
#
    #    print("VERSAO NOVA")
    #    print("src shape")
    #    print(src.shape)
    #    src = src.transpose(0, 1)
    #    tgt = tgt.transpose(0,1)
    #    tgt_input = tgt[:, :-1]
    #    src_mask, tgt_mask = create_masks(src, tgt_input, opt)
    #    print("src mask shape")
    #    print(src_mask.shape)
    #    print("tgt mask shape")
    #    print(tgt_mask.shape)
    #    break

