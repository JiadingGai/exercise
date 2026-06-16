#!/usr/bin/env python

import torch
import torch.nn as nn
from collections import Counter
from pathlib import Path
from typing import Tuple, List, Iterable
import json
import math
import os
import random
import time

import requests
import spacy


class Vocab:
    def __init__(self, tokens):
        self._itos = list(tokens)
        self._stoi = {token: index for index, token in enumerate(self._itos)}
        self._default_index = None
        self.vocab = self

    def __len__(self):
        return len(self._itos)

    def __call__(self, tokens):
        return [self[token] for token in tokens]

    def __getitem__(self, token):
        if token in self._stoi:
            return self._stoi[token]
        if self._default_index is not None:
            return self._default_index
        raise RuntimeError(f"Token {token!r} not found and no default index is set")

    def set_default_index(self, index):
        self._default_index = index

    def lookup_token(self, index):
        return self._itos[int(index)]

    def get_stoi(self):
        return self._stoi

    def get_itos(self):
        return self._itos


def build_vocab_from_iterator(iterator, min_freq=1, specials=None, special_first=True):
    counter = Counter()
    for tokens in iterator:
        counter.update(tokens)

    specials = list(specials or [])
    regular_tokens = sorted(token for token, count in counter.items() if count >= min_freq and token not in specials)
    if special_first:
        return Vocab(specials + regular_tokens)
    return Vocab(regular_tokens + specials)


def get_spacy_tokenizer(language):
    nlp = spacy.load(language)
    return lambda text: [token.text for token in nlp.tokenizer(text)]


MULTI30K_URLS = {
    "train": "https://huggingface.co/datasets/bentrevett/multi30k/resolve/main/train.jsonl",
    "valid": "https://huggingface.co/datasets/bentrevett/multi30k/resolve/main/val.jsonl",
    "test": "https://huggingface.co/datasets/bentrevett/multi30k/resolve/main/test.jsonl",
}
LOCAL_MULTI30K_DIR = Path(__file__).resolve().parent / "data" / "multi30k"


def load_multi30k_split(split, language_pair=("de", "en")):
    data_dir = Path(os.environ.get("MULTI30K_CACHE_DIR", LOCAL_MULTI30K_DIR))
    data_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = data_dir / Path(MULTI30K_URLS[split]).name

    if not jsonl_path.exists():
        with requests.get(MULTI30K_URLS[split], stream=True, timeout=60) as response:
            response.raise_for_status()
            with jsonl_path.open("wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

    examples = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            examples.append((row[language_pair[0]], row[language_pair[1]]))
    return examples


def ngrams_iterator(tokens, max_n):
    for token in tokens:
        yield token
    for n in range(2, max_n + 1):
        for i in range(len(tokens) - n + 1):
            yield " ".join(tokens[i : i + n])


def compute_ngram_counter(tokens, max_n):
    return Counter(tuple(x.split(" ")) for x in ngrams_iterator(tokens, max_n))


def corpus_bleu(candidate_corpus, references_corpus, max_n=4, weights=[0.25] * 4):
    clipped_counts = torch.zeros(max_n)
    total_counts = torch.zeros(max_n)
    weights = torch.tensor(weights)
    candidate_len = 0.0
    refs_len = 0.0

    for candidate, refs in zip(candidate_corpus, references_corpus):
        current_candidate_len = len(candidate)
        candidate_len += current_candidate_len
        refs_len += min((float(len(ref)) for ref in refs), key=lambda x: abs(current_candidate_len - x))

        reference_counters = compute_ngram_counter(refs[0], max_n)
        for ref in refs[1:]:
            reference_counters = reference_counters | compute_ngram_counter(ref, max_n)

        clipped_counter = compute_ngram_counter(candidate, max_n) & reference_counters
        for ngram, count in clipped_counter.items():
            clipped_counts[len(ngram) - 1] += count
        for i in range(max_n):
            total_counts[i] += max(current_candidate_len - i, 0)

    if candidate_len == 0 or min(clipped_counts) == 0:
        return 0.0
    score = torch.exp((weights * torch.log(clipped_counts / total_counts)).sum())
    brevity_penalty = math.exp(min(1 - refs_len / candidate_len, 0))
    return brevity_penalty * score.item()

class Encoder(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim,
                 dropout, 
                 device,
                 max_length = 100):
        super().__init__()

        self.device = device
        
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim,
                                                  dropout, 
                                                  device) 
                                     for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, src, src_mask):
        
        #src = [batch size, src len]
        #src_mask = [batch size, 1, 1, src len]
        
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        
        #pos = [batch size, src len]
        
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        
        #src = [batch size, src len, hid dim]
        
        for layer in self.layers:
            src = layer(src, src_mask)
            
        #src = [batch size, src len, hid dim]
            
        return src


class EncoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim,  
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        
        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, 1, 1, src len] 
                
        #self attention
        _src, _ = self.self_attention(src, src, src, src_mask)
        
        #dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        #positionwise feedforward
        _src = self.positionwise_feedforward(src)
        
        #dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        return src

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None):
        
        batch_size = query.shape[0]
        
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
                
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]
                
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]
                
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        #energy = [batch size, n heads, query len, key len]
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim = -1)
                
        #attention = [batch size, n heads, query len, key len]
                
        x = torch.matmul(self.dropout(attention), V)
        
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        
        #x = [batch size, query len, hid dim]
        
        return x, attention

class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        #x = [batch size, seq len, hid dim]
        
        x = self.dropout(torch.relu(self.fc_1(x)))
        
        #x = [batch size, seq len, pf dim]
        
        x = self.fc_2(x)
        
        #x = [batch size, seq len, hid dim]
        
        return x

class Decoder(nn.Module):
    def __init__(self, 
                 output_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device,
                 max_length = 100):
        super().__init__()
        
        self.device = device
        
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim, 
                                                  dropout, 
                                                  device)
                                     for _ in range(n_layers)])
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        #trg = [batch size, trg len]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]
                
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
                            
        #pos = [batch size, trg len]
            
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
                
        #trg = [batch size, trg len, hid dim]
        
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]
        
        output = self.fc_out(trg)
        
        #output = [batch size, trg len, output dim]
            
        return output, attention

class DecoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        #trg = [batch size, trg len, hid dim]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]
        
        #self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        
        #dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
            
        #trg = [batch size, trg len, hid dim]
            
        #encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        
        #dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
                    
        #trg = [batch size, trg len, hid dim]
        
        #positionwise feedforward
        _trg = self.positionwise_feedforward(trg)
        
        #dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return trg, attention

class Seq2Seq(nn.Module):
    def __init__(self, 
                 encoder, 
                 decoder, 
                 src_pad_idx, 
                 trg_pad_idx, 
                 device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
    def make_src_mask(self, src):
        
        #src = [batch size, src len]
        
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        #src_mask = [batch size, 1, 1, src len]

        return src_mask
    
    def make_trg_mask(self, trg):
        
        #trg = [batch size, trg len]
        
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        
        #trg_pad_mask = [batch size, 1, 1, trg len]
        
        trg_len = trg.shape[1]
        
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        
        #trg_sub_mask = [trg len, trg len]
            
        trg_mask = trg_pad_mask & trg_sub_mask
        
        #trg_mask = [batch size, 1, trg len, trg len]
        
        return trg_mask

    def forward(self, src, trg):
        
        #src = [batch size, src len]
        #trg = [batch size, trg len]
                
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        #src_mask = [batch size, 1, 1, src len]
        #trg_mask = [batch size, 1, trg len, trg len]
        
        enc_src = self.encoder(src, src_mask)
        
        #enc_src = [batch size, src len, hid dim]
                
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        
        #output = [batch size, trg len, output dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return output, attention

# end of transformer model definition

SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'
train = load_multi30k_split("train", language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
valid = load_multi30k_split("valid", language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
test = load_multi30k_split("test", language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
iter1 = iter(train)
print("Sample train sentence pair: ", next(iter1))
iter2 = iter(valid)
print("Sample Validation sentence pair: ", next(iter2))

for data_iter in train:
  print("Next train sentence pair: ", data_iter)
  break

print(f"Number of training examples = ", len(train))
print(f"Number of validation examples = ", len(valid))
print(f"Number of test examples = ", len(test))

token_transform = {}
vocab_transform = {}
token_transform[SRC_LANGUAGE] = get_spacy_tokenizer('de_core_news_sm')
token_transform[TGT_LANGUAGE] = get_spacy_tokenizer('en_core_web_sm')
def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}
    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train, ln),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True)

# Set ``UNK_IDX`` as the default index. The index is returned when the token is not found.
# If not set, it throws ``RuntimeError`` when the queried token is not found in the Vocabulary.
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    vocab_transform[ln].set_default_index(UNK_IDX)

# Print some sample tokens in vocab_transform:
print("vocab_transform size (DE) = ", vocab_transform['de'].vocab.__len__())
print("vocab_transform size (EN) = ", vocab_transform['en'].vocab.__len__())
print("the 678th DE token is ", vocab_transform['de'].vocab.lookup_token(678))
print("the 678th EN token is ", vocab_transform['en'].vocab.lookup_token(678))


# top-like gpu utility check: nvidia-smi -l 1
#assert torch.cuda.is_available(), "No CUDA found on your machine."
if torch.cuda.is_available():
  print("Training on this device: ", torch.cuda.get_device_name(0))
else:
  print("Training on CPU")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

INPUT_DIM = len(vocab_transform[SRC_LANGUAGE].vocab)
OUTPUT_DIM = len(vocab_transform[TGT_LANGUAGE].vocab)
HID_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

enc = Encoder(INPUT_DIM, 
              HID_DIM, 
              ENC_LAYERS, 
              ENC_HEADS, 
              ENC_PF_DIM, 
              ENC_DROPOUT, 
              device)

dec = Decoder(OUTPUT_DIM, 
              HID_DIM, 
              DEC_LAYERS, 
              DEC_HEADS, 
              DEC_PF_DIM, 
              DEC_DROPOUT, 
              device)

SRC_PAD_IDX = PAD_IDX
TRG_PAD_IDX = PAD_IDX
model = Seq2Seq(enc, dec, PAD_IDX, PAD_IDX, device).to(device)

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

model.apply(init_weights)


"""
Testing the model forward flow
"""
def sample_data(input_sentences: List[Tuple[str]]):
    print(input_sentences[0])
    sample_src = [x[0].strip().split() for x in input_sentences]
    sample_dst = [x[1].strip().split() for x in input_sentences]
    return sample_src, sample_dst

sample_de, sample_en = sample_data(list(train)[615:616])

x = []
x_toks = []
for sample in sample_de:
    t0 = [BOS_IDX] + vocab_transform['de'](sample) + [EOS_IDX]
    x.append(t0)
    x_toks.append([vocab_transform['de'].vocab.lookup_token(t) for t in t0])

y = []
y_toks = []
for sample in sample_en:
    t0 = [BOS_IDX] + vocab_transform['en'](sample) + [EOS_IDX]
    y.append(t0)
    y_toks.append([vocab_transform['en'].vocab.lookup_token(t) for t in t0])

print("x = ", x)
print("y = ", y)
print("x_toks = ", x_toks)
print("y_toks = ", y_toks)

tx_len = len(x[0])
ty_len = len(y[0])
tx = torch.tensor(x)
ty = torch.tensor(y)
sample_output, _attention = model(tx.to(device), ty.to(device))
predict_tokids = torch.argmax(sample_output, dim=-1).flatten()
predict_toks = []
for ptid in predict_tokids:
  predict_toks.append(vocab_transform['en'].vocab.lookup_token(ptid))
print("initial model predicted toks = ", predict_toks)

def translate_sentence(sentence, src_field: Vocab, trg_field, model, device, max_len = 50):
  model.eval()

  if isinstance(sentence, str):
      # nlp = spacy.load('de_core_news_sm')
      #tokens = [token.text.lower() for token in nlp(sentence)]
      #FIXME(lambda!!!!): should we lower cases here?
      #tokens = [token.text.lower() for token in token_transform[SRC_LANGUAGE](sentence)]
      tokens = [token for token in token_transform[SRC_LANGUAGE](sentence)]
  else:
      #FIXME(lambda!!!!): should we lower cases here?
      #tokens = [token.lower() for token in sentence]
      tokens = [token for token in sentence]

  tokens = [token for token in tokens if token not in ('<bos>', '<eos>')]
  tokens = ['<bos>'] + tokens + ['<eos>']
  src_indexes = src_field(tokens)
  src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
  src_mask = model.make_src_mask(src_tensor)
  
  with torch.no_grad():
      enc_src = model.encoder(src_tensor, src_mask)

  #trg_indexes = [trg_field.vocab.get_stoi()[trg_field.init_token]]
  trg_indexes = [trg_field.vocab.get_stoi()['<bos>']]

  for i in range(max_len):
    trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
    trg_mask = model.make_trg_mask(trg_tensor)
    with torch.no_grad():
      output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
    pred_token = output.argmax(2)[:,-1].item()
    trg_indexes.append(pred_token)

    if pred_token == trg_field.vocab.get_stoi()['<eos>']:
      break
  
  trg_tokens = [trg_field.vocab.get_itos()[i] for i in trg_indexes]
  return trg_tokens[1:], attention

src_sentence = x_toks[0]
translation, attention = translate_sentence(sentence=src_sentence, \
                                            src_field=vocab_transform['de'], \
                                            trg_field=vocab_transform['en'], \
                                            model=model, \
                                            device=device)
print(f'predicted trg = {translation}')

def calculate_bleu(data, src_field, trg_field, model, device, max_len = 50):
    
    trgs = []
    pred_trgs = []
    
    for datum in data:
        
        src = datum[0]
        trg = token_transform[TGT_LANGUAGE](datum[1])
        
        pred_trg, _ = translate_sentence(src, src_field, trg_field, model, device, max_len)
        
        if pred_trg and pred_trg[-1] == '<eos>':
            pred_trg = pred_trg[:-1]
        
        pred_trgs.append(pred_trg)
        trgs.append([trg])
        
    return corpus_bleu(pred_trgs, trgs)

BLEU_EVAL_LIMIT = int(os.environ.get("BLEU_EVAL_LIMIT", "100"))
initial_bleu = calculate_bleu(valid[:BLEU_EVAL_LIMIT], vocab_transform['de'], vocab_transform['en'], model, device)
print(f'Initial validation BLEU score = {initial_bleu*100:.2f}')

"""
Preparing data:
https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/transformer_tutorial.ipynb
"""
from torch.nn.utils.rnn import pad_sequence

#helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# Function to add BOS/ESOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]), torch.tensor(token_ids), torch.tensor([EOS_IDX])))

# ``src`` and ``tgt`` language text transforms to convert raw strings into tensors indices
text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln], # Tokenization
                                               vocab_transform[ln], # Numericalization
                                               tensor_transform) # Add BOS/EOS and create tensor

# function to collate data samples into batch tensors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

from torch.utils.data import DataLoader
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "128"))
#print(len(train_iter))
train_iter = train
valid_iter = valid
test_iter = test
train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)
valid_dataloader = DataLoader(valid_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)
test_dataloader = DataLoader(test_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)
#print(train_dataloader.__str__())
#for src, tgt in train_dataloader:
#     print(src.shape)
#     print(tgt.shape)
#     print(src[:,123])
#     print(tgt[:,123])
#     break

def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

def initialize_weights(m):
  if hasattr(m, 'weight') and m.weight.dim() > 1:
    nn.init.xavier_uniform_(m.weight.data)
model.apply(initialize_weights);

LEARNING_RATE = 0.0005
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)
print(f'Loss function = {criterion}')

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    numTrainDataPoints = 0
    total_batches = len(iterator) if hasattr(iterator, "__len__") else None
    for i, batch in enumerate(iterator):
        src = batch[0].to(device)
        trg = batch[1].to(device)

        src = torch.transpose(src, 0, 1)
        trg = torch.transpose(trg, 0, 1)

        optimizer.zero_grad()
        output, _ = model(src, trg[:,:-1])

        #output = [batch size, trg len - 1, output dim]
        #trg = [batch size, trg len]

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)

        #output = [batch size * trg len - 1, output dim]
        #trg = [batch size * trg len - 1]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        numTrainDataPoints += 1 #iterator.batch_size

        if LOSS_LOG_EVERY > 0 and ((i + 1) % LOSS_LOG_EVERY == 0 or i == 0):
            running_loss = epoch_loss / numTrainDataPoints
            total_batches_str = f"/{total_batches}" if total_batches is not None else ""
            print(
                f"\t[train step {i + 1}{total_batches_str}] "
                f"loss: {loss.item():.3f} | running avg loss: {running_loss:.3f}",
                flush=True,
            )
        
    return epoch_loss / numTrainDataPoints

def evaluate(model, iterator, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        numTrainDataPoints = 0
        for i, batch in enumerate(iterator):

            src = batch[0].to(device)
            trg = batch[1].to(device)

            src = torch.transpose(src, 0, 1)
            trg = torch.transpose(trg, 0, 1)

            output, _ = model(src, trg[:,:-1])
            
            #output = [batch size, trg len - 1, output dim]
            #trg = [batch size, trg len]
            
            output_dim = output.shape[-1]
            
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:,1:].contiguous().view(-1)
            
            #output = [batch size * trg len - 1, output dim]
            #trg = [batch size * trg len - 1]
            
            loss = criterion(output, trg)

            epoch_loss += loss.item()
            numTrainDataPoints += 1 #iterator.batch_size
        
    return epoch_loss / numTrainDataPoints

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

N_EPOCHS = int(os.environ.get("N_EPOCHS", "100"))
SAMPLE_TRANSLATION_EVERY = int(os.environ.get("SAMPLE_TRANSLATION_EVERY", "1"))
LOSS_LOG_EVERY = int(os.environ.get("LOSS_LOG_EVERY", "50"))
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    
    train_loss = train(model, train_dataloader, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_dataloader, criterion)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut6-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    if SAMPLE_TRANSLATION_EVERY > 0 and (epoch + 1) % SAMPLE_TRANSLATION_EVERY == 0:
        sample_translation, _ = translate_sentence(sample_de[0], vocab_transform['de'], vocab_transform['en'], model, device)
        if sample_translation and sample_translation[-1] == '<eos>':
            sample_translation = sample_translation[:-1]
        print("[SAMPLE SOURCE] = ", sample_de[0])
        print("[GOLD TRANSLATION] = ", sample_en[0])
        print("[GREEDY TRANSLATION] = ", sample_translation)

model.load_state_dict(torch.load('tut6-model.pt', map_location=device))
test_loss = evaluate(model, test_dataloader, criterion)
print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

test_bleu = calculate_bleu(test_iter[:BLEU_EVAL_LIMIT], vocab_transform['de'], vocab_transform['en'], model, device)
print(f'Test BLEU score = {test_bleu*100:.2f}')
