# import gluonnlp
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from config import CONFIG as conf
import random
import json
from collections import Counter
from transformers import BertTokenizer, BertModel

mask_pro = conf['mask_pro']
random_seed = conf['random_seed']
device = conf['device']
random.seed(random_seed)
torch.manual_seed(random_seed)
max_doc_len = 100

# Initialize BERT tokenizer and model globally (can be moved to a setup function if needed)
bert_model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = BertModel.from_pretrained(bert_model_name).to(device)
bert_model.eval()  # Set to eval mode by default

def get_all_words(data):
    all_text = []
    for sample in data:
        for sentence in sample:
            all_text += sentence
    return all_text

def build_vocab(data_list):
    # Not needed with BERT, but kept for compatibility
    return tokenizer

def save_vocab(vocab):
    # Save tokenizer config
    vocab.save_pretrained('data/bert_tokenizer/')

def load_vocab():
    # Load tokenizer config
    return BertTokenizer.from_pretrained(bert_model_name)

def build_paragraph(text_data, my_tokenizer):
    # Tokenize each utterance in the dialog
    text_data = [text[:max_doc_len] for text in text_data]
    indexs = []
    lengths = []
    for text in text_data:
        tokens = my_tokenizer.encode(text, add_special_tokens=True, max_length=max_doc_len, truncation=True)
        indexs.append(torch.tensor(tokens).long())
        lengths.append(len(tokens))
    return indexs, lengths

def filter_output(x, lengths):
    batch_size = max(lengths)
    indexes = []
    for i in range(len(lengths)):
        indexes += [i*batch_size+j for j in range(lengths[i])]
        # print(indexes)
    return x[indexes]

def mask_sentence(batch_data):
    mask_id = torch.tensor([tokenizer.mask_token_id])
    batch_mask = []
    new_batch_data = []
    cand_pool = []
    for para in batch_data:
        if len(para[0]) < 2:
            continue
        para_embed = list(para[0])
        para_len = list(para[1])
        this_cand_pool_embed = []
        this_cand_pool_length = []
        mask = torch.rand(len(para_len))
        mask = mask.le(mask_pro)
        if mask.sum() <= 1:
            idx = list(range(len(mask)))
            sel_idx = random.sample(idx, 2)
            mask[sel_idx] = 1
        batch_mask.append(mask)
        for i in range(len(mask)):
            if mask[i] == 1:
                this_cand_pool_embed.append(para_embed[i])
                this_cand_pool_length.append(para_len[i])
                para_embed[i] = mask_id
                para_len[i] = 1
        new_batch_data.append([para_embed, para_len])
        cand_pool.append([this_cand_pool_embed, this_cand_pool_length])
    return new_batch_data, batch_mask, cand_pool

def switch_within_para(mask, para_embed, para_len):
    para_size = len(para_len)
    cand_idx = [i for i in range(para_size) if mask[i]==1]
    origin_idx = list(cand_idx)
    random.shuffle(cand_idx)
    for i, idx in enumerate(cand_idx):
        if idx == origin_idx[i] and i!=0:
            cand_idx[i], cand_idx[i-1] = cand_idx[i-1], cand_idx[i]
        elif idx==origin_idx[i]:
            cand_idx[i], cand_idx[i+1] = cand_idx[i+1], cand_idx[i]
    new_cand_embed = list(para_embed)
    new_cand_len = list(para_len)
    for i, idx in enumerate(origin_idx):
        new_cand_embed[idx] = para_embed[cand_idx[i]]
        new_cand_len[idx] = para_len[cand_idx[i]]
    return new_cand_embed, new_cand_len

def local_sort_sentence(batch_data, cand_permutation):
    sorter_len = 3
    new_batch_data = []
    cand_labels = torch.LongTensor(len(batch_data)).random_(0, len(cand_permutation))
    sel_labels = []
    start_idx_list = []
    for i, para in enumerate(batch_data):
        if len(para[0]) < sorter_len:
            continue
        start_idx = random.randint(0, len(para[0])-sorter_len)
        start_idx_list.append(start_idx)
        para_embed = list(para[0])
        para_len = list(para[1])
        this_label = cand_labels[i]
        sel_labels.append(this_label)
        this_permut = cand_permutation[this_label]
        for j in range(len(this_permut)):
            para_embed[start_idx+j] = para[0][start_idx+this_permut[j]]
            para_len[start_idx+j] = para[1][start_idx+this_permut[j]]
        new_batch_data.append([para_embed, para_len])
    return new_batch_data, start_idx_list, sel_labels

def switch_sentence(batch_data, sentence_cands):
    batch_mask = []
    new_batch_data = []
    cand_pool = []
    cand_size = len(sentence_cands)
    for para in batch_data:
        if len(para[0]) < 2:
            continue
        para_embed = list(para[0])
        para_len = list(para[1])
        this_cand_pool_embed = []
        this_cand_pool_length = []
        mask = torch.rand(len(para_len))
        mask = mask.le(mask_pro)
        if mask.sum() <=1:
            idx = list(range(len(mask)))
            sel_idx = random.sample(idx, 2)
            mask[sel_idx] = 1
        batch_mask.append(mask)
        para_embed, para_len = switch_within_para(mask, para_embed, para_len)
        new_batch_data.append([para_embed, para_len])
    return new_batch_data, batch_mask

def replace_sentence(batch_data, sentence_cands):
    batch_mask = []
    new_batch_data = []
    cand_pool = []
    cand_size = len(sentence_cands)
    for para in batch_data:
        if len(para[0]) < 2:
            continue
        para_embed = list(para[0])
        para_len = list(para[1])
        this_cand_pool_embed = []
        this_cand_pool_length = []
        mask = torch.rand(len(para_len))
        mask = mask.le(mask_pro)
        if mask.sum() <=1:
            idx = list(range(len(mask)))
            sel_idx = random.sample(idx, 2)
            mask[sel_idx] = 1
        batch_mask.append(mask)
        for i in range(len(mask)):
            if mask[i] == 1:
                para_embed[i] = sentence_cands[random.randint(0, cand_size-1)]
                para_len[i] = len(para_embed[i])
        new_batch_data.append([para_embed, para_len])
    return new_batch_data, batch_mask

def gen_mask_based_length(batch_size, doc_size, lengths):
    masks = torch.ones(batch_size, doc_size)
    index_matrix = torch.arange(0, doc_size).expand(batch_size, -1)
    index_matrix = index_matrix.long()
    doc_lengths = torch.tensor(lengths).view(-1,1)
    doc_lengths_matrix = doc_lengths.expand(-1, doc_size)
    masks[torch.ge(index_matrix-doc_lengths_matrix, 0)] = 0
    return masks.to(device)

def get_fetch_idx(batch_size, start_idx):
    num_to_sort = 3
    start_idx = torch.tensor(start_idx)
    idx_1 = torch.arange(batch_size).view(-1,1).expand(-1, num_to_sort)
    idx_1 = idx_1.contiguous().view(-1)
    idx_2 = start_idx.view(-1,1).expand(-1, num_to_sort)
    idx_2 = idx_2.contiguous()
    for i in range(num_to_sort):
        idx_2[:,i] += i
    idx_2 = idx_2.contiguous().view(-1)
    return idx_1, idx_2
