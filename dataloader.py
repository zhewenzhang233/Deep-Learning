from torch.utils.data import Dataset
import torch
import json
import os
from transformers import PreTrainedTokenizer
import sys
from torch.utils.data import DataLoader

'''
class DartDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str,
                 src_len: int, tgt_len: int, is_test: bool = False):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"

        print("Loading data file at {}...".format(file_path))
        sys.stdout.flush()

        with open(file_path) as f:
            lines_dict = json.load(f)

        full_src_lst = []
        full_tgt_lst = []
        for example in lines_dict:
            temp_triples = ''
            for i, tripleset in enumerate(example['tripleset']):
                subj, rela, obj = tripleset
                if i > 0:
                    temp_triples += ' | '
                temp_triples += '{} : {} : {}'.format(subj, rela, obj)

            for sent in example['annotations']:
                full_tgt_lst.append(sent['text'])
                full_src_lst.append(temp_triples)

                if is_test:
                    break

        assert len(full_tgt_lst) == len(full_src_lst)

        print('data length : {}'.format(len(full_src_lst)))
        print("begin to tokenize {} data ...".format(self.__class__.__name__))
        sys.stdout.flush()

        batch_encoding = tokenizer(full_src_lst, add_special_tokens=False,
                                   truncation=True, max_length=src_len,
                                   return_tensors='pt',
                                   padding='max_length')

        self.src = batch_encoding["input_ids"]
        self.src_mask = batch_encoding["attention_mask"]

        full_tgt_lst = [tokenizer.bos_token + tgt + tokenizer.eos_token
                        for tgt in full_tgt_lst]

        batch_encoding = tokenizer(full_tgt_lst, add_special_tokens=True,
                                   truncation=True, max_length=tgt_len,
                                   return_tensors='pt',
                                   padding='max_length')

        self.tgt = batch_encoding["input_ids"]
        self.tgt_mask = batch_encoding["attention_mask"]

        print("tokenize finished ...".format(file_path))
        sys.stdout.flush()

    def __len__(self):
        return self.src.size(0)

    def __getitem__(self, i):
        return (
                torch.tensor(self.src[i], dtype=torch.long),
                torch.tensor(self.src_mask[i], dtype=torch.long),
                torch.tensor(self.tgt[i], dtype=torch.long),
                torch.tensor(self.tgt_mask[i], dtype=torch.long),
                )


class E2EDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str,
                 src_len: int, tgt_len: int, is_test: bool =False):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"

        print("Loading data file at {}...".format(file_path))
        sys.stdout.flush()
        with open(file_path, encoding="utf-8") as f:
            lines = [line.split('||') for line in f.read().splitlines() if (len(line) > 0 and not line.isspace()
                                                                             and len(line.split('||')) ==2)]
        src_lines, tgt_lines = list(zip(*lines))
        src_lines = list(src_lines)
        tgt_lines = list(tgt_lines)

        print('data length : {}'.format(len(src_lines)))
        print("begin to tokenize {} data ...".format(self.__class__.__name__))

        batch_encoding = tokenizer(src_lines, add_special_tokens=False,
                                   truncation=True, max_length=src_len,
                                   return_tensors='pt',
                                   padding='max_length')

        self.src = batch_encoding["input_ids"]
        self.src_mask = batch_encoding["attention_mask"]

        tgt_lines = [tokenizer.bos_token + tgt + tokenizer.eos_token
                     for tgt in tgt_lines]

        batch_encoding = tokenizer(tgt_lines, add_special_tokens=True,
                                   truncation=True, max_length=tgt_len,
                                   return_tensors='pt',
                                   padding='max_length')

        self.tgt = batch_encoding["input_ids"]
        self.tgt_mask = batch_encoding["attention_mask"]

        print("tokenize finished ...".format(file_path))
        sys.stdout.flush()

    def __len__(self):
        return self.src.size(0)

    def __getitem__(self, i):
        return (
                torch.tensor(self.src[i], dtype=torch.long),
                torch.tensor(self.src_mask[i], dtype=torch.long),
                torch.tensor(self.tgt[i], dtype=torch.long),
                torch.tensor(self.tgt_mask[i], dtype=torch.long))


class WebNLGDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str,
                 src_len: int, tgt_len: int, is_test: bool =False):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"

        print("Loading data file at {}...".format(file_path))
        sys.stdout.flush()
        with open(file_path) as f:
            lines_dict = json.load(f)

        full_rela_lst = []
        full_src_lst = []
        full_tgt_lst = []

        for i, example in enumerate(lines_dict['entries']):
            sents = example[str(i + 1)]['lexicalisations']
            triples = example[str(i + 1)]['modifiedtripleset']

            rela_lst = []
            temp_triples = ''
            for j, tripleset in enumerate(triples):
                subj, rela, obj = tripleset['subject'], tripleset['property'], tripleset['object']
                rela_lst.append(rela)
                temp_triples += ' | '
                temp_triples += '{} : {} : {}'.format(subj, rela, obj)

            for sent in sents:
                if sent["comment"] == 'good':
                    full_tgt_lst.append(sent["lex"])
                    full_src_lst.append(temp_triples)
                    full_rela_lst.append(rela_lst)

                    if is_test:
                        break

        assert len(full_rela_lst) == len(full_src_lst)
        assert len(full_rela_lst) == len(full_tgt_lst)

        print('data length : {}'.format(len(full_src_lst)))

        print("begin to tokenize {} data ...".format(self.__class__.__name__))
        sys.stdout.flush()
        batch_encoding = tokenizer(full_src_lst, add_special_tokens=False,
                                   truncation=True, max_length=src_len,
                                   return_tensors='pt',
                                   padding='max_length')

        self.src = batch_encoding["input_ids"]
        self.src_mask = batch_encoding["attention_mask"]

        full_tgt_lst = [tokenizer.bos_token + tgt + tokenizer.eos_token
                     for tgt in full_tgt_lst]

        batch_encoding = tokenizer(full_tgt_lst, add_special_tokens=True,
                                   truncation=True, max_length=tgt_len,
                                   return_tensors='pt',
                                   padding='max_length')

        self.tgt = batch_encoding["input_ids"]
        self.tgt_mask = batch_encoding["attention_mask"]

        print("tokenize finished ...".format(file_path))
        sys.stdout.flush()

    def __len__(self):
        return self.src.size(0)

    def __getitem__(self, i):
        return (
                torch.tensor(self.src[i], dtype=torch.long),
                torch.tensor(self.src_mask[i], dtype=torch.long),
                torch.tensor(self.tgt[i], dtype=torch.long),
                torch.tensor(self.tgt_mask[i], dtype=torch.long))


class SummarizationDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str,
                 src_len: int, tgt_len: int, is_test: bool =False):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"

        print("Loading data file at {}...".format(file_path))
        sys.stdout.flush()
        with open(file_path) as f:
            lines_dict = json.load(f)

        full_src_lst = []
        full_tgt_lst = []

        for i, example in enumerate(lines_dict):
            full_src_lst.append(example['document'])
            full_tgt_lst.append(example['summary'])
 

        assert len(full_tgt_lst) == len(full_src_lst)

        print('data length : {}'.format(len(full_src_lst)))

        print("begin to tokenize {} data ...".format(self.__class__.__name__))
        sys.stdout.flush()
        batch_encoding = tokenizer(full_src_lst, add_special_tokens=False,
                                   truncation=True, max_length=src_len,
                                   return_tensors='pt',
                                   padding='max_length')

        self.src = batch_encoding["input_ids"]
        self.src_mask = batch_encoding["attention_mask"]

        full_tgt_lst = [tokenizer.bos_token + tgt + tokenizer.eos_token
                            for tgt in full_tgt_lst]

        batch_encoding = tokenizer(full_tgt_lst, add_special_tokens=True,
                                   truncation=True, max_length=tgt_len,
                                   return_tensors='pt',
                                   padding='max_length')

        self.tgt = batch_encoding["input_ids"]
        self.tgt_mask = batch_encoding["attention_mask"]

        print("tokenize finished ...".format(file_path))
        sys.stdout.flush()

    def __len__(self):
        return self.src.size(0)

    def __getitem__(self, i):
        return (
                torch.tensor(self.src[i], dtype=torch.long),
                torch.tensor(self.src_mask[i], dtype=torch.long),
                torch.tensor(self.tgt[i], dtype=torch.long),
                torch.tensor(self.tgt_mask[i], dtype=torch.long))


class Samsum(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str,
                 src_len: int, tgt_len: int, is_test: bool = False):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"

        print("Loading data file at {}...".format(file_path))
        sys.stdout.flush()

        lines = json.load(open(file_path, 'r'))

        full_src_lst = []
        full_tgt_lst = []

        for i, example in enumerate(lines):
            full_src_lst.append(example['dialogue'])
            full_tgt_lst.append(example['summary'])

        assert len(full_tgt_lst) == len(full_src_lst)

        print('data length : {}'.format(len(full_src_lst)))

        print("begin to tokenize {} data ...".format(self.__class__.__name__))
        sys.stdout.flush()

        # cnt = []
        # for src in full_src_lst:
        #     cnt.append(len(tokenizer(src, add_special_tokens=False, truncation=True)["input_ids"]))
        # print(sum(cnt) / len(cnt))
        #
        # cnt = []
        # for tgt in full_tgt_lst:
        #     cnt.append(len(tokenizer(tgt, add_special_tokens=False, truncation=True)["input_ids"]))
        # print(sum(cnt) / len(cnt))
        #
        # exit()

        #156.54452891664403
        #25.50237578061363

        batch_encoding = tokenizer(full_src_lst, add_special_tokens=False,
                                   truncation=True, max_length=src_len,
                                   return_tensors='pt',
                                   padding='max_length')

        self.src = batch_encoding["input_ids"]
        self.src_mask = batch_encoding["attention_mask"]

        full_tgt_lst = [tokenizer.bos_token + tgt + tokenizer.eos_token
                        for tgt in full_tgt_lst]

        batch_encoding = tokenizer(full_tgt_lst, add_special_tokens=True,
                                   truncation=True, max_length=tgt_len,
                                   return_tensors='pt',
                                   padding='max_length')

        self.tgt = batch_encoding["input_ids"]
        self.tgt_mask = batch_encoding["attention_mask"]

        print("tokenize finished ...".format(file_path))
        sys.stdout.flush()

    def __len__(self):
        return self.src.size(0)

    def __getitem__(self, i):
        return (
            torch.tensor(self.src[i], dtype=torch.long),
            torch.tensor(self.src_mask[i], dtype=torch.long),
            torch.tensor(self.tgt[i], dtype=torch.long),
            torch.tensor(self.tgt_mask[i], dtype=torch.long))

'''


class WikiZh2019Dataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, tgt_len: int):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"

        print("Loading data file at {}...".format(file_path))
        sys.stdout.flush()

        lines = open(file_path, 'r', encoding='UTF-8').readlines()

        full_src_lst = []
        full_tgt_lst = []

        for line in lines:
            # full_src_lst.append(example['dialogue'])
            full_tgt_lst.append(line)

        assert len(full_tgt_lst)

        print('data length : {}'.format(len(full_tgt_lst)))

        print("begin to tokenize {} data ...".format(self.__class__.__name__))
        sys.stdout.flush()

        '''batch_encoding = tokenizer(full_src_lst, add_special_tokens=False,
                                   truncation=True, max_length=src_len,
                                   return_tensors='pt',
                                   padding='max_length')

        self.src = batch_encoding["input_ids"]
        self.src_mask = batch_encoding["attention_mask"]'''

        full_tgt_lst = [tokenizer.bos_token + tgt + tokenizer.eos_token
                        for tgt in full_tgt_lst]
        print('finished add start and end token')
        batch_encoding = tokenizer(full_tgt_lst, add_special_tokens=True,
                                   truncation=True, max_length=tgt_len,
                                   return_tensors='pt',
                                   padding='max_length')
        print('finished tokenizer')
        self.tgt = batch_encoding["input_ids"]
        self.tgt_mask = batch_encoding["attention_mask"]

        print("tokenize finished ...".format(file_path))
        sys.stdout.flush()

    def __len__(self):
        return self.tgt.size(0)

    def __getitem__(self, i):
        return (
            # torch.tensor(self.src[i], dtype=torch.long),
            # torch.tensor(self.src_mask[i], dtype=torch.long),
            torch.tensor(self.tgt[i], dtype=torch.long),
            torch.tensor(self.tgt_mask[i], dtype=torch.long))
