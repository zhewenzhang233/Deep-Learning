import shutil, os
import torch
import torch.nn as nn
from prefix import PrefixTuning, AffineNet
from dataloader import *
from prefix import *
from transformers.adapters import MAMConfig, PrefixTuningConfig, AdapterConfig
from rouge import Rouge
import time 



'''def postprocess(gens):
    gens_ = []
    for gen in gens:
        gen_ = []
        for i, t in enumerate(gen.lower()):
            if t.isdigit() or t.isalpha() or t == ' ':
                gen_.append(t)
            else:
                if i > 0 and gen_[-1] != ' ':
                    gen_.append(' ')
                gen_.append(t)
                if i + 1 < len(gen) and gen[i+1] != ' ':
                    gen_.append(' ')
        gens_.append(''.join(gen_))
    return gens_
'''


'''def cal_summarization_metric(gens, refs):
    rouge = Rouge()
    scores = rouge.get_scores(gens, refs, avg=True)
    return scores['rouge-1']['f'], scores['rouge-2']['f'], scores['rouge-l']['f']
'''

'''def normalize_gens_and_refs(inputs, gens, refs):
    pre_input = None

    new_gens, new_refs = [], []

    for input, gen, ref in zip(inputs, gens, refs):
        if pre_input is None:
            new_gens.append(gen)
        elif input != pre_input:
            new_refs.append('')
            new_gens.append(gen)

        new_refs.append(ref)
        pre_input = input

    return new_gens, new_refs
'''


'''
def eval_(data_loader, tokenizer, model, dataset, input_pad_size,  eval_gen, gen_file):
    torch.cuda.empty_cache()
    pos_tensor = torch.arange(input_pad_size + eval_gen).long()[None, :]
    inputs, gens, refs = [], [], []
    
    cnt_time = 0
    
    with torch.no_grad():
        # fo = open(gen_file, 'w+')
        for batch_data in data_loader:
            pre = time.time()
            outputs = model.infer(src=batch_data[0].cuda(),
                                  src_mask=batch_data[1].cuda(),
                                  pos=pos_tensor.cuda(),
                                  max_length=eval_gen,
                                  bos_id=tokenizer.bos_token_id,
                                  eos_id=tokenizer.eos_token_id)
            cnt_time += time.time() - pre

            gen = tokenizer.batch_decode(outputs,
                                         skip_special_tokens=True)
            input = tokenizer.batch_decode(batch_data[0],
                                           skip_special_tokens=True)
            ref = tokenizer.batch_decode(batch_data[2],
                                         skip_special_tokens=True)

            ref = [_.replace('\n', '') for _ in ref]
            gen = [_.replace('\n', '') for _ in gen]
            input = [_.replace('\n', '') for _ in input]

            refs += ref
            gens += gen
            inputs += input

            # for input_item, gen_item in zip(input, gen):
                # print(input_item + '</d>' + gen_item, file=fo)

    assert len(refs) == len(gens) == len(inputs)

    if dataset == 'E2E':
        gens, refs = normalize_gens_and_refs(inputs, gens, refs)
    elif dataset == 'DART':
        gens = postprocess(gens)
        
    for i in range(len(gens)):
        if gens[i].strip() == '':
            gens[i] = '1'    
        
    fo.close()
    fo = open('temp_gen.txt', 'w+')
    for gen in gens:
        print(gen, file=fo)
    fo.close()
    fo = open('temp_ref.txt', 'w+')
    for ref in refs:
        print(ref, file=fo)
    fo.close()
    
    
    torch.cuda.empty_cache()

    if dataset == 'DART':
        os.system("cp temp_gen.txt metric/dart_and_webnlg")
        scores = run_dart_evaluate()
    elif dataset == 'E2E':
        scores = run_e2e_evaluate('temp_ref.txt', 'temp_gen.txt')
    elif dataset == 'WebNLG':
        os.system("cp temp_gen.txt  metric/dart_and_webnlg")
        scores = run_webnlg_evaluate()
    else:
        scores = cal_summarization_metric(gens, refs)
        print('rouge-1:{}\nrouge-2:{}\nrouge-3:{}'.format(scores[0], scores[1], scores[2]))

    return cnt_time 
'''
file_path_dic = {'all': '/home/zzw/generate/data/all_type_dataset/all_data.txt',
                 'baike': '/home/zzw/generate/data/all_type_dataset/baike.txt',
                 'chat_qa': '/home/zzw/generate/data/all_type_dataset/chat_qa.txt',
                 'finance': '/home/zzw/generate/data/all_type_dataset/finance.txt',
                 'law': '/home/zzw/generate/data/all_type_dataset/law.txt',
                 'long_chat': '/home/zzw/generate/data/all_type_dataset/long_chat.txt',
                 'open_qa': '/home/zzw/generate/data/all_type_dataset/open_qa.txt',
                 'sentiment': '/home/zzw/generate/data/all_type_dataset/sentiment.txt',
                 'test': '/home/zzw/generate/data/all_type_dataset/test.txt',
                 'text_classification_dev': '/home/zzw/generate/data/all_type_dataset/text_classification_dev.txt',
                 'text_generate_dev.txt': '/home/zzw/generate/data/all_type_dataset/text_generate_dev.txt.txt',
                 'WikiZh': '/home/zzw/generate/data/wiki_zh_2019/combine_test_1000.txt'
                 }


def solve_pad_size(dataset):
    if dataset == 'WikiZh':
        pad_size = 300

    return pad_size


def solve_dataset(dataset, tokenizer, pad_size):

    if dataset == 'WikiZh':
        train_file = str(file_path_dic['all'])
        AutoDataset = WikiZh2019Dataset

    train_set = AutoDataset(tokenizer, train_file, pad_size)
    return train_set


def clear_dir(dir_path):
    shutil.rmtree(dir_path)
    os.mkdir(dir_path)



def unfreeze_all_modules(model: nn.Module) -> nn.Module:
    for param_name, param in model.named_parameters():
        param.requires_grad = True
    return model





 


