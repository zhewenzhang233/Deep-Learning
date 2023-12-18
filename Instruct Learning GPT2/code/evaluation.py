import os

from transformers import GPT2Tokenizer

import torch
from torch.utils.data import DataLoader
import argparse
from model import Model
from utils import eval_, solve_test, solve_pad_size,  send_complete_email, move_cur_task_trained_module

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='gpt2-medium')
    parser.add_argument('--load_prefix_modules_file', type=str,
                        default='prefix_trained_modules')
    parser.add_argument('--temp_store_prefix_modules', type=str,
                        default='temp_store')

    args = parser.parse_args()
    dataset = args.dataset.strip()
    mode = args.mode.strip()
    model_name = args.model_name.strip()
    load_prefix_modules_file = args.load_prefix_modules_file.strip()
    temp_store_prefix_modules = args.temp_store_prefix_modules.strip()

    if mode == 'prefixFusion':
        move_cur_task_trained_module(dataset, load_prefix_modules_file,
                                     temp_store_prefix_modules)

    assert mode in ['ft', 'mam',  'adapter', 'prefix', 'prefixFusion']
    assert dataset in ['DART', 'E2E', 'WebNLG', 'XSum', 'Wiki']

    gen_file = './gen/' + dataset + '_' + mode + '_gen.txt'

    save_model_path = 'gpt2-tuned.pkl'

    input_pad_size, output_pad_size = solve_pad_size(dataset)
    pos_tensor = torch.arange(input_pad_size + output_pad_size).long()[None, :]

    model = Model(mode, model_name, load_prefix_modules_file).cuda()
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.bos_token = tokenizer.eos_token

    test_set = solve_test(dataset, tokenizer, input_pad_size, output_pad_size)
    test_loader = DataLoader(test_set,  batch_size=6, num_workers=1)

    eval_gen = output_pad_size

    print('input length: {}, output length: {}'.format(input_pad_size, eval_gen))
    checkpoint = torch.load(save_model_path)
    model.load_state_dict(checkpoint)
    eval_(test_loader, tokenizer, model, dataset, input_pad_size, eval_gen, gen_file)

    if mode == 'prefix':
         kv = model.prefix_net.simple_forward()
         torch.save({'kv': kv}, 'prefix_trained_modules/{}.pt'.format(dataset))

    if mode == 'prefixFusion':
        move_cur_task_trained_module(dataset, temp_store_prefix_modules,
                                     load_prefix_modules_file)

    send_complete_email(task=dataset, mode=mode)