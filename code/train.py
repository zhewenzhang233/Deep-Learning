import time
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from torch.utils.data import DataLoader
import argparse
from model import Model
from utils import solve_dataset, solve_pad_size,  unfreeze_all_modules
from torch.utils.data.distributed import DistributedSampler
import sys
from code.prefix import traverse_net, traverse_fg_net, traverse_fgg_net, traverse_fm_net, traverse_lora_net
import torch.distributed as dist
from transformers import WEIGHTS_NAME, CONFIG_NAME

# os.environ['MKL_THREADING_LAYER'] = 'GNU'
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


'''def get_parameter_number(model):
    head_num = 1024 * 50257
    total_num = sum(p.numel() for p in model.parameters()) - head_num
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad) - head_num
    return {'Total(M)': total_num / 1e6, 'Trainable(K)': trainable_num / 1e3,
            'Ratio(%)': trainable_num * 100 / total_num}
'''

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, required=True)
    # parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='WikiZh')
    # parser.add_argument('--model_name', type=str, default='IDEA-CCNL/Wenzhong2.0-GPT2-3.5B-chinese')
    parser.add_argument('--model_name', type=str, default='/home/zzw/generate/Wenzhong-GPT2-110M')
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    args = parser.parse_args()
    dataset = args.dataset.strip()
    model_name = args.model_name.strip()
    epoch = args.epoch  # 循环学习的次数
    batch_size = args.batch_size
    seed = args.seed

    torch.random.manual_seed(seed)

    # 检查是否机器有GPU
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
        print('has device')
    else:
        print('not have device')

    assert dataset in ['WikiZh']

    # log_fo = open('./log/' + dataset + '_' + mode + '.txt', 'w+')
    # gen_file = './gen/' + dataset + '_' + mode + '_gen.txt'

    # save_model_path = 'gpt2-tuned.pkl'

    pad_size = solve_pad_size(dataset)
    pos_tensor = torch.arange(pad_size).long()[None, :]

    model = Model(model_name)
    

    model = unfreeze_all_modules(model)

    # model_p = torch.nn.DataParallel(model.cuda(), device_ids=[_ for _ in range(2)])

    '''
    if torch.distributed.get_rank() == 0:  
        print(get_parameter_number(model.gpt2))

    exit()
    '''
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-7, betas=(0.9, 0.98), eps=1e-9)

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.bos_token = tokenizer.eos_token

    train_set = solve_dataset(dataset, tokenizer, pad_size)

    train_sampler = DistributedSampler(train_set)
    # dev_sampler = DistributedSampler(dev_set)
    # test_sampler = DistributedSampler(test_set)

    train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=batch_size,
                              num_workers=4, pin_memory=True)

    model.to(device)
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print('use {} gpus!'.format(num_gpus))
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    # infer_size = 16
    # dev_loader = DataLoader(dev_set, sampler=dev_sampler, batch_size=infer_size, num_workers=4, pin_memory=True)
    # test_loader = DataLoader(test_set, batch_size=infer_size, num_workers=4)

    eval_gen = pad_size
    total_time = 0

    print('input length: {}, output length: {}'.format(pad_size, eval_gen))

    pre_loss = 1e5

    print('begin to train ...')
    sys.stdout.flush()


    # 开始训练模型
    now_1 = time.time()
    total_loss = 0
    cnt = 0
    for epoch_i in range(epoch):
        model.train()
        train_sampler.set_epoch(epoch)  # shuffle

        for i, batch_data in enumerate(train_loader):
            loss = model(tgt=batch_data[0].cuda(),
                         tgt_mask=batch_data[1].cuda(),
                         pos=pos_tensor.cuda())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            dist.all_reduce(loss.div_(torch.cuda.device_count()))
            total_loss += loss.item()
            cnt += 1
            # 每次epoch的最后输出下一个结果
            print("Epoch %d Step: %d | Loss: %f" % (epoch_i, i, loss.item()))
            sys.stdout.flush()

            
            
        '''
        # -----------* carculate evaluation metrics*---------
        torch.cuda.empty_cache()
        model.eval()
        total_loss = 0
        cnt = 0

        with torch.no_grad():
            for i, batch_data in enumerate(dev_loader):
                loss = model(src=batch_data[0].cuda(),
                               src_mask=batch_data[1].cuda(),
                               tgt=batch_data[2].cuda(),
                               tgt_mask=batch_data[3].cuda(),
                               pos=pos_tensor.cuda())
                dist.all_reduce(loss.div_(torch.cuda.device_count()))
                total_loss += loss.item()
                cnt += 1

        print('val loss : {}'.format(total_loss / cnt))
        # -----------* carculate evaluation metrics*---------
        if pre_loss > total_loss / cnt :
            pre_loss = total_loss / cnt 
            torch.save(model.module.state_dict(), save_model_path)
        total_time += time.time() - start
        '''
      # torch.save(model.module.state_dict(), save_model_path)
      # model_to_save.config.to_json_file(output_config_file)
      # tokenizer.save_vocabulary(save_model_path)
      
      
    now_2 = time.time()

    model = model.module.gpt2

    #    checkpoint = torch.load(save_model_path)
    #    model.load_state_dict(checkpoint)
    torch.cuda.empty_cache()
    # infer_time = eval_(train_loader, tokenizer, model, dataset, pad_size, eval_gen)
    # log_fo.close()

    print('train cost time:{}s'.format(now_2 - now_1))
    # print('infer cost time:{}s'.format(infer_time))
    print('val loss : {}'.format(total_loss / cnt))
    save_model_path = '/home/zzw/generate/test_GPT2_ep20/'
    if pre_loss > total_loss / cnt :
      pre_loss = total_loss / cnt
      '''output_model_file = os.path.join(save_model_path, WEIGHTS_NAME)
      output_config_file = os.path.join(save_model_path, CONFIG_NAME)
      model_to_save = model.module if hasattr(model, 'module') else model
      
      torch.save(model_to_save.state_dict(), output_model_file)
      model_to_save.config.to_json_file(output_config_file)
      tokenizer.save_vocabulary(save_model_path)'''
      tokenizer.save_pretrained(save_model_path)
      model.save_pretrained(save_model_path)
