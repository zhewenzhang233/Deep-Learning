#coding=gbk
import torch
from transformers import BertTokenizer


# tokenizer = BertTokenizer.from_pretrained(r'C:\Users\12902\Desktop\Topic-to-Essay\Adapter_Less2More\plm\gpt2')
#
# # c = tokenizer.encode_plus('[SEP][SEP][CLS][CLS]', padding='max_length', max_length=12,
# #                           truncation=True,
# #                           return_tensors='pt',
# #                           add_special_tokens=True)
#
# c = tokenizer.encode_plus('[CLS]', return_tensors='pt')
#
# input_ids = c['input_ids'][0]
# attention_mask = c['attention_mask'][0]
#
# print(input_ids)
# print(int(attention_mask.sum()))


# from transformers import GPT2LMHeadModel
#
#
# gpt2 = GPT2LMHeadModel.from_pretrained(r'C:\Users\12902\Desktop\Topic-to-Essay\Adapter_Less2More\plm\gpt2')
#
# # gpt2.add_adapter('ada-bert', config=cfg)
# # gpt2.train_adapter('ada-bert')
# # gpt2.set_active_adapters("ada-bert")
#
#
# for name, sub_module in gpt2.named_modules():
#     if 'mlp.c_fc' in name or 'mlp.c_proj' in name:
#         for param_name, param in sub_module.named_parameters():
#             if 'bias' in param_name:
#                 param.requires_grad = True

#
# def normalize_gens_and_refs(inputs, gens, refs):
#     pre_input = None
#
#     new_gens, new_refs = [], []
#
#     for input, gen, ref in zip(inputs, gens, refs):
#         if pre_input is None:
#             new_gens.append(gen)
#         elif input != pre_input:
#             new_refs.append('')
#             new_gens.append(gen)
#
#         new_refs.append(ref)
#         pre_input = input
#
#     return new_gens, new_refs
#
#
# inputs = ['aa', 'aa', 'bb', 'bb', 'bb']
# gens = ['1', '1', '2', '2', '2']
# refs = ['1', '1', '22', '22', '22']
#
# print(normalize_gens_and_refs(inputs, gens, refs))


# from transformers import GPT2Tokenizer
#
# tokenizer = GPT2Tokenizer.from_pretrained(r'D:\gpt2')
# tokenizer.add_special_tokens({'bos_token':'[CLS]'})
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.bos_token = tokenizer.eos_token
#
#
# e = tokenizer.eos_token
# c = tokenizer.batch_encode_plus([e + 'i am' + e, e + 'he is' + e],
#                                 padding='max_length',
#                                 max_length=6,
#                                 add_special_tokens=True)
# print(c)
# c = tokenizer.batch_decode(c['input_ids'], skip_special_tokens=True)
#
#
# print(c)


from transformers import GPT2Tokenizer, GPT2LMHeadModel
from model import Model

'''hf_model_path = '/home/zzw/generate/Wenzhong-GPT2-110M'
ckpt_path = '/home/zzw/generate/own-test-GPT2/gpt2_test.ckpt'
tokenizer = GPT2Tokenizer.from_pretrained(hf_model_path)
model = GPT2LMHeadModel.from_pretrained(hf_model_path)
# model.load_state_dict(torch.load(ckpt_path))'''
hf_model_path = '/home/zzw/generate/Wenzhong-GPT2-110M'
ckpt_path = '/home/zzw/generate/own-test-GPT2'
# model = Model(ckpt_path)
# model.load_state_dict(torch.load('/home/zzw/generate/own-test-GPT2/gpt2_test.ckpt'))
model = GPT2LMHeadModel.from_pretrained(ckpt_path )
tokenizer = GPT2Tokenizer.from_pretrained(ckpt_path)
model.eval()


# question = '请回答以下问题：北京是哪个国家的首都？ A.中国 B.美国 C.英国 D.俄罗斯'
# question = '请阅读以下问题的文本，16世纪以后，中国为什么没能产生近代根本原因'
# question = '回答以下问题：我公司一工作人员（农民工），在外地出差，但中途急诊，那么他的费用在北京社保能报销吗？'
# question = '请阅读下面的文本并给文本分类，我抽中了明天晚上可以去北京人艺看《窝头会馆》了！！，我太高兴了！！！该文本的分类应该属于。'
# question = '以下是一段A和他的朋友B之间的多轮对话：A：爷爷家养了一只小狗，它总冲我叫。'
# question = '11日下午，中共中央政治局常委、中央书记处书记刘云山登门看望了国家最高科技奖获得者于敏、张存浩。刘云山指出，广大科技工作者要学习老一辈科学家求真务实的钻研精神，淡泊名利、潜心科研，努力创造更多一流科研成果。，下面是该文本的总结'
# question = '请阅读下面的文本并给文本分类，如此精致的台球游戏你值得拥有。,,提升了稳定性。'
# question = 'U系列是最好的，采用国际顶尖技术（由格力自主研发）双级变频压缩机，提高压缩机运转效率，制冷制热能力更强劲；1赫兹变频技术，使空调相当于一个15 W电灯泡，更加节能省电；送风面积广，风力大；生态风，净化空气。非常不错，现在国美在做活动，可以了解一下。 根据文本回答以下问题：格力空调哪个系好'
question = '美利坚合众国（The United States of America），简称美国，首都华盛顿。位于北美洲中部，北与加拿大接壤，南靠墨西哥湾，西临太平洋，东濒大西洋。大部分地区属大陆性气候，南部属亚热带气候，地形总体西高东低，自然资源丰富，矿产资源总探明储量居世界首位。根据文本回答以下问题：美国的气候是什么？'
inputs = tokenizer(question, return_tensors='pt')
generation_output = model.generate(**inputs,
                                return_dict_in_generate=True,
                                output_scores=True,
                                max_length=1000,
                                do_sample=True,
                                top_p = 0.6,
                                num_return_sequences = 5)
for idx, sentence in enumerate(generation_output.sequences):
      print('next sentence %d:\n'%idx, tokenizer.decode(sentence).split('<|endoftext|>')[0])








