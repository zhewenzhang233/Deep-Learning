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


# question = '��ش��������⣺�������ĸ����ҵ��׶��� A.�й� B.���� C.Ӣ�� D.����˹'
# question = '���Ķ�����������ı���16�����Ժ��й�Ϊʲôû�ܲ�����������ԭ��'
# question = '�ش��������⣺�ҹ�˾һ������Ա��ũ�񹤣�������س������;�����ô���ķ����ڱ����籣�ܱ�����'
# question = '���Ķ�������ı������ı����࣬�ҳ������������Ͽ���ȥ�������տ�����ͷ��ݡ��ˣ�������̫�����ˣ��������ı��ķ���Ӧ�����ڡ�'
# question = '������һ��A����������B֮��Ķ��ֶԻ���A��үү������һֻС�������ܳ��ҽС�'
# question = '11�����磬�й��������ξֳ�ί��������Ǵ��������ɽ���ſ����˹�����߿Ƽ���������������Ŵ�ơ�����ɽָ�������Ƽ�������Ҫѧϰ��һ����ѧ��������ʵ�����о��񣬵���������Ǳ�Ŀ��У�Ŭ���������һ�����гɹ����������Ǹ��ı����ܽ�'
# question = '���Ķ�������ı������ı����࣬��˾��µ�̨����Ϸ��ֵ��ӵ�С�,,�������ȶ��ԡ�'
# question = 'Uϵ������õģ����ù��ʶ��⼼�����ɸ��������з���˫����Ƶѹ���������ѹ������תЧ�ʣ���������������ǿ����1���ȱ�Ƶ������ʹ�յ��൱��һ��15 W����ݣ����ӽ���ʡ�磻�ͷ�����㣬��������̬�磬�����������ǳ��������ڹ���������������˽�һ�¡� �����ı��ش��������⣺�����յ��ĸ�ϵ��'
question = '��������ڹ���The United States of America��������������׶���ʢ�١�λ�ڱ������в���������ô�������Ͽ�ī�����壬����̫ƽ�󣬶��������󡣴󲿷ֵ�������½�������ϲ������ȴ����򣬵����������߶��ͣ���Ȼ��Դ�ḻ�������Դ��̽��������������λ�������ı��ش��������⣺������������ʲô��'
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








