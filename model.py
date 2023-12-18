# from transformers.models.gpt2.modeling_gpt2 import *
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from prefix import *


class Model(nn.Module):
    def __init__(self,  model_name):
        super().__init__()
        # 'IDEA-CCNL/Wenzhong2.0-GPT2-3.5B-chinese'
        # self.gpt2 = GPT2LMHeadModel.from_pretrained(model_name)
        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_name)

        # tokenizer = GPT2Tokenizer.from_pretrained('IDEA-CCNL/Wenzhong2.0-GPT2-3.5B-chinese')
        # text = "Replace me by any text you'd like."
        # encoded_input = tokenizer(text, return_tensors='pt')
        # output = model(**encoded_input)

    def forward(self, tgt, tgt_mask, pos):
        bs, tgt_len = tgt.size()
        input_ids = tgt
        attention_mask = tgt_mask

        labels = input_ids.clone()
        labels = torch.where(attention_mask == 0, -100, labels)

        loss = self.gpt2(input_ids=input_ids,
                         attention_mask=attention_mask,
                         position_ids=pos,
                         labels=labels).loss
        return loss


    '''def infer(self, src, src_mask, pos, max_length, bos_id, eos_id):
        text = "Replace me by any text you'd like."
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)


        #set_seed(55)
        #generator = pipeline('text-generation', model='IDEA-CCNL/Wenzhong2.0-GPT2-3.5B-chinese')
        #generator("北京位于", max_length=30, num_return_sequences=1)

        return outputs'''
