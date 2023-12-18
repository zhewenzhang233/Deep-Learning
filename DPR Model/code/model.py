from torch import nn
from transformers import AutoModel


class DPR_CLF_Model(nn.Module):
    def __init__(self, model_name):

        super(DPR_CLF_Model, self).__init__()
        self.model_clf = AutoModel.from_pretrained(model_name)
        # self.tan = nn.Tanh()
        self.Linear1 = nn.Linear(768, 4)
        # self.c_e = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask):

        encoding = self.model_clf(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        print('encoding ', encoding.shape)
        encoding = encoding[:, 0, :]
        print('encoding1 ', encoding.shape)
        logits = self.Linear1(encoding)
        print('logitslogits ', logits.shape)

        return logits


