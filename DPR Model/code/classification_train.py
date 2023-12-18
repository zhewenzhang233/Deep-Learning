
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from dataloader_utils import Classification_Train_Dataset
from torch.utils.data import DataLoader
from model import DPR_CLF_Model
from torch.optim.lr_scheduler import LambdaLR
import wandb


# wandb.init(project="nlp_project", name="cls-roberta")
def train(args):
    label_list = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED"]
    label_int_dict = {"SUPPORTS": 0, "REFUTES": 1, "NOT_ENOUGH_INFO": 2, "DISPUTED": 3}
    label_int_dict_lower = {"supports": 0, "refutes": 1, "not_enough_info": 2, "disputed": 3}

    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        print('GPU is available! ')
        device = torch.device('cuda')

        # load tokenizer and model

        tokenizer_q = AutoTokenizer.from_pretrained('roberta-base')
        question_model = DPR_CLF_Model('roberta-base')
        question_model.load_state_dict(torch.load(args.question_model_name))

        train_dataset = Classification_Train_Dataset("new-train-claims-negative.json", tokenizer_q, args.max_length)
        dev_dataset = Classification_Train_Dataset("new-dev-claims.json", tokenizer_q, args.max_length)

        train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers=4,
                                      collate_fn=train_dataset.collate_cls)
        dev_dataloader = DataLoader(dev_dataset, batch_size=2, shuffle=False, num_workers=4,
                                    collate_fn=dev_dataset.collate_cls)

        question_model.to(device)

        epochs = args.epoch
        question_optimizer = torch.optim.Adam(question_model.parameters(), lr=args.lr, eps=2e-9)
        question_scheduler = LambdaLR(question_optimizer, lr_lambda=lambda ep: 1 / (epochs + 1))
        c_e = nn.CrossEntropyLoss()
        step_count = 0
        max_f_score = 0
        for epoch in range(20):
            question_model.train()

            print('epoch: ', epoch, '\n')
            loss_list = []
            for i, batch_dataset in enumerate(tqdm(train_dataloader)):
                question_optimizer.zero_grad()
                labels = batch_dataset['labels']
                labels_ixd = []
                for i in labels:
                    idx = label_int_dict_lower[i]
                    labels_ixd.append(idx)
                tensor_idx = torch.tensor(labels_ixd, dtype=torch.long)
                question_logits = question_model(
                    input_ids=batch_dataset['inputs_q_input_ids'].cuda(),
                    attention_mask=batch_dataset['inputs_q_input_ids'].cuda()
                )
                question_logits = torch.tensor(question_logits, dtype=torch.float)
                question_loss = c_e(question_logits, tensor_idx.cuda())
                question_loss.requires_grad = True
                question_loss.backward()
                loss_list.append(question_loss.item())

                nn.utils.clip_grad_norm_(question_model.parameters(), 1)

                question_optimizer.step()
                question_scheduler.step()
                question_optimizer.zero_grad()
                step_count += 1
                all_steps = len(train_dataloader) / args.batch_size
                del question_loss, question_logits
                question_model.eval()
                if step_count <= all_steps and step_count%2 == 0:
                    cnt = 0.
                    correct_cnt = 0.
                    for b in tqdm(dev_dataloader):
                        question_model.to(device)
                        logits = question_model(
                            input_ids=b['inputs_q_input_ids'].cuda(),
                            attention_mask=b['inputs_q_attention_mask'].cuda())
                        lo = - logits
                        predict_labels = torch.argmax(lo, dim=1)
                        b_list = list(b["labels"])
                        label_list = []
                        for j in b_list:
                            label_list.append(label_int_dict_lower[j])
                        predict_labels = predict_labels[0:len(label_list)]
                        for j in range(len(predict_labels)):
                            if predict_labels[j] == label_list[j]:
                                correct_cnt += 1
                        cnt += len(predict_labels)
                    acc = correct_cnt / cnt
                    # wandb.log({'acc: ': acc}, step=step_count)
                    if acc > max_f_score:
                        max_f_score = acc
                    avg_loss = sum(loss_list) / step_count
                    # wandb.log({"loss": avg_loss}, step=step_count)
                    lr = step_count * (args.lr - 2e-8) / all_steps + 2e-8
                    # wandb.log({"learning_rate": lr}, step=step_count)
        # torch.save(question_model.state_dict(), 'q_model/question_cls_ckpt.bin')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument('--context_model_name', type=str, default='q_model/question_ckpt.bin')
    parser.add_argument('--question_model_name', type=str, default='q_model/question_cls_ckpt.bin')
    parser.add_argument('--seed', type=int, default=70)
    parser.add_argument('--lr', type=float, default=2e-6)
    parser.add_argument('--max_length', type=int, default=500)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    train(args)
# nohup python -u classification_train.py >cls.out 2>&1 &