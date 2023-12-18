import json
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from dataloader_utils import Train_Dataset, Val_Dataset, Evidence_Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import random
import numpy as np


def evaluate_dev(question_model, context_model, dev_dataloader, evidence_dataloader, args):
    question_model.eval()
    context_model.eval()

    f2 = open("/content/drive/MyDrive/project_code2/project-data/evidence.json", "r")
    dataset_evidence = json.load(f2)
    f2.close()
    list1_text = list(dataset_evidence.values())
    list1_id = list(dataset_evidence.keys())

    # check GPU is available or not
    if torch.cuda.is_available():
        print('GPU is available! ')
        device = torch.device('cuda')

        question_model.to(device)
        context_model.to(device)
        tan_func = nn.Tanh()

        print('start model')
        # start encoding
        dev_embeddings = []
        dev_extend_e_ids = []
        dev_extend_e_texts = []
        for batch_dataset in tqdm(dev_dataloader):
            question_outputs = question_model(
                input_ids=batch_dataset['inputs_q_input_ids'].cuda(),
                attention_mask=batch_dataset['inputs_q_attention_mask'].cuda()).last_hidden_state
            question_outputs = question_outputs[:, 0, :]
            question_inputs = nn.functional.normalize(question_outputs, p=2, dim=1).cpu()

            dev_embeddings.append(question_inputs)
            dev_extend_e_ids.append(batch_dataset['evidence_ids'])
            dev_extend_e_texts.append(batch_dataset['evidence_texts'])

        extend_e_ids = []
        extend_e_texts = []
        evi_embeddings = []
        for batch_dataset in tqdm(evidence_dataloader):
            evidence_outputs = context_model(
                input_ids=batch_dataset['evidence_input_ids'].cuda(),
                attention_mask=batch_dataset['evidence_attention_mask'].cuda()).last_hidden_state
            evidence_outputs = evidence_outputs[:, 0, :].detach()
            evidence_inputs = nn.functional.normalize(evidence_outputs, p=2, dim=1).cpu()
            evi_embeddings.append(evidence_inputs)
            extend_e_ids.extend(batch_dataset['evidence_ids_extend'])
            extend_e_texts.extend(batch_dataset['evidence_texts_extend'])

        evi_embeddings = torch.cat(evi_embeddings, dim=0).t()
        tan_evi_embeddings = tan_func(evi_embeddings)
        f_scores = []

        for i in range(len(dev_embeddings)):
            tan_dev_embeddings = tan_func(dev_embeddings[i])
            eval_matrix = torch.mm(tan_dev_embeddings, tan_evi_embeddings)
            top_eval_ids = torch.topk(eval_matrix, k=5, dim=1).indices.tolist()
            for j in range(len(top_eval_ids)):
                evidence_correct = 0
                top_eval_id = []
                for e in top_eval_ids[j]:
                    if e < len(list1_id):
                        top_eval_id.append(list1_id[e])
                    else:
                        top_eval_id.append(random.choice(list1_id))
                for gr_ev in dev_extend_e_ids[i][j]:
                    if gr_ev in top_eval_id:
                        evidence_correct += 1
                evidence_f1score = 0
                if evidence_correct > 0:
                    evidence_recall = float(evidence_correct) / len(dev_extend_e_ids[i][j])
                    evidence_precision = float(evidence_correct) / len(top_eval_id)
                    evidence_f1score = (2 * evidence_precision * evidence_recall) / (
                            evidence_precision + evidence_recall)

                f_scores.append(evidence_f1score)
        mean_f = np.mean(f_scores if len(f_scores) > 0 else [0.0])
        del evi_embeddings, extend_e_ids, dev_embeddings, evidence_outputs, question_outputs
        return mean_f


def train(args):
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        print('GPU is available! ')
        device = torch.device('cuda')

        # load tokenizers and models
        tokenizer_q = AutoTokenizer.from_pretrained(args.question_model_name)
        tokenizer_c = AutoTokenizer.from_pretrained(args.context_model_name)
        question_model = AutoModel.from_pretrained(args.question_model_name)
        context_model = AutoModel.from_pretrained(args.context_model_name)

        train_dataset = Train_Dataset('new-train-claims-negative.json', tokenizer_q, tokenizer_c, args.max_length)
        evidence_dataset = Evidence_Dataset('reduced-evidences.json', tokenizer_c, args.max_length)
        dev_dataset = Val_Dataset('new-dev-claims.json', tokenizer_q, tokenizer_c, args.max_length)

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                      collate_fn=train_dataset.collate_train)
        evidence_dataloader = DataLoader(evidence_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                         collate_fn=evidence_dataset.collate_evidence)
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                    collate_fn=dev_dataset.collate_val)

        question_model.to(device)
        context_model.to(device)

        epochs = args.epoch
        # optimize
        question_optimizer = torch.optim.Adam(question_model.parameters(), lr=args.lr, eps=3e-8)
        context_optimizer = torch.optim.Adam(context_model.parameters(), lr=args.lr, eps=3e-8)
        question_scheduler = LambdaLR(question_optimizer, lr_lambda=lambda ep: 1 / (epochs + 1))
        context_scheduler = LambdaLR(context_optimizer, lr_lambda=lambda ep: 1 / (epochs + 1))

        tan_func = nn.Tanh()
        print('val evaluation: \n')
        avg_loss = 0
        step_count = 0
        max_f_score = 0


        # train

        for epoch in range(20):
            question_model.train()
            context_model.train()
            print('epoch: ', epoch, '\n')
            loss_list = []
            for i, batch_dataset in enumerate(tqdm(train_dataloader)):
                question_optimizer.zero_grad()
                context_optimizer.zero_grad()

                question_outputs = question_model(
                    input_ids=batch_dataset['inputs_q_input_ids'].cuda(),
                    attention_mask=batch_dataset['inputs_q_attention_mask'].cuda()).last_hidden_state
                context_outputs = context_model(
                    input_ids=batch_dataset['inputs_c_input_ids'].cuda(),
                    attention_mask=batch_dataset['inputs_c_attention_mask'].cuda()).last_hidden_state
                question_outputs = nn.functional.normalize(question_outputs[:, 0, :], p=2, dim=1).cpu()
                context_outputs = nn.functional.normalize(context_outputs[:, 0, :], p=2, dim=1).cpu()
                output_matrix = torch.mm(question_outputs, context_outputs.t())
                tan_matrix = tan_func(output_matrix)
                softmax_output = - nn.functional.log_softmax(tan_matrix / 0.05, dim=1)
                losses = []
                for j, p_count in enumerate(batch_dataset['positive_count']):
                    loss = torch.mean(softmax_output[j, 0:p_count])
                    losses.append(loss)
                losses = torch.stack(losses)
                losses = losses.mean()
                losses.backward()
                loss_list.append(losses.item())

                step_count += 1
                all_steps = len(train_dataloader) / args.batch_size

                nn.utils.clip_grad_norm_(question_model.parameters(), 1)
                nn.utils.clip_grad_norm_(context_model.parameters(), 1)
                q_lr = question_scheduler.get_last_lr()
                c_lr = context_scheduler.get_last_lr()
                sum_lr = q_lr + c_lr
                context_optimizer.step()
                question_optimizer.step()
                question_scheduler.step()
                context_scheduler.step()
                question_optimizer.zero_grad()
                context_optimizer.zero_grad()
                print('sum_lr', sum_lr)
                del loss, output_matrix, question_outputs, context_outputs, softmax_output
                if step_count <= all_steps and step_count % 2 == 0:
                    f_score = evaluate_dev(question_model, context_model, dev_dataloader, evidence_dataloader, args)
                    if f_score > max_f_score:
                        max_f_score = f_score
                    avg_loss = sum(loss_list) / step_count
                    print('avg_loss', avg_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument('--context_model_name', type=str, default='roberta-base')
    parser.add_argument('--question_model_name', type=str, default='roberta-base')
    parser.add_argument('--seed', type=int, default=70)
    parser.add_argument('--lr', type=float, default=4e-5)
    parser.add_argument('--max_length', type=int, default=20)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    train(args)

    # wandb.finish()

# nohup python -u main.py >train.out 2>&1 &