import json
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import BartTokenizer, BartModel
from dataloader_utils import Train_Dataset, Val_Dataset, Evidence_Dataset
from torch.utils.data import DataLoader


def evaluate_dev(question_model, context_model, dev_dataloader, evidence_dataloader, args):
    question_model.eval()
    context_model.eval()
    e_set = set()
    # check GPU is available or not
    if torch.cuda.is_available():
        print('GPU is available! ')
        device = torch.device('cuda')
        question_model.to(device)
        context_model.to(device)
        print('start model')
        # start encoding
        for epoch in range(10):
            dev_embeddings = []
            dev_extend_e_ids = []
            dev_extend_e_texts = []
            for batch_dataset in tqdm(dev_dataloader):
                question_outputs = question_model(
                    input_ids=batch_dataset['inputs_q_input_ids'].cuda(),
                    attention_mask=batch_dataset['inputs_q_attention_mask'].cuda()).last_hidden_state
                question_outputs = question_outputs[:, 0, :].detach()
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

            for i in range(len(dev_embeddings)):
                eval_matrix = torch.mm(dev_embeddings[i], evi_embeddings)
                top_eval_ids = torch.topk(eval_matrix, k=30, dim=1).indices.tolist()
                for j in range(len(top_eval_ids)):
                    for e in top_eval_ids[j]:
                        e_set.add(e)
            del evi_embeddings, extend_e_ids, dev_embeddings, evidence_outputs, question_outputs
        return e_set


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument('--context_model_name', type=str, default='facebook/bart-base')
    parser.add_argument('--question_model_name', type=str, default='facebook/bart-base')
    parser.add_argument('--seed', type=int, default=70)
    parser.add_argument('--lr', type=float, default=2e-6)
    parser.add_argument('--max_length', type=int, default=300)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    if torch.cuda.is_available():
        print('GPU is available! ')
        device = torch.device('cuda')

        # load tokenizers and models
        tokenizer_q = BartTokenizer.from_pretrained(args.question_model_name)
        tokenizer_c = BartTokenizer.from_pretrained(args.context_model_name)
        question_model = BartModel.from_pretrained(args.question_model_name)
        context_model = BartModel.from_pretrained(args.context_model_name)
        evidence_dataset = Evidence_Dataset('less-evidences.json', tokenizer_c, args.max_length)
        dev_dataset = Val_Dataset('new-train-dev-claims.json', tokenizer_q, tokenizer_c, args.max_length)

        evidence_dataloader = DataLoader(evidence_dataset, batch_size=30, shuffle=False, num_workers=4,
                                         collate_fn=evidence_dataset.collate_evidence)
        dev_dataloader = DataLoader(dev_dataset, batch_size=30, shuffle=False, num_workers=4,
                                    collate_fn=dev_dataset.collate_val)
    evi_set = evaluate_dev(question_model, context_model, dev_dataloader, evidence_dataloader, args)
    f = open("project-data/less-evidences.json", "r")
    dataset_evidence1 = json.load(f)
    f.close()
    f2 = open("project-data/evidence.json", "r")
    dataset_evidence2 = json.load(f2)
    f2.close()
    full_evidence_list = list(dataset_evidence2.keys())
    for i in evi_set:
        evi = full_evidence_list[i]
        dataset_evidence1[evi] = dataset_evidence2[evi]
    f_output = open("project-data/new_less_evidence.json", 'w')
    json.dump(dataset_evidence1, f_output)
    f_output.close()
# nohup python -u new-evidence.py >evi.out 2>&1 &
