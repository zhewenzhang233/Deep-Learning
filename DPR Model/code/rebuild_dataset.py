import json
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import BartTokenizer, BartModel
from dataloader_utils import Val_Dataset
from torch.utils.data import DataLoader


# collect all the evidence ids from the datasets
def find_all_evidence_in_files():
    evidence_id_set = set()

    f1 = open("project-data/train-claims.json", "r")
    dataset_train = json.load(f1)
    f1.close()
    f2 = open("project-data/dev-claims.json", "r")
    dataset_dev = json.load(f2)
    f2.close()
    f3 = open("project-data/dev-claims-baseline.json", "r")
    dataset_dev_base = json.load(f3)
    f3.close()

    for i in dataset_train.keys():
        for j in dataset_train[i]['evidences']:
            evidence_id_set.add(j)

    for i in dataset_dev.keys():
        for j in dataset_dev[i]['evidences']:
            evidence_id_set.add(j)

    for i in dataset_dev_base.keys():
        for j in dataset_dev_base[i]['evidences']:
            evidence_id_set.add(j)

    return evidence_id_set


def get_the_most_possible_evidences(args):
    evidence_index_set = set()
    # check GPU is available or not
    if torch.cuda.is_available():
        print('GPU is available! ')
        device = torch.device('cuda')

        # load tokenizers and models
        tokenizer_q = BartTokenizer.from_pretrained(args.question_model_name)
        tokenizer_c = BartTokenizer.from_pretrained(args.context_model_name)
        question_model = BartModel.from_pretrained(args.question_model_name)
        context_model = BartModel.from_pretrained(args.context_model_name)

        print('start dataloader')
        # load dataset and put it lin dataloader
        dataset1 = Val_Dataset('new-train-dev-claims.json', tokenizer_q, tokenizer_c, args.max_length)
        dataloader = DataLoader(dataset1, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                collate_fn=dataset1.collate_val)
        question_model.to(device)
        context_model.to(device)

        new_train_dataset = dataset1

        tan_func = nn.Tanh()

        print('start model')
        # start encoding
        for batch_dataset in tqdm(dataloader):

            # question encoding
            question_outputs = question_model(
                input_ids=batch_dataset['inputs_q_input_ids'].cuda(),
                attention_mask=batch_dataset['inputs_q_attention_mask'].cuda()).last_hidden_state
            # print(question_outputs.shape)
            question_outputs = question_outputs[:, 0, :]
            print('question_outputs', question_outputs.shape)
            question_inputs = nn.functional.normalize(question_outputs, p=2, dim=1).cpu()

            context_outputs_list = []
            for i in range(len(batch_dataset['inputs_c_input_ids_list'])):
                # context encoding
                context_output = context_model(
                    input_ids=batch_dataset['inputs_c_input_ids_list'][i].cuda(),
                    attention_mask=batch_dataset['inputs_c_attention_mask_list'][i].cuda()).last_hidden_state
                context_output = context_output[:, 0, :].detach()
                context_output = nn.functional.normalize(context_output.detach(), p=2, dim=1).cpu()
                context_outputs_list.append(context_output)
            context_outputs = torch.cat(context_outputs_list, dim=0).t()
            print('context_outputs', question_outputs.shape)

            # get the index of evidences
            t_q_output = tan_func(question_inputs)
            t_c_output = tan_func(context_outputs)
            question_context_matrix = torch.mm(t_q_output, t_c_output)
            matrix_shape = question_context_matrix.shape
            top_evidence_ids1 = torch.topk(question_context_matrix, k=matrix_shape[0], dim=1).indices.tolist()
            print(top_evidence_ids1)
            for iii in range(len(top_evidence_ids1)):
                for j in top_evidence_ids1[iii]:
                    evidence_index_set.add(j)

    return list(evidence_index_set)


def get_negative_evidences(args):
    
    f1 = open("project-data/new-train-claims.json", "r")
    dataset_train = json.load(f1)
    f1.close()
    
    new_train_dataset = dataset_train
    evidence_index_set = set()
    # check GPU is available or not
    if torch.cuda.is_available():
        print('GPU is available! ')
        device = torch.device('cuda')

        # load tokenizers and models
        tokenizer_q = BartTokenizer.from_pretrained(args.question_model_name)
        tokenizer_c = BartTokenizer.from_pretrained(args.context_model_name)
        question_model = BartModel.from_pretrained(args.question_model_name)
        context_model = BartModel.from_pretrained(args.context_model_name)

        print('start negative dataloader')
        # load dataset and put it in dataloader
        dataset1 = Val_Dataset('new-train-claims.json', tokenizer_q, tokenizer_c, args.max_length)
        dataloader = DataLoader(dataset1, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                collate_fn=dataset1.collate_val)
        question_model.to(device)
        context_model.to(device)

        print('start negative model')
        # start encoding
        for batch_dataset in tqdm(dataloader):

            # question encoding
            question_outputs = question_model(
                input_ids=batch_dataset['inputs_q_input_ids'].cuda(),
                attention_mask=batch_dataset['inputs_q_attention_mask'].cuda()).last_hidden_state
            print(question_outputs.shape)
            question_outputs = question_outputs[:, 0, :]
            print(question_outputs.shape)
            question_inputs = nn.functional.normalize(question_outputs, p=2, dim=1).cpu()

            context_outputs_list = []
            for i in range(len(batch_dataset['inputs_c_input_ids_list'])):
                # context encoding
                context_output = context_model(
                    input_ids=batch_dataset['inputs_c_input_ids_list'][i].cuda(),
                    attention_mask=batch_dataset['inputs_c_attention_mask_list'][i].cuda()).last_hidden_state
                context_output = context_output[:, 0, :].detach()
                context_output = nn.functional.normalize(context_output.detach(), p=2, dim=1).cpu()
                context_outputs_list.append(context_output)
            context_outputs = torch.cat(context_outputs_list, dim=0).t()

            # get the index of evidences
            question_context_matrix = torch.mm(question_inputs, context_outputs)
            top_evidence_ids = torch.topk(question_context_matrix, k=5, dim=1).indices.tolist()


            e_id_list = []
            e_text_list = []
            for i in range(len(batch_dataset['evidence_ids'])):
                e_id_list.extend(batch_dataset['evidence_ids'][i])
                e_text_list.extend(batch_dataset['evidence_texts'][i])

            # get negative evidences
            new_train_dataset = dataset_train
            for idx in range(len(batch_dataset['claim_ids'])):
                c_id = batch_dataset['claim_ids'][idx]
                new_train_dataset[c_id] = dataset_train[c_id]
                negative_id = []
                negative_text = []
                for i in top_evidence_ids[idx]:
                    if i < len(e_id_list):
                        negative_id.append(e_id_list[i])
                        negative_text.append(e_text_list[i])
                new_train_dataset[c_id]['negative_evidence'] = negative_id
                new_train_dataset[c_id]['negative_evidence_text'] = negative_text

    return new_train_dataset


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument('--context_model_name', type=str, default='facebook/bart-base')
    parser.add_argument('--question_model_name', type=str, default='facebook/bart-base')
    parser.add_argument('--seed', type=int, default=70)
    parser.add_argument('--lr', type=float, default=2e-6)
    parser.add_argument('--max_length', type=int, default=500)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    e_ids_set = find_all_evidence_in_files()
    e_index = get_the_most_possible_evidences(args)

    f = open("project-data/evidence.json", "r")
    dataset_evidence = json.load(f)
    f.close()

    dataset_evidence_key_list = list(dataset_evidence.keys())
    for i in e_index:
        e_ids_set.add(dataset_evidence_key_list[i])

    e_ids = list(e_ids_set)
    new_evidence_dict = dict()
    for i in e_ids:
        new_evidence_dict[i] = dataset_evidence[i]

    # output a new evidence dataset
    f_output = open("project-data/less-evidences.json", 'w')
    json.dump(new_evidence_dict, f_output)
    f_output.close()

    # output a new train dataset with negative evidence
    '''for i in new_t_dataset.keys():
        e_list = []
        for j in new_t_dataset[i]['negative_evidence']:
            e_list.append(dataset_evidence[j])
        new_t_dataset[i]['negative_evidence_text'] = e_list
    f_output = open("project-data/new-dev-train-claims-negative.json", 'w')
    json.dump(new_t_dataset, f_output)
    f_output.close()'''


# nohup python -u rebuild_dataset.py >neg.out 2>&1 &
