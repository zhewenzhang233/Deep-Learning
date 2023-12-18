import json
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer,AutoModel
from dataloader_utils import Train_Dataset, Val_Dataset, Evidence_Dataset, Test_Dataset
from torch.utils.data import DataLoader



def cos_similarity(guess_id_list, guess_text_list, true_id_list, true_text_list, dev_claim):
    predict_id_list = [] 
    predict_text_list = []
    simi_dict = dict()
    for jdx in range(len(guess_text_list)):
        s1 = dev_claim
        print('dev_claim ', dev_claim)
        s2 = guess_text_list[jdx]
        vocab = set(s1.split() + s2.split())
        word_to_idx = {word: i for i, word in enumerate(vocab)}
        vec1 = torch.zeros(len(vocab))
        vec2 = torch.zeros(len(vocab))
        for word in s1.split():
            vec1[word_to_idx[word]] = 1
        for word in s2.split():
            vec2[word_to_idx[word]] = 1
        similarity = float(nn.functional.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)))
        simi_dict[similarity] = guess_text_list[jdx]
    len_true = len(true_id_list)
    dict_list = list(simi_dict.keys())
    dict_list.sort(reverse=True)
    dict_list_max = dict_list[0:len_true]
    for i in dict_list_max:
        predict_text_list.append(simi_dict[i])
    print('dict_list_max', dict_list_max)
    print('predict_text_list', predict_text_list)
    return predict_text_list


def evaluate_dev(question_model, context_model, dev_dataloader, evidence_dataloader, args):
    question_model.eval()
    context_model.eval()
    
    predict_ids = []
    predict_texts = []

    f1 = open("project-data/new-train-claims.json", "r")
    dataset_train = json.load(f1)
    f1.close()
    
    f2 = open("project-data/evidence.json", "r")
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

        # start encoding
        dev_embeddings = []
        dev_extend_e_ids = []
        dev_extend_e_texts = []
        dev_claims = []
        for batch_dataset in tqdm(dev_dataloader):
            question_outputs = question_model(
                input_ids=batch_dataset['inputs_q_input_ids'].cuda(),
                attention_mask=batch_dataset['inputs_q_attention_mask'].cuda()).last_hidden_state
            question_outputs = question_outputs[:, 0, :].detach()
            question_inputs = nn.functional.normalize(question_outputs, p=2, dim=1).cpu()

            dev_embeddings.append(question_inputs)
            dev_extend_e_ids.append(batch_dataset['evidence_ids'])
            dev_extend_e_texts.append(batch_dataset['evidence_texts'])
            dev_claims.append(batch_dataset['combine'])
            del question_outputs

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
            extend_e_ids.extend(batch_dataset['evidence_ids'])
            extend_e_texts.extend(batch_dataset['evidence_texts'])
            del evidence_outputs

        evi_embeddings = torch.cat(evi_embeddings, dim=0).t()
        tan_evi_embeddings = tan_func(evi_embeddings)

        for i in range(len(dev_embeddings)):
            tan_dev_embeddings = tan_func(dev_embeddings[i])
            eval_matrix = torch.mm(tan_dev_embeddings, tan_evi_embeddings)
            top_eval_ids = torch.topk(eval_matrix, k=30, dim=1).indices.tolist()
            for j in range(len(top_eval_ids)):
                val_id = dev_extend_e_ids[i][j]
                val_text = dev_extend_e_texts[i][j]
                claim = dev_claims[i][j]
                print('claims ', claim)
                top_eval_id = []
                top_eval_text = []
                for e in top_eval_ids[j]:
                    top_eval_id.append(list1_id[e])
                    top_eval_text.append(list1_text[e])
                predict_text_list = cos_similarity(top_eval_id, top_eval_text, val_id, val_text, claim)
                predict_texts.append(predict_text_list)
                
        del evi_embeddings, dev_embeddings, extend_e_ids, extend_e_texts, dev_extend_e_ids, dev_extend_e_texts
        key_list = list(dataset_train.keys())
        if len(key_list) == len(predict_texts):
            for kidx in range(len(key_list)):
                dataset_train[key_list[kidx]]['negative_evidence_text'] = predict_texts[kidx]
    return dataset_train


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument('--context_model_name', type=str, default='roberta-base')
    parser.add_argument('--question_model_name', type=str, default='roberta-base')
    parser.add_argument('--seed', type=int, default=70)
    parser.add_argument('--lr', type=float, default=2e-6)
    parser.add_argument('--max_length', type=int, default=200)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    if torch.cuda.is_available():
        print('GPU is available! ')
        device = torch.device('cuda')

        # load tokenizers and models
        tokenizer_q = AutoTokenizer.from_pretrained(args.question_model_name)
        tokenizer_c = AutoTokenizer.from_pretrained(args.context_model_name)
        question_model = AutoModel.from_pretrained(args.question_model_name)
        context_model = AutoModel.from_pretrained(args.context_model_name)

        evidence_dataset = Evidence_Dataset('new_less_evidence.json', tokenizer_c, args.max_length)
        dev_dataset = Val_Dataset('new-train-claims.json', tokenizer_q, tokenizer_c, args.max_length)

        evidence_dataloader = DataLoader(evidence_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                         collate_fn=evidence_dataset.collate_evidence)
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                    collate_fn=dev_dataset.collate_val)

    new_dataset = evaluate_dev(question_model, context_model, dev_dataloader, evidence_dataloader, args)

    f_output = open("project-data/new_train_negative1.json", 'w')
    json.dump(new_dataset, f_output)
    f_output.close()
    
    # nohup python -u generate_negative_evidence.py >n1.out 2>&1 &