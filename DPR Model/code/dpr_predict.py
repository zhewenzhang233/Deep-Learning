import json
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer,AutoModel
from dataloader_utils import Evidence_Dataset, Test_Dataset
from torch.utils.data import DataLoader


def cos_similarity(e_list, text_list, claim):
    predict_id_list = []
    simi_dict = dict()
    for jdx in range(len(e_list)):
        s1 = claim
        s2 = text_list[jdx]
        vocab = set(s1.split() + s2.split())
        word_to_idx = {word: i for i, word in enumerate(vocab)}
        vec1 = torch.zeros(len(vocab))
        vec2 = torch.zeros(len(vocab))
        for word in s1.split():
            vec1[word_to_idx[word]] = 1
        for word in s2.split():
            vec2[word_to_idx[word]] = 1
        similarity = float(nn.functional.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)))
        simi_dict[similarity] = e_list[jdx]
    len_true = len(e_list)
    dict_list = list(simi_dict.keys())
    dict_list.sort(reverse=True)
    dict_list_max = dict_list[0:len_true]
    for i in dict_list_max:
        predict_id_list.append(simi_dict[i])
    return predict_id_list


def predict_result(args):
    
    f2 = open("project-data/evidence.json", "r")
    dataset_evidence = json.load(f2)
    f2.close()
    list1_text = list(dataset_evidence.values())
    list1_id = list(dataset_evidence.keys())
    
    tokenizer_q = AutoTokenizer.from_pretrained('roberta-base')
    tokenizer_c = AutoTokenizer.from_pretrained('roberta-base')
    question_model = AutoModel.from_pretrained('roberta-base')
    question_model.load_state_dict(torch.load(args.question_model_name))
    context_model = AutoModel.from_pretrained('roberta-base')
    context_model.load_state_dict(torch.load(args.context_model_name))

    test_dataset = Test_Dataset('test-claims-unlabelled.json', tokenizer_q, args.max_length)
    evidence_dataset = Evidence_Dataset('less-evidences.json', tokenizer_c, args.max_length)

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                 collate_fn=test_dataset.collate_test)
    evidence_dataloader = DataLoader(evidence_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                     collate_fn=evidence_dataset.collate_evidence)
    
    question_model.cuda()
    context_model.cuda()
    question_model.eval()
    context_model.eval()
    
    tan_func = nn.Tanh()
    
    extend_e_ids = []
    extend_e_texts = []
    evi_embeddings = []
    for batch_dataset in tqdm(evidence_dataloader):
        evidence_outputs = context_model(
            input_ids=batch_dataset['evidence_input_ids'].cuda(),
            attention_mask=batch_dataset['evidence_attention_mask'].cuda()).last_hidden_state
        # print(question_outputs.shape)
        evidence_outputs = evidence_outputs[:, 0, :].detach()
        # print(question_outputs.shape)
        evidence_inputs = nn.functional.normalize(evidence_outputs, p=2, dim=1).cpu()
        evi_embeddings.append(evidence_inputs)
        extend_e_ids.extend(batch_dataset['evidence_ids_extend'])
        extend_e_texts.extend(batch_dataset['evidence_texts'])
        del evidence_outputs
    evi_embeddings = torch.cat(evi_embeddings, dim=0).t()

    predict_dict = dict()
    for batch_dataset in tqdm(test_dataloader):
        question_outputs = question_model(
            input_ids=batch_dataset['test_input_ids'].cuda(),
            attention_mask=batch_dataset['test_input_ids'].cuda()).last_hidden_state
        question_outputs = question_outputs[:, 0, :]
        question_inputs = nn.functional.normalize(question_outputs, p=2, dim=1).cpu()
        output_matrix = torch.mm(question_inputs, evi_embeddings)
        outputs = tan_func(output_matrix)
        top_eval_ids = torch.topk(outputs, k=5, dim=1).indices.tolist()
        for idx in range(len(batch_dataset['claim_ids'])):
            predict_dict[batch_dataset['claim_ids'][idx]] = dict()
        for idx in range(len(batch_dataset['claim_ids'])):
            claim = batch_dataset['claim_texts'][idx]
            predict_dict[batch_dataset['claim_ids'][idx]]['claim_text'] = batch_dataset['claim_texts'][idx]
            predict_dict[batch_dataset['claim_ids'][idx]]['claim_label'] = ''
            e_list = []
            text_list = []
            for j in top_eval_ids[idx]:
                e_list.append(list1_id[j])
                text_list.append(list1_text[j])
            e_ids = cos_similarity(e_list, text_list, claim)
            predict_dict[batch_dataset['claim_ids'][idx]]['evidence'] = e_ids
    del question_outputs
    f = open("project-data/test-claims.json", 'w')
    json.dump(predict_dict, f)
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument('--context_model_name', type=str, default='c_model/context_ckpt.bin')
    parser.add_argument('--question_model_name', type=str, default='q_model/question_ckpt.bin')
    parser.add_argument('--seed', type=int, default=70)
    parser.add_argument('--lr', type=float, default=2e-6)
    parser.add_argument('--max_length', type=int, default=300)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    predict_result(args)
# nohup python -u dpr_predict.py >test.out 2>&1 &