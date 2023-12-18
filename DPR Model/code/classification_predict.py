import json
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from dataloader_utils import Classification_Test_Dataset
from torch.utils.data import DataLoader
from model import DPR_CLF_Model


def predict(args):
    label_list = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED"]
    label_int_dict = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT_ENOUGH_INFO", 3: "DISPUTED"}
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        print('GPU is available! ')
        device = torch.device('cuda')

        tokenizer_q = AutoTokenizer.from_pretrained('roberta-base')
        question_model = DPR_CLF_Model('roberta-base')
        question_model.load_state_dict(torch.load(args.question_model_name))

        test_dataset = Classification_Test_Dataset("test-claims.json", tokenizer_q, args.max_length)

        test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4,
                                      collate_fn=test_dataset.collate_cls)

        question_model.to(device)

        c_e = nn.CrossEntropyLoss()
        labels = []
        for batch_dataset in tqdm(test_dataloader):
            question_model.to(device)

            logits = question_model(
                input_ids=batch_dataset['inputs_q_input_ids'].cuda(),
                attention_mask=batch_dataset['inputs_q_attention_mask'].cuda())

            predict_labels = list(logits.argmax(-1))
            # b_list = list(batch_dataset["labels"])
            print('predict_labels', predict_labels)
            label_list = []
            for j in predict_labels:
                label_list.append(label_int_dict[int(j)])

            labels.extend(label_list)
        f2 = open("project-data/test-claims.json", "r")
        dataset_evidence = json.load(f2)
        f2.close()
        keys = list(dataset_evidence.keys())
        for key in range(len(keys)):
            dataset_evidence[keys[key]]['claim_label'] = labels[key]
        f = open("project-data/test-claims-prediction.json", 'w')
        json.dump(dataset_evidence, f)
        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters')
    # parser.add_argument('--context_model_name', type=str, default='c_model/context_ckpt.bin')
    parser.add_argument('--question_model_name', type=str, default='q_model/question_cls_ckpt.bin')
    parser.add_argument('--seed', type=int, default=70)
    parser.add_argument('--lr', type=float, default=2e-6)
    parser.add_argument('--max_length', type=int, default=500)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    predict(args)
# nohup python -u classification_predict.py >cls1.out 2>&1 &