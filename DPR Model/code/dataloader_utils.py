import json
from torch.utils.data import Dataset


class Train_Dataset(Dataset):
    def __init__(self, file_name, tokenizer_q, tokenizer_c, max_length):
        self.max_length = max_length
        # f = open("project-data/{}".format(file_name), "r")
        f = open("/content/drive/MyDrive/project_code2/project-data/{}".format(file_name), "r")
        self.val_dataset = json.load(f)
        self.tokenizer_q = tokenizer_q
        self.tokenizer_c = tokenizer_c
        f.close()

    def __len__(self):
        return len(self.val_dataset.keys())

    def __getitem__(self, i):
        claim_ids = list(self.val_dataset.keys())
        claim_id = claim_ids[i]
        claim_dict = self.val_dataset[claim_id]
        claim_text = claim_dict['claim_text']
        e_texts = claim_dict['evidence_text']
        e_ids = claim_dict['evidences']
        n_e_texts = claim_dict['negative_evidence_text']
        combine_list = [claim_id, claim_text, e_ids, e_texts, n_e_texts]
        return combine_list

    def collate_train(self, batch):

        claims = []
        claim_ids = []
        evidence_texts = []
        evidence_ids = []
        negative_evidence_texts = []
        question_combine_list = []
        context_combine_list = []
        positive_counts = []
        c_total_len = 0
        q_total_len = 0

        for claim_id, claim_text, e_ids, e_texts, n_e_texts in batch:
            claim_ids.append(claim_id)
            claims.append(claim_text)
            evidence_ids.append(e_ids)
            evidence_texts.append(e_texts)
            negative_evidence_texts.append(n_e_texts)

            # count the positive evidences
            positive_counts.append(len(e_ids))

            # combine all the e_ids and e_texts
            evidence_combines = []
            for i in range(len(e_ids)):
                evidence_combine = e_texts[i] + ' '
                evidence_combines.append(evidence_combine)

            negative_evidence_combines = []
            for i in range(len(n_e_texts)):
                negative_evidence_combine = n_e_texts[i] + ' '
                negative_evidence_combines.append(negative_evidence_combine)

            question_combine = claim_text + ' '
            context_combine = ''
            for i in evidence_combines:
                question_combine += i
                context_combine += i

            for i in negative_evidence_combines:
                context_combine += i

            q_total_len += len(question_combine)
            c_total_len += len(context_combine)
            question_combine_list.append(question_combine)
            context_combine_list.append(context_combine)

        inputs_q = self.tokenizer_q(
            question_combine_list,
            return_tensors="pt",
            padding=True,
            max_length=250,
            truncation=True
        )

        inputs_c = self.tokenizer_c(
            context_combine_list,
            return_tensors="pt",
            padding=True,
            max_length=250,
            truncation=True
        )

        batch_dict = dict()
        batch_dict['inputs_q_input_ids'] = inputs_q.input_ids
        batch_dict['inputs_q_attention_mask'] = inputs_q.attention_mask
        batch_dict['inputs_c_input_ids'] = inputs_c.input_ids
        batch_dict['inputs_c_attention_mask'] = inputs_c.input_ids
        batch_dict['claim_ids'] = claim_ids
        batch_dict['claim_text'] = claims
        batch_dict['evidence_ids'] = evidence_ids
        batch_dict['evidence_texts'] = evidence_texts
        batch_dict['negative_evidence_texts'] = negative_evidence_texts
        batch_dict['positive_count'] = positive_counts

        return batch_dict


class Val_Dataset(Dataset):
    def __init__(self, file_name, tokenizer_q, tokenizer_c, max_length):
        self.max_length = max_length
        f = open("/content/drive/MyDrive/project_code2/project-data/{}".format(file_name), "r")
        self.val_dataset = json.load(f)
        self.tokenizer_q = tokenizer_q
        self.tokenizer_c = tokenizer_c
        f.close()

    def __len__(self):
        return len(self.val_dataset.keys())

    def __getitem__(self, i):
        claim_ids = list(self.val_dataset.keys())
        claim_id = claim_ids[i]
        claim_dict = self.val_dataset[claim_id]
        claim_text = claim_dict['claim_text']
        e_texts = claim_dict['evidence_text']
        e_ids = claim_dict['evidences']
        combine_list = [claim_id, claim_text, e_ids, e_texts]
        return combine_list

    def collate_val(self, batch):

        claims = []
        claim_ids = []
        evidence_texts = []
        evidence_ids = []
        input_c_ids = []
        input_c_masks = []
        combine_list = []

        for claim_id, claim_text, e_ids, e_texts in batch:
            claim_ids.append(claim_id)
            claims.append(claim_text)
            evidence_ids.append(e_ids)
            evidence_texts.append(e_texts)
            evidence_combines = []
            s = claim_text + ' '
            for i in range(len(e_ids)):
                evidence_combine = e_texts[i] + ' '
                s += evidence_combine
                evidence_combines.append(evidence_combine)
            combine_list.append(s.lower())
            inputs_c = self.tokenizer_c(
                evidence_combines,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True
            )
            input_c_ids.append(inputs_c.input_ids)
            input_c_masks.append(inputs_c.attention_mask)

        inputs_q = self.tokenizer_q(
            claims,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True
        )

        batch_dict = dict()
        batch_dict['inputs_q_input_ids'] = inputs_q.input_ids
        batch_dict['inputs_q_attention_mask'] = inputs_q.attention_mask
        batch_dict['inputs_c_input_ids_list'] = input_c_ids
        batch_dict['inputs_c_attention_mask_list'] = input_c_masks
        batch_dict['claim_ids'] = claim_ids
        batch_dict['claim_text'] = claims
        batch_dict['evidence_ids'] = evidence_ids
        batch_dict['evidence_texts'] = evidence_texts
        batch_dict['combine'] = combine_list

        return batch_dict


class Evidence_Dataset(Dataset):
    def __init__(self, file_name, tokenizer, max_length):
        f = open("/content/drive/MyDrive/project_code2/project-data/{}".format(file_name), "r")
        self.max_length = max_length
        self.e_dataset = json.load(f)
        self.tokenizer = tokenizer
        f.close()

    def __len__(self):
        return len(self.e_dataset.keys())

    def __getitem__(self, i):
        e_ids = list(self.e_dataset.keys())
        e_id = e_ids[i]
        e_texts = self.e_dataset[e_id].lower()
        combine_list = [e_id, e_texts]
        return combine_list

    def collate_evidence(self, batch):
        evidence_texts = []
        evidence_ids = []
        evidence_texts_extend = []
        evidence_ids_extend = []
        evidence_combine = []
        for e_ids, e_texts in batch:
            evidence_ids.append(e_ids)
            evidence_texts.append(e_texts)
            evidence_ids_extend.extend(e_ids)
            evidence_texts_extend.extend(e_texts)
            '''for j in range(len(e_ids)):
                if j<=len(e_ids) and j<=len(e_texts):
                    temp_s = e_ids[j] + ' ' + e_texts[j]
                    evidence_combine.append(temp_s)'''
        evi_combine = []
        for i in evidence_texts:
            evi_combine.extend(i)

        inputs = self.tokenizer(
            evi_combine,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True
        )

        batch_dict = dict()
        batch_dict['evidence_ids'] = evidence_ids
        batch_dict['evidence_texts'] = evidence_texts
        batch_dict['evidence_ids_extend'] = evidence_ids_extend
        batch_dict['evidence_texts_extend'] = evi_combine
        batch_dict['evidence_input_ids'] = inputs.input_ids
        batch_dict['evidence_attention_mask'] = inputs.attention_mask

        return batch_dict


class Test_Dataset(Dataset):
    def __init__(self, file_name, tokenizer, max_length):
        self.max_length = max_length
        f = open("/content/drive/MyDrive/project_code2/project-data/{}".format(file_name), "r")
        self.val_dataset = json.load(f)
        self.tokenizer = tokenizer
        f.close()

    def __len__(self):
        return len(self.val_dataset.keys())

    def __getitem__(self, i):
        claim_ids = list(self.val_dataset.keys())
        claim_id = claim_ids[i]
        claim_dict = self.val_dataset[claim_id]
        claim_text = claim_dict['claim_text']
        combine_list = [claim_id, claim_text]
        return combine_list

    def collate_test(self, batch):
        claims = []
        claim_ids = []

        for claim_id, claim_text in batch:
            claim_ids.append(claim_id)
            claims.append(claim_text)

        inputs = self.tokenizer(
            claims,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True
        )

        batch_dict = dict()
        batch_dict['test_input_ids'] = inputs.input_ids
        batch_dict['test_attention_mask'] = inputs.attention_mask
        batch_dict['claim_ids'] = claim_ids
        batch_dict['claim_texts'] = claims
        batch_dict['evidence_ids'] = []
        batch_dict['labels'] = []

        return batch_dict


# todo 写下面的代码
class Classification_Train_Dataset(Dataset):
    def __init__(self, file_name, tokenizer, max_length):
        self.max_length = max_length
        f = open("/content/drive/MyDrive/project_code2/project-data/{}".format(file_name), "r")
        self.val_dataset = json.load(f)
        self.tokenizer = tokenizer
        f.close()

    def __len__(self):
        return len(self.val_dataset.keys())

    def __getitem__(self, i):
        claim_ids = list(self.val_dataset.keys())
        claim_id = claim_ids[i]
        claim_dict = self.val_dataset[claim_id]
        claim_text = claim_dict['claim_text']
        e_texts = claim_dict['evidence_text']
        e_ids = claim_dict['evidences']
        label = claim_dict['claim_label']
        combine_list = [claim_id, claim_text, label, e_ids, e_texts]
        return combine_list

    def collate_cls(self, batch):

        claims = []
        claim_ids = []
        evidence_texts = []
        evidence_ids = []
        labels = []
        for claim_id, claim_text, label, e_ids, e_texts in batch:
            claim_ids.append(claim_id)
            evidence_ids.append(e_ids)
            evidence_texts.append(e_texts)
            labels.append(label)
            c = claim_text + ' '
            for i in range(len(e_ids)):
                c += e_texts[i] + ' '
            claims.append(c)

        inputs_q = self.tokenizer(
            claims,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True
        )

        batch_dict = dict()
        batch_dict['inputs_q_input_ids'] = inputs_q.input_ids
        batch_dict['inputs_q_attention_mask'] = inputs_q.attention_mask
        batch_dict['claim_ids'] = claim_ids
        batch_dict['claim_text'] = claims
        batch_dict['evidence_ids'] = evidence_ids
        batch_dict['labels'] = labels

        return batch_dict


class Classification_Test_Dataset(Dataset):
    def __init__(self, file_name, tokenizer, max_length):
        self.max_length = max_length
        f = open("/content/drive/MyDrive/project_code2/project-data/{}".format(file_name), "r")
        self.val_dataset = json.load(f)
        self.tokenizer = tokenizer
        f.close()

    def __len__(self):
        return len(self.val_dataset.keys())

    def __getitem__(self, i):
        claim_ids = list(self.val_dataset.keys())
        claim_id = claim_ids[i]
        claim_dict = self.val_dataset[claim_id]
        claim_text = claim_dict['claim_text']
        e_ids = claim_dict['evidence']
        label = claim_dict['claim_label']
        combine_list = [claim_id, claim_text, label, e_ids]
        return combine_list

    def collate_cls(self, batch):
        claims = []
        claim_ids = []
        evidence_ids = []
        labels = []

        for claim_id, claim_text, label, e_ids in batch:
            claim_ids.append(claim_id)
            claims.append(claim_text)
            evidence_ids.append(e_ids)
            labels.append(label)

        inputs_q = self.tokenizer(
            claims,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True
        )

        batch_dict = dict()
        batch_dict['inputs_q_input_ids'] = inputs_q.input_ids
        batch_dict['inputs_q_attention_mask'] = inputs_q.attention_mask
        batch_dict['claim_ids'] = claim_ids
        batch_dict['claim_text'] = claims
        batch_dict['evidence_ids'] = evidence_ids
        batch_dict['labels'] = labels
        return batch_dict
