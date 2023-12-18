import json


def evidence_analysis(dataset_evi):
    max_len = 0
    max_count = 0
    sum_evidence_len = 0

    for i in dataset_evi.keys():
        e_list = dataset_evi[i].split()
        if max_len < len(e_list):
            max_len = len(e_list)
        sum_evidence_len += len(e_list)

    avg_len = int(sum_evidence_len / len(dataset_evi.keys()))

    for i in dataset_evi.keys():
        if len(dataset_evi[i]) < 200:
            max_count += 1
    print('evidence analysis: ')
    print('total_count ', len(dataset_evi.keys()), ' max len: ', max_len, ' lower than avg count: ', max_count, ' avg: ', avg_len)


def evidence_analysis(dataset_evi):
    max_len = 0
    max_count = 0
    sum_evidence_len = 0

    for i in dataset_evi.keys():
        e_list = dataset_evi[i].split()
        if max_len < len(e_list):
            max_len = len(e_list)
        sum_evidence_len += len(e_list)

    avg_len = int(sum_evidence_len / len(dataset_evi.keys()))

    for i in dataset_evi.keys():
        if len(dataset_evi[i]) < 200:
            max_count += 1
    print('evidence analysis: ')
    print('total_count ', len(dataset_evi.keys()), ' max len: ', max_len, ' lower than avg count: ', max_count, ' avg: ', avg_len)
    print()


def train_analysis(dataset_train):
    s_count = 0
    r_count = 0
    n_count = 0
    d_count = 0
    # label_list = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED"]
    for i in dataset_train.keys():
        s = dataset_train[i]['claim_label']
        if s == "SUPPORTS":
            s_count += 1
        elif s == "REFUTES":
            r_count += 1
        elif s == "NOT_ENOUGH_INFO":
            n_count += 1
        elif s == "DISPUTED":
            d_count += 1

    print('train analysis: ')
    print('total_count ', len(dataset_train.keys()))
    print("SUPPORTS", s_count, "REFUTES", r_count, "NOT_ENOUGH_INFO", n_count, "DISPUTED", d_count)
    print()


def dev_analysis(dataset_train):
    s_count = 0
    r_count = 0
    n_count = 0
    d_count = 0
    # label_list = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED"]
    for i in dataset_train.keys():
        s = dataset_train[i]['claim_label']
        if s == "SUPPORTS":
            s_count += 1
        elif s == "REFUTES":
            r_count += 1
        elif s == "NOT_ENOUGH_INFO":
            n_count += 1
        elif s == "DISPUTED":
            d_count += 1

    print('dev analysis: ')
    print('total_count ', len(dataset_train.keys()))
    print("SUPPORTS", s_count, "REFUTES", r_count, "NOT_ENOUGH_INFO", n_count, "DISPUTED", d_count)
    print()


def create_dataset_with_text(dataset, file_name):
    for i in dataset.keys():

        claim = dataset_dev[i]['claim_text'].lower()
        dataset[i]['claim_text'] = claim

        label = dataset_dev[i]['claim_label'].lower()
        dataset[i]['claim_label'] = label

        e_ids = dataset_dev[i]['evidences']
        eid_list = []
        for j in e_ids:
            s = dataset[j].lower()
            eid_list.append(s)
        dataset[i]['evidence_text'] = eid_list
    dev_output = open("project-data/{}".format(file_name), 'w')
    json.dump(dataset, dev_output)
    dev_output.close()


if __name__ == '__main__':
    f1 = open("./project-data/evidence.json", "r")
    dataset_evidence = json.load(f1)
    f1.close()
    f2 = open("./project-data/train-claims.json", "r")
    dataset_train = json.load(f2)
    f2.close()
    f3 = open("./project-data/dev-claims.json", "r")
    dataset_dev = json.load(f3)
    f3.close()
    f4 = open("./project-data/dev-claims-baseline.json", "r")
    dataset_dev_base = json.load(f4)
    f4.close()

    evidence_analysis(dataset_evidence)
    train_analysis(dataset_train)
    dev_analysis(dataset_dev)
    create_dataset_with_text(dataset_train, 'new-train-claims.json')
    create_dataset_with_text(dataset_dev, 'new-dev-claims.json')

