from rouge import Rouge


def cal_summarization_metric():
    rouge = Rouge()
    gens = open('temp_gen.txt', 'r').readlines()
    refs = open('temp_ref.txt', 'r').readlines()
    scores = rouge.get_scores(gens, refs, avg=True)
    return scores['rouge-1']['f'], scores['rouge-2']['f'], scores['rouge-l']['f']

print(cal_summarization_metric())