"""
update from https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py and https://github.com/mandarjoshi90/triviaqa/blob/master/evaluation/triviaqa_evaluation.py
"""

import re
import string
from collections import Counter
import jsonlines
import argparse

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation + "".join(["‘", "’", "´", "`"]))
        return "".join(ch if ch not in exclude else " " for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace("_", " ")

    return white_space_fix(remove_articles(remove_punc(lower(replace_underscore(s)))))


def bool_mapping(s):
    if s == "True":
        return "yes"
    elif s == "False":
        return "no"
    else:
        return s


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(bool_mapping(prediction))
    normalized_ground_truth = normalize_answer(bool_mapping(ground_truth))

    ZERO_METRIC = (0, 0, 0)

    if (
        normalized_prediction in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return ZERO_METRIC
    if (
        normalized_ground_truth in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return normalize_answer(bool_mapping(prediction)) == normalize_answer(
        bool_mapping(ground_truth)
    )


def cover_exact_match_score_1(prediction, ground_truth):

    pre_list = normalize_answer(bool_mapping(prediction)).split(" ")
    ground_list = normalize_answer(bool_mapping(ground_truth)).split(" ")
    # print("prediction: ",prediction)
    # print("ground_truth: ",ground_truth)
    # print("pre_list: ",pre_list)
    # print("ground_list: ",ground_list)
    # 不考虑顺序和连续
    return all(ground in pre_list for ground in ground_list)


# 滑动窗口em
def cover_exact_match_score_2(prediction, ground_truth):
    pre_list = normalize_answer(bool_mapping(prediction)).split(" ")
    ground_list = normalize_answer(bool_mapping(ground_truth)).split(" ")

    for i in range(len(pre_list) - len(ground_list) + 1):  # "hello world", "hello"
        if pre_list[i : i + len(ground_list)] == ground_list:
            return True
    # 假如gt比answer长的话，直接执行下面的操作
    pre_str = " ".join(pre_list)
    ground_str = " ".join(ground_list)
    if ground_str in pre_str:
        return True
    return False


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    if metric_fn.__name__ == "exact_match_score":
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths)
    elif metric_fn.__name__ == "f1_score":
        for ground_truth in ground_truths:
            f1, prec, recall = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append((f1, prec, recall))
        f1, prec, recall = max(scores_for_ground_truths, key=lambda x: x[0])
        return f1, prec, recall
    elif metric_fn.__name__ == "cover_exact_match_score_1":
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths)
    elif metric_fn.__name__ == "cover_exact_match_score_2":
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths)


def update_answer(metrics, prediction, gold):
    em = metric_max_over_ground_truths(exact_match_score, prediction, gold)
    f1, prec, recall = metric_max_over_ground_truths(f1_score, prediction, gold)
    cover_em_1 = metric_max_over_ground_truths(
        cover_exact_match_score_1, prediction, gold
    )
    cover_em_2 = metric_max_over_ground_truths(
        cover_exact_match_score_2, prediction, gold
    )

    metrics["em"] += float(em)
    metrics["cover_em_1"] += float(cover_em_1)
    metrics["cover_em_2"] += float(cover_em_2)
    metrics["f1"] += f1
    metrics["prec"] += prec
    metrics["recall"] += recall

    if cover_em_1:
        metrics["acc_num"] += 1
    return em, prec, recall, f1, cover_em_1, cover_em_2


def read_jsonl(file_path):
    data = []
    with jsonlines.open(file_path, "r") as reader:
        for obj in reader:
            data.append(obj)
    return data


def eval(file):
    data = read_jsonl(file)
    print(len(data))
    print(f"Eval {len(data)} from {file}")
    metrics = {
        "em": 0,
        "f1": 0,
        "cover_em_1": 0,
        "cover_em_2": 0,
        "prec": 0,
        "recall": 0,
        "acc_num": 0,
    }
    
    for d in data:

        pred_answer = d["pred_ans"]
        if isinstance(d["answer"], list):
            em, prec, recall, f1, cover_em_1, cover_em_2 = update_answer(
                metrics, pred_answer, d["answer"]
            )
        else:
            em, prec, recall, f1, cover_em_1, cover_em_2 = update_answer(
                metrics, pred_answer, [d["answer"]]
            )
    N = len(data)
    for k in metrics.keys():
        if k == "acc_num":
            continue
        metrics[k] /= N

    final_metrics = [
        str(round(metrics["em"] * 100, 1)),
        str(round(metrics["cover_em_1"] * 100, 1)),
        str(round(metrics["f1"] * 100, 1)),
        str(metrics["acc_num"]),
        str(round(metrics["cover_em_1"] * 100, 1)),
    ]
    print("Eval File: ", file)
    print("EM: ", final_metrics[0])
    print("Cover-EM: ", final_metrics[1])
    print("Cover-EM_2: ", final_metrics[4])
    print("F1: ", final_metrics[2])
    print("Acc_Num: ", final_metrics[3])
    # return ' & '.join(final_metrics)
    return f"\nfile:{file}\nem:{final_metrics[0]}, cem:{final_metrics[1]}, f1:{final_metrics[2]}, acc_num:{final_metrics[3]}"


if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(description="Direct batch vLLM page generator")
    # Model configuration
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    print(args)
    result = eval(args.input_path)
    print(result)
