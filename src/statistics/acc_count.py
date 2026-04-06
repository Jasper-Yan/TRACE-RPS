import json

def calculate_top_k_accuracy(jsonl_file_path):
    top1_correct_exact = 0
    top2_correct_exact = 0
    top3_correct_exact = 0

    top1_correct_inexact = 0
    top2_correct_inexact = 0
    top3_correct_inexact = 0

    total_entries = 0
    skipped_entries = [] 

    with open(jsonl_file_path, 'r', encoding='utf-8') as file:
        for idx, line in enumerate(file):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                skipped_entries.append({"index": idx, "reason": "Invalid JSON"})
                continue

            review_key = list(data.get("reviews", {}).get("synth", {}).keys())
            if not review_key:
                skipped_entries.append({"index": idx, "reason": "Missing 'reviews.synth' key"})
                continue
            key = review_key[0]

            name_list = list(data.get("predictions").keys())
            model_name = name_list[0]
            evaluations = data.get("evaluations", {}).get(model_name, {}).get("synth", {}).get(key)
            if not evaluations or not isinstance(evaluations, list):
                skipped_entries.append({"index": idx, "reason": "Missing or invalid 'evaluations' data"})
                continue
            eval_length = len(evaluations)

            if eval_length >= 1 and evaluations[0] == 1:
                top1_correct_exact += 1
            if eval_length >= 2 and 1 in evaluations[:2]:
                top2_correct_exact += 1
            if eval_length >= 3 and 1 in evaluations[:3]:
                top3_correct_exact += 1

            if eval_length >= 1 and evaluations[0] >= 0.5:
                top1_correct_inexact += 1
            if eval_length >= 2 and any(x >= 0.5 for x in evaluations[:2]):
                top2_correct_inexact += 1
            if eval_length >= 3 and any(x >= 0.5 for x in evaluations[:3]):
                top3_correct_inexact += 1

            total_entries += 1

    top1_accuracy_exact = top1_correct_exact / total_entries if total_entries > 0 else 0.0
    top2_accuracy_exact = top2_correct_exact / total_entries if total_entries > 0 else 0.0
    top3_accuracy_exact = top3_correct_exact / total_entries if total_entries > 0 else 0.0

    top1_accuracy_inexact = top1_correct_inexact / total_entries if total_entries > 0 else 0.0
    top2_accuracy_inexact = top2_correct_inexact / total_entries if total_entries > 0 else 0.0
    top3_accuracy_inexact = top3_correct_inexact / total_entries if total_entries > 0 else 0.0

    return {
        "exact": {
            "top1_accuracy": top1_accuracy_exact,
            "top2_accuracy": top2_accuracy_exact,
            "top3_accuracy": top3_accuracy_exact
        },
        "inexact": {
            "top1_accuracy": top1_accuracy_inexact,
            "top2_accuracy": top2_accuracy_inexact,
            "top3_accuracy": top3_accuracy_inexact
        },
        "total_entries": total_entries,
        "skipped_entries": skipped_entries
    }