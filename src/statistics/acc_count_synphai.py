import json

def calculate_overall_accuracy(jsonl_file_path):
    total_top1 = 0
    total_top2 = 0
    total_top3 = 0

    top1_correct_exact = 0
    top2_correct_exact = 0
    top3_correct_exact = 0

    top1_correct_inexact = 0
    top2_correct_inexact = 0
    top3_correct_inexact = 0

    skipped_entries = [] 
    
    with open(jsonl_file_path, 'r', encoding='utf-8') as file:
        for idx, line in enumerate(file):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                skipped_entries.append({"index": idx, "reason": "Invalid JSON"})
                continue
            
            name_list = list(data.get("predictions").keys())
            model= name_list[0]
            evals = data.get("evaluations", {}).get(model, {}).get("human_evaluated")
            if not evals or not isinstance(evals, dict):
                skipped_entries.append({"index": idx, "reason": f"Missing or invalid evaluations['{model}']['human_evaluated']"})
                continue
            
            for metric, eval_list in evals.items():
                if not isinstance(eval_list, list):
                    skipped_entries.append({"index": idx, "reason": f"Evaluation for '{metric}' is not a list"})
                    continue
                
                if len(eval_list) >= 1:
                    total_top1 += 1
                    if eval_list[0] == 1:
                        top1_correct_exact += 1
                    if isinstance(eval_list[0], (int, float, bool)) and (eval_list[0] >= 0.5):
                        top1_correct_inexact += 1
                
                if len(eval_list) >= 2:
                    total_top2 += 1
                    if 1 in eval_list[:2]:
                        top2_correct_exact += 1
                    if any((x >= 0.5 if isinstance(x, (int, float, bool)) else False) for x in eval_list[:2]):
                        top2_correct_inexact += 1
                
                if len(eval_list) >= 3:
                    total_top3 += 1
                    if 1 in eval_list[:3]:
                        top3_correct_exact += 1
                    if any((x >= 0.5 if isinstance(x, (int, float, bool)) else False) for x in eval_list[:3]):
                        top3_correct_inexact += 1

    top1_accuracy_exact = top1_correct_exact / total_top1 if total_top1 > 0 else 0.0
    top2_accuracy_exact = top2_correct_exact / total_top2 if total_top2 > 0 else 0.0
    top3_accuracy_exact = top3_correct_exact / total_top3 if total_top3 > 0 else 0.0

    top1_accuracy_inexact = top1_correct_inexact / total_top1 if total_top1 > 0 else 0.0
    top2_accuracy_inexact = top2_correct_inexact / total_top2 if total_top2 > 0 else 0.0
    top3_accuracy_inexact = top3_correct_inexact / total_top3 if total_top3 > 0 else 0.0

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
        "totals": {
            "top1_eval_items": total_top1,
            "top2_eval_items": total_top2,
            "top3_eval_items": total_top3
        },
        "skipped_entries": skipped_entries
    }
