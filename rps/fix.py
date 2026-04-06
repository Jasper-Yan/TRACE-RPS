import os
os.environ["HF_HOME"] = "/tmlscratch/andriush/models/"
from src.utils.initialization import (
    read_config_from_yaml
)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.configs import REDDITConfig
from src.reddit.reddit_utils import load_data
from src.reddit.reddit import create_prompts, parse_answer
from src.reddit.reddit_types import Profile
from src.reddit.normalize import normalize
import json

def map_synthpai_to_pii(synthpair: dict[str, any]) -> str:
    mapped_feature = {
        "income_level": "income",
        "age": "age",
        "sex": "gender",
        "city_country": "location",
        "birth_city_country": "birth_city_country",
        "education_category": "education",
        "occupation": "occupation",
        "relationship_status": "relationship_status",
        "age": "age",
        "gender": "gender",
        "location": "location",
        "pobp": "birth_city_country",
        "education": "education",
        "occupation": "occupation",
        "married": "relationship_status",
        "income": "income",
    }

    new_pii = {}
    for key, value in synthpair.items():
        if key in mapped_feature:
            new_pii[mapped_feature[key]] = value

    return new_pii

if __name__ == '__main__':

    class Args:
        seed = 1
        config_path = "configs/reddit/defense/fix.yaml"

    args = Args()
    cfg = read_config_from_yaml(args.config_path)

    assert isinstance(cfg.task_config, REDDITConfig)
    profiles = load_data(cfg.task_config.path)
    if "comments_eval_revised" in cfg.task_config.path:
        for profile in profiles:
            """for comment in profile.comments:
                comment.review_pii = {
                        "human_evaluated": map_synthpai_to_pii(
                            comment.review_pii["human_evaluated"]
                        )
                    }"""
            if len(profile.comments) > 25:
                profile.comments = profile.comments[:25]
                profile.num_comments = 25
            profile.review_pii = {
                    "human_evaluated": map_synthpai_to_pii(
                        profile.review_pii["human_evaluated"]
                    )
                }
    if "synthpai" in cfg.task_config.path:
        for profile in profiles:
            profile.review_pii = {
                    "human_evaluated": map_synthpai_to_pii(
                        profile.review_pii["human_evaluated"]
                    )
                }
    prompts= []
    for profile in profiles:
        prompts += create_prompts(profile, cfg.task_config)

    with open(cfg.task_config.path, "r", encoding="utf-8") as source_f, \
    open("defense_rps/synthpai_predicted_llama2_13b_trace_gpt4o.jsonl","a",encoding="utf-8") as predict_f,\
    open("defense_rps/synthpai_predicted_llama2_13b_trace_gpt4o_fix.jsonl","a",encoding="utf-8") as fix_f:
        source_data = [json.loads(line.strip()) for line in source_f]

        for i, prompt in enumerate(prompts):
           if source_data[i]['rule_is_success'] == False:
                op = prompt.original_point
                assert isinstance(op, Profile)
                op.predictions[cfg.gen_model.name] = parse_answer(source_data[i]['final_response_text'], prompt.gt)
                op.predictions[cfg.gen_model.name]["full_answer"] = source_data[i]['final_response_text']
                predict_f.write(json.dumps(op.to_json()) + "\n")
                predict_f.flush()
        
        normalize(
        in_paths=[predict_f.name],
        outpath=fix_f.name,
        fix=True,
        merge=True)