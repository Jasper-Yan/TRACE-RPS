# Copyright 2026 TRACE-RPS Team (https://arxiv.org/abs/2602.11528)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from transformers import LlamaForCausalLM, AutoTokenizer
import torch
import numpy as np
from typing import List, Dict, Tuple
from prompts import (
    ADVERSARIAL_ANONYMIZATION_SYSTEM_PROMPT,
    ADVERSARIAL_INFERENCE_QUERY_PROMPT_TEMPLATE,
    ADVERSARIAL_INFERENCE_SYSTEM_PROMPT,
    functional_words,
    PRIVACY_LEAKAGE_CHAIN_PROMPT_TEMPLATE,
    ADVERSARIAL_REASONING_CHAIN_ANONYMIZATION_QUERY_PROMPT_TEMPLATE,
)
from utils import call_openai_chat_completion, parse_inference_response_with_fallback
from src.configs import Config
from src.reddit.reddit_utils import load_data
from src.reddit.reddit_types import Profile
from src.reddit.reddit import filter_profiles
import json
import argparse
from src.utils.initialization import (
    read_config_from_yaml,
    seed_everything,
    set_credentials,
    get_out_file,
)
import string

SENSITIVE_ATTRIBUTES = ["Income", "Education", "Place of Birth", "Location", "Relationship Status", "Age"]

model_path = "llama_hf/llama_7b_chat"
model = LlamaForCausalLM.from_pretrained(model_path, output_attentions=True, device_map="auto", torch_dtype=torch.float16) 
tokenizer = AutoTokenizer.from_pretrained(model_path)


def get_attribute_options(pii_type: str) -> str:
    if pii_type == "income":
        pii_type_str = " Choose from these options: No income, Low (<30k USD), Medium (30-60k USD), High (60-150k USD), Very High (>150k USD)."
    elif pii_type == "education":
        pii_type_str = " Choose from these options: No Highschool, In Highschool, HS Diploma, In College, College Degree, PhD."
    elif pii_type == "gender":
        pii_type_str = " Choose from these options: Male, Female."
    elif pii_type == "relationship_status":
        pii_type_str = (
            " Choose from these options: No relation, In Relation, Married, Divorced."
        )
    elif pii_type == "age":
        pii_type_str = " Use the age of the author when he wrote the comment."
    else:
        pii_type_str = ""

    return pii_type_str

def inference_model(text: str, attribute: str) -> Dict[str, any]:
    options = get_attribute_options(attribute)
    user_prompt = ADVERSARIAL_INFERENCE_QUERY_PROMPT_TEMPLATE.format(
        target_attribute=attribute,
        target_attribute_options=options,
        comments=text
    )
    response = call_openai_chat_completion(
        system_prompt=ADVERSARIAL_INFERENCE_SYSTEM_PROMPT,
        user_prompt=user_prompt
    )
    parsed = parse_inference_response_with_fallback(response, attribute, is_certainty=True)
    return parsed

def get_attention_weights(text, question):
    

    prompt = text + " " + question
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)


    with torch.no_grad():
        attention_output = model(**inputs, output_attentions=True)

    all_attentions = attention_output.attentions
    last_layer_attention = all_attentions[-1]
    pooled_attention = torch.mean(last_layer_attention, dim=1)
    last_token_attention = pooled_attention[:, -1, :].cpu().numpy()


    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].cpu().numpy()[0])


    text_tokens = tokenizer.encode(text, add_special_tokens=False)
    text_end_index = len(text_tokens) + 1
    

    text_tokens_only = tokens[:text_end_index]
    text_attention_weights = last_token_attention[0][:text_end_index]

    return text_tokens_only, text_attention_weights


def group_tokens_to_words(tokens: list[str], attention_weights: np.ndarray) -> tuple[list[str], list[float]]:
    if tokens and tokens[0] == '<s>':
        tokens = tokens[1:]
        attention_weights = attention_weights[1:]

    words = []
    word_attention_weights = []
    
    current_word_tokens = []
    current_word_weights = []
    
    punctuation_marks = set('.,!?-;:()[]{}""\'\'')
    
    for i, token in enumerate(tokens):
        
        decoded_current = tokenizer.decode(tokenizer.convert_tokens_to_ids([token]), skip_special_tokens=True)
        

        if decoded_current.strip() in punctuation_marks:

            if current_word_tokens:
                word = tokenizer.decode(tokenizer.convert_tokens_to_ids(current_word_tokens), skip_special_tokens=True).strip()
                if word:
                    words.append(word)
                    word_attention_weights.append(np.sum(current_word_weights))
                current_word_tokens = []
                current_word_weights = []
            

            words.append(decoded_current)
            word_attention_weights.append(attention_weights[i])
            continue

        current_word_tokens.append(token)
        current_word_weights.append(attention_weights[i])

        if token == '▁' and i < len(tokens) - 1:
            continue
        
        decoded_word = tokenizer.decode(tokenizer.convert_tokens_to_ids(current_word_tokens), skip_special_tokens=True)
        
        # Check if we have a complete word
        next_decoded = decoded_word
        if i < len(tokens) - 1:
            next_token = tokens[i + 1]
            next_decoded = tokenizer.decode(tokenizer.convert_tokens_to_ids(current_word_tokens + [next_token]), skip_special_tokens=True)
        
        should_split = (
            ' ' in decoded_word or
            (i < len(tokens) - 1 and next_decoded.count(' ') > decoded_word.count(' ')) or
            decoded_word.strip() in '.,!?-' or
            i == len(tokens) - 1
        )
        
        if should_split and current_word_tokens:
            # Clean up the word
            word = decoded_word.strip()
            if word:
                words.append(word)
                word_attention_weights.append(np.sum(current_word_weights))
            current_word_tokens = []
            current_word_weights = []

    return words, word_attention_weights

def get_top_k_words(words: List[str], word_attention_weights: List[float], k: int = 10) -> List[Tuple[str, float]]:
    """
    Returns the top k words with the highest attention weights.
    """

    word_weight_pairs = list(zip(words, word_attention_weights))
    
    punctuation_marks = set(string.punctuation)
    

    filtered_word_weight_pairs = [
        (word, weight) for word, weight in word_weight_pairs
        if (not all(char in punctuation_marks for char in word)) and (word.lower() not in functional_words)
    ]
    

    sorted_word_weight_pairs = sorted(
        filtered_word_weight_pairs,
        key=lambda x: x[1],
        reverse=True
    )
    

    top_k = sorted_word_weight_pairs[:k]
    

    top_k_words = [word for word, weight in top_k]
    
    return top_k_words

def get_attribute_questions(pii_type: str) -> str:

    if pii_type == "income":
        pii_type_q = "What is their approximate income?"
    elif pii_type == "education":
        pii_type_q = "What is their highest level of education?"
    elif pii_type == "gender":
        pii_type_q = "What is their gender?"
    elif pii_type == "relationship_status":
        pii_type_q = "Are they single, inrelation, married, or divorced?"
    elif pii_type == "age":
        pii_type_q = "How old are they?"
    elif pii_type == "location":
        pii_type_q = "Where do they currently live?"
    elif pii_type == "birth_city_country":
        pii_type_q = "Where were they born?"
    else:
        pii_type_q = ""

    return pii_type_q

def privacy_leakage_chain(comments: str, inference: str, guess: str, target_attribute: str) -> str:
    user_prompt = PRIVACY_LEAKAGE_CHAIN_PROMPT_TEMPLATE.format(
        comments=comments,
        inference=inference,
        guess=guess,
        target_attribute=target_attribute
    )

    system_prompt = "You are a helpful assistant trained to identify privacy risks in text." 

    try:
        response = call_openai_chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens = 500
        )
        response = response.replace("*", "").replace("#","")
        if response.startswith("Inference Chain:\n"):
            response = response[len("Inference Chain:\n"):].strip()
        return response
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return ""

def anonymization_model(comments: str, inference: str, important_words: str, chain: str) -> str:
    
    user_prompt = ADVERSARIAL_REASONING_CHAIN_ANONYMIZATION_QUERY_PROMPT_TEMPLATE.format(
        comments=comments,
        inference=inference,
        important_words=important_words,
        reasoning_chain=chain
    )

    response = call_openai_chat_completion(
        system_prompt=ADVERSARIAL_ANONYMIZATION_SYSTEM_PROMPT,
        user_prompt=user_prompt
    )
    
    lines = response.splitlines()
    if lines and "explanation" in lines[0].lower():
        response = "\n".join(lines[1:]).strip()

    if '#' in response:
        _, anonymized_text = response.split('#', 1)
        anonymized_text = anonymized_text.strip()
        if anonymized_text.startswith("Comments:\n"):
            anonymized_text = anonymized_text[len("Comments:\n"):]  
        if 'Inference for comments:' in anonymized_text:
            anonymized_text = anonymized_text.split('Inference for comments:', 1)[0]
        return anonymized_text

    else:
        return comments  

def adversarial_anonymization(text: str, attributes: List[str], max_iterations: int = 5) -> Tuple[str, List[Dict[str, any]]]:

    current_text = text
    inference_history = []

    if isinstance(attributes, str):
        attributes = [attributes]

    for iteration in range(1, max_iterations + 1):
        print(f"\n---Iteration: {iteration} anonymization ---")
        print(f"curent text: {current_text}\n")

        inferred_attrs = {}
        for attr in attributes:
            inference = inference_model(current_text, attr)
            if inference['guesses']:
                inferred_attrs[attr] = inference
            question = get_attribute_questions(attr)

        inference_history.append(inferred_attrs)

        if not inferred_attrs:
            print("No more sensitive attributes detected. Stopping anonymization.")
            break

        if inference['certainty'] <= 2:
            print("Certainty is no more than 2")
            break

        tokens, attention_weights = get_attention_weights(current_text, question)
        words, word_attention_weights = group_tokens_to_words(tokens, attention_weights)

        top_k_words = get_top_k_words(words, word_attention_weights)
        
        print(f"\nTop 10 words with highest attention weights:")
        important_words_str = ", ".join(top_k_words)
        print(important_words_str)
        infferred_text = inference['inference'] + '\nGuess: ' + inference['guesses']

        chain = privacy_leakage_chain(
                    comments=current_text,
                    inference=inference['inference'],
                    guess=inference['guesses'],
                    target_attribute=attr
                )

        anonymized_text = anonymization_model(current_text, infferred_text, important_words_str, chain)

        if anonymized_text == current_text:
            break

        current_text = anonymized_text

    return current_text, inference_history

def run_reddit_anonymization(cfg: Config) -> None:
    profiles: List[Profile] = load_data(cfg.task_config.path)

    profiles = filter_profiles(profiles, cfg.task_config.profile_filter)

    
    with open(cfg.task_config.path, "r", encoding="utf-8") as source_f, \
     open(cfg.task_config.outpath, "a", encoding="utf-8") as target_f: 
        
        source_data = [json.loads(line.strip()) for line in source_f]

        for i, profile in enumerate(profiles):

            print("***********")
            print(f"Profile {i}")

            combined_comments = "\n".join([comment.text for comment in profile.comments])


            anonymized_text, history = adversarial_anonymization(combined_comments, list(profile.review_pii["synth"].keys())[0])


            profile.anonymized_text = anonymized_text
            profile.anonymization_history = history


            source_data[i]['response'] = anonymized_text
            target_f.write(json.dumps(source_data[i], ensure_ascii=False) + '\n')
            target_f.flush()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/reddit/defense/rps.yaml",
        help="Path to the config file",
    )
    args = parser.parse_args()

    cfg = read_config_from_yaml(args.config_path)

    run_reddit_anonymization(cfg)

    seed_everything(cfg.seed)
    set_credentials(cfg)
    f, path = get_out_file(cfg)