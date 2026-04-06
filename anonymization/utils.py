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

from openai import OpenAI
from typing import Dict
from prompts import fix_response_system_prompt, fix_response_user_prompt, fix_response_user_prompt_with_certainty
from credentials import openai_api_key, openai_base_url
base_url = openai_base_url
api_key = openai_api_key

def call_openai_chat_completion(system_prompt: str, user_prompt: str, temperature: float = 0, max_tokens: int = 2000) -> str:
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI API call error: {e}")
        return ""


def parse_inference_response(response: str) -> Dict[str, any]:
    lines = response.split('\n')
    inference = ""
    guesses = []
    certainty = 1

    for line in lines:
        if line.lower().startswith("inference:"):
            inference = line.partition(":")[2].strip()
        elif line.lower().startswith("guess:"):
            guesses = line.partition(":")[2].strip()
        elif line.lower().startswith("certainty:"):
            try:
                certainty = int(line.partition(":")[2].strip())
            except:
                certainty = 1

    return {
        "inference": inference,
        "guesses": guesses,
        "certainty": certainty
    }

def fix_response_format(original_answer: str, target_attribute: str , is_certainty: bool = False) -> str:

    system_prompt = fix_response_system_prompt
    
    if is_certainty==False:
        user_prompt = fix_response_user_prompt.format(original_answer=original_answer, target_attribute=target_attribute)
    
    else:
        user_prompt = fix_response_user_prompt_with_certainty.format(original_answer=original_answer, target_attribute=target_attribute)


    fixed_response = call_openai_chat_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.1,
        max_tokens=1024
    )

    return fixed_response


def parse_inference_response_with_fallback(
    original_answer: str, 
    target_attribute: str,
    is_certainty: bool = False
) -> Dict[str, any]:
    
    original_answer = original_answer.replace("*", "").replace("#","")

    result = parse_inference_response(original_answer)


    if not result["inference"] or not result["guesses"]:

        fixed = fix_response_format(original_answer, target_attribute, is_certainty)

        result = parse_inference_response(fixed)

    return result

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