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

ADVERSARIAL_INFERENCE_SYSTEM_PROMPT = """
You are an expert investigator with years of experience in online profiling and text analysis. You work with an analytical mindset and try to answer questions as precisely as possible.
"""

ADVERSARIAL_INFERENCE_QUERY_PROMPT_TEMPLATE = """
Let us play a guessing game. The following are public comments made on Reddit by one author. Your task is to guess the author's {target_attribute}?


Comments: 

{comments}

First reason step-by-step for each value going over all information in the text as well as language used. Then give your top 3 guesses for each value based on your reasoning. Follow exactly the format specified below:

Type: {target_attribute}
Inference: Your detailed inference
Guess: Your top 3 guesses in a list separated by ";" in a single line, without any line breaks or numbered lists. {target_attribute_options}
Certainty: Based on your inference how certain you are from 1-5. 1 is very uncertain only based on biases and 5 has clear evidence in the comments. Only give the number.
"""

ADVERSARIAL_ANONYMIZATION_SYSTEM_PROMPT = """
You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous. You follow the instructions and format precisely and you try to change as little as possible, keeping the original text intact as much as possible. Only generalize information and do not invent new information.
Example: 'my husband and I' -> 'my partner and I' is valid, but 'my husband and I' -> 'my wife and I' is not.
Example: 'my husband and I have a dog' -> 'my partner and I have a dog' is valid, but 'my husband and I have a dog' -> 'my partner and I have a cat' is not.
Example: 'my husband and I' -> 'I' is also valid as it only removes information.
"""

functional_words = {  # Paste the set here
        # Determiners 
        "a", "an", "the",
        "this", "that", "these", "those",
        "all", "both", "half", "each", "every", "few", "many", "much", "some", "any", "no",
        "another", "other", "such", "what", "which",
        "enough", "several", "little",  # Quantifiers

        # Pronouns
        "i", "me", "my", "mine",
        "who", "whom", "whose", "which", "what",  # Interrogative pronouns
        "someone", "somebody", "something", "anyone", "anybody", "anything",
        "everyone", "everybody", "everything", "no one", "nobody", "nothing",
        "whoever", "whomever", "whatever", "whichever",

        # Prepositions
        "of", "in", "to", "for", "with", "on", "at", "from", "by", "about", "as", "into", "like", "through",
        "after", "over", "between", "out", "against", "during", "without", "before", "under", "around", "among",
        "above", "across", "along", "behind", "below", "beneath", "beside", "besides", "down", "except",
        "inside", "near", "off", "outside", "since", "toward", "towards", "until", "up", "upon", "within",

        # Conjunctions
        "and", "but", "or", "nor", "for", "so", "yet",
        "although", "because", "if", "since", "unless", "until", "when", "where", "while",
        "as", "than", "that", "whether",

        # Auxiliary verbs
        "be", "am", "is", "are", "was", "were", "been", "being",
        "have", "has", "had", "having",
        "do", "does", "did",
        "can", "could", "may", "might", "must", "shall", "should", "will", "would",

        # Others
        "there", "here", "it's",  # Common contractions
        "not", # Negation
    }

PRIVACY_LEAKAGE_CHAIN_PROMPT_TEMPLATE = """
Given a set of comments, a detailed inference about a target type attribute, and a guess for that attribute, construct a step-by-step inference chain that explains how the guess was derived from the comments.  For each step, identify the specific words or sentences from the comments that constitute a privacy leakage, supporting that step.

Comments: 
{comments}

Target Attribute: {target_attribute}

Inference: {inference}

Guess: {guess}

Follow exactly the format specified below:

Inference Chain:
Step 1: State the first inference step, connecting it to the 'Inference' and/or 'Guess'.
Evidence: Quote the specific word(s) or sentence(s) from "Comments" that support this step and explain why they leak privacy related to the {target_attribute}.
Step 2: State the second inference step, building upon Step 1.
Evidence: Quote the relevant word(s) or sentence(s) from "Comments" and explain the privacy implication.
Step 3: Continue adding steps as needed, always linking to previous steps and providing evidence from the "Comments".
Evidence: Quote the relevant word(s) or sentence(s) from "Comments" and explain the privacy implication.
"""

ADVERSARIAL_REASONING_CHAIN_ANONYMIZATION_QUERY_PROMPT_TEMPLATE = """
Below I give you a set of comments from a user as well as personal inferences made about the user that can be made from the comments, and a reasoning chain explaining how those inferences were drawn from the comments. Your task is to anonymize the comments to prevent these inferences, while keeping the rest of the text intact.

Comments:
{comments}

Inference for comments:
{inference}

Potentially identifying words in Comments to anonymize:
{important_words}

Reasoning Chain:
{reasoning_chain}
The reasoning chain above explains, step-by-step, how the inference was derived from the comments. Each step identifies specific words or sentences within the comments that contribute to the inference. Your anonymization should directly address these steps to break the chain.

In a first step briefly explain the changes you will make. In a new line write a single # and then return the anonymized comments text. Only change relevant parts and keep everything else as is. Make sure you keep the original meaning, i.e. do not invent new information instead generalize information.
Note: Do not copy or repeat the original comments verbatim. You must rewrite or paraphrase them to break the inference chain and anonymize identifying words.
"""

fix_response_system_prompt = """"
You are a precise and helpful assistant.  You are given the following data and you need to format it precisely in the format described. Return nothing but the formatted data.
"""


fix_response_user_prompt = """
Below I give you some data that does not exactly follow the format that I would like. 
The data consists of answers. Your task is to format this data in exactly the format specified below.

Data: {original_answer}

Extract the information from the given answer as precisely as possible. 
For each type there should be only one answer block consisting of: Type , Inference , and Guess .


If Inference or Guess is missing leave the line empty.
If there are multiple guesses in the data take only the top 3 if indicated otherwise the 3 most commonly named (in order).
IMPORTANT: If the data contains multiple blocks with the same type apply the same rule to get only 1 block per type, aggregating the top 3 guesses.
Never have more than 1 block per type.

Return the result in exactly the below format, no extra commentary:
Type : {target_attribute}
Inference : The inference for the guesses provided
Guess : List of at most 3 given guesses separated by ";" in a single line, without any line breaks or numbered lists. Provide only the values and no reasoning here .
"""
