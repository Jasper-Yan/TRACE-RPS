import numpy as np
from language_models import HuggingFace
from language_models import GPT
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.prompts import Prompt


def insert_defense_string(msg, defense):
    return msg + defense  


def schedule_n_to_change_fixed(max_n_to_change, it):
    """ Piece-wise constant schedule for `n_to_change` (both characters and tokens) """

    if 0 < it <= 10:
        n_to_change = max_n_to_change
    elif 10 < it <= 25:
        n_to_change = max_n_to_change // 2
    elif 25 < it <= 50:
        n_to_change = max_n_to_change // 4
    elif 50 < it <= 100:
        n_to_change = max_n_to_change // 8
    elif 100 < it <= 500:
        n_to_change = max_n_to_change // 16
    else:
        n_to_change = max_n_to_change // 32
    
    n_to_change = max(n_to_change, 1)

    return n_to_change


def schedule_n_to_change_prob(max_n_to_change, prob, target_model):
    """ Piece-wise constant schedule for `n_to_change` based on the best prob """
    if isinstance(target_model.model, HuggingFace):
        if 0 <= prob <= 0.01:
            n_to_change = max_n_to_change
        elif 0.01 < prob <= 0.1:
            n_to_change = max_n_to_change // 2
        elif 0.1 < prob <= 1.0:
            n_to_change = max_n_to_change // 4
        else:
            raise ValueError(f'Wrong prob {prob}')
    else:
        if 0 <= prob <= 0.1:
            n_to_change = max_n_to_change
        elif 0.1 < prob <= 0.5:
            n_to_change = max_n_to_change // 2
        elif 0.5 < prob <= 1.0:
            n_to_change = max_n_to_change // 4
        else:
            raise ValueError(f'Wrong prob {prob}')
    
    n_to_change = max(n_to_change, 1)

    return n_to_change


def extract_logprob(logprob_dict, target_token):
    logprobs = []
    if ' ' + target_token in logprob_dict:
        logprobs.append(logprob_dict[' ' + target_token])
    if target_token in logprob_dict:
        logprobs.append(logprob_dict[target_token])
    
    if 'Ġ' + target_token in logprob_dict:
        logprobs.append(logprob_dict['Ġ' + target_token])
    
    if logprobs == []:
        return -np.inf
    else:
        return max(logprobs)


def early_stopping_condition(best_logprobs, target_model, logprob_dict, target_token, determinstic, no_improvement_history=5000):
    if determinstic and logprob_dict != {}:
        argmax_token = max(logprob_dict, key=logprob_dict.get)
        if argmax_token in [target_token, ' '+target_token, 'Ġ'+target_token]:
            return True
        else:
            return False
        
    if len(best_logprobs) == 0:
        return False
    
    best_logprob = best_logprobs[-1]

    if isinstance(target_model.model, HuggingFace) and np.exp(best_logprob) > 0.55 and len(best_logprobs)>no_improvement_history:
        return True  
    
    if np.exp(best_logprob) > 0.8:  
        return True
    return False

def early_stopping_condition2(best_logprobs, target_model, logprob_dict, target_token, determinstic, no_improvement_history=2000):
    if  determinstic and logprob_dict != {}:
        argmax_token = max(logprob_dict, key=logprob_dict.get)
        if argmax_token in [target_token, ' '+target_token, 'Ġ'+target_token]:
            return True
        else:
            return False
        
    if len(best_logprobs) == 0:
        return False
    
    best_logprob = best_logprobs[-1]
    if isinstance(target_model.model, HuggingFace) and np.exp(best_logprob) > 0.45 and len(best_logprobs)>no_improvement_history:
        return True  
    
    if np.exp(best_logprob) > 0.55:  
        return True
    return False