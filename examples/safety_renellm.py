import os, sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from easyjailbreak.datasets import JailbreakDataset
from easyjailbreak.attacker.ReNeLLM_ding_2023 import *
from easyjailbreak.models.openai_model import OpenaiModel
from easyjailbreak.models.huggingface_model import HuggingfaceModel

sys.path.append(os.getcwd())

generation_config = {'max_new_tokens': 512}
target_model_path = 'microsoft/Phi-3-mini-4k-instruct'
model_name = 'llama-2'
model = AutoModelForCausalLM.from_pretrained(target_model_path)
tokenizers = AutoTokenizer.from_pretrained(target_model_path)
target_model = HuggingfaceModel(model=model, tokenizer=tokenizers, model_name=model_name,
                                generation_config=generation_config)

chat_name = 'microsoft/Phi-3-small-8k-instruct'
model = AutoModelForCausalLM.from_pretrained(chat_name)
tokenizers = AutoTokenizer.from_pretrained(chat_name)
attack_model = HuggingfaceModel(model=model, tokenizer=tokenizers, model_name='llama-2',
                                generation_config=generation_config)

dataset_name = 'AdvBench'
dataset = JailbreakDataset(dataset_name)

attacker = ReNeLLM(attack_model=attack_model,
                   target_model=target_model,
                   eval_model=attack_model,
                   jailbreak_datasets=dataset)
attacker.attack()
attacker.jailbreak_datasets.save_to_jsonl('AdvBench_renellm.jsonl')
