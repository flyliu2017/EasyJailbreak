import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from easyjailbreak.attacker.ReNeLLM_ding_2023 import *
from easyjailbreak.datasets import JailbreakDataset
from easyjailbreak.models import ServerAPIModel
from easyjailbreak.models.huggingface_model import HuggingfaceModel

sys.path.append(os.getcwd())
device = 'cuda:0'

generation_config = {'max_new_tokens': 512}
target_model_path = 'microsoft/Phi-3-mini-4k-instruct'
model_name = 'phi3'
model = AutoModelForCausalLM.from_pretrained(target_model_path,
                                             trust_remote_code=True,
                                             torch_dtype=torch.bfloat16,
                                             attn_implementation="flash_attention_2")

model = model.to(device)
tokenizers = AutoTokenizer.from_pretrained(target_model_path)
target_model = HuggingfaceModel(model=model, tokenizer=tokenizers, model_name=model_name,
                                generation_config=generation_config)

eval_model_path = 'thu-coai/ShieldLM-7B-internlm2'
attack_model = ServerAPIModel(model_name=eval_model_path, conversation_template='internlm2',
                              base_url='http://10.11.140.104:23333/v1/chat/completions',
                              generation_config={'max_tokens': 512, 'stop': ["<|im_end|>"]})
attack_model.set_system_message(
    "You are an AI assistant whose name is InternLM (书生·浦语).\n"
    "- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n"
    "- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.")

tokenizers = AutoTokenizer.from_pretrained(eval_model_path, trust_remote_code=True)

dataset_name = 'AdvBench'
dataset = JailbreakDataset(dataset_name)

attacker = ReNeLLM(attack_model=attack_model,
                   target_model=target_model,
                   eval_model=attack_model,
                   jailbreak_datasets=dataset)
attacker.attack()
attacker.jailbreak_datasets.save_to_jsonl('AdvBench_renellm.jsonl')
