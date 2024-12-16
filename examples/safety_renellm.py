import os, sys

import torch
from fastchat.conversation import register_conv_template, Conversation, SeparatorStyle
from transformers import AutoModelForCausalLM, AutoTokenizer
from easyjailbreak.datasets import JailbreakDataset
from easyjailbreak.attacker.ReNeLLM_ding_2023 import *
from easyjailbreak.models.openai_model import OpenaiModel
from easyjailbreak.models.huggingface_model import HuggingfaceModel

register_conv_template(
    Conversation(
        name="phi3",
        system_template="<system>\n{system_message}",
        roles=("<|user|>", "<|assistant|>"),
        sep_style=SeparatorStyle.CHATML,
        sep="<|end|>",
        stop_token_ids=[32000, 32007],
        stop_str="<|end|>",
    )
)

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

# chat_name = 'microsoft/Phi-3-small-8k-instruct'
# chat_name = target_model_path
# model = AutoModelForCausalLM.from_pretrained(chat_name, trust_remote_code=True, attn_implementation="flash_attention_2")
# tokenizers = AutoTokenizer.from_pretrained(chat_name, trust_remote_code=True)
attack_model = HuggingfaceModel(model=model, tokenizer=tokenizers, model_name=model_name,
                                generation_config=generation_config)

dataset_name = 'AdvBench'
dataset = JailbreakDataset(dataset_name)

attacker = ReNeLLM(attack_model=attack_model,
                   target_model=target_model,
                   eval_model=attack_model,
                   jailbreak_datasets=dataset)
attacker.attack()
attacker.jailbreak_datasets.save_to_jsonl('AdvBench_renellm.jsonl')
