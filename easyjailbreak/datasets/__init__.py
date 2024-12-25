from fastchat.conversation import register_conv_template, Conversation, SeparatorStyle

from .instance import Instance
from .jailbreak_datasets import JailbreakDataset

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

register_conv_template(
    Conversation(
        name="internlm2",
        system_template="<system>\n{system_message}",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
        stop_token_ids=[92542],
        stop_str="<|end|>",
    )
)