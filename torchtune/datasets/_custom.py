from torchtune.modules.tokenizers import Tokenizer
from torchtune.datasets import ChatDataset
from torchtune.data import Message
from typing import Mapping, Any, List

def message_converter(sample: Mapping[str, Any], train_on_input: bool) -> List[Message]:
    input_msg = sample["input"]
    output_msg = sample["output"]

    user_message = Message(
        role="user",
        content=input_msg,
        masked=True,  # Mask if not training on prompt
    )
    assistant_message = Message(
        role="assistant",
        content=output_msg,
        masked=False,
    )
    # A single turn conversation
    messages = [user_message, assistant_message]

    return messages

def custom_dataset(
    *,
    tokenizer: Tokenizer,
    max_seq_len: int = 2048,  # You can expose this if you want to experiment
) -> ChatDataset:

    return ChatDataset(
        tokenizer=tokenizer,
        # For local csv files, we specify "csv" as the source, just like in
        # load_dataset
        source="csv",
        convert_to_messages=message_converter,
        # Llama3 does not need a chat format
        chat_format=None,
        max_seq_len=max_seq_len,
        # To load a local file we specify it as data_files just like in
        # load_dataset
        data_files="/home/lib/code/llm/llama3/input_file.csv",
    )