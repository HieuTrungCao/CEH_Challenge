
import os
import torch
import wandb
import argparse
import yaml
import pandas as pd

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import setup_chat_format

from preprocess.util import preprocess


def load_model(config, args):
    # QLoRA config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config["qlora"]["load_in_4bit"],
        bnb_4bit_quant_type=config["qlora"]["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=config["qlora"]["bnb_4bit_use_double_quant"]
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map=config["model"]["device_map"],
        attn_implementation=config["model"]["attn_implementation"]
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model, tokenizer = setup_chat_format(model, tokenizer)

    return model, tokenizer

def infer(config, args):

    model, tokenizer = load_model(config, args)
    
    data = pd.read_csv(config["data"]["path"], encoding= 'unicode_escape')

    for question in data["question"]:
        messages = [
            {
                "role": "user",
                "content": "answer this question about security: " + preprocess(question)
            }
        ]
    
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, 
                                        add_generation_prompt=True)

        inputs = tokenizer(prompt, return_tensors='pt', padding=True, 
                        truncation=True).to("cuda")

        outputs = model.generate(**inputs, max_length=config["output"]["max_length"], 
                                num_return_sequences=1)

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(text.split("assistant")[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="config/infer.yaml", help="Enter config file path!")
    parser.add_argument("-m", "--model", help="Enter model path!")

    args = parser.parse_args()

    with open(args.config) as config_file:
        config = yaml.safe_load(config_file)
    
    infer(config=config, args=args)