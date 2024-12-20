
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
    pipeline
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

    return model, tokenizer

def infer(config, args):

    model, tokenizer = load_model(config, args)
    
    data = pd.read_csv(args.question, encoding= 'unicode_escape')

    pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    
    result = {}
    result["index"] = data["index"]
    result["question"] = data["question"]

    answer = []
    print("Infering")
    for question in data["question"]:
        messages = [
            {
                "role": "user",
                "content": "answer this question about security: " + preprocess(question)
            }
        ]
    
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, 
                                        add_generation_prompt=True)

        outputs = pipe(prompt, max_new_tokens=120, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        answer.append(outputs[0]["generated_text"].split("assistant")[-1])
    
    result["llm_answer"] = answer

    result = pd.DataFrame(result)
    if not os.path.exists(config["output"]["path"]):
        os.mkdir(config["output"]["path"])

    path_result = os.path.join(config["output"]["path"], "result.csv")
    print("Result file: ", path_result)
    result.to_csv(path_result, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="config/infer.yaml", help="Enter config file path!")
    parser.add_argument("-m", "--model", help="Enter model path!")
    parser.add_argument("-q", "--question", help="Enter question files!")

    args = parser.parse_args()

    with open(args.config) as config_file:
        config = yaml.safe_load(config_file)
    
    infer(config=config, args=args)