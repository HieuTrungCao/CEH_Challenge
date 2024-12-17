import os
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import warnings
import argparse
import yaml 

from trl import SFTTrainer
from transformers import (
    TrainingArguments,
    TextStreamer,
    AutoTokenizer, 
    AutoModelForSequenceClassification
)
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import Dataset

warnings.filterwarnings("ignore")

def load_data(config, tokenizer):
    def formatting_prompt(examples):
        questions = examples["question"]
        answers = examples["llm_answer"]
        texts = []
        for _question, _answer in zip(questions, answers):
            text = config["data"]["prompt"].format(_question, _answer) + tokenizer.eos_token
            texts.append(texts)
        
        return {"text": texts, }
    
    training_data = pd.read_csv(config["data"["path"]])
    training_data = Dataset.from_pandas(training_data)
    training_data = training_data.map(formatting_prompt, batched=True)

    return training_data

def get_model_tokenizer(config):
    # Get model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model"]["name"],
        max_seq_length=config["model"]["max_seq_length"],
        load_in_4bit=config["model"]["load_in_4bit"],
        dtype=config["model"]["dtype"]
    )

    # Get fast model with LoRA
    model = FastLanguageModel.get_peft_model(
        model=model,
        r=config["peft"]["r"],
        lora_alpha=config["peft"]["lora_alpha"],
        lora_dropout=config["peft"]["lora_dropout"],
        target_modules=config["peft"]["target_modules"],
        use_rslora=config["peft"]["use_rslora"],
        use_gradient_checkpointing=config["peft"]["use_gradient_checkpointing"],
        random_state=config["peft"]["random_state"],
        loftq_config=config["peft"]["loftq_config"]
    )

    return model, tokenizer

def train(config):
    print("Loading model!...................")
    model, tokenizer = get_model_tokenizer(config=config)
    print("Trainable paprameters: ", model.print_trainable_parameters())
    
    print("Loading training data!...........")
    training_data = load_data(config=config, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=training_data,
        dataset_text_field=config["trainer"]["dataset_text_field"],
        max_seq_length=config["trainer"]["max_seq_length"],
        dataset_num_proc=config["trainer"]["dataset_num_proc"],
        packing=config["trainer"]["packing"],
        args=TrainingArguments(
            learning_rate=config["training_args"]["learning_rate"],
            lr_scheduler_type=config["training_args"]["lr_scheduler_type"],
            per_device_train_batch_size=config["training_args"]["per_device_train_batch_size"],
            gradient_accumulation_steps=config["training_args"]["gradient_accumulation_steps"],
            num_train_epochs=config["training_args"]["num_train_epochs"],
            logging_steps=config["training_args"]["logging_steps"],
            optim=config["training_args"]["optim"],
            weight_decay=config["training_args"]["weight_decay"],
            warmup_steps=config["training_args"]["warmup_steps"],
            output_dir=config["training_args"]["output_dir"],
            seed=config["training_args"]["seed"]
        )
    )

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="config/train.yaml", help="Enter config file path!")

    args = parser.parse_args()

    with open(args.config) as config_file:
        config = yaml.safe_load(config_file)
    
    train(config=config)