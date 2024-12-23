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
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
from datasets import Dataset
from trl import SFTTrainer, setup_chat_format
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient

def load_dataset(path, tokenizer):
    data = pd.read_csv(path)
    data = data.loc[:, ["question", "llm_answer"]]

    def format_chat_template(row):
        row_json = [{"role": "user", "content": "answer this question about security: " + row["question"]},
                {"role": "assistant", "content": row["llm_answer"]}]
        row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
        return row

    dataset = Dataset.from_pandas(data)
    dataset = dataset.map(
        format_chat_template,
        num_proc=config["data"]["num_proc"]
        )
    
    return dataset

def load_model(config):
    # QLoRA config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config["qlora"]["load_in_4bit"],
        bnb_4bit_quant_type=config["qlora"]["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=config["qlora"]["bnb_4bit_use_double_quant"]
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        quantization_config=bnb_config,
        device_map=config["model"]["device_map"],
        attn_implementation=config["model"]["attn_implementation"]
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    tokenizer.chat_template = None
    model, tokenizer = setup_chat_format(model, tokenizer)

    # LoRA config
    peft_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["lora_alpha"],
        lora_dropout=config["lora"]["lora_dropout"],
        bias=config["lora"]['bias'],
        task_type=config["lora"]["task_type"],
        target_modules=config["lora"]["target_modules"]
    )
    model = get_peft_model(model, peft_config)

    return model, tokenizer, peft_config

def save_mode(base_model, new_model_path, huggingface_repo):
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    base_model_reload = AutoModelForCausalLM.from_pretrained(
            base_model,
            return_dict=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
    )

    tokenizer.chat_template = None
    base_model_reload, tokenizer = setup_chat_format(base_model_reload, tokenizer)

    # Merge adapter with base model
    model = PeftModel.from_pretrained(base_model_reload, new_model_path)

    model = model.merge_and_unload()
    model.push_to_hub(huggingface_repo)
    tokenizer.push_to_hub(huggingface_repo)

def train(config):
    user_secrets = UserSecretsClient()
    hf_token = user_secrets.get_secret("HUGGINGFACE_KEY")
    wb_token = user_secrets.get_secret("WANDB_KEY")

    login(token = hf_token)
    wandb.login(key=wb_token)
    run = wandb.init(
        project='Viettel Challenge', 
        job_type="training", 
        anonymous="allow"
    )

    model, tokenizer, peft_config = load_model(config=config)
    train_dataset = load_dataset(path=config["data"]["train"], tokenizer=tokenizer)
    valid_dataset = load_dataset(path=config["data"]["valid"], tokenizer=tokenizer)

    training_arguments = TrainingArguments(
        output_dir=config["training_args"]["output_dir"],
        per_device_train_batch_size=config["training_args"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training_args"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["training_args"]["gradient_accumulation_steps"],
        optim=config["training_args"]["optim"],
        num_train_epochs=config["training_args"]["num_train_epochs"],
        evaluation_strategy=config["training_args"]["evaluation_strategy"],
        eval_steps=config["training_args"]["eval_steps"],
        logging_steps=config["training_args"]["logging_steps"],
        warmup_steps=config["training_args"]["warmup_steps"],
        logging_strategy=config["training_args"]["logging_strategy"],
        learning_rate=config["training_args"]["learning_rate"],
        fp16=config["training_args"]["fp16"],
        bf16=config["training_args"]["bf16"],
        group_by_length=config["training_args"]["group_by_length"],
        report_to=config["training_args"]["report_to"]
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        peft_config=peft_config,
        max_seq_length=config["trainer"]["max_seq_length"],
        dataset_text_field=config["trainer"]["dataset_text_field"],
        tokenizer=tokenizer,
        args=training_arguments,
        packing= False
    )

    trainer.train()
    trainer.save_model(os.path.join(config["training_args"]['output_dir'], "best"))
    wandb.finish()
    save_mode(
        config["model"]["name"], 
        os.path.join(os.path.join(config["training_args"]['output_dir'], "best")),
        config["huggingface"]["repo_name"]
        )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="config/train.yaml", help="Enter config file path!")

    args = parser.parse_args()

    with open(args.config) as config_file:
        config = yaml.safe_load(config_file)
    
    train(config=config)