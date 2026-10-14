import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel

max_seq_length = 2048
load_in_4bit = True

print("🚀 Loading base model and tokenizer (Llama 3 8B)...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-Instruct-bnb-4bit",
    max_seq_length=max_seq_length,
    load_in_4bit=load_in_4bit,
)

# Attach LoRA adapters for parameter-efficient fine-tuning
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# Text-to-SQL Prompt Template matching BIRD style
bird_prompt = """Below is a natural language question about a database, paired with schema evidence. Write the correct SQL query to answer the question.

### Question:
{question}

### Schema Evidence:
{evidence}

### SQL:
{sql}"""

EOS_TOKEN = tokenizer.eos_token


def formatting_func(examples):
  questions = examples["question"]
  evidences = examples["evidence"]
  sqls = examples["SQL"]

  texts = []
  for q, ev, sql in zip(questions, evidences, sqls):
    ev_text = ev if ev else "None provided"
    text = (
        bird_prompt.format(question=q, evidence=ev_text, sql=sql) + EOS_TOKEN
    )
    texts.append(text)
  return {"text": texts}


print("📥 Downloading BIRD dataset from Hugging Face...")
dataset = load_dataset("birdsql/bird_mini_dev", split="mini_dev_mysql")
dataset = dataset.map(formatting_func, batched=True)

print("⚙️ Configuring Trainer...")
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,  # 👈 Updated here from tokenizer=tokenizer
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,  # Quick test run steps
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        output_dir="outputs",
    ),
)

print("🔥 Starting fine-tuning...")
trainer_stats = trainer.train()

print("💾 Saving fine-tuned model adapters...")
model.save_pretrained_merged(
    "bird_sql_model", tokenizer, save_method="merged_16bit"
)
print("✅ Training complete! Model saved locally to 'bird_sql_model'.")
