# Healthcare GPT-2 Fine-Tuning with PEFT and QLoRA

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![HuggingFace](https://img.shields.io/badge/🤗-Hugging%20Face-yellow)](https://huggingface.co/)

This repository demonstrates the fine-tuning of a GPT-2 model on medical data using advanced techniques such as PEFT (Parameter-Efficient Fine-Tuning) and QLoRA (Quantized Low-Rank Adaptation). The model is designed to generate healthcare-related text based on the [medalpaca/medical_meadow_wikidoc_patient_information](https://huggingface.co/datasets/medalpaca/medical_meadow_wikidoc_patient_information) dataset.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Techniques Used](#techniques-used)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Features

- Fine-tune GPT-2 on medical data using PEFT and QLoRA
- Efficient training with reduced memory footprint
- Generate healthcare-related text
- Utilizes Hugging Face's Transformers and Datasets libraries

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/healthcare-gpt2-finetuning.git
   cd healthcare-gpt2-finetuning
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install peft bitsandbytes transformers accelerate trl
   pip install --upgrade pyarrow
   ```

## Dataset

The training dataset is sourced from the [medalpaca/medical_meadow_wikidoc_patient_information](https://huggingface.co/datasets/medalpaca/medical_meadow_wikidoc_patient_information) dataset available on Hugging Face Datasets. This dataset contains medical information suitable for training healthcare language models.

## Techniques Used

### PEFT (Parameter-Efficient Fine-Tuning)
PEFT enables fine-tuning of large pre-trained models by updating only a small subset of their parameters, reducing computational requirements and allowing for efficient adaptation to specific tasks.

### QLoRA (Quantized Low-Rank Adaptation)
QLoRA is a quantization technique that reduces the model's memory footprint by representing model weights with lower precision (e.g., 4-bit). This allows for handling larger models on resource-constrained hardware.

### Hugging Face Transformers
We use the `transformers` library to load pre-trained models, manage tokenization, and facilitate fine-tuning. Our model is based on GPT-2, a popular generative model for text.

### Datasets Library
The `datasets` library is utilized to load and preprocess the dataset, providing tools for easy manipulation of data.

### Trainer API
We employ the Trainer API to handle the fine-tuning process, simplifying the training loop and providing a high-level interface for training, evaluation, and model saving.

## Usage

To use this project, you'll need to create Python scripts or Jupyter notebooks that implement the fine-tuning process and inference. Here's a general outline of the steps involved:

1. Load the dataset:
   ```python
   from datasets import load_dataset
   
   dataset = load_dataset("medalpaca/medical_meadow_wikidoc_patient_information")
   ```

2. Prepare the model and tokenizer:
   ```python
   from transformers import AutoTokenizer, AutoModelForCausalLM
   from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

   model_name = "openai-community/gpt2"
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True)
   ```

3. Configure and apply PEFT:
   ```python
   peft_config = LoraConfig(
       r=16,
       lora_alpha=32,
       target_modules=["q_proj", "v_proj"],
       lora_dropout=0.05,
       bias="none",
       task_type="CAUSAL_LM"
   )
   model = prepare_model_for_kbit_training(model)
   model = get_peft_model(model, peft_config)
   ```

4. Set up the Trainer and train the model:
   ```python
   from transformers import Trainer, TrainingArguments

   training_args = TrainingArguments(
       output_dir="./results",
       num_train_epochs=3,
       per_device_train_batch_size=4,
       save_steps=1000,
       save_total_limit=2,
   )

   trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=dataset["train"],
   )

   trainer.train()
   ```

5. Save the fine-tuned model:
   ```python
   model.save_pretrained("./healthcare_gpt2model")
   ```

6. For inference, load the model and generate text:
   ```python
   from transformers import pipeline

   generator = pipeline('text-generation', model="./healthcare_gpt2model")
   response = generator("Patient symptoms include fever and cough.", max_length=100)
   print(response[0]['generated_text'])
   ```
