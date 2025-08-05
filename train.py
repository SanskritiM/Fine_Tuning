import os
import re
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    BitsAndBytesConfig, 
    WhisperTokenizer, 
    WhisperFeatureExtractor
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
import numpy as np
from datasets import DatasetDict
from accelerate import Accelerator
from peft.utils.other import prepare_model_for_kbit_training
from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model
import torch.multiprocessing as mp
from datasets import Audio

AWS_KEY = os.getenv("AWS_KEY")
AWS_SECRET = os.getenv("AWS_SECRET")
HF_TOKEN = os.getenv("HF_TOKEN")

def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz

    model_name_or_path = "openai/whisper-large-v2"
    task = "transcribe"
    language = "Hindi"

    processor = WhisperProcessor.from_pretrained(model_name_or_path, language=language, task=task)
    audio = batch["audio_bytes"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = processor.tokenizer(batch["transcript"]).input_ids
    return batch


def is_female(example):
    return example['gender'].lower() == 'female'


def get_data():
    s3_path_train = "s3://faqs-audio/train_hindi_question/*batch*.parquet"
    s3_path_val = "s3://faqs-audio/val_hindi_question/*batch*.parquet"
    s3_path_test = "s3://faqs-audio/test_hindi_question/*batch*.parquet"
    
    aws_key = AWS_KEY
    aws_secret = AWS_SECRET
    storage_options = {"key": aws_key,
                       "secret": aws_secret,
                       "client_kwargs": {
                           "region_name": "ap-south-1"  # e.g., "us-east-1"
                       }
                      }
    
    train_dataset = load_dataset("parquet", data_files=s3_path_train, storage_options=storage_options, streaming=True)
    validation_dataset = load_dataset("parquet", data_files=s3_path_val, storage_options=storage_options, streaming=True)
    test_dataset = load_dataset("parquet", data_files=s3_path_test, storage_options=storage_options, streaming=True)
    
    common_voice = DatasetDict()
    common_voice["train"]=train_dataset['train']
    common_voice["validation"]=validation_dataset['train']
    common_voice["test"]=test_dataset['train']
    
    
    common_voice["train"] = common_voice["train"].remove_columns(["language", "speaker" ,"duration"])
    common_voice["validation"] = common_voice["validation"].remove_columns(["language", "speaker" ,"duration"])
    common_voice["test"] = common_voice["test"].remove_columns(["language", "speaker" ,"duration"])    

    train_dataset = common_voice["train"].cast_column("audio_bytes", Audio(sampling_rate=16000))
    test_dataset = common_voice["test"].cast_column("audio_bytes", Audio(sampling_rate=16000))
    validation_dataset = common_voice["validation"].cast_column("audio_bytes", Audio(sampling_rate=16000))
    
    # Apply preprocessing
    train_features = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names)
    test_features = test_dataset.map(prepare_dataset, remove_columns=test_dataset.column_names)
    val_features = validation_dataset.map(prepare_dataset, remove_columns=validation_dataset.column_names)
    
    # train_subset = train_features.take(64)
    # val_subset = val_features.take(16)

    return train_features, val_features, test_features



@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Generate attention mask for inputs
        batch["attention_mask"] = torch.ones(batch["input_features"].shape[:-1], dtype=torch.long)

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        # print(batch)

        return batch


def compute_metrics(pred):
    wer_metric = evaluate.load("wer")
    model_name_or_path = "openai/whisper-large-v2"
    task = "transcribe"
    language = "Hindi"
    language_abbr = "hi"

    processor = WhisperProcessor.from_pretrained(model_name_or_path, language=language, task=task)

    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100*wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


def main():
    # print("Hello")
    # os.environ["MASTER_ADDR"] = "127.0.0.1"
    # os.environ["MASTER_PORT"] = "29501"
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    print(f"[rank {local_rank}/{world_size}]" f"on device {torch.cuda.current_device()}")

    train_subset, val_subset, test_subset = get_data()

    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

    # 3) Load & prepare your quantized model
    model_name_or_path = "openai/whisper-large-v2"
    task = "transcribe"
    language = "Hindi"
    processor = WhisperProcessor.from_pretrained(model_name_or_path, language=language, task=task)

    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    model = WhisperForConditionalGeneration.from_pretrained(
        model_name_or_path,
        quantization_config=quant_config,
        device_map={"": local_rank})

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=False
    )

    # Hook to allow gradient on the first conv
    def make_inputs_require_grad(module, inp, out):
        out.requires_grad_(True)
    model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)

    # 4) Wrap with LoRA
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none"
    )
    train_model = get_peft_model(model, peft_config)
    train_model.print_trainable_parameters()

    # Create data collator

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # 5) Prepare Trainer & TrainingArguments
    acc_config = {'split_batches':True, 'dispatch_batches':False}

    training_args = Seq2SeqTrainingArguments(
        output_dir="./v2_hindi_faq_whisper_large_v2",
        hub_private_repo=True,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        weight_decay=0.01,
        warmup_steps=50,
        max_steps=30300,
        num_train_epochs=500,
        gradient_checkpointing=False,
        fp16=True,
        eval_strategy="epoch",
        per_device_eval_batch_size=4,
        predict_with_generate=True,
        save_strategy="epoch",
        resume_from_checkpoint=True,
        logging_strategy="epoch",
        report_to=["tensorboard"],
        push_to_hub=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        eval_on_start=True,
        dataloader_num_workers=2,
        label_names=["labels"],
        accelerator_config=acc_config,
    )
    
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=train_model,
        train_dataset=train_subset,
        eval_dataset=val_subset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    try:
        trainer.train()
    finally:
        peft_model_id = "v2_hindi_faq_whisper_large_v2"
        model.push_to_hub(peft_model_id)
    
    test_predictions = trainer.predict(test_subset)
    test_metrics = compute_metrics(test_predictions)
    print(f"Detailed Test WER: {test_metrics['wer']:.2f}%")


    trainer.save_model("./whisper-finetuned-final")
    
    


if __name__ == "__main__":
    main()
