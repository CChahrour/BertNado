from loguru import logger
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification


def get_model(
     
    task, 
    label2id,
    device, 
    dropout, 
    lora_r, 
    lora_alpha, 
    lora_dropout,
    model_parent="PoetschLab/GROVER",
):
    """
    Initialize the model with LoRA configuration.
    Args:
        model_parent: Parent model name or path.
        task: Task type (e.g., "multi_label_classification").
        label2id: Mapping of labels to IDs.
        device: Device for training (e.g., "cuda").
        dropout: Dropout rate for the model.
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha.
        lora_dropout: LoRA dropout rate.
    Returns:
        model: Initialized model with LoRA configuration.
    """
    logger.info(f"Initializing model: {model_parent}")
    logger.info(f"Task: {task}")
    if task == "multi_label_classification":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_parent,
            num_labels=len(label2id),
            problem_type=task,
            label2id=label2id,
            id2label={v: k for k, v in label2id.items()},
            trust_remote_code=True,
            hidden_dropout_prob=dropout,
        )
    elif task == "binary_classification":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_parent,
            num_labels=2,
            problem_type=task,
            label2id=label2id,
            id2label={v: k for k, v in label2id.items()},
            trust_remote_code=True,
            hidden_dropout_prob=dropout,
        )
    elif task == "regression":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_parent,
            num_labels=1,
            problem_type=task,
            trust_remote_code=True,
            hidden_dropout_prob=dropout,
        )
    else:
        raise ValueError(f"Unsupported task: {task}")
    
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=["query", "value"],
    )

    model = get_peft_model(model, peft_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of trainable parameters: {trainable:,}")

    return model.to(device)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Initialize model with LoRA configuration.")
    parser.add_argument("--model_parent", type=str, required=True, help="Parent model name or path.")
    parser.add_argument("--task", type=str, required=True, help="Task type (e.g., 'multi_label_classification').")
    parser.add_argument("--label2id", type=dict, required=True, help="Mapping of labels to IDs.")
    parser.add_argument("--device", type=str, required=True, help="Device for training (e.g., 'cuda').")
    parser.add_argument("--dropout", type=float, required=True, help="Dropout rate for the model.")
    parser.add_argument("--lora_r", type=int, required=True, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, required=True, help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float, required=True, help="LoRA dropout rate.")

    args = parser.parse_args()

    get_model(
        model_parent=args.model_parent,
        task=args.task,
        label2id=args.label2id,
        device=args.device,
        dropout=args.dropout,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )