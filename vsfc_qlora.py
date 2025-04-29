import argparse
import time
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from torchmetrics import Accuracy, F1Score
from mint.uit_vsfc_helpers import VSFCLoader, load_aivivn, AIVIVNDataset, AIVIVNLoader
from underthesea import word_tokenize
from torch.utils.data import random_split, DataLoader
import GPUtil

torch.set_float32_matmul_precision('high')

def lora_parse_args():
    parser = argparse.ArgumentParser(description="QLoRA Fine-tuning for Vietnamese Sentiment Analysis")
    parser.add_argument("--model", type=int, choices=[1,2,3,4], required=True,
                        help="1=PhoBERT-base-v2,2=PhoBERT-large,3=BARTpho,4=ViT5")
    parser.add_argument("--dataset", type=str, choices=['uit','aivi'], default='uit',
                        help="'uit' for UIT-VSFC, 'aivi' for AIVIVN-2019")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def get_4bit_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.bfloat16
    )


def get_lora_config(model_name: str):
    target_modules = {
        'vinai/phobert-base-v2': ["query","value"],
        'vinai/phobert-large': ["query","value"],
        'vinai/bartpho-word': ["q_proj","v_proj"],
        'VietAI/vit5-large': ["q","v"]
    }
    return LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, bias="none",
                      target_modules=target_modules[model_name], task_type="SEQ_CLS")


class QLoRA4VSA(L.LightningModule):
    def __init__(self, model_name, num_labels, lr):
        super().__init__()
        # Save lr in self.hparams.lr
        self.save_hyperparameters()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels,
            quantization_config=get_4bit_config()
        )
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, get_lora_config(model_name))
        self.train_acc = Accuracy(task="multiclass", num_classes=num_labels)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_labels)
        self.val_f1  = F1Score(task="multiclass", num_classes=num_labels, average='macro')
        self.test_acc = Accuracy(task="multiclass", num_classes=num_labels)
        self.test_f1  = F1Score(task="multiclass", num_classes=num_labels, average='macro')

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        out = self(**batch)
        loss = out.loss
        preds = torch.argmax(out.logits, dim=1)
        self.train_acc(preds, batch['labels'])
        self.log('train_loss', loss)
        self.log('train_acc', self.train_acc, prog_bar=True)
        if torch.cuda.is_available():
            for gpu in GPUtil.getGPUs(): self.log(f"GPU_{gpu.id}", gpu.memoryUsed)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(**batch)
        loss = out.loss
        preds = torch.argmax(out.logits, dim=1)
        self.val_acc(preds, batch['labels']); self.val_f1(preds, batch['labels'])
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, prog_bar=True)
        self.log('val_f1', self.val_f1, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        out = self(**batch)
        preds = torch.argmax(out.logits, dim=1)
        self.test_acc(preds, batch['labels']); self.test_f1(preds, batch['labels'])
        self.log('test_acc', self.test_acc, prog_bar=True)
        self.log('test_f1', self.test_f1, prog_bar=True)
        return {'acc': self.test_acc, 'f1': self.test_f1}

    def configure_optimizers(self):
        # correctly reference lr
        return torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr, weight_decay=0.01)


if __name__ == '__main__':
    args = lora_parse_args()
    model_map = {1:"vinai/phobert-base-v2",2:"vinai/phobert-large",3:"vinai/bartpho-word",4:"VietAI/vit5-large"}
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(model_map[args.model])

    if args.dataset == 'aivi':
        texts, labels = load_aivivn('train')
        texts = [word_tokenize(t, format='text') for t in texts]
        ds = AIVIVNDataset(texts, labels, tokenizer, max_length=128)
        v = int(len(ds)*0.1); train_ds, val_ds = random_split(ds,[len(ds)-v, v])
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4)
        test_loader  = AIVIVNLoader(tokenizer, batch_size=args.batch_size).load_data('test')
    else:
        loader = VSFCLoader(tokenizer, batch_size=args.batch_size)
        train_loader, val_loader, test_loader = loader.load_data('train'), loader.load_data('val'), loader.load_data('test')

    model = QLoRA4VSA(model_name=model_map[args.model], num_labels=3, lr=args.learning_rate)
    GPUs = [int(x) for x in args.gpus.split(',')]
    trainer = L.Trainer(max_epochs=args.epochs, accelerator='gpu', devices=GPUs,
                        callbacks=[EarlyStopping('val_loss',patience=3)], precision='bf16-mixed')
    t0=time.time(); trainer.fit(model, train_loader, val_loader)
    if trainer.is_global_zero: print(f"Time: {time.time()-t0:.1f}s")
    trainer.test(model, test_loader)
