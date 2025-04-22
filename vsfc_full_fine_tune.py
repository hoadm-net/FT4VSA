import time
import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks import DeviceStatsMonitor
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torchmetrics import Accuracy, F1Score
from underthesea import word_tokenize
import argparse
import GPUtil
from mint.uit_vsfc_helpers import load_uit_vsfc, VSFCDataset


torch.set_float32_matmul_precision('high')


def fft_parse_args():
    parser = argparse.ArgumentParser(
        description="Full Fine-tuning for Vietnamese Sentiment Analysis"
    )
    
    # Model selection
    parser.add_argument(
        "--model",
        type=int,
        choices=[1, 2, 3, 4],
        required=True,
        help="Select model: 1=PhoBERT-base-v2, 2=PhoBERT-large, 3=BARTpho, 4=ViT5"
    )
    
    # Batch size
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )
    
    # Number of epochs
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)"
    )
    
    # GPU indices
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="Comma-separated list of GPU indices to use, e.g., '0,1,2' (default: '0')"
    )
    
    # Learning rate
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)"
    )
    
    # Optional: Seed for reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    # Optional: Output directory
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save checkpoints and logs (default: 'outputs')"
    )
    
    return parser.parse_args()


class FFT4SentimentAnalysis(L.LightningModule):
    def __init__(self, model_name, num_labels, lr=2e-5):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, ignore_mismatched_sizes=True
            )
        self.lr = lr

        self.train_acc = Accuracy(task="multiclass", num_classes=num_labels)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_labels)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_labels, average='macro')
        self.test_acc = Accuracy(task="multiclass", num_classes=num_labels)
        self.test_f1 = F1Score(task="multiclass", num_classes=num_labels, average='macro')

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        preds = torch.argmax(outputs.logits, dim=1)
        self.train_acc(preds, batch['labels'])
        self.log('train_loss', loss)
        self.log('train_acc', self.train_acc, prog_bar=True)

        if torch.cuda.is_available():
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                self.log(f"GPU {gpu.id}", gpu.memoryUsed)
 
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        preds = torch.argmax(outputs.logits, dim=1)
        self.val_acc(preds, batch['labels'])
        self.val_f1(preds, batch['labels'])
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, prog_bar=True)
        self.log('val_f1', self.val_f1, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        outputs = self(**batch)
        preds = torch.argmax(outputs.logits, dim=1)
        acc = self.test_acc(preds, batch['labels'])
        f1 = self.test_f1(preds, batch['labels'])
        self.log('test_acc', acc, prog_bar=True)
        self.log('test_f1', f1, prog_bar=True)
        return {'test_acc': acc, 'test_f1': f1}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


if __name__ == '__main__':
    args = fft_parse_args()
    
    train_texts, train_labels = load_uit_vsfc('train')
    val_texts, val_labels = load_uit_vsfc('val')
    test_texts, test_labels = load_uit_vsfc('test')

    train_texts = [word_tokenize(text, format='text') for text in train_texts]
    val_texts = [word_tokenize(text, format='text') for text in val_texts]
    test_texts = [word_tokenize(text, format='text') for text in test_texts]

    # Load the tokenizer
    if args.model == 1:
        model_name = "vinai/phobert-base-v2"
    elif args.model == 2:
        model_name = "vinai/phobert-large"
    elif args.model == 3:
        model_name = "facebook/bartpho"
    elif args.model == 4:
        model_name = "google/vit5-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set the random seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Load data
    train_dataset = VSFCDataset(train_texts, train_labels, tokenizer)
    val_dataset = VSFCDataset(val_texts, val_labels, tokenizer)
    test_dataset = VSFCDataset(test_texts, test_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)

    # Initialize the model
    model = FFT4SentimentAnalysis(model_name=model_name, num_labels=3, lr=args.learning_rate)
    print('\n\n')

    # Print resource usage
    GPUs = [int(gpu) for gpu in args.gpus.split(',')]

    trainer = L.Trainer(
        max_epochs=args.epochs, 
        accelerator='gpu', 
        devices=GPUs,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=3)
        ])
    
    # Train the model
    start_time = time.time()
    trainer.fit(model, train_loader, val_loader)
    end_time = time.time()
    print(f"Training time: {end_time - start_time} seconds")

    # Test the model
    results = trainer.test(model, test_loader)
    print(f"Test accuracy: {results[0]['test_acc']}")
    print(f"Test F1 score: {results[0]['test_f1']}")
