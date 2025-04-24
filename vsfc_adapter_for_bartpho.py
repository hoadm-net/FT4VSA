import time
import argparse
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.strategies import DDPStrategy
from transformers import AutoTokenizer
from adapters import AutoAdapterModel, SeqBnConfig
from torchmetrics import Accuracy, F1Score
from mint.uit_vsfc_helpers import VSFCLoader
import GPUtil


torch.set_float32_matmul_precision('high')


def bartpho_adapter_parse_args():
    """Parse command line arguments for Adapter fine-tuning"""
    parser = argparse.ArgumentParser(
        description="Adapter Fine-tuning for Vietnamese Sentiment Analysis"
    )
    parser.add_argument(
        "--adapter_size",
        type=int,
        default=64,
        help="Hidden dimension size of adapter (default: 64)"
    )
    parser.add_argument(
        "--batch_size",
        type=int, 
        default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="Comma-separated list of GPU indices to use, e.g., '0,1,2' (default: '0')"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4)"
    )
    parser.add_argument(
        "--seed", 
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    return parser.parse_args()


class BARTphoAdapter(L.LightningModule):
    """Adapter-based Fine-tuning cho BARTpho"""
    def __init__(self, num_labels=3, adapter_size=64, lr=2e-5):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = AutoAdapterModel.from_pretrained(
            "vinai/bartpho-word",
        )

        # Set adapter size and reduction factor
        hidden_size = self.model.config.hidden_size
        reduction_factor = hidden_size // adapter_size

        # Check if hidden size is divisible by adapter size
        if hidden_size % adapter_size != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by adapter size {adapter_size}."
            )
        if reduction_factor < 1:
            raise ValueError(
                f"Reduction factor {reduction_factor} must be at least 1."
            )
        
        # 2. Configure adapter for BARTpho
        adapter_config = SeqBnConfig(
            mh_adapter=True,
            output_adapter=True,
            reduction_factor=reduction_factor,
            non_linearity="relu",
            original_ln_before=True
        )
        
        # Add adapter for encoder and decoder
        self.model.add_adapter("vsbc_adapter", config=adapter_config)
        self.model.train_adapter("vsbc_adapter")
        
        # 3. Classification head uses the last hidden state of the encoder
        self.classifier = torch.nn.Linear(
            self.model.config.d_model,  # hidden_size -> d_model
            num_labels
        )
        
        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_labels)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_labels)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_labels, average='macro')
        self.test_acc = Accuracy(task="multiclass", num_classes=num_labels)
        self.test_f1 = F1Score(task="multiclass", num_classes=num_labels, average='macro')


    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=torch.zeros_like(input_ids),  # Giả lập decoder input
            output_hidden_states=True
        )
        encoder_hidden = outputs.encoder_hidden_states[-1]
        pooled_output = encoder_hidden[:, 0, :]
        return self.classifier(pooled_output)

    def training_step(self, batch, batch_idx):
        logits = self(batch['input_ids'], batch['attention_mask'])
        loss = torch.nn.functional.cross_entropy(logits, batch['labels'])
        
        self.train_acc(logits.argmax(dim=1), batch['labels'])

        self.log('train_loss', loss)
        self.log('train_acc', self.train_acc, prog_bar=True)
        
        if torch.cuda.is_available():
            for gpu in GPUtil.getGPUs():
                self.log(f"GPU {gpu.id}", gpu.memoryUsed)

        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch['input_ids'], batch['attention_mask'])
        loss = torch.nn.functional.cross_entropy(logits, batch['labels'])
        
        self.val_acc(logits.argmax(dim=1), batch['labels'])
        self.val_f1(logits.argmax(dim=1), batch['labels'])
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, prog_bar=True)
        self.log('val_f1', self.val_f1, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        logits = self(batch['input_ids'], batch['attention_mask'])

        self.test_acc(logits.argmax(dim=1), batch['labels'])
        self.test_f1(logits.argmax(dim=1), batch['labels'])
        self.log('test_acc', self.test_acc, prog_bar=True)
        self.log('test_f1', self.test_f1, prog_bar=True)
        return {'test_acc': self.test_acc, 'test_f1': self.test_f1}

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=0.01
        )
    
if __name__ == '__main__':
    args = bartpho_adapter_parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(
        "vinai/bartpho-word",
        use_fast=False,
        model_max_length=256,
    )
    
    # Loading data
    loader = VSFCLoader(tokenizer, batch_size=args.batch_size)
    train_loader = loader.load_data(subset='train')
    val_loader = loader.load_data(subset='val')
    test_loader = loader.load_data(subset='test')


    # Initialize model
    model = BARTphoAdapter(lr=args.learning_rate)
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # GPU configuration
    GPUs = [int(gpu) for gpu in args.gpus.split(',')]

    # Setup for distributed training
    trainer = L.Trainer(
        strategy=DDPStrategy(find_unused_parameters=True),
        max_epochs=args.epochs,
        accelerator='gpu', 
        devices=GPUs,
        callbacks=[EarlyStopping(monitor='val_loss', patience=3)],
        enable_progress_bar=True,
        precision="16-mixed"
    )
    
    # Training execution
    if trainer.is_global_zero:
        start_time = time.time()

    trainer.fit(model, train_loader, val_loader)

    if trainer.is_global_zero:
        end_time = time.time()
        print(f"Training time: {end_time - start_time} seconds")
    
    # Final evaluation
    trainer.test(model, test_loader)
