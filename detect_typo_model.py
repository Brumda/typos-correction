import pandas as pd
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from helpers import DATA_PATH


class TypoDetectionModel:
    def __init__(self,
                 model_name="nreimers/MiniLM-L6-H384-uncased",
                 batch_size=32,
                 max_len=96,
                 save_path="detect_typo_models/best_model.pt",
                 wandb_project="typo-detection-probability",
                 wandb_run_name="minilm-typo-detection"
                 ):
        # Configuration
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_len = max_len
        self.save_path = save_path
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = None
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the tokenizer and model architecture"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = self._create_model().to(self.device)

    def _create_model(self):
        """Create the neural network model"""

        class TypoDetection(nn.Module):
            def __init__(self, model_name):
                super().__init__()
                self.encoder = AutoModel.from_pretrained(model_name)
                self.head = nn.Linear(self.encoder.config.hidden_size, 1)

            def forward(self, input_ids, attention_mask):
                outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                cls_token = outputs.last_hidden_state[:, 0]
                prob = torch.sigmoid(self.head(cls_token)).squeeze(1)
                return prob

        return TypoDetection(self.model_name)

    def load_data(self):
        """Load and prepare the dataset"""
        train_df = pd.read_csv(DATA_PATH + "train_prob_df.csv", dtype={0: str, 1: float})
        dev_df = pd.read_csv(DATA_PATH + "dev_prob_df.csv", dtype={0: str, 1: float})
        test_df = pd.read_csv(DATA_PATH + "test_prob_df.csv", dtype={0: str, 1: float})
        return train_df, dev_df, test_df

    class _SentenceDataset(Dataset):
        """Dataset with tokenization"""

        def __init__(self, df, tokenizer, max_len=96):
            self.sentences = df["text"].tolist()
            self.labels = df["prob"].values.astype("float32")
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            sentence = self.sentences[idx]
            inputs = self.tokenizer(
                    sentence,
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_len,
                    return_tensors='pt'
            )
            item = {key: val.squeeze(0) for key, val in inputs.items()}
            item["label"] = torch.tensor(self.labels[idx])
            return item

    class _EarlyStopping:
        def __init__(self, patience, delta, path):
            self.patience = patience
            self.delta = delta
            self.path = path
            self.counter = 0
            self.best_score = None
            self.early_stop = False

        def __call__(self, dev_loss, model):
            if self.best_score is None:
                self.best_score = dev_loss
                self._save_checkpoint(dev_loss, model)
            elif dev_loss > self.best_score + self.delta:  # No improvement
                self.counter += 1
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
                if self.counter >= self.patience:
                    self.early_stop = True
            else:  # Could be worse
                self.counter = 0
                if self.best_score > dev_loss: # actual improvement
                    self._save_checkpoint(dev_loss, model)
                    self.best_score = dev_loss

        def _save_checkpoint(self, dev_loss, model):
            print(f"Dev loss decreased ({self.best_score:.6f} --> {dev_loss:.6f}). Saving model...")
            torch.save(model.state_dict(), self.path)
            print(f"Model saved to {self.path}")

    def train(self, train_df, dev_df, epochs=10, learning_rate=1e-5, patience=3, delta=0.001, use_wandb=True):
        """Train the model with early stopping"""
        if use_wandb:
            wandb.init(project=self.wandb_project, name=self.wandb_run_name)

        ####################################
        # Data initialization
        ####################################
        train_ds = self._SentenceDataset(train_df, self.tokenizer, self.max_len)
        dev_ds = self._SentenceDataset(dev_df, self.tokenizer, self.max_len)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        dev_loader = DataLoader(dev_ds, batch_size=self.batch_size)

        ####################################
        # optimizer and loss function
        ####################################
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()  # Mean Squared Error for probability regression

        ####################################
        # Initialize early stopping
        ####################################
        early_stopping = self._EarlyStopping(
                patience=patience,
                delta=delta,
                path=self.save_path
        )
        ####################################
        # Training loop
        ####################################
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            ####################################
            # Train phase
            ####################################
            # for batch in tqdm(train_loader):
            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                optimizer.zero_grad()
                preds = self.model(input_ids, attention_mask)
                loss = loss_fn(preds, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            print(f"[Epoch {epoch + 1}] Train Loss: {avg_train_loss:.4f}")

            ####################################
            # Validation phase
            ####################################
            dev_metrics = self._validate(dev_loader, loss_fn)
            avg_dev_loss = dev_metrics["dev_loss"]
            mae = dev_metrics["mae"]

            ####################################
            # Log metrics
            ####################################
            metrics = {
                    "epoch":      epoch + 1,
                    "train_loss": avg_train_loss,
                    "dev_loss":   avg_dev_loss,
                    "dev_mae":    mae
            }

            if use_wandb:
                wandb.log(metrics)

            print(
                    f"[Epoch {epoch + 1}] Train Loss: {avg_train_loss:.4f} | Dev Loss: {avg_dev_loss:.4f} | MAE: {mae:.4f}")

            ####################################
            # Check early stopping
            ####################################
            early_stopping(avg_dev_loss, self.model)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

        if use_wandb:
            wandb.finish()

    def _validate(self, dev_loader, loss_fn):
        """Validate the model and return metrics"""
        self.model.eval()
        dev_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dev_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                preds = self.model(input_ids, attention_mask)
                dev_loss += loss_fn(preds, labels).item()

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_dev_loss = dev_loss / len(dev_loader)

        ####################################
        # Calculate MAE
        ####################################
        all_preds_tensor = torch.tensor(all_preds)
        all_labels_tensor = torch.tensor(all_labels)
        mae = nn.functional.l1_loss(all_preds_tensor, all_labels_tensor).item()

        return {
                "dev_loss": avg_dev_loss,
                "mae":      mae
        }

    def evaluate(self, test_df):
        """Evaluate the model on test data"""
        test_ds = self._SentenceDataset(test_df, self.tokenizer, self.max_len)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size)

        loss_fn = nn.MSELoss()
        metrics = self._validate(test_loader, loss_fn)

        print(f"Test Loss: {metrics['dev_loss']:.4f} | Test MAE: {metrics['mae']:.4f}")
        return metrics

    def predict(self, sentence):
        """Make a prediction for a single sentence"""
        self.model.eval()

        ####################################
        # Tokenize the input
        ####################################
        inputs = self.tokenizer(
                sentence,
                padding='max_length',
                truncation=True,
                max_length=self.max_len,
                return_tensors='pt'
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        ####################################
        # Get prediction
        ####################################
        with torch.no_grad():
            prob = self.model(input_ids, attention_mask)

        return prob.item()

    def load_model(self, path=None):
        """Load the model from a file"""
        if path is None:
            path = self.save_path
            self.model.load_state_dict(torch.load(path, strict=False))
        print(f"Model loaded from {path}")


if __name__ == "__main__":
    typo_model = TypoDetectionModel()
    train_df, dev_df, test_df = typo_model.load_data()
    typo_model.train(train_df, dev_df, epochs=20)
    # Load the best model for test evaluation
    typo_model.load_model()
    metrics = typo_model.evaluate(test_df)
    with open("detect_typo_result.txt", "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: '{value}'\n")
