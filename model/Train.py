import torch
from Bert import Bert
from torch.utils.data import DataLoader

class Trainer:
    """A class to represent the training process for the U-Net model for vessel segmentation.
    """
    def __init__(self) -> None:
        self.model : torch.nn.Module = None
        self.train_loader : DataLoader = None
        self.val_loader : DataLoader = None
        self.loss_fn : torch.nn.Module = None
        self.optimizer : torch.optim.Optimizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.history = {
            "validation": {
                "accuracy": [],
                "loss": [],
                "precision": [],
                "recall": []
            },
            "training": {
                "accuracy": [],
                "loss": [],
                "precision": [],
                "recall": []
            },
            "params": {
                "learning_rate": None,
                "weight_decay": None,
                "epochs": None,
                "smoothing" : None,
            }
        }

    
    def load_and_set_model(self, model_name : str, path : str):
        """Method to load and set the model"""
        state = torch.load(path)
        model = Bert(model_name)
        model.load_state_dict(state)
        self.set_model(model)
        return model
    
    def save_model(self, path : str = "model.pth", history_path : str = "history.txt"):
        """Method to save the model.
        """
        print("Saving the model...")
        torch.save(self.model.state_dict(), path)
        torch.save(self.history, history_path)
        print("Model saved.")

    def set_optimizer(self, optimizer : str):
        """Method to set the optimizer for the model.
        @param optimizer, The optimizer to be used for training the model.
        """
        self.optimizer : torch.optim.Optimizer = optimizer
        return self
    
    def set_model(self, model : torch.nn.Module):
        """Method to set the model for training.
        @param model, The model to be trained.
        """
        self.model = model
        return self
    
    def set_loader(self, train_loader : DataLoader, val_loader : DataLoader, test_loader : DataLoader):
        """Method to set the training and validation data loaders.
        @param train_loader : DataLoader, The training data loader.
        @param val_loader : DataLoader, The validation data loader.
        @param test_loader : DataLoader, The test data loader.
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        return self
    
    def set_loss_fn(self, loss_fn : torch.nn.Module):
        """Method to set the loss function for training the model.
        @param loss_fn, The loss function to be used for training the model.
        """
        self.loss_fn = loss_fn
        return self
    
    def train(self):
        """Method to train the model"""
        print("Training...")
        self.model.train()
        train_loss = 0
        for batch in self.train_loader:
            inputs = batch['input_ids'].to(self.device)
            masks = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            self.optimizer.zero_grad()
            y_pred = self.model(inputs, masks)
            loss = self.loss_fn(y_pred, labels, self.smoothing)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        return train_loss
    
    def validate(self):
        """Methode to validate the model"""
        print("Validating...")
        self.model.eval()
        with torch.no_grad():
            val_loss = 0
            for batch in self.val_loader:
                inputs = batch['input_ids'].to(self.device)
                masks = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                self.optimizer.zero_grad()
                y_pred = self.model(inputs, masks)
                loss = self.loss_fn(y_pred, labels, self.smoothing)
                val_loss += loss.item()
            return val_loss
        
    def fit(self, learning_rate = 1e-4, epochs : int = 100, weight_decay : float = 0.01, smoothing : float = 0.1, CL=False):
        """Method to train the model.
        @param learning_rate : float, The learning rate for the optimizer.
        @param epochs : int, The number of epochs for training the model.
        """
        print(f"Training the model on {self.device}...")
        self.model.to(self.device)
        self.optimizer = self.optimizer(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.smoothing = smoothing
        self.history["params"]["learning_rate"] = learning_rate
        self.history["params"]["weight_decay"] = weight_decay
        self.history["params"]["epochs"] = epochs
        self.history["params"]["smoothing"] = smoothing

        val_loss, train_loss = [], []
        for epoch in range(epochs):
            train_loss_epoch = self.train()
            val_loss_epoch = self.validate()
            train_loss.append(train_loss_epoch/len(self.train_loader))
            val_loss.append(val_loss_epoch / len(self.val_loader))
            if epoch == epochs//2 and CL:
              self.smoothing = 0 
            print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {train_loss[-1]}, Validation Loss: {val_loss[-1]}")

        print("Training complete.")
        self.history["training"]["loss"] = train_loss
        self.history["validation"]["loss"] = val_loss

        return self.history