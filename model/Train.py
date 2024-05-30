import torch
from torch.utils.data import DataLoader

class Trainer:
    """Class pour entrainer un modèle.
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

    def save_model(self, path : str = "model.pth", history_path : str = "history.txt"):
        print("Saving the model...")
        torch.save(self.model.state_dict(), path)
        torch.save(self.history, history_path)
        print("Model saved.")

    def set_optimizer(self, optimizer : str):
        self.optimizer : torch.optim.Optimizer = optimizer
        return self
    
    def set_model(self, model : torch.nn.Module):
        self.model = model
        return self
    
    def set_loader(self, train_loader : DataLoader, val_loader : DataLoader , test_loader : DataLoader ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        return self
    
    def set_loss_fn(self, loss_fn : torch.nn.Module):
        self.loss_fn = loss_fn
        return self
    
    def train(self):
        """Méthode pour entrainer le modèle"""
        print("Training...")
        self.model.train()
        train_loss = 0
        for inputs, labels, ns_scores in self.train_loader:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            labels = labels.to(self.device)
            ns_scores = ns_scores.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(**inputs)
            loss = self.loss_fn(outputs, labels, ns_scores, self.smoothing)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        return train_loss
    
    def validate(self):
        """Méthode pour valider le modèle"""
        print("Validating...")
        self.model.eval()
        with torch.no_grad():
            val_loss = 0
            for inputs, labels, ns_scores in self.val_loader:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                labels = labels.to(self.device)
                ns_scores = ns_scores.to(self.device)
                outputs = self.model(**inputs)
                loss = self.loss_fn(outputs, labels, ns_scores, self.smoothing)
                val_loss += loss.item()
            return val_loss
        
    def fit(self, learning_rate = 1e-4, epochs : int = 100, weight_decay : float = 0.01, smoothing : float = 0.1, CL=False):
        """Method to train the model.
        @param learning_rate : float, The learning rate for the optimizer.
        @param epochs : int, The number of epochs for training the model.
        """
        print(f"Training the model on {self.device}...")
        self.model.to(self.device)
        self.smoothing = smoothing
        self.optimizer = self.optimizer(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        self.history["params"]["learning_rate"] = learning_rate
        self.history["params"]["weight_decay"] = weight_decay
        self.history["params"]["epochs"] = epochs
        self.history["params"]["smoothing"] = smoothing

        val_loss, train_loss = [], []
        for epoch in range(epochs):
            train_loss_epoch = self.train()
            if self.val_loader is not None:
                val_loss_epoch = self.validate()
                val_loss.append(val_loss_epoch / len(self.val_loader))

            train_loss.append(train_loss_epoch/len(self.train_loader))

            # for curriculum learning, mettre le smoothing à 0 (two-stage training selon le papier)
            if CL and epoch == epochs//2:
                self.smoothing = 0.0

            print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {train_loss[-1]}, Validation Loss: {val_loss[-1]}")

        print("Training complete.")
        self.history["training"]["loss"] = train_loss
        self.history["validation"]["loss"] = val_loss

        return self.history