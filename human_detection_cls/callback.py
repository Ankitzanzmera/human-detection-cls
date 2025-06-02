import torch

class EarlyStopping:
    def __init__(self, patience=3, delta=0, verbose=False, save_path="checkpoint.pth"):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.save_path = save_path
        self.best_model = None

    def __call__(self, val_loss, model, logger):
        score = -val_loss  # Minimize loss (maximize negative loss)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, logger)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, logger)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, logger):
        if self.verbose:
            print(f"Validation loss improved. Saving model to {self.save_path} ...")
            logger.info(f"Validation loss improved. Saving model to {self.save_path} ...")
            
        torch.save(model.state_dict(), self.save_path)

