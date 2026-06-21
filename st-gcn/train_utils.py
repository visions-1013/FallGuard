import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import eval


from torch.utils.data import Dataset, DataLoader, Sampler
from model.st_graph import get_distance_adjacency, get_uniform_adjacency, get_adjacency
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FallDataset(Dataset):
    def __init__(self, falls, non_falls, window_size=None, valid=False):
        self.falls = falls
        self.non_falls = non_falls
        self.valid = valid
        
        # use random shift to prevent overfitting
        # randomly sample a sub-sequence of length 'window_size' from each sequence
        self.window_size = window_size
        self.seq_len = falls.shape[2]
        
        assert self.window_size == None or self.window_size <= self.seq_len

    def __len__(self):
        # The dataset length is twice the length of the smaller list
        return len(self.falls) + len(self.non_falls)

    def __getitem__(self, idx):
        if idx < len(self.falls):
            if self.window_size:
                if self.valid:
                    start_index = self.seq_len - self.window_size
                else:
                    start_index = np.random.randint(0, self.seq_len - self.window_size + 1)
                return self.falls[idx][:, start_index:start_index + self.window_size, :], 1.0
            else:
                return self.falls[idx], 1.0
        else:
            if self.window_size:
                if self.valid:
                    start_index = self.seq_len - self.window_size
                else:
                    start_index = np.random.randint(0, self.seq_len - self.window_size + 1)
                return self.non_falls[idx - len(self.falls)][:, start_index:start_index + self.window_size, :], 0.0
            else:
                return self.non_falls[idx - len(self.falls)], 0.0

class BalancedBatchSampler(Sampler):
    def __init__(self, dataset):
        self.num_falls = len(dataset.falls)
        self.num_non_falls = len(dataset.non_falls)
        self.data_size = self.num_falls + self.num_non_falls
        self.batch_size = 2 * self.num_falls

    def __iter__(self):
        # Create an array of indices representing balanced classes
        non_fall_indices = np.arange(self.num_non_falls)
        np.random.shuffle(non_fall_indices)  # Shuffle the indices to have random batches
        batch = np.arange(self.num_falls).tolist()
        for idx in non_fall_indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = np.arange(self.num_falls).tolist()
        if len(batch) > 0:  # Yield remaining items not fitting into a full batch
            yield batch

    def __len__(self):
        return (self.data_size + self.batch_size - 1) // self.batch_size


def evaluate(model, loader, print_acc=False):
    model.eval()
    loss_func = nn.CrossEntropyLoss()
    acc, loss = 0.0, 0.0
    count = 0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.type(torch.LongTensor).to(device)
        with torch.no_grad():
            y_pred = model(X_batch)
            loss += loss_func(y_pred, y_batch).detach().cpu().item() * X_batch.size(0)
            acc += torch.sum(torch.argmax(y_pred, dim=1) == y_batch).detach().cpu().item()
            count += X_batch.size(0)
 
    loss /= count
    acc /= count
    return loss, acc

class Trainer:
    def __init__(self, model, opt_method, lr, batch_size, epochs, weight_decay=0, momentum=0) -> None:
        self.model = model
        self.model.to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
    
    def train(self, training_set, validation_set, early_stop=False):
        loss_func = nn.CrossEntropyLoss()
        training_loader = torch.utils.data.DataLoader(training_set, batch_sampler=BalancedBatchSampler(training_set))
        # training_loader = torch.utils.data.DataLoader(training_set, batch_size=self.batch_size, shuffle=True)
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=self.batch_size, shuffle=False)

        train_loss_list, train_acc_list = [], []
        val_loss_list, val_acc_list = [], []

        progress = tqdm(np.arange(self.epochs))
        for n in progress:
            self.model.train()
            running_loss = 0
            running_acc = 0
            count = 0
            for X_batch, y_batch in training_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.type(torch.LongTensor).to(device)

                y_pred = self.model(X_batch)
                batch_loss = loss_func(y_pred, y_batch)
                
                running_loss += batch_loss.item() * X_batch.size(0)
                running_acc += torch.sum(torch.argmax(y_pred, axis=-1) == y_batch).detach().cpu().item()

                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
                
                count += X_batch.size(0)

            train_loss = running_loss / count
            train_acc = running_acc / count
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)

            val_loss, val_acc = evaluate(self.model, validation_loader)
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)
            
            progress.set_description(f'Training Loss: {train_loss:.4f}')

        x_axis = np.arange(self.epochs)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(x_axis, train_loss_list, label="Training")
        axes[0].plot(x_axis, val_loss_list, label="Validation")
        axes[0].set_title("Loss")
        axes[0].set_xlabel('Epoch')
        axes[0].legend()
        axes[1].plot(x_axis, train_acc_list, label='Training')
        axes[1].plot(x_axis, val_acc_list, label='Validation')
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel('Epoch')
        axes[1].legend()

        print(f"Training loss: {train_loss_list[-1]}")
        print(f"Validation loss: {val_loss_list[-1]}")
        print(f"Training accuracy: {train_acc_list[-1]}")
        print(f"Validation accuracy: {val_acc_list[-1]}")
        
        return {'train_loss': train_acc_list, 'val_loss': val_loss_list, 'train_acc': train_acc_list, 'val_acc': val_acc_list}
    
    def evaluate(self, loader):
        self.model.eval()
        loss_func = nn.CrossEntropyLoss()
        acc, loss = 0.0, 0.0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.type(torch.LongTensor).to(device)
            with torch.no_grad():
                y_pred = self.model(X_batch)
                loss += loss_func(y_pred, y_batch).detach().cpu().item()
                acc += torch.sum(torch.argmax(y_pred, axis=-1) == y_batch).detach().cpu().item()
                
        loss /= len(loader.dataset)
        acc /= len(loader.dataset)
        return loss, acc

def KFoldCrossValidation(
    model_class, k, 
    X_falls, X_non_falls, X_test, y_test,
    opt_method='adam', lr=1e-3, batch_size=128, epochs=50, weight_decay=0.0,
    early_break=False, shift_window_size=None,
    **model_args
 ):
    test_set = (X_test, y_test)
    
    # Setting up two KFold instances
    kf_falls = KFold(n_splits=k, shuffle=True)
    kf_non_falls = KFold(n_splits=k, shuffle=True)

    train_acc_list, val_acc_list, test_acc_list = [], [], []
    sp_list, ss_list = [], []
    for i, ((train_idx_f, valid_idx_f), (train_idx_nf, valid_idx_nf)) in enumerate(zip(kf_falls.split(X_falls), kf_non_falls.split(X_non_falls))):
        print(f"Fold {i}:")
        model = model_class(**model_args)
        trainer = Trainer(model, opt_method, lr, batch_size, epochs, weight_decay=weight_decay, momentum=0)
        training_set = FallDataset(falls=X_falls[train_idx_f], non_falls=X_non_falls[train_idx_nf], window_size=shift_window_size)
        validation_set = FallDataset(falls=X_falls[valid_idx_f], non_falls=X_non_falls[valid_idx_nf], window_size=shift_window_size, valid=True)
        res = trainer.train(training_set, validation_set)
        train_acc_best = np.max(res['train_acc'])
        val_acc_best = np.max(res['val_acc'])
        accuracy, specificity, sensitivity, test_results = eval.evaluate(model, device, testset=test_set, profile=False, in_channels=model_args.get('in_channels'))
        train_acc_list.append(train_acc_best)
        val_acc_list.append(val_acc_best)
        test_acc_list.append(accuracy)
        sp_list.append(specificity)
        ss_list.append(sensitivity)
        print(f"Best training accuracy: {train_acc_best}")
        print(f"Best validation accuracy: {val_acc_best}")
        print(f"Test accuracy: {accuracy}")
        if early_break:
            break
        
    if not early_break:
        print("================Final results================")
        print(f"Training accuracy: {np.mean(train_acc_list)}+/-{np.std(train_acc_list)}")
        print(f"Validation accuracy: {np.mean(val_acc_list)}+/-{np.std(val_acc_list)}")
        print(f"Test accuracy: {np.mean(test_acc_list)}+/-{np.std(test_acc_list)}")
        print(f"Specificity: {np.mean(sp_list)}+/-{np.std(sp_list)}")
        print(f"Sensitivity: {np.mean(ss_list)}+/-{np.std(ss_list)}")