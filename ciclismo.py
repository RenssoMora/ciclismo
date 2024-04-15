import os
import pandas as pd
import tqdm
import numpy as np
import torch

from torch import nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils import *
from torch.utils.data import DataLoader

################################################################################
################################################################################
################################################################################

class SequenceDataset(Dataset):
    def __init__(self, feats, targets, columns, target_cols, drop = False):
        self.ss       = feats.shape[1]
        self.feats    = pd.DataFrame(feats.reshape(-1, len(columns)), columns=columns)
        self.targets  = pd.DataFrame(targets, columns=columns)[target_cols]

        if drop:
          self.feats.drop(target_cols, inplace= True, axis=1)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        x = self.feats.iloc[i*self.ss:i*self.ss+self.ss].to_numpy()
        return x, self.targets.iloc[i].to_numpy()



################################################################################
################################################################################
################################################################################
class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers, device):
        super(LSTM, self).__init__()
        
        self.num_classes    = num_classes
        self.num_layers     = num_layers
        self.input_size     = input_size
        self.hidden_size    = hidden_size
        self.device         = device
        
        self.lstm           = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                        num_layers=num_layers, batch_first=True)
        
        self.fc             = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = torch.zeros( self.num_layers, x.size(0), self.hidden_size).to(self.device)        
        c_0 = torch.zeros( self.num_layers, x.size(0), self.hidden_size).to(self.device)        
        
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        
        h_out = h_out.view(-1, self.hidden_size)
        
        out = self.fc(h_out)
        
        return out

################################################################################
################################################################################
def train_model(data_loader, model, loss_function, 
                optimizer, writer, device):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()

    for X, y in data_loader:

        X = X.to(device)
        y = y.to(device)

        output = model(X)
        loss = loss_function(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        

    avg_loss = total_loss / num_batches
    #print(f"Train loss: {avg_loss}")
    writer.append(avg_loss)

def test_model(data_loader, model, loss_function, writer, device):

    print('testing ...')
    num_batches = len(data_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in data_loader:

            X = X.to(device)
            y = y.to(device)

            output = model(X)
            total_loss += loss_function(output, y).item()

    avg_loss = total_loss / num_batches
    
    if writer != None:
      writer.append(avg_loss)

################################################################################
################################################################################
################################################################################
def main():

    refined_file      = 'refined.json'
    blocks_info_file  = 'blocks_info.json'
    info              = u_loadJson(blocks_info_file)
    refined           = np.array(u_loadJson(refined_file))
    
    nsegments   = refined.shape[1]
    mean_bound  = 15                # este valor tiene que ser obvio menor a n_rows,
                                    # ahora estamos usando 45

    X = []
    y = []

    for lap in refined:
        for it in range(nsegments-1):
            X.append(lap[it])
            y.append(np.mean(lap[it+1][:mean_bound], axis=0))


    batch_size        = 1
    learning_rate     = 5e-5
    num_hidden_units  = 16
    drop              = False
    columns           = info['columns']
    target_cols       = ['watts_calc', 'cadence']

    X = np.array(X, dtype="float32")
    y = np.array(y, dtype="float32")


    scaler  = StandardScaler()
    raw_X   = X.reshape(-1, len(columns))
    scaler.fit(raw_X)
    n_X     = scaler.transform(raw_X)
    n_X     = n_X.reshape(-1, X.shape[1], len(columns))
    scaler  = StandardScaler()
    scaler.fit(y)
    n_y     = scaler.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
    n_X, n_y, test_size=0.33, random_state=42)

    torch.manual_seed(101)

    in_feats      = len(columns) - len(target_cols) if drop else len(columns)

    train_dataset = SequenceDataset( X_train, y_train, columns, target_cols)
    test_dataset  = SequenceDataset( X_test, y_test, columns, target_cols)

    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader   = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    learning_rate = 5e-5
    num_hidden_units = 16

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #self, num_classes, input_size, hidden_size, num_layers

    model         = LSTM(num_classes= len(target_cols), input_size = in_feats, 
                        hidden_size=num_hidden_units, num_layers=10, device=device).to(device)
    loss_function = nn.MSELoss()
    optimizer     = torch.optim.Adam(model.parameters(), lr=learning_rate)


    epochs = 15 
    train_loss  = []
    test_loss   = []

    for ix_epoch in range(epochs):
        print(f"Epoch {ix_epoch}\n---------")
        train_model(train_loader, model, loss_function, optimizer, train_loss, device)
        test_model(test_loader, model, loss_function, test_loss, device)


    model_dir   = './saves'
    os.makedirs(model_dir)
    model_file  = model_dir + '/model_tmp.pt'
    torch.save(model.state_dict(), model_file)

    model_ltrain    = model_dir + '/ltrain.json'
    model_ltest     = model_dir + '/ltest.json'

    u_saveDict2File(model_ltrain, train_loss)    
    u_saveDict2File(model_ltest, test_loss)    

################################################################################
################################################################################
################################################################################
if __name__ == "__main__":
    main()
