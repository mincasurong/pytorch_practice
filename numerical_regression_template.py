#%%

import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.datasets import fetch_california_housing


class TabularNNModel:
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], lr=0.0001, batch_size=32, epochs=5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model(input_dim, hidden_dims).to(self.device)
        self.criterion = nn.MSELoss() 
        '''
            BCELoss()
            BCEWithLogitsLoss() -- Combines Sigmoid + BCELoss
            MSELoss() -- Mean Squared Error -> binary classification
            CrossEntropyLoss() -- Softmax -> Multiclass classification
        ''' 
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        '''
            Adam()
            RMSprop -- adapt the learning rate for each parameter
            Adagrad -- Adative learning rate -> Good for sparse data
            Adadelta -- extension of Adagrad.  Reduce its aggressive, decreasing learning rate
        '''
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_losses = []
        
    def preprocess_data(self, df, target_column, drop_columns=None):
        if drop_columns is not None:
            df = df.drop(columns=drop_columns)
            
        # Fill missing values
        for column in df.columns:
            if df[column].dtype == 'object':
                df[column].fillna(df[column].mode()[0])
                df[column] = LabelEncoder().fit_transform(df[column])
            else:
                df[column].fillna(df[column].median())
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        self.train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                                     torch.tensor(y_train.values, dtype=torch.float32).view(-1,1)),
                                       batch_size=self.batch_size, shuffle=True)
        
        self.test_data = (torch.tensor(X_test, dtype=torch.float32).to(self.device),
                          torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1).to(self.device))
        
        return  X_train, X_test, y_train, y_test
        
        
    def _build_model(self, input_dim, hidden_dims):
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        # layers.append(nn.Sigmoid())  # No activation function here for regression
        return nn.Sequential(*layers)
    
    def train(self):
        best_loss = float('inf')
        patience, trials = 5, 0
        
        for epoch in range(self.epochs):
            running_loss = 0.0
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
            avg_loss = running_loss / len(self.train_loader)
            self.train_losses.append(avg_loss)
            print(f"Epoch {epoch + 1}, Loss: {avg_loss}")
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                trials = 0
            else:
                trials += 1
                if trials >= patience:
                    print('Early stopping!')
                    break
        print('Finished Training')
        return trials, epoch, avg_loss
        
    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            inputs, y_test = self.test_data
            outputs = self.model(inputs)
            
            # Convert to CPU tensors for sklearn
            y_test = y_test.cpu().numpy()
            predicted = outputs.cpu().numpy()
            
            
            # Report
            mse = mean_squared_error(y_test, predicted)
            print(f'mean_squared_error: {mse:.2f}')
            
            mae = mean_absolute_error(y_test, predicted)
            print(f'mean_absolute_error:\n{mae:.2f}')
                
            r2 = r2_score(y_test, predicted)
            print(f'r2_score:\n{r2:.2f}')
            
            
            # Plotting
            plt.figure(figsize=(12, 5))
            
            # Loss curve
            plt.subplot(1, 2, 1)
            plt.plot(self.train_losses, label='Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Loss Curve')
            plt.legend()
            
            # Predictions vs Actual
            plt.subplot(1, 2, 2)
            plt.scatter(y_test, predicted, alpha=0.7)
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title('Predicted vs Actual Values')
            plt.plot([min(y_test), max(y_test)], [min(predicted), max(predicted)], 'r')
            plt.legend()
            
            plt.tight_layout()
            plt.show()
            
        return mse, mae, r2
        
# Usage example with the Titanic dataset

if __name__ == "__main__":
    
    # Insert the dataset
    california = fetch_california_housing()
    df = pd.DataFrame(california.data, columns=california.feature_names)
    df['MedHousVal'] = california.target
        
    target_column = 'MedHousVal'
    drop_columns = []
    
    # Train and test the model    
    model = TabularNNModel(input_dim=len(df.columns) - len(drop_columns) - 1)
    X_train, X_test, y_train, y_test = model.preprocess_data(df, target_column, drop_columns)
    trials, epoch, avg_loss = model.train()
    mse, mae, r2 = model.evaluate()
    
# %%
