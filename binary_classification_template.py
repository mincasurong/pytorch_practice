#%%

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from torch.utils.data import DataLoader, TensorDataset


class TabularNNModel:
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], lr=0.0001, batch_size=16, epochs=20):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model(input_dim, hidden_dims).to(self.device)
        self.criterion = nn.BCELoss() 
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
        
    def preprocess_data(self, df, target_column, drop_columns=None):
        if drop_columns is not None:
            df = df.drop(columns=drop_columns)
            
        # Fill missing values
        for column in df.columns:
            if df[column].dtype == 'object':
                df[column].fillna(df[column].mode()[0], inplace=True)
                df[column] = LabelEncoder().fit_transform(df[column])
            else:
                df[column].fillna(df[column].median(), inplace=True)
        
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
        
    def _build_model(self, input_dim, hidden_dims):
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
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
            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(self.train_loader)}")
            
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
        
    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            inputs, labels = self.test_data
            outputs = self.model(inputs)
            predicted = (outputs > 0.5).float()
            
            # Convert to CPU tensors for sklearn
            labels = labels.cpu().numpy()
            predicted = predicted.cpu().numpy()
            
            # Report
            accuracy = accuracy_score(labels, predicted)
            print(f'Accuracy: {accuracy * 100:.2f}%')
            
            cm = confusion_matrix(labels, predicted)
            print(f'Confusion Matrix:\n{cm}')
                
            report = classification_report(labels, predicted, target_names=['Not Survived', 'Survived'])
            print(f'Classification Report:\n{report}')
            
        return accuracy, cm, report
        
# Usage example with the Titanic dataset

if __name__ == "__main__":
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    target_column = 'Survived'
    drop_columns = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    
    model = TabularNNModel(input_dim=len(df.columns) - len(drop_columns) - 1)
    model.preprocess_data(df, target_column, drop_columns)
    model.train()
    model.evaluate()
    
# %%