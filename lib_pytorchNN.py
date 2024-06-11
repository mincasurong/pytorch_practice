import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor


class TabularNNModel:
    def __init__(self, 
                 hidden_dims=[128, 64, 32], 
                 lr=0.0001, 
                 batch_size=32, 
                 epochs=5, 
                 dropout_rate=0.0
                 ):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_losses = []
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.dropout_rate = dropout_rate
        
    def feature_selection(self, X, y):
        df = X.copy()
        df['target'] = y
        
        correlation_matrix = df.corr()
        
        # Plotting the correlation matrix
        plt.figure(figsize=(12,8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix')
        plt.show()
        
        # Display correlations with the target
        print("Correlation with target:")
        target_correlation = correlation_matrix['target'].sort_values(ascending=False)
        print(target_correlation)
        
        # Drop features with low correlation with the target (threshold can be adjusted)
        threshold = 0.1
        selected_features = target_correlation[abs(target_correlation) > threshold].index.tolist()
        selected_features.remove('target')
        
        # Calculate VIF to check multicollinearity
        X_selected = df[selected_features]
        vif_data = pd.DataFrame()
        vif_data["feature"] = X_selected.columns
        vif_data["VIF"] = [variance_inflation_factor(X_selected.values, i) for i in range(X_selected.shape[1])]
        
        # Drop features with high VIF (threshold can be adjusted)
        high_vif_threshold = 5
        features_to_drop = vif_data[vif_data["VIF"] > high_vif_threshold]["feature"]
        selected_features = [f for f in selected_features if f not in features_to_drop]
        
        # Print VIF data
        print("\nVariance Inflation Factor (VIF) for selected features:")
        print(vif_data)
        
        # Print selected features after VIF check
        print("\nSelected features after VIF check:")
        print(selected_features)
        
        return X[selected_features], selected_features
        
    def _build_model(self, input_dim, hidden_dims, dropout_rate):
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        # layers.append(nn.Sigmoid())  # No activation function here for regression
        return nn.Sequential(*layers)
    
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
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Feature selection and correlation analysis
        X_train, selected_features = self.feature_selection(X_train, y_train)
        X_test = X_test[selected_features]
        
        # Update the model's input dimension based on selected features
        self.model = self._build_model(len(selected_features), hidden_dims=self.hidden_dims, dropout_rate=self.dropout_rate).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Convert the data into tensor
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        self.train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                                     torch.tensor(y_train.values, dtype=torch.float32).view(-1,1)),
                                       batch_size=self.batch_size, shuffle=True)
        
        self.test_data = (torch.tensor(X_test, dtype=torch.float32).to(self.device),
                          torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1).to(self.device))
        
        return X_train, X_test, y_train, y_test, selected_features
        
    def train(self):
        best_loss = float('inf')
        patience, trials = 10, 0
        
        for epoch in range(self.epochs):
            running_loss = 0.0
            self.model.train()
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
            print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
            
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
            print(f'Mean Squared Error: {mse:.2f}')
            
            mae = mean_absolute_error(y_test, predicted)
            print(f'Mean Absolute Error: {mae:.2f}')
                
            r2 = r2_score(y_test, predicted)
            print(f'R^2 Score: {r2:.2f}')
            
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
            plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r')
            
            plt.tight_layout()
            plt.show()
            
        return mse, mae, r2

''' How to use '''

'''
# Include the lib_pytorchNN
from lib_pytorch import TabularNNModel

# void main()
if __name__ == "__main__":
    
    # === Include dataset here ===
    # Define target_column:  target_column = 'name'
    # Define drop_columns:   drop_columns = []
        
    # Train and test the model.  Modify the hyperparameters.
    model = TabularNNModel(
                 hidden_dims=[128, 64, 32], 
                 lr=0.0001, 
                 batch_size=16, 
                 epochs=50, 
                 dropout_rate=0.0
                 )
    X_train, X_test, y_train, y_test, selected_features = model.preprocess_data(df, target_column, drop_columns)
    trials, epoch, avg_loss = model.train()
    mse, mae, r2 = model.evaluate()

'''
