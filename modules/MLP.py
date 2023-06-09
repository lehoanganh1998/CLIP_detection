import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler 
class MLPClassifier(nn.Module):
    def __init__(self, input_size, output_size, learning_rate,num_epochs):
        super(MLPClassifier, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        
        self.fc1 = nn.Linear(input_size, 50)
        self.relu = nn.ReLU()
        
        self.fc2 = nn.Linear(50, output_size)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)
        # self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,T_max=self.num_epochs-1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

    def train_model(self, features, labels):
        self.train()
        for epoch in range(self.num_epochs):
            # Forward pass
            outputs = self(features).squeeze()

            loss = self.criterion(outputs, labels)

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            # Print the loss at every 10th epoch
            if epoch == 10 or epoch == self.num_epochs - 1:
                print('Epoch [{}/{}], Loss: {}'.format(epoch+1, self.num_epochs, loss.item()))
                print("Learning rate: ", self.scheduler.get_lr())
                
    def evaluate_model(self, features, labels):
        self.eval()
        with torch.no_grad():
            outputs = self(features).squeeze()
            predicted = torch.round(outputs)
            accuracy = (predicted == labels).float().mean()
            
            # count the number of correct predictions for each class
            correct_fake = ((predicted == labels) & (predicted == 1)).sum().item()
            correct_real = ((predicted == labels) & (predicted == 0)).sum().item()

            return accuracy,  correct_fake, correct_real  