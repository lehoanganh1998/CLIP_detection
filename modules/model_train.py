import torch
import clip
import os

class ModelTrainer:
    def __init__(self, model, device,output_location, k_shots_list):
        self.model = model
        self.loss_img = torch.nn.CrossEntropyLoss()
        self.loss_txt = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-7, betas=(0.9,0.98), eps=1e-6, weight_decay=0.002)
        self.device = device
        self.output_location = output_location
        self.k_shots_list = k_shots_list
        self.saver = ModelSaver(output_location=output_location, k_shots_list=k_shots_list)
        
    def convert_models_to_fp32(self):
        for p in self.model.parameters():
            p.data = p.data.float()
            p.grad.data = p.grad.data.float()
    
    def evaluate(self, val_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                images, labels, prompts = batch
                images = images.to(self.device)
                prompts = prompts.to(self.device)
                labels = labels.to(self.device)
                logits_per_image,_ = self.model(images, prompts)
                preds = torch.argmax(logits_per_image, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        accuracy = correct / total * 100
        return accuracy

    def train(self, train_loader, val_loader, num_epochs):
        print(f"Training started for {self.k_shots_list}-shot learning")
        accuracy_list = []
        for epoch in range(num_epochs):
            for batch in train_loader:
                self.optimizer.zero_grad()
                images, labels, prompts = batch 
                images = images.to(self.device)
                prompts = prompts.to(self.device)
                labels = labels.to(self.device)
                logits_per_image, _ = self.model(images, prompts)
                loss = self.loss_img(logits_per_image, labels)
                
                loss.backward() 
                if self.device == "cpu":
                    self.optimizer.step()
                else:
                    self.convert_models_to_fp32()
                    self.optimizer.step()
                    clip.model.convert_weights(self.model)
                if epoch % 100 == 0:
                    val_accuracy = self.evaluate(val_loader=val_loader)
                    accuracy_list.append(val_accuracy)
                    print("Epoch: ", epoch, "\tLoss: ", loss.item(), "\tAccuracy: ", val_accuracy)    
        print(f"Accuracy for {self.k_shots_list}-shot learning: {max(accuracy_list)}")
        print("Training finished")
        print("Saving model")
        self.saver.save(epoch, self.model, self.optimizer, loss.item())


class ModelSaver:
    def __init__(self, output_location, k_shots_list):
        self.output_location = output_location
        self.k_shots_list = k_shots_list

    def save(self, epoch, model, optimizer, loss):
        output = f"{self.k_shots_list}_shot.pth"
        os.makedirs(self.output_location, exist_ok=True)
        output_weight_loc = os.path.join(self.output_location, output)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, output_weight_loc)