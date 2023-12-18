import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
from dataloader import Train_Dataset
from model import vgg16_model, vgg16_model_attention, resnet50_model_attention
from tqdm import tqdm
import wandb

# Initialize wandb
wandb.init(project="your_project_name", name="your_run_name")

def train(json_path, left_folder, right_folder, model_save_path,
          num_epochs=10, batch_size=6, learning_rate=5e-4, weight_decay=1e-5):
    
    # GPU setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print('GPU is available! ')

    # Dataset & DataLoader
    transform = transforms.Compose([
        transforms.ToTensor()
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = Train_Dataset(json_path=json_path, left_folder=left_folder,
                                  right_folder=right_folder, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model
    # model = vgg16_model().to(device)
    # model = vgg16_model_attention().to(device)
    model = resnet50_model_attention().to(device)

    # Cosine Similarity module
    cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    # The loss function can be mean squared error to compare similarity values
    # criterion = torch.nn.MSELoss()
    criterion = torch.nn.TripletMarginLoss(margin=1.0, p=2)
    # criterion = torch.nn.MarginRankingLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        print('epoch: ', epoch, '\n')
        running_loss = 0.0
        for i, (left_img, right_imgs) in enumerate(tqdm(train_loader)):
            left_img, right_imgs = left_img.to(device), right_imgs.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            left_features, right_features = model(left_img, right_imgs)

               # Compute the cosine similarity
            # This will give a matrix of shape [batch_size, 20]
            '''similarities = cosine_similarity(left_features.unsqueeze(1), right_features)
            # We aim for a similarity of 1 for the correct match (index 0) and 0 for others
            target_similarities = torch.zeros_like(similarities).to(device)
            target_similarities[:, 0] = 1.00

            # Compute loss
            loss = criterion(similarities, target_similarities)'''
            triplet_losses = [criterion(left_features, right_features[:, 0, :], right_features[:, i, :]) for i in range(1, 20)]
            loss = sum(triplet_losses) / len(triplet_losses)
            
            # Log loss to wandb
            wandb.log({"loss": loss.item()})

            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Print loss every 10 batches
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                
        # Update the learning rate based on the loss
        scheduler.step(running_loss / len(train_loader))
        
        # Log loss to wandb
        # wandb.log({"loss": running_loss / len(train_loader)})

    print('Training finished.')

    # Save the trained model
    torch.save(model.state_dict(), 'resnet_trained_01.pth')
    print(f"Model saved to resnet_trained.pth")
    
    wandb.finish()


# Sample usage:
if __name__ == "__main__":
    train(json_path='data.json', left_folder='train/left',
          right_folder='train/right', model_save_path='vgg16.pth')
    # nohup python -u train.py >train.out 2>&1 &
