import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class vgg16_model(nn.Module):
    def __init__(self, feature_dim=512, load_pretrained=True):
        super(vgg16_model, self).__init__()

        # Pre-trained VGG16 model without the classification head
        if load_pretrained:
            vgg16 = models.vgg16(pretrained=True)
        else:
            vgg16 = models.vgg16(pretrained=False)

        self.features = nn.Sequential(*list(vgg16.children())[:-1])

        # Flatten the output
        self.flatten = nn.Flatten()

        # Linear layer to reduce dimensionality for computational efficiency
        self.fc = nn.Linear(25088, feature_dim)  # VGG16 features are 25088 in size

    def forward_one(self, x):
        x = self.features(x)  # Shape: [batch_size, 512, 7, 7]
        x = self.flatten(x)  # Shape: [batch_size, 25088]
        x = self.fc(x)  # Shape: [batch_size, feature_dim]
        return x

    def forward(self, left_img, right_imgs):
        # Shape of right_imgs: [batch_size, 20, C, H, W]
        left_features = self.forward_one(left_img)  # Shape: [batch_size, feature_dim]
        # Reshape to [batch_size*20, C, H, W]
        right_features = self.forward_one(right_imgs.view(-1, *right_imgs.shape[2:]))
        # Reshape to [batch_size, 20, feature_dim]
        return left_features, right_features.view(-1, 20, right_features.size(-1))

    def load_checkpoint(self, checkpoint_path):
        # Detect if we have a GPU available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load the weights into the device
        self.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.to(device)  # Move the model to the device (GPU if available)


class Attention(nn.Module):
    def __init__(self, in_channels):
        super(Attention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0, bias=False)
    
    def forward(self, x):
        # Generate attention map
        attention_map = torch.sigmoid(self.conv(x))
        # Apply attention map on input
        out = x * attention_map
        return out

class vgg16_model_attention(nn.Module):
    def __init__(self, feature_dim=512, load_pretrained=True):
        super(vgg16_model_attention, self).__init__()

        # Pre-trained VGG16 model without the classification head
        if load_pretrained:
            vgg16 = models.vgg16(pretrained=True)
        else:
            vgg16 = models.vgg16(pretrained=False)

        self.features = nn.Sequential(*list(vgg16.children())[:-1])

        self.flatten = nn.Flatten()

        self.fc = nn.Linear(4608, feature_dim)  # Adjusted input dimension

        self.attention = Attention(512)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward_one(self, x):
        x = self.features(x)  # Shape: [batch_size, 512, 7, 7]
        x = self.attention(x)  # Shape: [batch_size, 512, 7, 7]
        x = self.maxpool(x)  # Shape: [batch_size, 512, 3, 3]
        x = self.flatten(x)  # Shape: [batch_size, 4608]
        x = self.fc(x)  # Shape: [batch_size, feature_dim]
        return x

    def forward(self, left_img, right_imgs):
        # Shape of right_imgs: [batch_size, 20, C, H, W]
        left_features = self.forward_one(left_img)  # Shape: [batch_size, feature_dim]
        # Reshape to [batch_size*20, C, H, W]
        right_features = self.forward_one(right_imgs.view(-1, *right_imgs.shape[2:]))
        # Reshape to [batch_size, 20, feature_dim]
        return left_features, right_features.view(-1, 20, right_features.size(-1))

    def load_checkpoint(self, checkpoint_path):
        # Detect if we have a GPU available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load the weights into the device
        self.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.to(device)  # Move the model to the device (GPU if available)


class resnet50_Attention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(Attention, self).__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1, padding=0)
        self.fc2 = nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1, padding=0)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Channel Attention
        avg_pool_result = self.avg_pool(x).view(batch_size, channels)
        max_pool_result = self.max_pool(x).view(batch_size, channels)
        
        avg_pool_result_fc = self.fc2(F.relu(self.fc1(avg_pool_result.unsqueeze(2).unsqueeze(3)), inplace=True)).squeeze(3).squeeze(2)
        max_pool_result_fc = self.fc2(F.relu(self.fc1(max_pool_result.unsqueeze(2).unsqueeze(3)), inplace=True)).squeeze(3).squeeze(2)

        channel_attention_weights = F.sigmoid(avg_pool_result_fc + max_pool_result_fc)
        channel_attention_weights = channel_attention_weights.unsqueeze(2).unsqueeze(3)

        # Applying the attention weights
        weighted_channel_features = x * channel_attention_weights.expand_as(x)

        return weighted_channel_features
        
class resnet50_model_attention(nn.Module):
    def __init__(self, feature_dim=512, load_pretrained=True):
        super(resnet50_model_attention, self).__init__()

        # Pre-trained ResNet50 model without the classification head
        if load_pretrained:
            resnet50 = models.resnet50(pretrained=True)
        else:
            resnet50 = models.resnet50(pretrained=False)

        # Remove the classification head (fc layer) from ResNet50
        self.features = nn.Sequential(*list(resnet50.children())[:-2])  # Exclude AvgPool and FC

        self.attention = Attention(2048)  # ResNet50 has 2048 channels in the final convolutional layer
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        # Adjust the input dimension of the fully connected layer
        self.fc = nn.Linear(24576, feature_dim)  # Adjusted input dimension

    def forward_one(self, x):
        x = self.features(x)  # Shape: [batch_size, 2048, 7, 7] for an input size of [batch_size, 3, 224, 224]
        x = self.attention(x)  # Shape: [batch_size, 2048, 7, 7]
        x = self.maxpool(x)  # Shape: [batch_size, 2048, 3, 3]
        x = self.flatten(x)  # Shape: [batch_size, 18432] (2048*3*3)
        x = self.fc(x)  # Shape: [batch_size, feature_dim]
        return x

    def forward(self, left_img, right_imgs):
        # Shape of right_imgs: [batch_size, 20, C, H, W]
        left_features = self.forward_one(left_img)  # Shape: [batch_size, feature_dim]
        # Reshape to [batch_size*20, C, H, W]
        right_features = self.forward_one(right_imgs.view(-1, *right_imgs.shape[2:]))
        # Reshape to [batch_size, 20, feature_dim]
        return left_features, right_features.view(-1, 20, right_features.size(-1))

    def load_checkpoint(self, checkpoint_path):
        # Detect if we have a GPU available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load the weights into the device
        self.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.to(device)  # Move the model to the device (GPU if available)

