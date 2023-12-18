import csv
import torch
from dataloader import Test_Dataset
from model import vgg16_model, vgg16_model_attention, resnet50_model_attention
from torchvision import transforms
import torch.nn.functional as F


def evaluate(json_path, left_folder, right_folder, model_path, csv_path='similarities.csv'):
    if torch.cuda.is_available():
        print('GPU is available! ')
        # Initialization
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model = vgg16_model(load_pretrained=False).to(device)
        # model = vgg16_model_attention(load_pretrained=False).to(device)
        model = resnet50_model_attention(load_pretrained=False).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        # Dataset and DataLoader setup
        test_dataset = Test_Dataset(json_path=json_path, left_folder=left_folder, right_folder=right_folder, transform=transform)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1,
                                                  shuffle=False)  # Evaluating one pair at a time

        cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

        # Open CSV file for writing
        with open(csv_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)

            # Write the header
            csvwriter.writerow(["left"] + [f"c{i}" for i in range(20)])

            # Evaluation loop
            with torch.no_grad():
                
                # Use loop index to generate names for left images
                for left_name, left_img, right_imgs in test_loader:
                    left_img, right_imgs = left_img.to(device), right_imgs.to(device)

                    left_features, right_features = model(left_img, right_imgs)
                    # print('right_features ',right_features )
                    # Compute the cosine similarity
                    # similarities = cosine_similarity(left_features, right_features.squeeze(1))
                    # similarities = cosine_similarity(left_features.unsqueeze(0), right_features)
                    # print('s', similarities)
                    # print('sss', left_features.shape, right_features.shape)
                    # Compute the cosine similarity for each pair (one left image to 20 right images)
                    similarities = F.cosine_similarity(left_features.unsqueeze(1), right_features, dim=2)
                    similarities = 0.5 * (similarities + 1)
                    similarity_list = similarities.squeeze(0).tolist()
                    # print('s', similarities.shape)
                    # for s in similarities:
                        # print('Length of similarities for one item:', len(s))

                    # Write to CSV

                    row_data = [str(left_name[0])] + [round(sim, 10) for sim in similarity_list]
                    csvwriter.writerow(row_data)

        print("Similarities saved to", csv_path)

if __name__ == "__main__":
    evaluate(json_path='test.json', left_folder='test/left',
             right_folder='test/right', model_path='resnet_trained_01.pth')
# nohup python -u evaluation.py >test.out 2>&1 &
