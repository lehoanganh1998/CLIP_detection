import os
import clip
import torch
from PIL import Image
from tqdm import tqdm
from modules import *
# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)
configs = Setting("data/configsLP.yaml")
# dataset
dataset = CustomDataset(dataset=configs.train_location,prompts=configs.prompts, train=True, testsize=1, seed=1)
count = 0
count_fake = 0
count_real = 0
real = []
fake = []
for i in tqdm(range(len(dataset))):
    image = Image.open(dataset[i][0])

    # Prepare the inputs

    image_input = preprocess(image).unsqueeze(0).to(device)
    classes = ['real', 'fake']
    text_inputs = torch.cat([clip.tokenize(f"this is a {c} photo") for c in classes]).to(device)

    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

    # Pick the top 5 most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(2)
    # Print the result
    value = values[0] .item() * 100.
    if value >= 50:
        count_real += 1
        pred = 0
    elif value < 50:
        count_fake += 1
        pred = 1
    if pred == dataset[i][1]:
        count += 1
    real.append(values[0].item() * 100.)
    fake.append(values[1].item() * 100.)
acc = count / (len(dataset)) * 100.
print("Accuracy: " + str(acc) + "%")
print(f"{count_real} real images over {len(dataset)} images")
print(f"{count_fake} fake images over {len(dataset)} images")
print(f"real avevage %: {sum(real) / len(real)}")
print(f"fake avevage %: {sum(fake) / len(fake)}")