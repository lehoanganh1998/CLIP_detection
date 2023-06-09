import clip
import torch
from torch.utils.data import DataLoader
from modules import *
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np
import csv
from sklearn.metrics import confusion_matrix, roc_auc_score
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
configs = Setting("data/configsLP.yaml")
result_acc = []
result_std = []
f = open("Result/result.csv", 'a', newline='')  # Append mode
writer = csv.writer(f)

# Write the headers
writer.writerow(["Model", "K-Shot", "Accuracy (Mean)", "Accuracy (Std)", "Correct Real (Mean)", "Correct Real (Std)", "Correct Fake (Mean)", "Correct Fake (Std)", "AUC (Mean)", "AUC (Std)"])

repeat_time = 10
for model_name in configs.model_name:
    print(model_name + "!!!!")
    model, transform = clip.load(model_name, device=device, jit=False)
    final_acc = []
    final_count = []
    final_auc = []
    for k in configs.k_shots_list:
        print("k = " + str(k) + "!!!!")
        accuracies = []
        count_true = []
        count_false = []
        aucs = []
        for i in tqdm(range(repeat_time)):
            # Import train, test dataset and k-shot dataset
            train_dataset = CustomDataset(dataset=configs.train_location, prompts=configs.prompts, train=True, testsize=0.2, seed=configs.random_seed)
            test_dataset = CustomDataset(dataset=configs.test_location, prompts=configs.prompts, train=False, testsize=0.2, seed=configs.random_seed)
            few_shot_dataset = k_samples(train_dataset, k)

            # Loading dataset into batches
            train_load = DataLoader(DataTransformForLinear(few_shot_dataset, transform), batch_size=k * 2, shuffle=False)
            test_load = DataLoader(DataTransformForLinear(test_dataset, transform), batch_size=32, shuffle=False)
            
            # Initialize Extractor
            train_extractor = Extract(model=model, device=device, dataset=train_load)
            test_extractor = Extract(model=model, device=device, dataset=test_load)
            
            # Extract feature
            train_features, train_labels = train_extractor.extract_features()
            test_features, test_labels = test_extractor.extract_features()
            classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=0)
            classifier.fit(train_features, train_labels)
            probabilities = classifier.predict_proba(test_features)
            predictions = np.argmax(probabilities, axis=1)
            
            accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
            conf_matrix = confusion_matrix(test_labels, predictions)
            correct_fake = conf_matrix[1, 1]
            correct_real = conf_matrix[0, 0]
            accuracies.append(accuracy)
            count_true.append(correct_real)
            count_false.append(correct_fake)
            auc = roc_auc_score(test_labels, probabilities[:, 1])
            aucs.append(auc)
            print(correct_fake, correct_real, auc)
        final_acc.append(accuracies)
        final_count.append([round(np.mean(count_true),2), round(np.std(count_true),2), round(np.mean(count_false),2), round(np.std(count_false),2)])
        final_auc.append(aucs)
    
    # Write the results for each model and k-shot
    for k in range(len(final_acc)):
        writer.writerow([model_name, configs.k_shots_list[k], round(np.mean(final_acc[k]),2), round(np.std(final_acc[k]),2),
                         final_count[k][0], final_count[k][1], final_count[k][2], final_count[k][3],
                         round(np.mean(final_auc[k]),2), round(np.std(final_auc[k]),2)])

f.close()  # Close the file after writing
