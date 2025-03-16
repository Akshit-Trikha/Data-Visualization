import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from skimage.feature import local_binary_pattern

# Paths to Dataset (update these paths as per your system)
train_csv_path = r"C:\data visualization\Messidor Dataset\train.csv"
test_csv_path = r"C:\data visualization\Messidor Dataset\test.csv"
train_img_path = r"C:\data visualization\Messidor Dataset\trainimg"
test_img_path = r"C:\data visualization\Messidor Dataset\testimg"

# Load CSV Files
train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

# Image Preprocessing
def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    
    # Resize to 64x64
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    
    # Extract Green Channel
    green = img[:, :, 1]
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(green)
    
    # Local Binary Patterns
    lbp = local_binary_pattern(clahe_img, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=np.arange(0, 10), range=(0, 9))
    
    # Combine features
    return np.concatenate([clahe_img.flatten(), lbp_hist])

# Extract Features and Labels
def extract_features_and_labels(df, img_path):
    features, labels = [], []
    for _, row in df.iterrows():
        img_filename = row['Image']
        img_full_path = os.path.join(img_path, img_filename)
        img_features = preprocess_image(img_full_path)
        
        if img_features is not None:
            features.append(img_features)
            labels.append(0 if row['Id'] == 0 else 1)  # Binary Classification
    
    return np.array(features), np.array(labels)

# Load and Convert Data to Binary Classification
X_train, y_train = extract_features_and_labels(train_df, train_img_path)
X_test, y_test = extract_features_and_labels(test_df, test_img_path)

print(f"Original Class Distribution: {Counter(y_train)}")

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Feature Selection
selector = SelectKBest(mutual_info_classif, k=70)  # Select top 70 features
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Apply SMOTE for Balancing
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_selected, y_train)

print(f"Balanced Class Distribution: {Counter(y_train_bal)}")

# Train Naïve Bayes Model
model = GaussianNB(var_smoothing=1e-9)
model.fit(X_train_bal, y_train_bal)

# Predictions
y_pred = model.predict(X_test_selected)
y_prob = model.predict_proba(X_test_selected)

# Compute Metrics
accuracy = accuracy_score(y_test, y_pred) * 100
conf_matrix = confusion_matrix(y_test, y_pred)

# Sensitivity & Specificity Calculation
def compute_metrics(cm):
    sensitivity = np.round(cm.diagonal() / cm.sum(axis=1), 2)
    specificity = []
    for i in range(len(cm)):
        TN = cm.sum() - (cm[i,:].sum() + cm[:,i].sum() - cm[i,i])
        FP = cm[:,i].sum() - cm[i,i]
        specificity.append(np.round(TN/(TN+FP), 2) if (TN+FP)!=0 else 0)
    return sensitivity, np.array(specificity)

sens, spec = compute_metrics(conf_matrix)

# Print Results
print(f"\nFinal Accuracy: {accuracy:.2f}%")
print("Confusion Matrix:\n", conf_matrix)
print(f"Sensitivity: {sens}")
print(f"Specificity: {spec}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Visualizations

# Training Accuracy Line Chart
train_accuracies = [accuracy_score(y_train_bal, model.predict(X_train_bal)) * 100]
plt.figure(figsize=(10,6))
plt.plot(range(1,6), train_accuracies * 5, marker='o', color='darkcyan')
plt.title("Training Accuracy Progression")
plt.xlabel("Iterations")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.show()

# Confusion Matrix Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Class Distribution Bar Chart
plt.figure(figsize=(6,4))
sns.countplot(x=y_test, palette="coolwarm")
plt.title("Class Distribution in Test Data")
plt.xlabel("Class Labels")
plt.ylabel("Count")
plt.show()

# Histogram of Prediction Probabilities
plt.figure(figsize=(8,6))
plt.hist(y_prob.flatten(), bins=20, alpha=0.7, color='g', edgecolor='black')
plt.xlabel("Prediction Probability")
plt.ylabel("Frequency")
plt.title("Histogram of Prediction Probabilities")
plt.show()

from sklearn.manifold import TSNE

# Reduce features to 2D using t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_test_tsne = tsne.fit_transform(X_test_selected)

# Scatter Plot
plt.figure(figsize=(8,6))
plt.scatter(X_test_tsne[:,0], X_test_tsne[:,1], c=y_test, cmap="coolwarm", alpha=0.7)
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.title("Feature Scatter Plot (Light Blue to Red)")
plt.colorbar(label="Class Labels")
plt.show()
