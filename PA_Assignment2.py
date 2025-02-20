import json
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, roc_curve, auc
import matplotlib.pyplot as plt

# Load dataset (replace 'dataset.json' with actual file path)
def load_json(file_path, sample_size=12000):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return pd.DataFrame(data).sample(n=sample_size, random_state=42)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove special characters
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

# Feature Engineering
def extract_features(df):
    df['review_length'] = df['reviewText'].apply(lambda x: len(str(x).split()))
    df['helpful_ratio'] = df['helpful'].apply(lambda x: x[0] / x[1] if x[1] > 0 else 0)
    return df.dropna()

# Load Data
nltk.download('stopwords')
df = load_json('reviews_Video_Games.json', sample_size=12000)

# Exclude reviews with fewer than two helpfulness ratings
df['helpfulness_count'] = df['helpful'].apply(lambda x: x[1])
df = df[df['helpfulness_count'] >= 2]
df = df.drop(columns=['helpfulness_count'])

# Proceed with preprocessing
df = extract_features(df)
df['cleaned_text'] = df['reviewText'].astype(str).apply(preprocess_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=1200)
X_text = vectorizer.fit_transform(df['cleaned_text'])
X_meta = df[['review_length', 'overall']]
X = np.hstack((X_text.toarray(), X_meta))
y = df['helpful_ratio'] > 0.5  # Convert helpful ratio into binary label

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Model
model = RandomForestClassifier(n_estimators=128, random_state=42)
model.fit(X_train, y_train)

# Predictions & Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
mse = mean_squared_error(y_test.astype(float), y_pred.astype(float))
print(f'Accuracy: {accuracy:.4f}')
print(f'Mean Squared Error: {mse:.4f}')

# ROC Curve and AUC
y_pred_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
print(f'Area Under the ROC Curve (AUC): {roc_auc:.4f}')

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Variable Importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
features = vectorizer.get_feature_names_out()[:5000].tolist() + ['review_length', 'overall']

print("Top 20 most influential variables:")
for i in range(20):
    print(f"{features[indices[i]]}: {importances[indices[i]]}")
