# train_model.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os

# -----------------------------
# 1. Load datasets
# -----------------------------
fake_file = "Fake.csv"
main_file = "main_data.csv"

# --- Load fake news ---
df_fake = pd.read_csv(fake_file)
df_fake['label'] = 0
df_fake['text_combined'] = df_fake['title'].fillna('') + " " + df_fake['text'].fillna('')

# --- Load main_data (API-fetched news) ---
df_main = pd.read_csv(main_file)
df_main['label'] = 1

# Determine text column for main_data
if 'content' in df_main.columns:
    text_col_main = 'content'
elif 'description' in df_main.columns:
    text_col_main = 'description'
else:
    text_col_main = None

if text_col_main:
    df_main['text_combined'] = df_main['title'].fillna('') + " " + df_main[text_col_main].fillna('')
else:
    df_main['text_combined'] = df_main['title'].fillna('')

# -----------------------------
# 2. Combine datasets
# -----------------------------
df_combined = pd.concat([df_main, df_fake], ignore_index=True)

# -----------------------------
# 3. Remove duplicates within each class
# -----------------------------
df_final = pd.concat([
    df_combined[df_combined['label'] == 0].drop_duplicates(subset=['title']),
    df_combined[df_combined['label'] == 1].drop_duplicates(subset=['title'])
], ignore_index=True)

print(f"Total samples after removing duplicates: {len(df_final)}")

# -----------------------------
# 4. Check class distribution
# -----------------------------
print("Class distribution:\n", df_final['label'].value_counts())
if df_final['label'].nunique() < 2:
    raise ValueError("Dataset must contain at least two classes (0 and 1) for training!")

# -----------------------------
# 5. Prepare features and labels
# -----------------------------
X = df_final['text_combined']
y = df_final['label']

# -----------------------------
# 6. TF-IDF vectorization
# -----------------------------
tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=20000)
X_tfidf = tfidf.fit_transform(X)

# -----------------------------
# 7. Train model
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=500, class_weight='balanced', solver='saga')
model.fit(X_train, y_train)

# Evaluate
score = model.score(X_test, y_test)
print(f"Validation Accuracy: {score*100:.2f}%")

# -----------------------------
# 8. Save model and vectorizer
# -----------------------------
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
print("Model and vectorizer saved successfully!")
