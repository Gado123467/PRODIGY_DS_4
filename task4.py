import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# -----------------------------
# Setup
# -----------------------------
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# -----------------------------
# Load Twitter Sentiment Dataset
# (NO HEADER IN FILE)
# -----------------------------
train_df = pd.read_csv(
    "twitter_training.csv",
    header=None,
    encoding="utf-8"
)

val_df = pd.read_csv(
    "twitter_validation.csv",
    header=None,
    encoding="utf-8"
)

# Assign correct column names
columns = ['id', 'topic', 'label', 'text']
train_df.columns = columns
val_df.columns   = columns

# -----------------------------
# Keep only text column
# -----------------------------
train_df = train_df[['text']].dropna()
val_df   = val_df[['text']].dropna()

# -----------------------------
# Sentiment Analysis (VADER)
# -----------------------------
def analyze_sentiment(df):
    df['compound'] = df['text'].apply(
        lambda x: sia.polarity_scores(str(x))['compound']
    )
    df['sentiment'] = df['compound'].apply(
        lambda s: 'Positive' if s >= 0.05 else
                  'Negative' if s <= -0.05 else
                  'Neutral'
    )
    return df

train_df = analyze_sentiment(train_df)
val_df   = analyze_sentiment(val_df)

# -----------------------------
# Visualization
# -----------------------------
plt.figure()
train_df['sentiment'].value_counts().plot(kind='bar')
plt.title("Training Sentiment Distribution (Twitter)")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

plt.figure()
val_df['sentiment'].value_counts().plot(kind='bar')
plt.title("Validation Sentiment Distribution (Twitter)")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# -----------------------------
# Summary
# -----------------------------
print("\nTraining Sentiment (%)")
print(train_df['sentiment'].value_counts(normalize=True) * 100)

print("\nValidation Sentiment (%)")
print(val_df['sentiment'].value_counts(normalize=True) * 100)
