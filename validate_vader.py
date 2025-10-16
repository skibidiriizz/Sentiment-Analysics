import pandas as pd
import sys

PATH = r"c:\Users\asiqi\OneDrive\Desktop\Collage\SEM_6\AI_Analysics\archive (3)\Reviews.csv"
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except Exception as e:
    print('vaderSentiment not installed or failed to import:', e)
    print('Run: pip install vaderSentiment')
    sys.exit(1)

print('Reading 50 rows from', PATH)
df = pd.read_csv(PATH, nrows=50, encoding='utf-8', low_memory=False)
print('Columns:', df.columns.tolist())
print('Rows read:', len(df))

analyzer = SentimentIntensityAnalyzer()
print('\nSample VADER scores for first 5 reviews (Text column):\n')
for i, text in enumerate(df['Text'].fillna('').astype(str).head(5)):
    print(f'--- Review {i+1} ---')
    print(text[:400])
    print('Scores:', analyzer.polarity_scores(text))
    print()