"""Show samples from the training data."""
import pandas as pd

df = pd.read_csv("data/raw/train.csv")
print("Columns:", list(df.columns))
print(f"Shape: {df.shape}")
print()

# Show first 5 samples
for i in range(min(8, len(df))):
    row = df.iloc[i]
    print(f"=== Row {i} (id={row['id']}) ===")
    prompt = row["prompt"]
    print(f"Prompt ({len(prompt)} chars):")
    print(prompt[:1000])
    if len(prompt) > 1000:
        print("... [truncated]")
    print(f"\nAnswer: {row['answer']}")
    print("-" * 80)
    print()

# Check for category info in prompts
print("\n=== ANSWER PATTERNS ===")
import re
# Look for patterns
roman = df[df['answer'].str.match(r'^[IVXLCDM]+$', na=False)]
binary = df[df['answer'].str.match(r'^[01]{8}$', na=False)]
decimal = df[df['answer'].str.match(r'^\d+\.\d+$', na=False)]
print(f"Roman numeral answers: {len(roman)}")
print(f"8-bit binary answers: {len(binary)}")
print(f"Decimal answers: {len(decimal)}")
print(f"Other answers: {len(df) - len(roman) - len(binary) - len(decimal)}")

# Show some 'other' answers
other = df[~df['answer'].str.match(r'^[IVXLCDM]+$|^[01]{8}$|^\d+\.\d+$', na=False)]
print(f"\nSample 'other' answers:")
for ans in other['answer'].head(10):
    print(f"  '{ans}'")
