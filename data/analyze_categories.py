"""Analyze each puzzle type in detail to understand the exact rules for synthetic data generation."""
import pandas as pd
import re
import numpy as np

df = pd.read_csv("data/raw/train.csv")

# ===================== NUMERAL SYSTEM (Roman) =====================
print("=" * 60)
print("NUMERAL SYSTEM ANALYSIS")
print("=" * 60)
roman_rows = df[df['answer'].str.match(r'^[IVXLCDM]+$', na=False)]
print(f"Count: {len(roman_rows)}")

# Extract all integer->roman pairs from examples
for idx, row in roman_rows.head(3).iterrows():
    print(f"\nPrompt:\n{row['prompt']}")
    print(f"Answer: {row['answer']}")

# ===================== GRAVITY PHYSICS =====================
print("\n" + "=" * 60)
print("GRAVITY PHYSICS ANALYSIS")
print("=" * 60)
# gravity problems have "gravitational constant" in them
gravity_rows = df[df['prompt'].str.contains('gravitational', na=False)]
print(f"Count: {len(gravity_rows)}")

for idx, row in gravity_rows.head(3).iterrows():
    prompt = row['prompt']
    # Extract all (t, d) pairs
    pairs = re.findall(r't\s*=\s*([\d.]+)s.*?distance\s*=\s*([\d.]+)', prompt)
    # Extract query t
    query_t = re.search(r't\s*=\s*([\d.]+)s\s+given', prompt)
    if pairs and query_t:
        qt = float(query_t.group(1))
        # Compute g from first pair
        t0, d0 = float(pairs[0][0]), float(pairs[0][1])
        g = 2 * d0 / (t0 ** 2)
        predicted = 0.5 * g * qt ** 2
        print(f"g={g:.4f}, query_t={qt}, predicted={predicted:.2f}, actual={row['answer']}")

# ===================== UNIT CONVERSION =====================
print("\n" + "=" * 60)
print("UNIT CONVERSION ANALYSIS")
print("=" * 60)
unit_rows = df[df['prompt'].str.contains('unit conversion', na=False)]
print(f"Count: {len(unit_rows)}")

for idx, row in unit_rows.head(3).iterrows():
    prompt = row['prompt']
    # Extract all (input, output) pairs
    pairs = re.findall(r'([\d.]+)\s*m\s+becomes\s+([\d.]+)', prompt)
    query = re.search(r'convert.*?:\s*([\d.]+)\s*m', prompt)
    if pairs and query:
        qv = float(query.group(1))
        # Compute ratio from first pair
        in0, out0 = float(pairs[0][0]), float(pairs[0][1])
        ratio = out0 / in0
        predicted = qv * ratio
        print(f"ratio={ratio:.6f}, query={qv}, predicted={predicted:.2f}, actual={row['answer']}")

# ===================== ENCRYPTION =====================
print("\n" + "=" * 60)
print("ENCRYPTION ANALYSIS")
print("=" * 60)
encrypt_rows = df[df['prompt'].str.contains('encryption rules', na=False)]
print(f"Count: {len(encrypt_rows)}")

# Show structure
for idx, row in encrypt_rows.head(2).iterrows():
    print(f"\nPrompt:\n{row['prompt']}")
    print(f"Answer: {row['answer']}")

# ===================== BIT MANIPULATION =====================
print("\n" + "=" * 60)
print("BIT MANIPULATION ANALYSIS")
print("=" * 60)
bit_rows = df[df['prompt'].str.contains('bit manipulation', na=False)]
print(f"Count: {len(bit_rows)}")

# ===================== EQUATION TRANSFORM =====================
print("\n" + "=" * 60)
print("EQUATION TRANSFORM ANALYSIS")
print("=" * 60)
eq_rows = df[~df['prompt'].str.contains('bit manipulation|encryption|gravitational|unit conversion|numeral system', na=False)]
print(f"Count: {len(eq_rows)}")
for idx, row in eq_rows.head(3).iterrows():
    print(f"\nPrompt:\n{row['prompt']}")
    print(f"Answer: {row['answer']}")
