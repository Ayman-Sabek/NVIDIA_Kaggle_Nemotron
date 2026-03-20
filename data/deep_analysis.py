"""
Deep analysis of each puzzle category to understand what reasoning is needed.
This helps decide: prompt engineering vs LoRA training.
"""
import pandas as pd
import re
import numpy as np

train = pd.read_csv(r'd:\AI\Nemotron_Challange\data\raw\train.csv')

def categorize(prompt):
    p = prompt.lower()
    if 'bit manipulation' in p: return 'bit_manipulation'
    elif 'encryption' in p or 'decrypt' in p: return 'encryption'
    elif 'numeral system' in p: return 'numeral_system'
    elif 'gravitational constant' in p: return 'gravity_physics'
    elif 'unit conversion' in p: return 'unit_conversion'
    elif 'transformation rules' in p and 'equation' in p: return 'equation_transform'
    else: return 'other'

train['category'] = train['prompt'].apply(categorize)

# === 1. NUMERAL SYSTEM: Is it always Roman numerals? ===
print("=" * 80)
print("1. NUMERAL SYSTEM ANALYSIS")
print("=" * 80)
ns = train[train['category'] == 'numeral_system']
# Check if all answers are valid Roman numerals
roman_chars = set('IVXLCDM')
all_roman = ns['answer'].apply(lambda x: all(c in roman_chars for c in x.strip()))
print(f"All Roman numeral answers: {all_roman.all()} ({all_roman.sum()}/{len(ns)})")
# What numbers are being converted?
for i in range(min(5, len(ns))):
    row = ns.iloc[i]
    # Extract the number to convert
    number_match = re.search(r'number (\d+)', row['prompt'])
    if number_match:
        print(f"  Convert {number_match.group(1)} -> {row['answer']}")

# === 2. GRAVITY PHYSICS: What's the formula? ===
print("\n" + "=" * 80)
print("2. GRAVITY PHYSICS ANALYSIS")
print("=" * 80)
gp = train[train['category'] == 'gravity_physics']
# Parse examples from a few problems
for i in range(3):
    row = gp.iloc[i]
    # Extract t,d pairs
    pairs = re.findall(r't = ([\d.]+)s, distance = ([\d.]+) m', row['prompt'])
    if pairs:
        print(f"\nProblem {i} examples:")
        ts = [float(t) for t, d in pairs]
        ds = [float(d) for t, d in pairs]
        # d = 0.5 * g * t^2 => g = 2*d/t^2
        gs = [2*d/(t**2) for t, d in zip(ts, ds)]
        print(f"  Inferred g values: {[f'{g:.2f}' for g in gs]}")
        print(f"  Mean g: {np.mean(gs):.4f}")
        # Find the query
        query_match = re.search(r'determine.*?t = ([\d.]+)', row['prompt'])
        if query_match:
            qt = float(query_match.group(1))
            predicted = 0.5 * np.mean(gs) * qt**2
            print(f"  Query t={qt}, predicted d={predicted:.2f}, actual={row['answer']}")

# === 3. UNIT CONVERSION: What's the pattern? ===
print("\n" + "=" * 80)
print("3. UNIT CONVERSION ANALYSIS")
print("=" * 80)
uc = train[train['category'] == 'unit_conversion']
for i in range(3):
    row = uc.iloc[i]
    # Extract conversion pairs
    pairs = re.findall(r'([\d.]+) m becomes ([\d.]+)', row['prompt'])
    if pairs:
        print(f"\nProblem {i} examples:")
        ratios = [float(b)/float(a) for a, b in pairs]
        print(f"  Ratios: {[f'{r:.4f}' for r in ratios]}")
        print(f"  Mean ratio: {np.mean(ratios):.4f}, Std: {np.std(ratios):.4f}")
        # Find query
        query_match = re.search(r'convert.*?([\d.]+) m', row['prompt'])
        if query_match:
            qval = float(query_match.group(1))
            predicted = qval * np.mean(ratios)
            print(f"  Query {qval}m, predicted={predicted:.2f}, actual={row['answer']}")

# === 4. ENCRYPTION: What's the cipher type? ===
print("\n" + "=" * 80)
print("4. ENCRYPTION ANALYSIS")
print("=" * 80)
enc = train[train['category'] == 'encryption']
for i in range(3):
    row = enc.iloc[i]
    lines = row['prompt'].split('\n')
    example_lines = [l for l in lines if '->' in l and 'example' not in l.lower()]
    print(f"\nProblem {i}:")
    for el in example_lines[:2]:
        print(f"  {el.strip()}")
    # Check if it's a simple substitution cipher
    # Compare first char mappings across examples
    print(f"  Answer: {row['answer']}")

# === 5. BIT MANIPULATION: Patterns? ===
print("\n" + "=" * 80)
print("5. BIT MANIPULATION ANALYSIS")
print("=" * 80)
bm = train[train['category'] == 'bit_manipulation']
for i in range(3):
    row = bm.iloc[i]
    pairs = re.findall(r'(\d{8})\s*->\s*(\d{8})', row['prompt'])
    if pairs:
        print(f"\nProblem {i}:")
        for inp, out in pairs[:3]:
            xor = bin(int(inp, 2) ^ int(out, 2))[2:].zfill(8)
            print(f"  {inp} -> {out}  (XOR={xor})")

# === 6. EQUATION TRANSFORM ===
print("\n" + "=" * 80)
print("6. EQUATION TRANSFORM ANALYSIS")
print("=" * 80)
et = train[train['category'] == 'equation_transform']
for i in range(5):
    row = et.iloc[i]
    print(f"\nProblem {i}:")
    print(f"  Prompt: {row['prompt'][:300]}")
    print(f"  Answer: {row['answer']}")

# === DIFFICULTY ASSESSMENT ===
print("\n" + "=" * 80)
print("DIFFICULTY ASSESSMENT FOR LLM")
print("=" * 80)
print("""
numeral_system:  EASY - Standard number->Roman numeral conversion. LLM should know this.
gravity_physics: MEDIUM - Infer g from d=0.5*g*t^2, then compute. Needs arithmetic.
unit_conversion: MEDIUM - Infer linear ratio, then multiply. Needs arithmetic.
encryption:      HARD - Must decode substitution cipher from examples. Pattern matching.
bit_manipulation: HARD - Must identify bitwise operation from examples. Complex patterns.
equation_transform: HARD - Must decode symbol substitution rules. Abstract reasoning.
""")
