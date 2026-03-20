"""Quick data analysis script."""
import pandas as pd
import re

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

def count_examples(prompt):
    return len(re.findall(r'->', prompt))

train['num_examples'] = train['prompt'].apply(count_examples)

print('=== EXAMPLES PER PROBLEM BY CATEGORY ===')
for cat in train['category'].unique():
    subset = train[train['category'] == cat]
    ex = subset['num_examples']
    print(f'  {cat}: mean={ex.mean():.1f}, min={ex.min()}, max={ex.max()}')

train['est_tokens'] = train['prompt'].str.len() / 4
print()
print('=== ESTIMATED TOKEN COUNTS ===')
mean_tok = train['est_tokens'].mean()
max_tok = train['est_tokens'].max()
print(f'  Mean: {mean_tok:.0f} tokens')
print(f'  Max: {max_tok:.0f} tokens')
print(f'  With max_tokens=7680: ~{7680 - mean_tok:.0f} tokens for reasoning+answer')

print()
print('=== ANSWER FORMAT SUMMARY ===')
for cat in sorted(train['category'].unique()):
    subset = train[train['category'] == cat]
    answers = subset['answer'].astype(str)
    numeric = answers.apply(lambda x: bool(re.match(r'^-?[\d.]+$', x.strip())))
    binary = answers.apply(lambda x: bool(re.match(r'^[01]+$', x.strip())))
    alpha = answers.apply(lambda x: bool(re.match(r'^[a-zA-Z\s]+$', x.strip())))
    
    print(f'\n  {cat} ({len(subset)} problems):')
    print(f'    Numeric: {numeric.sum()}, Binary: {binary.sum()}, Alpha: {alpha.sum()}')
    print(f'    Avg len: {answers.str.len().mean():.1f}, Unique: {answers.nunique()}')
    print(f'    Samples: {answers.sample(min(3, len(answers)), random_state=42).tolist()}')

print()
print('=== KEY INSIGHT ===')
print('All 6 categories are PATTERN RECOGNITION / IN-CONTEXT LEARNING tasks.')
print('The model must:')
print('  1. Observe input->output example pairs')
print('  2. Identify the hidden transformation rule')
print('  3. Apply it to a new input')
print('This is a reasoning benchmark focused on rule induction and application.')
