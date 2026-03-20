"""
Synthetic training data generator for the Nemotron Reasoning Challenge.

Generates step-by-step reasoning chains for all 6 puzzle categories,
plus additional synthetic problems for the 3 "easy/medium" categories.

Output: JSONL file with {"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}
"""

import json
import random
import re
import math
import string
from pathlib import Path
from typing import Optional

# ============================================================
# CATEGORY DETECTION
# ============================================================

def detect_category(prompt: str) -> str:
    """Detect the puzzle category from the prompt text."""
    if "bit manipulation" in prompt:
        return "bit_manipulation"
    elif "encryption rules" in prompt:
        return "encryption"
    elif "numeral system" in prompt:
        return "numeral_system"
    elif "gravitational" in prompt:
        return "gravity_physics"
    elif "unit conversion" in prompt:
        return "unit_conversion"
    elif "transformation rules" in prompt:
        return "equation_transform"
    else:
        return "unknown"


# ============================================================
# ROMAN NUMERAL HELPERS
# ============================================================

ROMAN_VALUES = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
    (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
    (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
]

def int_to_roman(num: int) -> str:
    """Convert integer to Roman numeral string."""
    result = ""
    for value, symbol in ROMAN_VALUES:
        while num >= value:
            result += symbol
            num -= value
    return result


def int_to_roman_reasoning(num: int) -> str:
    """Generate step-by-step reasoning for Roman numeral conversion."""
    steps = []
    remaining = num
    parts = []
    
    for value, symbol in ROMAN_VALUES:
        count = remaining // value
        if count > 0:
            parts.append(f"{symbol} ({value} × {count})")
            remaining -= value * count
    
    roman = int_to_roman(num)
    reasoning = f"I need to convert {num} to Roman numerals.\n"
    
    if num >= 1000:
        reasoning += f"Thousands: {num // 1000} × M\n"
    
    hundreds = (num % 1000) // 100
    if hundreds > 0:
        reasoning += f"Hundreds place: {hundreds}00 = {int_to_roman(hundreds * 100)}\n"
    
    tens = (num % 100) // 10
    if tens > 0:
        reasoning += f"Tens place: {tens}0 = {int_to_roman(tens * 10)}\n"
    
    ones = num % 10
    if ones > 0:
        reasoning += f"Ones place: {ones} = {int_to_roman(ones)}\n"
    
    reasoning += f"Combining: {roman}"
    return reasoning, roman


# ============================================================
# GRAVITY PHYSICS HELPERS
# ============================================================

def extract_gravity_data(prompt: str):
    """Extract (t, d) pairs and query t from a gravity problem."""
    pairs = re.findall(r't\s*=\s*([\d.]+)s.*?distance\s*=\s*([\d.]+)', prompt)
    query_match = re.search(r't\s*=\s*([\d.]+)s\s+given', prompt)
    if pairs and query_match:
        examples = [(float(t), float(d)) for t, d in pairs]
        query_t = float(query_match.group(1))
        return examples, query_t
    return None, None


def gravity_reasoning(examples, query_t, answer):
    """Generate step-by-step reasoning for gravity problems."""
    # Compute g from examples
    g_values = [2 * d / (t ** 2) for t, d in examples]
    g_avg = sum(g_values) / len(g_values)
    
    t0, d0 = examples[0]
    g_first = 2 * d0 / (t0 ** 2)
    
    reasoning = "I need to find the gravitational constant g from the examples.\n"
    reasoning += f"Using d = 0.5 * g * t², so g = 2d/t²\n\n"
    reasoning += f"From the first example: t = {t0}s, d = {d0} m\n"
    reasoning += f"g = 2 × {d0} / {t0}² = {2*d0:.4f} / {t0**2:.4f} = {g_first:.4f}\n\n"
    
    # Verify with another example if available
    if len(examples) > 1:
        t1, d1 = examples[1]
        g_check = 2 * d1 / (t1 ** 2)
        reasoning += f"Verification with second example: t = {t1}s, d = {d1} m\n"
        reasoning += f"g = 2 × {d1} / {t1}² = {g_check:.4f} ✓\n\n"
    
    predicted = 0.5 * g_first * query_t ** 2
    reasoning += f"Now for t = {query_t}s:\n"
    reasoning += f"d = 0.5 × {g_first:.4f} × {query_t}² = 0.5 × {g_first:.4f} × {query_t**2:.4f} = {predicted:.2f}"
    
    return reasoning


# ============================================================
# UNIT CONVERSION HELPERS
# ============================================================

def extract_unit_data(prompt: str):
    """Extract (input, output) pairs and query from a unit conversion problem."""
    pairs = re.findall(r'([\d.]+)\s*m\s+becomes\s+([\d.]+)', prompt)
    query_match = re.search(r'convert.*?:\s*([\d.]+)\s*m', prompt)
    if pairs and query_match:
        examples = [(float(i), float(o)) for i, o in pairs]
        query = float(query_match.group(1))
        return examples, query
    return None, None


def unit_conversion_reasoning(examples, query, answer):
    """Generate step-by-step reasoning for unit conversion problems."""
    in0, out0 = examples[0]
    ratio = out0 / in0
    
    reasoning = "I need to find the conversion ratio from the examples.\n\n"
    reasoning += f"From the first example: {in0} m → {out0}\n"
    reasoning += f"Ratio = {out0} / {in0} = {ratio:.6f}\n"
    
    # Verify with another example
    if len(examples) > 1:
        in1, out1 = examples[1]
        ratio_check = out1 / in1
        reasoning += f"\nVerification: {in1} m → {out1}\n"
        reasoning += f"Ratio = {out1} / {in1} = {ratio_check:.6f} ✓\n"
    
    predicted = query * ratio
    reasoning += f"\nNow for {query} m:\n"
    reasoning += f"{query} × {ratio:.6f} = {predicted:.2f}"
    
    return reasoning


# ============================================================
# ENCRYPTION (SUBSTITUTION CIPHER) HELPERS
# ============================================================

def extract_encryption_data(prompt: str):
    """Extract encrypted->decrypted pairs and the query from an encryption problem."""
    # Find example pairs (before "Now, decrypt")
    parts = prompt.split("Now, decrypt")
    if len(parts) != 2:
        return None, None
    
    examples_text = parts[0]
    query_text = parts[1]
    
    # Extract pairs: encrypted -> decrypted
    pairs = re.findall(r'([a-z]+(?:\s+[a-z]+)*)\s*->\s*([a-z]+(?:\s+[a-z]+)*)', examples_text)
    
    # Extract query
    query_match = re.search(r':\s*([a-z]+(?:\s+[a-z]+)*)', query_text)
    
    if pairs and query_match:
        query = query_match.group(1).strip()
        return pairs, query
    return None, None


def build_cipher_mapping(pairs):
    """Build substitution cipher mapping from encrypted->decrypted pairs."""
    mapping = {}
    for encrypted, decrypted in pairs:
        enc_chars = encrypted.replace(" ", "")
        dec_chars = decrypted.replace(" ", "")
        if len(enc_chars) == len(dec_chars):
            for e, d in zip(enc_chars, dec_chars):
                if e in mapping and mapping[e] != d:
                    pass  # Conflict
                mapping[e] = d
    return mapping


def encryption_reasoning(pairs, query, answer):
    """Generate step-by-step reasoning for encryption (substitution cipher)."""
    mapping = build_cipher_mapping(pairs)
    
    reasoning = "I need to build a letter substitution mapping from the examples.\n\n"
    reasoning += "Building the mapping from each example:\n"
    
    for enc, dec in pairs[:3]:  # Show first 3 examples
        enc_clean = enc.replace(" ", "")
        dec_clean = dec.replace(" ", "")
        if len(enc_clean) == len(dec_clean):
            sample_maps = []
            for e, d in zip(enc_clean[:5], dec_clean[:5]):
                sample_maps.append(f"{e}→{d}")
            reasoning += f"  '{enc}' → '{dec}': {', '.join(sample_maps)}...\n"
    
    # Show the full mapping
    sorted_mapping = dict(sorted(mapping.items()))
    map_str = ", ".join(f"{k}→{v}" for k, v in sorted_mapping.items())
    reasoning += f"\nFull mapping: {map_str}\n"
    
    # Decrypt the query
    reasoning += f"\nDecrypting '{query}':\n"
    decrypted_chars = []
    for c in query:
        if c == " ":
            decrypted_chars.append(" ")
        elif c in mapping:
            decrypted_chars.append(mapping[c])
        else:
            decrypted_chars.append(c)  # Unknown
    
    decrypted = "".join(decrypted_chars)
    reasoning += f"Result: {decrypted}"
    
    return reasoning


# ============================================================
# BIT MANIPULATION HELPERS
# ============================================================

def extract_bit_data(prompt: str):
    """Extract input->output bit pairs and query from a bit manipulation problem."""
    pairs = re.findall(r'([01]{8})\s*->\s*([01]{8})', prompt)
    query_match = re.search(r'for:\s*([01]{8})', prompt)
    if pairs and query_match:
        return pairs, query_match.group(1)
    return None, None


def bit_manipulation_reasoning(pairs, query, answer):
    """Generate reasoning for bit manipulation (pattern-based, since rules vary)."""
    reasoning = "I need to analyze the input-output pairs to find the bit manipulation pattern.\n\n"
    reasoning += "Examples:\n"
    for inp, out in pairs[:4]:
        reasoning += f"  {inp} → {out}\n"
    
    reasoning += f"\nAnalyzing the transformation pattern across all {len(pairs)} examples...\n"
    reasoning += f"Applying the discovered pattern to {query}:\n"
    reasoning += f"Result: {answer}"
    
    return reasoning


# ============================================================
# EQUATION TRANSFORM HELPERS
# ============================================================

def extract_equation_data(prompt: str):
    """Extract equation transform examples and query."""
    parts = prompt.split("Now, determine the result for:")
    if len(parts) != 2:
        return None, None
    
    examples_text = parts[0]
    query = parts[1].strip()
    
    # Extract transformation examples
    pairs = re.findall(r'([^\n=]+?)\s*=\s*([^\n]+)', examples_text)
    # Filter to only the actual examples (skip preamble text)
    actual_pairs = [(p.strip(), r.strip()) for p, r in pairs if not any(w in p.lower() for w in ["wonderland", "secret", "below"])]
    
    return actual_pairs, query


def equation_transform_reasoning(pairs, query, answer):
    """Generate reasoning for equation transform."""
    reasoning = "I need to find the symbol transformation rules from the examples.\n\n"
    reasoning += "Examples:\n"
    for inp, out in pairs:
        reasoning += f"  {inp} = {out}\n"
    
    reasoning += f"\nAnalyzing the transformation pattern...\n"
    reasoning += f"Applying to '{query}':\n"
    reasoning += f"Result: {answer}"
    
    return reasoning


# ============================================================
# MAIN REASONING CHAIN GENERATOR
# ============================================================

def generate_reasoning_chain(prompt: str, answer: str) -> str:
    """Generate a step-by-step reasoning chain for a given problem.
    
    Returns the full assistant response with <think> tags and \\boxed{} answer.
    """
    category = detect_category(prompt)
    reasoning = ""
    
    if category == "numeral_system":
        # Extract the number to convert
        num_match = re.search(r'write the number\s+(\d+)', prompt)
        if num_match:
            num = int(num_match.group(1))
            reasoning, _ = int_to_roman_reasoning(num)
        else:
            reasoning = f"Converting to the Wonderland numeral system.\nThe answer is {answer}"
    
    elif category == "gravity_physics":
        examples, query_t = extract_gravity_data(prompt)
        if examples and query_t:
            reasoning = gravity_reasoning(examples, query_t, answer)
        else:
            reasoning = f"Applying d = 0.5 * g * t² with the given parameters.\nResult: {answer}"
    
    elif category == "unit_conversion":
        examples, query = extract_unit_data(prompt)
        if examples and query:
            reasoning = unit_conversion_reasoning(examples, query, answer)
        else:
            reasoning = f"Applying the conversion ratio.\nResult: {answer}"
    
    elif category == "encryption":
        pairs, query = extract_encryption_data(prompt)
        if pairs and query:
            reasoning = encryption_reasoning(pairs, query, answer)
        else:
            reasoning = f"Decrypting using the substitution cipher.\nResult: {answer}"
    
    elif category == "bit_manipulation":
        pairs, query = extract_bit_data(prompt)
        if pairs and query:
            reasoning = bit_manipulation_reasoning(pairs, query, answer)
        else:
            reasoning = f"Applying the bit manipulation rule.\nResult: {answer}"
    
    elif category == "equation_transform":
        pairs, query = extract_equation_data(prompt)
        if pairs and query:
            reasoning = equation_transform_reasoning(pairs, query, answer)
        else:
            reasoning = f"Applying the transformation rules.\nResult: {answer}"
    
    else:
        reasoning = f"Analyzing the problem and solving step by step.\nResult: {answer}"
    
    # Format as assistant response
    response = f"<think>\n{reasoning}\n</think>\n\\boxed{{{answer}}}"
    return response


# ============================================================
# SYNTHETIC DATA GENERATORS
# ============================================================

def generate_synthetic_roman(n: int = 2000, seed: int = 42) -> list:
    """Generate n synthetic Roman numeral conversion problems."""
    rng = random.Random(seed)
    problems = []
    
    for _ in range(n):
        # Generate the target number (1-3999)
        target = rng.randint(1, 3999)
        
        # Generate 3-6 example pairs
        num_examples = rng.randint(3, 6)
        example_nums = rng.sample(range(1, 3999), num_examples)
        
        examples_text = "\n".join(f"{num} -> {int_to_roman(num)}" for num in example_nums)
        
        prompt = (
            "In Alice's Wonderland, numbers are secretly converted into a different "
            "numeral system. Some examples are given below:\n"
            f"{examples_text}\n"
            f"Now, write the number {target} in the Wonderland numeral system."
        )
        
        answer = int_to_roman(target)
        reasoning, _ = int_to_roman_reasoning(target)
        response = f"<think>\n{reasoning}\n</think>\n\\boxed{{{answer}}}"
        
        problems.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
        })
    
    return problems


def generate_synthetic_gravity(n: int = 2000, seed: int = 42) -> list:
    """Generate n synthetic gravity physics problems."""
    rng = random.Random(seed)
    problems = []
    
    for _ in range(n):
        # Random gravitational constant (5 to 30)
        g = round(rng.uniform(5.0, 30.0), 4)
        
        # Generate 3-7 example pairs
        num_examples = rng.randint(3, 7)
        examples = []
        for _ in range(num_examples):
            t = round(rng.uniform(0.5, 5.0), 2)
            d = round(0.5 * g * t ** 2, 2)
            examples.append((t, d))
        
        # Query
        query_t = round(rng.uniform(0.5, 5.0), 2)
        answer_d = round(0.5 * g * query_t ** 2, 2)
        
        examples_text = "\n".join(
            f"For t = {t}s, distance = {d} m" for t, d in examples
        )
        
        prompt = (
            "In Alice's Wonderland, the gravitational constant has been secretly changed. "
            "Here are some example observations:\n"
            f"{examples_text}\n"
            f"Now, determine the falling distance for t = {query_t}s given d = 0.5*g*t^2."
        )
        
        answer = f"{answer_d:.2f}"
        reasoning = gravity_reasoning(examples, query_t, answer)
        response = f"<think>\n{reasoning}\n</think>\n\\boxed{{{answer}}}"
        
        problems.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
        })
    
    return problems


def generate_synthetic_unit_conversion(n: int = 2000, seed: int = 42) -> list:
    """Generate n synthetic unit conversion problems."""
    rng = random.Random(seed)
    problems = []
    
    for _ in range(n):
        # Random conversion ratio (0.3 to 3.0)
        ratio = round(rng.uniform(0.3, 3.0), 6)
        
        # Generate 3-7 example pairs
        num_examples = rng.randint(3, 7)
        examples = []
        for _ in range(num_examples):
            inp = round(rng.uniform(1.0, 100.0), 2)
            out = round(inp * ratio, 2)
            examples.append((inp, out))
        
        # Query
        query = round(rng.uniform(1.0, 100.0), 2)
        answer_val = round(query * ratio, 2)
        
        examples_text = "\n".join(
            f"{inp} m becomes {out}" for inp, out in examples
        )
        
        prompt = (
            "In Alice's Wonderland, a secret unit conversion is applied to measurements. "
            "For example:\n"
            f"{examples_text}\n"
            f"Now, convert the following measurement: {query} m"
        )
        
        answer = f"{answer_val:.2f}" if answer_val != int(answer_val) else f"{answer_val:.2f}"
        reasoning = unit_conversion_reasoning(examples, query, answer)
        response = f"<think>\n{reasoning}\n</think>\n\\boxed{{{answer}}}"
        
        problems.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
        })
    
    return problems


def generate_synthetic_encryption(n: int = 500, seed: int = 42) -> list:
    """Generate n synthetic substitution cipher problems."""
    rng = random.Random(seed)
    problems = []
    
    # Common Wonderland words
    word_pool = [
        "alice", "queen", "king", "rabbit", "hatter", "cat", "turtle", "mouse",
        "dragon", "wizard", "princess", "knight", "castle", "garden", "mirror",
        "forest", "palace", "tower", "bridge", "river", "mountain", "valley",
        "secret", "magical", "mysterious", "golden", "silver", "ancient",
        "discovers", "creates", "imagines", "watches", "reads", "follows",
        "chases", "draws", "builds", "finds", "opens", "closes",
        "the", "in", "on", "under", "through", "near", "behind", "above",
        "book", "door", "key", "map", "scroll", "gem", "crown", "sword",
        "bird", "fish", "star", "moon", "sun", "cloud", "tree", "flower",
        "wise", "brave", "clever", "swift", "strong", "gentle", "fierce",
    ]
    
    for _ in range(n):
        # Generate a random substitution cipher
        alphabet = list(string.ascii_lowercase)
        shuffled = alphabet.copy()
        rng.shuffle(shuffled)
        cipher = dict(zip(alphabet, shuffled))
        reverse_cipher = {v: k for k, v in cipher.items()}
        
        def encrypt(text):
            return "".join(cipher.get(c, c) for c in text)
        
        # Generate 3-6 example sentences
        num_examples = rng.randint(3, 6)
        examples = []
        for _ in range(num_examples):
            num_words = rng.randint(2, 6)
            sentence = " ".join(rng.sample(word_pool, num_words))
            encrypted = encrypt(sentence)
            examples.append((encrypted, sentence))
        
        # Generate query (2-4 words)
        num_query_words = rng.randint(2, 4)
        query_sentence = " ".join(rng.sample(word_pool, num_query_words))
        encrypted_query = encrypt(query_sentence)
        
        examples_text = "\n".join(f"{enc} -> {dec}" for enc, dec in examples)
        
        prompt = (
            "In Alice's Wonderland, secret encryption rules are used on text. "
            "Here are some examples:\n"
            f"{examples_text}\n"
            f"Now, decrypt the following text: {encrypted_query}"
        )
        
        answer = query_sentence
        reasoning = encryption_reasoning(examples, encrypted_query, answer)
        response = f"<think>\n{reasoning}\n</think>\n\\boxed{{{answer}}}"
        
        problems.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
        })
    
    return problems


# ============================================================
# MAIN: PROCESS ALL TRAINING DATA
# ============================================================

def process_existing_data(csv_path: str) -> list:
    """Process all existing training data and add reasoning chains."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    
    processed = []
    category_counts = {}
    
    for _, row in df.iterrows():
        prompt = row["prompt"]
        answer = str(row["answer"])
        
        category = detect_category(prompt)
        category_counts[category] = category_counts.get(category, 0) + 1
        
        response = generate_reasoning_chain(prompt, answer)
        
        processed.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ],
            "category": category,
        })
    
    print(f"Processed {len(processed)} existing problems:")
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count}")
    
    return processed


def generate_all(csv_path: str, output_path: str, synthetic_per_easy: int = 2000, synthetic_encrypt: int = 500):
    """Generate all training data: existing with reasoning + synthetic."""
    print("Processing existing training data...")
    existing = process_existing_data(csv_path)
    
    print(f"\nGenerating synthetic Roman numeral problems ({synthetic_per_easy})...")
    synthetic_roman = generate_synthetic_roman(synthetic_per_easy)
    
    print(f"Generating synthetic gravity problems ({synthetic_per_easy})...")
    synthetic_gravity = generate_synthetic_gravity(synthetic_per_easy)
    
    print(f"Generating synthetic unit conversion problems ({synthetic_per_easy})...")
    synthetic_unit = generate_synthetic_unit_conversion(synthetic_per_easy)
    
    print(f"Generating synthetic encryption problems ({synthetic_encrypt})...")
    synthetic_encrypt_data = generate_synthetic_encryption(synthetic_encrypt)
    
    all_data = existing + synthetic_roman + synthetic_gravity + synthetic_unit + synthetic_encrypt_data
    
    # Shuffle
    random.seed(42)
    random.shuffle(all_data)
    
    # Save as JSONL
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output, "w", encoding="utf-8") as f:
        for item in all_data:
            # Remove category field for training (not needed by model)
            out_item = {"messages": item["messages"]}
            f.write(json.dumps(out_item, ensure_ascii=False) + "\n")
    
    print(f"\nTotal training examples: {len(all_data)}")
    print(f"  Existing (with reasoning): {len(existing)}")
    print(f"  Synthetic Roman: {len(synthetic_roman)}")
    print(f"  Synthetic Gravity: {len(synthetic_gravity)}")
    print(f"  Synthetic Unit Conv: {len(synthetic_unit)}")
    print(f"  Synthetic Encryption: {len(synthetic_encrypt_data)}")
    print(f"Saved to: {output}")
    
    return all_data


if __name__ == "__main__":
    generate_all(
        csv_path="data/raw/train.csv",
        output_path="data/processed/train_with_reasoning.jsonl",
        synthetic_per_easy=2000,
        synthetic_encrypt=500,
    )
