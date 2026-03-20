"""Verify training data quality."""
import json

with open("data/processed/train_with_reasoning.jsonl", "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"Total examples: {len(lines)}")
print()

# Show one example of each type by checking the content
categories_shown = set()
for line in lines:
    item = json.loads(line)
    user_msg = item["messages"][0]["content"]
    asst_msg = item["messages"][1]["content"]
    
    # Detect category
    if "bit manipulation" in user_msg and "bit_manipulation" not in categories_shown:
        cat = "bit_manipulation"
    elif "encryption" in user_msg and "encryption" not in categories_shown:
        cat = "encryption"
    elif "numeral system" in user_msg and "numeral_system" not in categories_shown:
        cat = "numeral_system"
    elif "gravitational" in user_msg and "gravity_physics" not in categories_shown:
        cat = "gravity_physics"
    elif "unit conversion" in user_msg and "unit_conversion" not in categories_shown:
        cat = "unit_conversion"
    elif "transformation rules" in user_msg and "equation_transform" not in categories_shown:
        cat = "equation_transform"
    else:
        continue
    
    categories_shown.add(cat)
    print(f"=== {cat} ===")
    print(f"User: {user_msg[:200]}...")
    print(f"Assistant: {asst_msg[:400]}...")
    print("-" * 60)
    print()
    
    if len(categories_shown) == 6:
        break

# Check answer extraction
import re
boxed_count = 0
for line in lines:
    item = json.loads(line)
    asst = item["messages"][1]["content"]
    if "\\boxed{" in asst:
        boxed_count += 1

print(f"\nExamples with \\boxed{{}}: {boxed_count}/{len(lines)} ({100*boxed_count/len(lines):.1f}%)")
