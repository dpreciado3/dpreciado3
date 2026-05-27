import json
import random
import re
import pandas as pd
from llama_cpp import Llama

# -------------------------------------------------------------------------
# 1. Initialization
# -------------------------------------------------------------------------
# Use an Instruct/IT variant of Gemma 4 (e.g., gemma-4-e4b-it-Q4_K_M.gguf)
MODEL_PATH = "path/to/gemma-4-e4b-it-Q4_K_M.gguf"

print("Loading GGUF model onto CPU...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,         # Context window (Gemma 4 supports up to 128K if needed)
    n_threads=4,        # Set this to your physical CPU core count
    verbose=False
)

# Chat template formatter for Gemma 4
def format_gemma_prompt(system_instructions: str, user_input: str) -> str:
    prompt = f"{system_instructions}\n\nINPUT:\n{user_input}"
    return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

def clean_and_parse_json(text: str):
    """Extracts and parses JSON if the model wraps it in markdown code fences."""
    match = re.search(r"\{.*\}|\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None

# -------------------------------------------------------------------------
# 2. Phase 1: Taxonomy Discovery
# -------------------------------------------------------------------------
def discover_categories(texts: list, sample_size: int = 30) -> list:
    """Analyzes a batch of sample data to build a uniform classification schema."""
    print(f"Sampling {sample_size} items to discover underlying categories...")
    
    # Take a random sample to represent the dataset diversity
    sampled_texts = random.sample(texts, min(len(texts), sample_size))
    formatted_samples = "\n".join([f"- {text}" for text in sampled_texts])
    
    system_instructions = (
        "You are an expert data analyst. Review the provided sample texts and "
        "identify between 4 and 8 distinct, mutually exclusive categories that best "
        "classify them. Return your response ONLY as a valid JSON list of strings. "
        "Do not include explanations outside of the JSON array.\n"
        "Example Output format: [\"Category A\", \"Category B\", \"Category C\"]"
    )
    
    prompt = format_gemma_prompt(system_instructions, formatted_samples)
    
    # Generate schema
    response = llm(prompt, max_tokens=250, temperature=0.2)
    response_text = response["choices"][0]["text"]
    
    categories = clean_and_parse_json(response_text)
    if not categories or not isinstance(categories, list):
        print("Warning: Failed to parse model output into a valid JSON list. Falling back to default list.")
        return ["Technical Support", "Billing/Invoices", "General Inquiry", "Feedback"]
        
    return categories

# -------------------------------------------------------------------------
# 3. Phase 2: Classification Loop
# -------------------------------------------------------------------------
def classify_text(text: str, categories: list) -> str:
    """Classifies a single text string using the discovered categories."""
    categories_str = ", ".join([f"'{cat}'" for cat in categories])
    
    system_instructions = (
        f"Classify the input text into exactly one of these categories: [{categories_str}]. "
        "Your response must consist ONLY of the selected category name. "
        "Do not provide explanations, preamble, or punctuation."
    )
    
    prompt = format_gemma_prompt(system_instructions, text)
    
    # Low max_tokens keeps CPU processing exceptionally fast per row
    response = llm(prompt, max_tokens=15, temperature=0.0)
    predicted_category = response["choices"][0]["text"].strip()
    
    # Fallback cleanup in case the model outputs a slight variant or extra words
    for cat in categories:
        if cat.lower() in predicted_category.lower():
            return cat
            
    return "Unclassified"

# -------------------------------------------------------------------------
# 4. Execution Example
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy dataset simulating mixed feedback/support tickets
    dataset = [
        "My order hasn't arrived yet, it's been over two weeks.",
        "The mobile app keeps crashing whenever I try to upload a photo.",
        "Can I get a refund if the shoes don't fit me properly?",
        "Is there a dark mode option available for the desktop dashboard?",
        "I was double charged on my visa card for this month's premium tier.",
        "Your delivery driver left the box out in the pouring rain.",
        "How do I reset my account password? The reset link isn't arriving.",
        "The new interface update looks incredibly clean, great job.",
        "Where can I download the tax invoice for my subscription payment?",
    ]
    
    # Run Phase 1: Discover the categories
    discovered_tags = discover_categories(dataset, sample_size=6)
    print(f"\nDiscovered Categories: {discovered_tags}\n")
    
    # Run Phase 2: Process and classify all rows
    print("Classifying dataset...")
    results = []
    for item in dataset:
        label = classify_text(item, discovered_tags)
        results.append({"Text": item, "Category": label})
        print(f"-> Classified: \"{item[:30]}...\" As [{label}]")
        
    # Save results to a clean DataFrame
    df = pd.DataFrame(results)
    # df.to_csv("classified_dataset.csv", index=False)
  
