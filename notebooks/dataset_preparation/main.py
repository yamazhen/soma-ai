from typing import List, Literal
from google import genai
from datasets import Dataset
import pandas as pd
import os
import random
import time

from google.genai.types import json
from pydantic import BaseModel


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

class Conversaton(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class DatasetExample(BaseModel):
    conversations: List[Conversaton]

class DatasetBatch(BaseModel):
    examples: List[DatasetExample]

def generateDatasetBatch(batch_size: int = 10, content_type: str = "mixed") -> List[dict]:

    subjects = [
        "biology", "chemistry", "physics", "mathematics", "history", 
        "geography", "computer science", "psychology", "economics", 
        "literature", "environmental science", "astronomy"
    ]

    content_types = ["study notes", "quiz questions", "flashcards"]

    prompt = f"""Generate {batch_size} diverse educational training examples for an AI assistant. 

REQUIREMENTS:
- Create examples across different subjects: {', '.join(random.sample(subjects, 6))}
- Include all content types: {', '.join(content_types)}
- Make content appropriate for high school to early college level
- Ensure diversity in topics, difficulty levels, and cultural perspectives
- Each example should follow this exact format:

Example structure:
{{
  "conversations": [
    {{"role": "system", "content": "You are an educational assistant that [generates study notes/creates quiz questions/creates flashcards] from educational content."}},
    {{"role": "user", "content": "[Request to create study notes/quiz/flashcards] from this text: '[Educational content about specific topic]'"}},
    {{"role": "assistant", "content": "[Properly formatted response with emojis, clear structure, and educational value]"}}
  ]
}}

CONTENT GUIDELINES:
- Study notes: Use emojis, clear headings, bullet points, key concepts
- Quiz questions: Include multiple choice, true/false, short answer, fill-in-blank
- Flashcards: Clear front/back format with concise Q&A

Generate diverse topics covering science, humanities, technology, and social sciences."""

    response = None
    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": DatasetBatch.model_json_schema(),
            }
        )

        if not response or not response.text:
            print("No response from gemini")
            return []

        result = json.loads(response.text)
        examples = result.get("examples", [])

        if not examples:
            print("No examples generated")
            return []

        return examples
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print("Response text:", { response.text if response else "No response" })
        return []
    except Exception as e:
        print(f"Error generating dataset batch: {e}")
        return []

def generateFullDataset(total_examples: int = 100, batch_size: int = 10) -> List[dict]:

    all_examples = []
    batches_needed = (total_examples + batch_size -1) // batch_size

    print(f"Generating {batches_needed} batches of {batch_size} examples each...")

    for i in range(batches_needed):
        current_batch_size = min(batch_size, total_examples - len(all_examples))
        print(f"Generating batch {i+1}/{batches_needed} ({current_batch_size} examples)...")

        max_retries = 3
        batch_examples = []

        for retry in range(max_retries):
            batch_examples = generateDatasetBatch(current_batch_size)

            if batch_examples:
                break
            elif retry < max_retries - 1:
                print(f"  Retry {retry + 1}/{max_retries}")
                time.sleep(2)

        if batch_examples:
            all_examples.extend(batch_examples)
            print(f"Generated {len(batch_examples)} examples. Total so far: {len(all_examples)}")
        else:
            print(f"Failed to generate batch {i + 1} after {max_retries} retries")

        time.sleep(1)

    return all_examples

def saveDataset(examples: List[dict], filename: str = "educational_dataset"):
    
    if not examples:
        print("No examples to save")
        return

    try:
        with open(f"{filename}.json", "w", encoding="utf-8") as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)
        print(f"Dataset saved to {filename}.json")
    except Exception as e:
        print(f"Error saving JSON: {e}")
        return

    try:
        dataset = Dataset.from_pandas(pd.DataFrame(examples))
        dataset.save_to_disk(filename)
        print(f"Dataset saved to {filename} in Hugging Face format")
    except Exception as e:
        print(f"Error saving HF dataset: {e}")

def printDataset (examples: List[dict]):
    content_types = {"notes": 0, "quiz": 0, "flashcards": 0, "other": 0}

    for example in examples:
        try:
            user_content = example["conversations"][1]["content"].lower()
            if "notes" in user_content:
                content_types["notes"] += 1
            elif "quiz" in user_content:
                content_types["quiz"] += 1
            elif "flashcards" in user_content:
                content_types["flashcards"] += 1
            else:
                content_types["other"] += 1
        except (KeyError, IndexError):
            content_types["other"] += 1

    print(f"\nDataset Statistics:")
    print(f"Total examples: {len(examples)}")
    for content_type, count in content_types.items():
        if count > 0:
            print(f"- {content_type}: {count} examples ({count/len(examples)*100:.1f}%)")

if __name__ == "__main__":
    if not GEMINI_API_KEY:
        print("GEMINI_API_KEY environment variable not set!")
        exit(1)
    
    
    print("Starting dataset generation...")
    dataset = generateFullDataset(total_examples=100, batch_size=10)
    
    if dataset:
        saveDataset(dataset, "educational_content_dataset")
        print("Dataset generation complete!")
    else:
        print("Failed to generate dataset")
