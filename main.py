import argparse
import torch
from transformers import pipeline
import time

from utils.read_doc import getText

model_creator = "facebook"
model_name = "bart-large-cnn"

start_time = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\033[94m Device used for inference: {device}\033[0m")

if device == "cuda":
    summarizer = pipeline(
        "summarization", model=f"{model_creator}/{model_name}", device=0
    )
else:
    summarizer = pipeline("summarization", model=f"{model_creator}/{model_name}")

print(f"Took {time.time() - start_time} seconds to load summarizer\n")

parser = argparse.ArgumentParser(description="Summarize a document")
parser.add_argument("filename", type=str, help="Path to the document file")
args = parser.parse_args()

print(f"Reading {args.filename}")
ARTICLE = getText(args.filename)
print("Finished reading doc file")

max_length = len(ARTICLE) / 5
max_length = int(max_length)
print(f"Max summary tokens: {max_length}\n\n")

summary = summarizer(ARTICLE, max_length=max_length, min_length=30, do_sample=False)
print(f"Took {time.time() - start_time} seconds to complete summarization\n\n")

print(f"Summary:\n\033[92m{summary.pop()['summary_text']}\033[0m")
