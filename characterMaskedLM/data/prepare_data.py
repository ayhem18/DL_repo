"""
I downloaded the realnewslike dataset by following the instructions found here:
https://github.com/allenai/allennlp/discussions/5056
"""

import gzip
import json
import os
from pathlib import Path


if __name__ == "__main__":

    from datasets import load_dataset, ReadInstruction
    dataset = load_dataset("allenai/c4", "realnewslike", streaming=True, split="train") 
    
        