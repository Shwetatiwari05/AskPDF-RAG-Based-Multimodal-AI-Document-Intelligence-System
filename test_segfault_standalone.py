#!/usr/bin/env python3
"""
Standalone script to test if the exact resume text causes segfault
outside of FastAPI/Uvicorn process.
"""

import os
import sys
import torch
import multiprocessing
import joblib

# Set before any transformers/sentence_transformers import
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    import loky
    LOKY_VERSION = loky.__version__
except ImportError:
    LOKY_VERSION = "not installed"

print("=" * 60)
print("STANDALONE SEGFAULT TEST")
print("PID:", os.getpid())
print("Parent PID:", os.getppid())
print("CPU count:", multiprocessing.cpu_count())
print("Torch threads:", torch.get_num_threads())
print("Torch interop threads:", torch.get_num_interop_threads())
print("TOKENIZERS_PARALLELISM:", os.environ.get("TOKENIZERS_PARALLELISM"))
print("joblib version:", joblib.__version__)
print("loky version:", LOKY_VERSION)
print("=" * 60)

# Import after env var is set
from sentence_transformers import SentenceTransformer

print("Loading model...")
model = SentenceTransformer(
    "sentence-transformers/paraphrase-MiniLM-L3-v2",
    device="cpu"
)
print("Model loaded.")

# These are the EXACT raw chunk texts from the latest upload
# We need to extract them from the upload logs or re-run the upload
# For now, we'll use placeholder - but the script should be updated
# with the actual chunk text after we see the logs

# Example: replace these with actual chunk text from logs
raw0 = "PASTE_CHUNK_0_TEXT_HERE"
raw1 = "PASTE_CHUNK_1_TEXT_HERE"

chunk0 = "passage: " + raw0
chunk1 = "passage: " + raw1

print(f"chunk0 len={len(chunk0)}")
print(f"chunk1 len={len(chunk1)}")
print(f"chunk0[:500] = {repr(chunk0[:500])}")
print(f"chunk1[:500] = {repr(chunk1[:500])}")

# Test 1: chunk0 only, batch_size=1
print("TEST 1: before encode (chunk0 only, batch_size=1)")
try:
    result1 = model.encode(
        [chunk0],
        batch_size=1,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    print("TEST 1: after encode, shape =", result1.shape)
except Exception as e:
    print("TEST 1: EXCEPTION =", repr(e))
    sys.exit(1)

# Test 2: chunk1 only, batch_size=1
if chunk1.strip():
    print("TEST 2: before encode (chunk1 only, batch_size=1)")
    try:
        result2 = model.encode(
            [chunk1],
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        print("TEST 2: after encode, shape =", result2.shape)
    except Exception as e:
        print("TEST 2: EXCEPTION =", repr(e))
        sys.exit(1)

# Test 3: both chunks, batch_size=1
print("TEST 3: before encode (both chunks, batch_size=1)")
try:
    result3 = model.encode(
        [chunk0, chunk1],
        batch_size=1,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    print("TEST 3: after encode, shape =", result3.shape)
except Exception as e:
    print("TEST 3: EXCEPTION =", repr(e))
    sys.exit(1)

# Test 4: both chunks, batch_size=8
print("TEST 4: before encode (both chunks, batch_size=8)")
try:
    result4 = model.encode(
        [chunk0, chunk1],
        batch_size=8,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    print("TEST 4: after encode, shape =", result4.shape)
except Exception as e:
    print("TEST 4: EXCEPTION =", repr(e))
    sys.exit(1)

# Test 5: progressively shorter versions of chunk0
for length in [500, 1000, 1500, 2000, 2500, 3000, 3500]:
    if length < len(chunk0):
        test_chunk = chunk0[:length]
        print(f"TEST 5-{length}: before encode (chunk0[:{length}], batch_size=1)")
        try:
            result5 = model.encode(
                [test_chunk],
                batch_size=1,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True
            )
            print(f"TEST 5-{length}: after encode, shape =", result5.shape)
        except Exception as e:
            print(f"TEST 5-{length}: EXCEPTION =", repr(e))
            sys.exit(1)

print("=" * 60)
print("ALL TESTS PASSED - No segfault in standalone process")
print("=" * 60)