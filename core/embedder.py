import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# from sentence_transformers import SentenceTransformer
# import numpy as np

# _model = None

# def get_model():
#     global _model

#     if _model is None:
#         print("Loading embedding model...")
#         _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

#     return _model

# print("Loading embedding model...")
# def generate_embeddings(chunks):

#     model = get_model()

#     texts = [
#         "passage: " + chunk["text"]
#         for chunk in chunks
#     ]

#     all_embeddings = []
#     print("Encoding starts")
#     for i in range(0, len(texts), 32):
#         batch = texts[i:i+32]
#         emb = model.encode(
#             batch,
#             normalize_embeddings=True,
#             show_progress_bar=False
#         )
#         all_embeddings.extend(emb)
#     print("Encoding finished")

#     embeddings = np.array(all_embeddings, dtype=np.float32)

#     return np.array(embeddings, dtype=np.float32)
import gc
import numpy as np

_model = None


def get_model():
    global _model

    if _model is None:
        from sentence_transformers import SentenceTransformer

        print("LOADING EMBEDDING MODEL")

        _model = SentenceTransformer(
            "sentence-transformers/paraphrase-MiniLM-L3-v2",
            device="cpu"
        )

        import torch
        torch.set_num_threads(1)

        print("MODEL LOADED")

    return _model



def generate_embeddings(chunks):

    model = get_model()


    texts = [
        "passage: " + c["text"]
        for c in chunks
    ]


    print("TOTAL TEXTS:", len(texts))


    embeddings = []


    for start in range(0, len(texts), 8):

        end = min(start + 8, len(texts))

        print(
            f"Embedding batch {start}-{end}"
        )


        batch = texts[start:end]


        emb = model.encode(
            batch,
            batch_size=8,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        )


        embeddings.append(emb)
        print(f"DEBUG F: after append, embeddings len={len(embeddings)}")

        del batch
        del emb

        gc.collect()



    print("STACKING EMBEDDINGS")


    result = np.vstack(embeddings)

    result = result.astype(
        np.float32,
        copy=False
    )


    del embeddings
    del texts

    gc.collect()


    print(
        "FINAL EMBEDDING:",
        result.shape
    )


    return result