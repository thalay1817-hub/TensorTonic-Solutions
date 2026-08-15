import unicodedata
import hashlib
import re

def stable_exact_deduplication(documents, lowercase, collapse_whitespace, hash_bits):
    def normalize(text):
        t = unicodedata.normalize("NFKC", text)
        if lowercase:
            t = t.casefold()
        if collapse_whitespace:
            t = t.strip()
            t = re.sub(r"\s+", " ", t)
        return t

    def bucket_key(norm_text):
        digest = hashlib.sha256(norm_text.encode("utf-8")).digest()
        full_int = int.from_bytes(digest, "big")
        mask = (1 << hash_bits) - 1
        return full_int & mask

    buckets = {}  # bucket_key -> list of (normalized_text, retained_id)
    retained_ids = []
    removed_to_retained = {}

    for doc in documents:
        doc_id = doc["id"]
        text = doc["text"]
        norm = normalize(text)
        key = bucket_key(norm)

        bucket = buckets.setdefault(key, [])
        match_id = None
        for stored_norm, stored_id in bucket:
            if stored_norm == norm:
                match_id = stored_id
                break

        if match_id is not None:
            removed_to_retained[doc_id] = match_id
        else:
            bucket.append((norm, doc_id))
            retained_ids.append(doc_id)

    return {
        "retained_ids": retained_ids,
        "removed_to_retained": removed_to_retained,
    }