import numpy as np
from typing import List, Tuple

def create_nsp_pairs(
    documents: List[List[str]],
    pair_specs: List[dict]
) -> List[Tuple[str, str, int]]:

    pairs = []
    for spec in pair_specs:
        doc_a = spec["doc_a"]
        doc_b = spec["doc_b"]
        sent_a_idx = spec["sent_a"]
        sent_b_idx = spec["sent_b"]

        sent_a = documents[doc_a][sent_a_idx]
        sent_b = documents[doc_b][sent_b_idx]

        # IsNext=1 if same document and sent_b immediately follows sent_a
        is_next = int(doc_a == doc_b and sent_b_idx == sent_a_idx + 1)

        pairs.append((sent_a, sent_b, is_next))

    return pairs


class NSPHead:
    """Next Sentence Prediction classification head."""
    def __init__(self, hidden_size: int):
        self.W = np.random.randn(hidden_size, 2) * 0.02
        self.b = np.zeros(2)

    def forward(self, cls_hidden: np.ndarray) -> np.ndarray:
        return cls_hidden @ self.W + self.b


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)