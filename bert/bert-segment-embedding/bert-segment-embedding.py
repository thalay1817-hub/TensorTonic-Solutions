import numpy as np

class BertEmbeddings:
    """
    BERT Embeddings = Token + Position + Segment
    """
    def __init__(self, vocab_size: int, max_position: int, hidden_size: int):
        self.hidden_size = hidden_size
        self.token_embeddings = np.random.randn(vocab_size, hidden_size) * 0.02
        self.position_embeddings = np.random.randn(max_position, hidden_size) * 0.02
        self.segment_embeddings = np.random.randn(2, hidden_size) * 0.02

    def forward(self, token_ids: np.ndarray, segment_ids: np.ndarray) -> np.ndarray:
        batch_size, seq_len = token_ids.shape

        token_emb = self.token_embeddings[token_ids]           # (batch, seq, hidden)
        position_emb = self.position_embeddings[np.arange(seq_len)]  # (seq, hidden)
        segment_emb = self.segment_embeddings[segment_ids]     # (batch, seq, hidden)

        return token_emb + position_emb + segment_emb