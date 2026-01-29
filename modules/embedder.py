from typing import Any

from sentence_transformers import SentenceTransformer
from chromadb import Documents, EmbeddingFunction, Embeddings


class SentenceTransformerEF(EmbeddingFunction):
    '''Class for receiving embeddings, passed as a function to the `create_collection` method'''
    def __init__(
        self,
        model_name_or_path: str,
        model_kwargs: dict[str, Any],
        encode_kwargs: dict[str, Any],
    ):
        # if not torch.cuda.is_available():
        #     model_kwargs['device'] = 'cpu'
        model_kwargs['model_name_or_path'] = model_name_or_path
        self.model = SentenceTransformer(**model_kwargs)
        self.encode_kwargs = encode_kwargs

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = self.model.encode(input, **self.encode_kwargs)
        return embeddings.tolist()

