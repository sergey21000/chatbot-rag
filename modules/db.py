from abc import ABC, abstractmethod

import chromadb
from chromadb import EmbeddingFunction, QueryResult, Collection


class DbAbc(ABC):
    @abstractmethod
    def create_collection_and_add_texts(self, texts: list[str], *args, **kwargs):
        pass

    @abstractmethod
    def search_similar_texts(self, *args, **kwargs):
        pass


class ChromaDb(DbAbc):
    '''Database for texts, embeddings and vector search'''
    def __init__(self):
        settings = chromadb.config.Settings(anonymized_telemetry=False, allow_reset=True)
        self.client = chromadb.EphemeralClient(settings=settings)

    def collection_exists(self, collection_name: str) -> bool:
        collections = [collection.name for collection in self.client.list_collections()]
        return collection_name in collections

    def delete_collection(self, collection_name: str) -> None:
        if self.collection_exists(collection_name):
            self.client.delete_collection(name=collection_name)

    def get_collection(
            self,
            collection_name: str,
            embedding_function: EmbeddingFunction,
    ) -> Collection | None:
        if self.collection_exists(collection_name):
            collection = self.client.get_collection(
                name=collection_name,
                embedding_function=embedding_function,
            )
            return collection

    def create_collection_and_add_texts(
        self,
        collection_name: str,
        texts: list[str],
        embedding_function: EmbeddingFunction,
        create_collection_kwargs: dict[str, str | int],
    ) -> None:
        self.delete_collection(collection_name)
        collection = self.client.create_collection(
            name=collection_name,
            embedding_function=embedding_function,
            **create_collection_kwargs,
        )
        ids = list(map(str, range(len(texts))))
        collection.add(
            ids=ids,
            documents=texts,
        )

    @classmethod
    def search_similar_texts(
        cls,
        collection: str,
        query_text: str,
        query_kwargs: dict,
    ) -> QueryResult:
        n_results = query_kwargs['n_results']
        max_distance_treshold = query_kwargs['max_distance_treshold']
        if n_results == 'all':
            results = collection.get()
            return results
        elif n_results == 'max':
            n_results = collection.count()
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
        )
        filtered_results = cls.filter_by_distance(
            results=results,
            max_distance_treshold=max_distance_treshold,
        )
        return filtered_results

    @staticmethod
    def filter_by_distance(
        results: QueryResult,
        max_distance_treshold: float,
    ) -> dict[str, str | float]:
        filtered_results = {k: [] for k in results}
        for i in range(len(results['ids'][0])):
            if results['distances'][0][i] <= max_distance_treshold:
                filtered_results['ids'].append(results['ids'][0][i])
                filtered_results['documents'].append(results['documents'][0][i])
                filtered_results['distances'].append(results['distances'][0][i])
                filtered_results['metadatas'].append(results['metadatas'][0][i])
        return filtered_results