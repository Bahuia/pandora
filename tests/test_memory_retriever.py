import numpy as np

from utils.memory_retriever import MemoryExample, SemanticMemoryRetriever


class FakeEncoder:
    def encode(self, sentences, **kwargs):
        vectors = []
        for sentence in sentences:
            text = sentence.casefold()
            vectors.append([float("count" in text), float("language" in text), 0.1])
        return np.asarray(vectors, dtype=np.float32)


def make_example(dataset, example_id, question):
    return MemoryExample(dataset, example_id, question, "schema", "reason", "result = []", "test")


def test_cross_task_and_same_dataset_retrieval():
    examples = [
        make_example("spider", "1", "count the records"),
        make_example("webqsp", "2", "which language is spoken"),
    ]
    retriever = SemanticMemoryRetriever(examples, encoder=FakeEncoder())
    cross = retriever.retrieve("count all rows", 1, "webqsp", "cross_task")
    same = retriever.retrieve("count all rows", 1, "webqsp", "same_dataset")
    assert cross[0][0].dataset_id == "spider"
    assert same[0][0].dataset_id == "webqsp"
