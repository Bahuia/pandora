from core.voting import VoteAggregator


def test_tie_resolution_is_independent_of_completion_order():
    first = [
        {"vote_id": 2, "answer": [["b"]]},
        {"vote_id": 1, "answer": [["a"]]},
    ]
    second = list(reversed(first))
    assert VoteAggregator().aggregate(first)["answer"] == VoteAggregator().aggregate(second)["answer"]
