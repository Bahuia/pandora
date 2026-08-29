import pandas as pd

from utils.kg_box_builder import KGBoxBuilder, Triple


def test_bidirectional_hop_traversal_and_materialization(tmp_path):
    triples = [
        Triple("alice", "type", "person"),
        Triple("alice", "born_in", "paris"),
        Triple("paris", "type", "city"),
        Triple("paris", "country", "france"),
    ]
    builder = KGBoxBuilder(triples)
    subgraph = builder.extract_subgraph(["alice"], {"born_in", "country"}, max_hops=2)
    assert len(subgraph) == 2
    schema, foreign_keys = builder.materialize(subgraph, tmp_path)
    assert "person = pd.DataFrame" in schema
    assert "city = pd.DataFrame" in schema
    assert (tmp_path / "person.csv").exists()
    assert pd.read_csv(tmp_path / "person.csv").iloc[0]["born_in"] == "paris"
    assert foreign_keys
