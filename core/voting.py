#!/usr/bin/env python3
"""
Pandora Voting Module

Majority voting aggregation for multi-vote inference with parallel execution.
"""

import hashlib
from collections import Counter
from typing import Any, Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
sys.path.append("..")
from utils.logger import setup_logger


class VoteAggregator:
    """
    Aggregates results from multiple inference votes using majority voting.

    Features:
    - Hash-based result comparison
    - Resolves ties deterministically by result hash and vote ID
    - Provides vote distribution statistics
    """

    def __init__(self):
        self.logger = setup_logger("pandora.voting")

    def aggregate(self, votes: List[dict]) -> dict[str, Any]:
        """
        Aggregate votes using majority voting based on result hashing.

        Votes with answer=None or empty answer are EXCLUDED from voting.
        If all votes fail, returns the first vote with error info.

        Args:
            votes: List of vote result dictionaries, each containing:
                   - example_id: str
                   - answer: list (the predicted answer)
                   - Other metadata...

        Returns:
            Aggregated result with winning vote's data plus:
            - vote_count: Number of votes for winning result
            - total_votes: Total number of votes
            - vote_distribution: Distribution of votes
        """
        if not votes:
            return {"success": False, "error": "No votes to aggregate"}

        # Completion order is nondeterministic under ThreadPoolExecutor.
        votes = sorted(votes, key=lambda vote: int(vote.get("vote_id", 0)))

        # Filter out failed votes (answer is None or empty)
        valid_votes = [v for v in votes if v.get("answer") is not None and v.get("answer") != []]
        failed_votes = [v for v in votes if v.get("answer") is None or v.get("answer") == []]

        if not valid_votes:
            # All votes failed — return first failed vote with error info
            self.logger.warning(f"All {len(votes)} votes failed, returning first failed vote")
            winner = failed_votes[0].copy()
            winner["vote_count"] = 0
            winner["total_votes"] = len(votes)
            winner["failed_votes"] = len(failed_votes)
            winner["success"] = False
            return winner

        if len(valid_votes) == 1:
            # Only one valid vote — return it
            winner = valid_votes[0].copy()
            winner["vote_count"] = 1
            winner["total_votes"] = len(votes)
            winner["failed_votes"] = len(failed_votes)
            winner["vote_distribution"] = {str(winner.get("vote_id", "unknown")): 100.0}
            return winner

        # Hash results for comparison (valid votes only)
        result_hashes = []
        for vote in valid_votes:
            result_hash = self._hash_result(vote.get("answer"))
            result_hashes.append((result_hash, vote))

        # Count votes
        hash_counts = Counter(h[0] for h in result_hashes)

        # Resolve count ties by hash so repeated runs aggregate identically.
        winning_hash = sorted(hash_counts, key=lambda h: (-hash_counts[h], h))[0]

        # Get all votes with winning hash
        winning_votes = [v for h, v in result_hashes if h == winning_hash]

        # Select the first winning vote as representative
        winner = winning_votes[0].copy()
        winner["vote_count"] = hash_counts[winning_hash]
        winner["total_votes"] = len(votes)
        winner["failed_votes"] = len(failed_votes)

        # Calculate vote distribution (including failed votes as a group)
        vote_distribution = {}
        if failed_votes:
            vote_distribution["failed"] = len(failed_votes) / len(votes) * 100
        for h, count in hash_counts.items():
            percentage = count / len(votes) * 100
            # Find a representative vote ID for this hash
            rep_vote = next(v for h2, v in result_hashes if h2 == h)
            vote_distribution[f"vote_{rep_vote.get('vote_id', 'unknown')}"] = percentage

        winner["vote_distribution"] = vote_distribution

        self.logger.info(
            f"Voting complete: {hash_counts[winning_hash]}/{len(votes)} votes "
            f"({hash_counts[winning_hash]/len(votes)*100:.1f}%) for winning result "
            f"({len(failed_votes)} failed votes excluded)"
        )

        return winner

    def get_vote_distribution(self, votes: List[dict]) -> dict:
        """
        Get detailed vote distribution statistics.

        Args:
            votes: List of vote result dictionaries

        Returns:
            Distribution dictionary with hash counts and percentages
        """
        if not votes:
            return {}

        result_hashes = [self._hash_result(vote.get("answer")) for vote in votes]
        hash_counts = Counter(result_hashes)

        distribution = {
            "total_votes": len(votes),
            "unique_results": len(hash_counts),
            "counts": {h: count for h, count in hash_counts.items()},
            "percentages": {
                h: f"{count / len(votes) * 100:.1f}%"
                for h, count in hash_counts.items()
            },
        }

        return distribution

    def _hash_result(self, result: Any) -> str:
        """
        Create a hash of a result for comparison.

        Normalizes the result to ensure consistent hashing regardless of:
        - List ordering (sorts for consistency)
        - Type differences (converts to string)

        Args:
            result: Result to hash (typically list of lists)

        Returns:
            MD5 hash string
        """
        if result is None:
            return "null"

        try:
            # Normalize result for hashing
            if isinstance(result, list):
                # Convert to sorted tuple of tuples for consistent hashing
                normalized = tuple(
                    sorted(
                        tuple(str(x) for x in item) if isinstance(item, list) else str(item)
                        for item in result
                    )
                )
            else:
                normalized = str(result)

            # Create MD5 hash
            result_str = str(normalized)
            return hashlib.md5(result_str.encode()).hexdigest()

        except Exception as e:
            self.logger.warning(f"Failed to hash result: {e}")
            return str(hash(str(result)))


def run_with_voting(
    agent,
    example: dict,
    n_votes: int = 5,
    shot_k: int = 10,
    max_workers: int = 4
) -> dict[str, Any]:
    """
    Run inference with majority voting using parallel execution.

    This function runs the agent's inference `n_votes` times in parallel,
    then aggregates the results using majority voting.

    Args:
        agent: PandoraAgent instance
        example: Example dictionary containing:
                 - question: str
                 - db_id: str
                 - gold_answer: list (optional)
                 - Other metadata...
        n_votes: Number of parallel inference runs (default: 5)
        shot_k: Number of few-shot examples (default: 10)
        max_workers: Maximum number of parallel workers (default: 4)

    Returns:
        Aggregated result dictionary containing:
        - example_id: str
        - answer: list (winning answer)
        - vote_count: int (votes for winning answer)
        - total_votes: int
        - vote_distribution: dict
        - All individual vote results
    """
    example_id = example.get('id', example.get('question_id', 'unknown'))
    logger = setup_logger("pandora.voting")
    logger.info(f"Running {n_votes}-vote inference for example {example_id}")

    votes = []

    # Run inference in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all votes
        future_to_vote = {
            executor.submit(agent.run, example, shot_k): i + 1
            for i in range(n_votes)
        }

        # Collect results as they complete
        for future in as_completed(future_to_vote):
            vote_id = future_to_vote[future]
            try:
                result = future.result()
                result["vote_id"] = vote_id
                votes.append(result)
                logger.info(f"Vote {vote_id}/{n_votes} completed")
            except Exception as e:
                logger.error(f"Vote {vote_id} failed: {e}")
                votes.append({
                    "vote_id": vote_id,
                    "success": False,
                    "error": str(e),
                    "answer": None
                })

    if not votes:
        return {
            "example_id": example_id,
            "success": False,
            "error": "All votes failed",
            "n_votes": n_votes
        }

    # Aggregate votes
    aggregator = VoteAggregator()
    aggregated = aggregator.aggregate(votes)
    aggregated["example_id"] = example_id
    aggregated["n_votes"] = n_votes
    aggregated["all_votes"] = sorted(votes, key=lambda vote: int(vote.get("vote_id", 0)))

    return aggregated
