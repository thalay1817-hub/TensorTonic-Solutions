def canonical_preference_pairs(records):
    pairs = []

    for record in records:
        prompt_id = record["prompt_id"]
        candidates = record["candidates"]

        # Sort a copy by ascending rank; do not mutate the original list
        sorted_candidates = sorted(candidates, key=lambda c: c["rank"])

        n = len(sorted_candidates)
        for winner_index in range(n):
            winner_id = sorted_candidates[winner_index]["id"]
            for loser_index in range(winner_index + 1, n):
                loser_id = sorted_candidates[loser_index]["id"]
                pairs.append({
                    "prompt_id": prompt_id,
                    "winner_id": winner_id,
                    "loser_id": loser_id,
                })

    return {"pairs": pairs}
