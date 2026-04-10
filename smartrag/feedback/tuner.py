"""RetrievalTuner — self-tuning RRF weights from accumulated feedback."""

from __future__ import annotations

import json
import time


class RetrievalTuner:
    """Adjust retrieval ranking weights based on feedback data."""

    def __init__(self, base_weights: dict[str, float]):
        self._base_weights = dict(base_weights)
        self._previous_weights: dict[str, float] = dict(base_weights)

    def tune(self, feedback_store) -> dict[str, float] | None:
        """Compute new weights from feedback. Returns None if insufficient data."""
        data = feedback_store.get_tuning_data(min_queries=50)
        if data is None:
            return None

        win_rates = data["win_rates"]
        if not win_rates:
            return None

        avg_win_rate = sum(win_rates.values()) / len(win_rates)

        new_weights: dict[str, float] = {}
        for source, base_w in self._base_weights.items():
            wr = win_rates.get(source, avg_win_rate)
            calculated = base_w * (1 + 0.5 * (wr - avg_win_rate))
            # Clamp
            calculated = max(0.1, min(3.0, calculated))
            # Smooth
            prev = self._previous_weights.get(source, base_w)
            new_weights[source] = 0.7 * calculated + 0.3 * prev

        # Clamp again after smoothing
        for source in new_weights:
            new_weights[source] = max(0.1, min(3.0, new_weights[source]))

        # Store in tuning_history
        feedback_store._conn.execute(
            """INSERT INTO tuning_history (timestamp, weights, data_points, trigger)
               VALUES (?, ?, ?, ?)""",
            (
                time.time(),
                json.dumps(new_weights),
                data["data_points"],
                "auto",
            ),
        )
        feedback_store._conn.commit()

        self._previous_weights = dict(new_weights)
        return new_weights
