from __future__ import annotations


class FrozenRelayReleaseGate:
    """Frozen v9 counterfactual Relay release semantics."""

    def __init__(
        self,
        *,
        release_threshold: float = 0.95,
        stable_release_steps: int = 3,
    ):
        self.release_threshold = float(
            release_threshold
        )

        self.stable_release_steps = int(
            stable_release_steps
        )

        self.stable_counter = 0
        self.release_count = 0

        self.last_counterfactual_connectivity = 1.0
        self.last_counterfactual_target_ratio = 1.0

    def reset_stability(self) -> None:
        self.stable_counter = 0

    def observe(
        self,
        *,
        assignments,
        positions,
        relay_builder,
        base_position,
    ) -> bool:
        """Return True exactly when all Relay assignments should release."""

        if not assignments:
            self.stable_counter = 0
            return False

        relay_uav_ids = {
            int(helper_uid)
            for helper_uid, _anchor
            in assignments.values()
        }

        target_uav_ids = [
            int(target_uid)
            for target_uid
            in assignments
        ]

        counterfactual = relay_builder.snapshot(
            positions,
            base_position=base_position,
            exclude_uav_ids=relay_uav_ids,
        )

        self.last_counterfactual_connectivity = float(
            counterfactual.connectivity_ratio
        )

        connected_ids = set(
            counterfactual.connected_uav_ids
        )

        connected_targets = 0

        for target_uid in target_uav_ids:
            # Defensive parity with frozen v9.
            if target_uid in relay_uav_ids:
                continue

            if target_uid in connected_ids:
                connected_targets += 1

        if target_uav_ids:
            self.last_counterfactual_target_ratio = (
                float(connected_targets)
                / float(len(target_uav_ids))
            )
        else:
            self.last_counterfactual_target_ratio = 1.0

        targets_recovered = (
            connected_targets
            == len(target_uav_ids)
        )

        stable = (
            counterfactual.connectivity_ratio
            >= self.release_threshold
            and targets_recovered
        )

        if stable:
            self.stable_counter += 1

            if (
                self.stable_counter
                >= self.stable_release_steps
            ):
                self.release_count += 1
                self.stable_counter = 0
                return True

        else:
            self.stable_counter = 0

        return False
