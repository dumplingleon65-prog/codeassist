"""
Dense reward mixer + sparse rewards mixer for CodeAssistZero.
"""

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, Any, List, Optional, Set, Tuple

ACTION_TO_IDX = {
    "NO-OP": 0,
    "Fill Partial Line": 1,
    "Write Single Line Code": 2,
    "Write Multi Line Code": 3,
    "Edit Existing Lines": 4,
    "Explain Single Lines": 5,
    "Explain Multi Lines": 6,
}
IDX_TO_ACTION = {v: k for k, v in ACTION_TO_IDX.items()}


@dataclass
class RewardWeights:
    """Configurable weights for different reward components"""

    # Test and compilation rewards
    test_pass_delta: float = 1.0  # Reward for passing tests
    test_regress: float = -1.0  # Penalty for breaking tests
    compile_fix: float = 0.75  # Reward for fixing compilation
    compile_break: float = -0.5  # Penalty for breaking compilation
    compile_maintain: float = 0.1  # Small reward for maintaining compilation
    # Interaction rewards
    undo_penalty: float = -0.75  # Penalty when human undoes/takes over a significant portion of the assistant work
    retain_reward: float = (
        0.25  # Bonus when assistant contributions survive the next human turn
    )
    extinction_penalty: float = (
        -0.25
    )  # Delayed penalty if assistant work fails to survive to episode end
    survival_reward: float = (
        1.0  # Delayed reward if assistant work survives to episode end
    )
    # TODO: If we can figure out a clean way of simulating explicit undo flags, we should consider a special penalty for that since it "scales differently" than implicit undos (i.e., both good and bad lines may get taken over without regard for their actual contents)
    # Shaping rewards (telescoping NO-OP)
    noop_shaping: float = 0.05  # Initial positive reward for short NO-OP sequences
    noop_positive_warmup: int = (
        1  # Number of consecutive NO-OPs that remain (decreasingly) positive
    )
    noop_penalty_base: float = 0.05  # Base magnitude once sequence turns negative
    noop_penalty_growth: float = (
        1.5  # Multiplicative growth per additional NO-OP beyond warmup
    )
    noop_penalty_cap: float = 5.0  # Cap on negative magnitude before clipping


@dataclass
class RewardConfig:
    weights: RewardWeights = field(
        default_factory=RewardWeights
    )  # Use default_factory for mutable defaults
    lang: str = "python"
    debug_mode: bool = False  # Enable detailed reward logging
    normalize_by_episode_length: bool = True  # Normalize rewards by episode length
    reward_clipping: Tuple[float, float] = (
        -5.0,
        5.0,
    )  # Clip rewards to prevent explosion


class RewardMixer:
    def __init__(self, cfg: Optional[RewardConfig] = None):
        self.cfg = cfg if cfg is not None else RewardConfig()
        self.episode_step_count = 0
        self.reward_history = []
        self.last_debug_info = {}
        # Reward memory parameters for shaping behavior
        self.noop_sequence_length = 0  # For tracking consecutive NO-OPs

    def reset_episode(self):
        self.episode_step_count = 0
        self.reward_history = []
        self.last_debug_info = {}
        self.noop_sequence_length = 0

    def _debug(self, message: str) -> None:
        if self.cfg.debug_mode:
            print(f"[RewardMixer] {message}")

    @staticmethod
    def _format_line(line: Any, limit: int = 80) -> str:
        if not isinstance(line, str):
            return "<non-str>"
        text = line.replace("\n", "\\n")
        if len(text) > limit:
            text = f"{text[: limit - 3]}..."
        return text

    def step_reward(
        self,
        prev_state: Dict[str, Any],
        action_idx: int,
        next_state: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Compute step-wise reward with improved reward shaping

        Args:
            prev_state: Previous state dict
            action_idx: Action taken (0-6)
            next_state: Resulting state dict
            context: Optional context with additional info (sparse signals, special flags, etc.)
        """
        w = self.cfg.weights
        total_reward = 0.0
        reward_components = {}  # For debugging/logging

        # Get environment info
        prev_env = prev_state.get("env", {})
        next_env = next_state.get("env", {})

        # Compilation rewards
        compile_reward = self._compute_compile_rewards(prev_env, next_env, w)
        total_reward += compile_reward
        reward_components["compile"] = compile_reward

        # Test rewards
        test_reward = self._compute_test_rewards(prev_env, next_env, w)
        total_reward += test_reward
        reward_components["test"] = test_reward

        # Myopic interaction rewards (e.g., undo/deletion penalties due to immediate human feedback)
        interaction_reward = self._compute_interaction_rewards(
            prev_state, next_state, action_idx, w, mode="myopic"
        )
        total_reward += interaction_reward
        reward_components["interaction"] = interaction_reward

        # Survival interaction rewards (e.g. undo/deletion penalties due to human feedback on final state)
        final_state = None
        if context:
            final_state = context.get("episode_final_state") or context.get(
                "final_state"
            )
        survival_reward = 0.0
        if isinstance(final_state, dict):
            survival_reward = self._compute_interaction_rewards(
                prev_state, final_state, action_idx, w, mode="survival"
            )
            total_reward += survival_reward
        reward_components["survival"] = survival_reward

        # Diminishing NOOP shaping reward/penalty
        if IDX_TO_ACTION.get(action_idx, "Unknown") == "NO-OP":
            self.noop_sequence_length += 1
            noop_shaping_reward = self._compute_noop_shaping_rewards(w)
        else:
            self.noop_sequence_length = 0
            noop_shaping_reward = 0.0
        total_reward += noop_shaping_reward
        reward_components["noop_shaping"] = noop_shaping_reward

        # TODO: Add in code quality rewards (e.g., linting, complexity)
        # TODO: Add in code style rewards (e.g., lines aren't too long, good comment-to-code ratio)
        # TODO: Add a proximity penalty if actions are taken too far from the cursor
        # TODO: Add a small boost for actions that lead to non-trivially faster test execution times

        # Apply reward clipping
        total_reward = max(
            self.cfg.reward_clipping[0], min(self.cfg.reward_clipping[1], total_reward)
        )

        # Debugging/logging info
        self.episode_step_count += 1
        self.reward_history.append(total_reward)
        self.last_debug_info = reward_components.copy()
        self.last_debug_info["total"] = total_reward
        self.last_debug_info["step"] = self.episode_step_count

        if self.cfg.debug_mode:
            print("-> Reward Debug Info:")
            print(f"----> Step {self.episode_step_count} reward: {total_reward:.3f}")
            for component, value in reward_components.items():
                print(f"--------> {component}: {value:.3f}")
            print()
        return float(total_reward)

    def get_last_debug_info(self) -> Dict[str, Any]:
        """Get debug information from the last reward computation"""
        return self.last_debug_info.copy()

    @staticmethod
    def _safe_int(value: Any, default: int = -1) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _span_len(span: Any) -> int:
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            return 0
        try:
            start = int(span[0])
            end = int(span[1])
        except (TypeError, ValueError):
            return 0
        return max(0, end - start)

    @staticmethod
    def _span_overlap(span_a: Any, span_b: Any) -> int:
        if not isinstance(span_a, (list, tuple)) or not isinstance(
            span_b, (list, tuple)
        ):
            return 0
        if len(span_a) != 2 or len(span_b) != 2:
            return 0
        try:
            a_start, a_end = int(span_a[0]), int(span_a[1])
            b_start, b_end = int(span_b[0]), int(span_b[1])
        except (TypeError, ValueError):
            return 0
        overlap_start = max(a_start, b_start)
        overlap_end = min(a_end, b_end)
        return max(0, overlap_end - overlap_start)

    @staticmethod
    def _line_similarity(source: Any, target: Any) -> float:
        if not isinstance(source, str) or not isinstance(target, str):
            return 0.0
        src = source.strip()
        tgt = target.strip()
        if not src and not tgt:
            return 1.0
        if not src or not tgt:
            return 0.0
        return SequenceMatcher(None, src, tgt).ratio()

    @staticmethod
    def _map_line_indices(
        source_lines: Any, target_lines: Any, min_similarity: float = 1.0
    ) -> Dict[int, int]:
        """
        Map line indices from source to target by tracking identical or similar content.

        The SequenceMatcher-based alignment handles insertions/deletions before the
        tracked line so assistant ownership follows the line when it shifts. When
        ``min_similarity`` is lower than ``1.0`` we attempt to align lines based on
        textual similarity, which enables tracking moved-but-tweaked content.
        """
        if not isinstance(source_lines, (list, tuple)) or not isinstance(
            target_lines, (list, tuple)
        ):
            return {}

        matcher = SequenceMatcher(
            a=list(source_lines), b=list(target_lines), autojunk=False
        )
        mapping: Dict[int, int] = {}
        used_targets: Set[int] = set()
        for block in matcher.get_matching_blocks():
            for offset in range(block.size):
                src_idx = block.a + offset
                tgt_idx = block.b + offset
                mapping[src_idx] = tgt_idx
                used_targets.add(tgt_idx)

        if min_similarity >= 1.0:
            return mapping

        unmatched_sources = [
            idx for idx in range(len(source_lines)) if idx not in mapping
        ]
        if not unmatched_sources:
            return mapping

        unmatched_targets = [
            idx for idx in range(len(target_lines)) if idx not in used_targets
        ]
        for src_idx in unmatched_sources:
            best_idx: Optional[int] = None
            best_score = 0.0
            source_line = source_lines[src_idx]
            for tgt_idx in unmatched_targets:
                score = RewardMixer._line_similarity(source_line, target_lines[tgt_idx])
                if score > best_score:
                    best_score = score
                    best_idx = tgt_idx
            if best_idx is not None and best_score >= min_similarity:
                mapping[src_idx] = best_idx
                used_targets.add(best_idx)
                unmatched_targets.remove(best_idx)
        return mapping

    @staticmethod
    def _trim_line_span(line: str, span: Any) -> Tuple[int, int]:
        if not isinstance(line, str):
            return (0, 0)
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            return (0, 0)
        n = len(line)
        try:
            start = int(span[0])
            end = int(span[1])
        except (TypeError, ValueError):
            return (0, 0)
        start = max(0, min(n, start))
        end = max(start, min(n, end))
        while start < end and line[start].isspace():
            start += 1
        while end > start and line[end - 1].isspace():
            end -= 1
        return (start, end)

    @staticmethod
    def _has_signal(lines: Any, line_idx: int, span: Tuple[int, int]) -> bool:
        if (
            not isinstance(lines, (list, tuple))
            or line_idx < 0
            or line_idx >= len(lines)
        ):
            return False
        if not isinstance(span, tuple) or len(span) != 2:
            return False
        start, end = span
        if end <= start:
            return False
        segment = lines[line_idx][start:end]
        return any(not ch.isspace() for ch in segment)

    def _compute_test_rewards(
        self, prev_env: Dict, next_env: Dict, w: RewardWeights, binary: bool = True
    ) -> float:
        """Compute test-related rewards"""
        prev_tests = prev_env.get("tests", {})
        next_tests = next_env.get("tests", {})

        prev_passed = int(prev_tests.get("passed", 0))
        next_passed = int(next_tests.get("passed", 0))

        prev_total = int(prev_tests.get("total", 0) or 0)
        next_total = int(next_tests.get("total", 0) or 0)
        if prev_total and next_total and prev_total != next_total:
            # Some zero-style episodes only execute a subset of tests on certain timesteps.
            # Rather than failing, if verbose, pick the larger pool so proportional rewards remain bounded.
            if self.verbose:
                print(
                    f"[RewardMixer] Warning: total tests changed from {prev_total} to {next_total}; "
                    "using max for reward computation."
                )
            else:
                raise ValueError(
                    f"Total tests changed from {prev_total} to {next_total}"
                )
        total_tests = max(prev_total, next_total)

        reward = 0.0
        if next_passed > prev_passed:  # Tests improved
            if binary:  # Binary reward: +w.test_pass_delta for any improvement
                reward += w.test_pass_delta
            else:  # Proportional reward based on number of tests improved
                improvement = next_passed - prev_passed
                reward += w.test_pass_delta * (
                    improvement / total_tests if total_tests > 0 else improvement
                )
        elif next_passed < prev_passed:  # Tests regressed
            if binary:  # Binary penalty: -w.test_regress for any regression
                reward += w.test_regress
            else:  # Proportional penalty based on number of tests regressed
                regression = prev_passed - next_passed
                reward += w.test_regress * (
                    regression / total_tests if total_tests > 0 else regression
                )
        # TODO: Add stagnation penalty?
        return reward

    def _compute_compile_rewards(
        self, prev_env: Dict, next_env: Dict, w: RewardWeights
    ) -> float:
        """Compute compilation-related rewards"""
        prev_compiled = bool(prev_env.get("compiled", False))
        next_compiled = bool(next_env.get("compiled", False))

        reward = 0.0

        if not prev_compiled and next_compiled:
            # Fixed compilation
            reward += w.compile_fix
        elif prev_compiled and not next_compiled:
            # Broke compilation
            reward += w.compile_break
        elif next_compiled:
            # Maintained compilation
            reward += w.compile_maintain

        # TODO: Add stagnation penalty?
        return reward

    def _compute_interaction_rewards(
        self,
        prev_state: Dict,
        next_state: Dict,
        action_idx: int,
        w: RewardWeights,
        takeover_threshold: float = 0.7,
        retention_threshold: float = 0.7,
        mode: str = "myopic",
    ) -> float:
        """
        Compute interaction rewards, e.g. penalties for human undoing assistant actions
        Mode can be 'myopic' (next state only) or 'survival' (final state comparison)
        """
        reward = 0.0

        # Get line attribution data
        prev_attribs = prev_state.get("line_attribs", {})
        next_attribs = next_state.get("line_attribs", {})

        if not prev_attribs or not next_attribs:
            return reward

        prev_assistant = prev_attribs.get("A", []) or []
        prev_human = prev_attribs.get("H", []) or []
        next_human = next_attribs.get("H", []) or []
        prev_lines = prev_state.get("lines_text", []) or []
        next_lines = next_state.get("lines_text", []) or []

        # Track assistant-owned content even if the human inserts or deletes lines before it.
        line_map = self._map_line_indices(
            prev_lines, next_lines, min_similarity=takeover_threshold
        )
        if self.cfg.debug_mode:
            self._debug(f"{mode} interaction line map: {line_map}")

        undo_count = 0
        retained_count = 0

        for prev_idx, prev_A in enumerate(prev_assistant):
            prev_H = prev_human[prev_idx] if prev_idx < len(prev_human) else {}
            prev_A_time = self._safe_int(prev_A.get("t_last", -1))
            prev_H_time = self._safe_int(prev_H.get("t_last", -1))

            if prev_A_time <= prev_H_time:
                continue

            prev_line = prev_lines[prev_idx] if prev_idx < len(prev_lines) else ""
            prev_line_repr = self._format_line(prev_line)  # Debug only
            prev_A_span = self._trim_line_span(prev_line, prev_A.get("span", [0, 0]))
            assist_span_len = self._span_len(prev_A_span)
            if assist_span_len <= 0:
                continue

            prev_has_signal = self._has_signal(prev_lines, prev_idx, prev_A_span)
            if not prev_has_signal:  # No meaningful assistant content, skip
                continue

            next_idx = line_map.get(prev_idx)
            if next_idx is None or next_idx < 0 or next_idx >= len(next_lines):
                undo_count += 1
                self._debug(
                    f"{mode} interaction: prev line {prev_idx} lost in next state; counting undo. "
                    f"assistant_line='{prev_line_repr}'"
                )
                continue

            next_line = next_lines[next_idx]
            next_line_repr = self._format_line(next_line)  # For debugging only
            next_H = next_human[next_idx] if next_idx < len(next_human) else {}
            next_H_time = self._safe_int(next_H.get("t_last", -1))
            similarity = self._line_similarity(prev_line, next_line)

            if next_H_time > prev_A_time:  # Human has edited this line after assistant
                if prev_line.strip() == next_line.strip():
                    retained_count += 1
                    self._debug(
                        f"{mode} interaction: prev line {prev_idx} moved to {next_idx} with identical content; "
                        f"counting retention. line='{prev_line_repr}'"
                    )
                    continue

                if similarity < takeover_threshold:
                    undo_count += 1
                    self._debug(
                        f"{mode} interaction: human takeover on prev line {prev_idx} -> next {next_idx} "
                        f"(similarity={similarity:.2f}). assistant_line='{prev_line_repr}' "
                        f"human_line='{next_line_repr}'"
                    )
                elif similarity >= retention_threshold:
                    retained_count += 1
                    self._debug(
                        f"{mode} interaction: prev line {prev_idx} retained after human touch at {next_idx} "
                        f"(similarity={similarity:.2f}). assistant_line='{prev_line_repr}' "
                        f"updated_line='{next_line_repr}'"
                    )
                else:
                    self._debug(
                        f"{mode} interaction: prev line {prev_idx} partially edited at {next_idx}; "
                        f"no reward change (similarity={similarity:.2f}). assistant_line='{prev_line_repr}' "
                        f"updated_line='{next_line_repr}'"
                    )
                continue

            # No human takeover; treat the line as retained by default.
            retained_count += 1
            if self.cfg.debug_mode:
                self._debug(
                    f"{mode} interaction: prev line {prev_idx} cleanly retained at {next_idx} "
                    f"(similarity={similarity:.2f}). assistant_line='{prev_line_repr}' "
                    f"retained_line='{next_line_repr}'"
                )
        # Apply penalties based on undos detected
        if undo_count > 0:
            if mode == "myopic":
                reward += w.undo_penalty  # * undo_count
            else:
                reward += w.extinction_penalty  # * undo_count
            self._debug(
                f"{mode} interaction undo_count={undo_count}, applied penalty {w.undo_penalty}."
            )
        if retained_count > 0:
            if mode == "myopic":
                reward += w.retain_reward  # * retained_count
            else:
                reward += w.survival_reward  # * retained_count
            self._debug(
                f"{mode} interaction retained_count={retained_count}, applied reward {w.retain_reward}."
            )
        return reward

    def _compute_noop_shaping_rewards(self, w: RewardWeights) -> float:
        """Telescoping shaping signal for consecutive NO-OP actions.

        Short sequences keep a small positive reward that linearly decays across
        the warmup window. Once the streak exceeds the warmup, the reward turns
        negative and its magnitude grows multiplicatively until capped. Any
        non-NO-OP action resets the streak externally.
        """

        length = self.noop_sequence_length
        if length <= 0:
            return 0.0

        warmup = max(0, int(getattr(w, "noop_positive_warmup", 0)))
        base_reward = float(getattr(w, "noop_shaping", 0.0))

        if warmup > 0 and length <= warmup:
            progress = (length - 1) / warmup
            scale = max(0.0, 1.0 - progress)
            return base_reward * scale

        # Beyond warmup: transition to penalties that grow with streak length.
        steps_into_penalty = length - (warmup if warmup > 0 else 1)
        steps_into_penalty = max(1, steps_into_penalty)

        penalty_base = float(getattr(w, "noop_penalty_base", base_reward))
        growth = float(getattr(w, "noop_penalty_growth", 1.5))
        cap = float(getattr(w, "noop_penalty_cap", self.cfg.reward_clipping[1]))

        penalty = penalty_base * (growth ** (steps_into_penalty - 1))
        penalty = min(penalty, cap)
        return -penalty

    def get_episode_summary(self) -> Dict[str, Any]:
        """Get summary of rewards for this episode"""
        if not self.reward_history:
            return {}

        return {
            "total_steps": len(self.reward_history),
            "total_reward": sum(self.reward_history),
            "average_reward": sum(self.reward_history) / len(self.reward_history),
            "max_reward": max(self.reward_history),
            "min_reward": min(self.reward_history),
            "positive_steps": sum(1 for r in self.reward_history if r > 0),
            "negative_steps": sum(1 for r in self.reward_history if r < 0),
        }
