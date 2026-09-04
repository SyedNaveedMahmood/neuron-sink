from __future__ import annotations

import unittest

import numpy as np
import torch

from neuron_sink.sink_metrics import (
    REGISTERED_SINK_FLOOR,
    SinkPreflightError,
    build_sink_scope,
    eligible_mlp_layers,
    future_sink_layers_by_mlp_layer,
    layer_scores,
    per_layer_head_position0_attention,
    select_sink_heavy_heads,
    select_sink_heavy_layers,
    sink_scalar_from_map,
    top_quartile_size,
)
from neuron_sink.upstream_bridge import is_available, sink_repro_module


def causal_attention(num_layers: int, num_heads: int, seq_len: int, seed: int = 0):
    """A batch of valid causal, row-normalised attention tensors."""

    generator = torch.Generator().manual_seed(seed)
    tensors = []
    for _ in range(num_layers):
        logits = torch.randn(num_heads, seq_len, seq_len, generator=generator)
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool).tril()
        logits = logits.masked_fill(~mask, float("-inf"))
        tensors.append(torch.softmax(logits, dim=-1))
    return tensors


class MapDecompositionTests(unittest.TestCase):
    def test_map_matches_hand_computed_values(self) -> None:
        seq_len = 4  # second half is queries 2 and 3
        attention = torch.zeros(2, seq_len, seq_len)
        # layer 0, head 0: attention to key 0 from queries 2, 3 = 0.5 and 0.1 -> mean 0.3
        attention[0, 2, 0] = 0.5
        attention[0, 3, 0] = 0.1
        # head 1: 0.2 and 0.4 -> mean 0.3000000...
        attention[1, 2, 0] = 0.2
        attention[1, 3, 0] = 0.4
        result = per_layer_head_position0_attention([attention])
        self.assertEqual(result.shape, (1, 2))
        self.assertAlmostEqual(result[0, 0], 0.3, places=6)
        self.assertAlmostEqual(result[0, 1], 0.3, places=6)

    def test_target_position_is_respected(self) -> None:
        attention = torch.zeros(1, 4, 4)
        attention[0, 2:, 1] = 1.0
        at_zero = per_layer_head_position0_attention([attention], target_pos=0)
        at_one = per_layer_head_position0_attention([attention], target_pos=1)
        self.assertAlmostEqual(at_zero[0, 0], 0.0, places=9)
        self.assertAlmostEqual(at_one[0, 0], 1.0, places=9)

    def test_accepts_batched_tensor_and_rejects_real_batches(self) -> None:
        attention = causal_attention(1, 2, 6)[0]
        batched = attention.unsqueeze(0)
        self.assertTrue(
            np.allclose(
                per_layer_head_position0_attention([attention]),
                per_layer_head_position0_attention([batched]),
            )
        )
        with self.assertRaises(ValueError):
            per_layer_head_position0_attention([attention.unsqueeze(0).repeat(2, 1, 1, 1)])

    def test_scalar_over_band_equals_direct_mean(self) -> None:
        attentions = causal_attention(12, 12, 40, seed=3)
        layer_head = per_layer_head_position0_attention(attentions)
        band = list(range(3, 11))
        direct = np.mean([
            attentions[layer][:, 20:, 0].mean().item() for layer in band
        ])
        self.assertAlmostEqual(sink_scalar_from_map(layer_head, band), float(direct), places=6)

    def test_scalar_rejects_out_of_range_layers(self) -> None:
        layer_head = per_layer_head_position0_attention(causal_attention(4, 2, 8))
        with self.assertRaises(IndexError):
            sink_scalar_from_map(layer_head, [4])
        with self.assertRaises(ValueError):
            sink_scalar_from_map(layer_head, [])

    def test_inconsistent_shapes_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            per_layer_head_position0_attention([
                torch.zeros(2, 4, 4), torch.zeros(2, 6, 6)
            ])


@unittest.skipUnless(
    is_available("sink_repro"), "upstream/sink-repro submodule is not checked out"
)
class UpstreamParityTests(unittest.TestCase):
    """The decomposition must reproduce the pinned upstream sink scalar exactly."""

    def test_band_scalar_matches_compute_bos_attention_metric(self) -> None:
        legacy = sink_repro_module("intervention_analysis_legacy")
        engine = sink_repro_module("nnsight_engine")
        num_layers, num_heads, seq_len = 12, 12, 40
        attentions = causal_attention(num_layers, num_heads, seq_len, seed=11)
        band_start, band_end = legacy.compute_band(num_layers, "scaled")
        self.assertEqual((band_start, band_end), (3, 11))

        upstream = legacy.compute_bos_attention_metric(
            attentions, num_layers, "mid", target_pos=0,
            layer_start=band_start, layer_end=band_end,
        )
        mine = sink_scalar_from_map(
            per_layer_head_position0_attention(attentions),
            list(range(band_start, band_end)),
        )
        limit = engine.METRIC_ATOL + engine.METRIC_RTOL * abs(upstream)
        self.assertLessEqual(abs(mine - upstream), limit)

    def test_all_layer_scope_also_matches(self) -> None:
        legacy = sink_repro_module("intervention_analysis_legacy")
        engine = sink_repro_module("nnsight_engine")
        attentions = causal_attention(12, 12, 40, seed=5)
        upstream = legacy.compute_bos_attention_metric(
            attentions, 12, "all", target_pos=0
        )
        mine = sink_scalar_from_map(per_layer_head_position0_attention(attentions), None)
        limit = engine.METRIC_ATOL + engine.METRIC_RTOL * abs(upstream)
        self.assertLessEqual(abs(mine - upstream), limit)


class QuartileRuleTests(unittest.TestCase):
    def test_quartile_size(self) -> None:
        self.assertEqual(top_quartile_size(12), 3)   # GPT-2-small
        self.assertEqual(top_quartile_size(24), 6)   # GPT-2-medium
        self.assertEqual(top_quartile_size(4), 1)
        self.assertEqual(top_quartile_size(1), 1)
        with self.assertRaises(ValueError):
            top_quartile_size(0)


class SinkHeavyLayerRuleTests(unittest.TestCase):
    def test_top_quartile_and_floor(self) -> None:
        scores = [0.01] * 9 + [0.60, 0.70, 0.50]  # layers 9, 10, 11 are the top quartile
        layers, rule, above, partial = select_sink_heavy_layers(scores)
        self.assertEqual(layers, (9, 10, 11))
        self.assertEqual(rule, "top_quartile_and_floor")
        self.assertEqual(above, 3)
        self.assertFalse(partial)

    def test_layers_in_quartile_below_floor_are_excluded(self) -> None:
        # Only two of the top three clear 0.15, and two is still >= 2, so no fallback.
        scores = [0.01] * 9 + [0.60, 0.70, 0.14]
        layers, rule, above, partial = select_sink_heavy_layers(scores)
        self.assertEqual(layers, (9, 10))
        self.assertEqual(rule, "top_quartile_and_floor")
        self.assertEqual(above, 2)
        self.assertFalse(partial)

    def test_two_above_floor_stay_in_the_quartile_branch(self) -> None:
        # Any layer above the floor outscores every layer below it, so when at most a
        # quartile of layers clear the floor they ARE the top quartile. Two above-floor
        # layers therefore satisfy both criteria and never reach the fallback.
        scores = [0.30] * 2 + [0.01] * 10
        layers, rule, above, partial = select_sink_heavy_layers(scores)
        self.assertEqual(layers, (0, 1))
        self.assertEqual(rule, "top_quartile_and_floor")
        self.assertEqual(above, 2)
        self.assertFalse(partial)

    def test_fallback_is_reachable_only_when_one_layer_clears_the_floor(self) -> None:
        # Consequence of the property above: across every arrangement of scores, the
        # registered "top two above the floor" fallback fires exactly when a single layer
        # is above the floor -- in which case it can only return that one layer.
        for above_floor in range(1, 13):
            scores = [0.9] * above_floor + [0.01] * (12 - above_floor)
            layers, rule, above, partial = select_sink_heavy_layers(scores)
            with self.subTest(above_floor=above_floor):
                self.assertEqual(above, above_floor)
                if above_floor == 1:
                    self.assertEqual(rule, "fallback_top_two_above_floor")
                    self.assertEqual(len(layers), 1)
                    self.assertTrue(partial)
                else:
                    self.assertEqual(rule, "top_quartile_and_floor")
                    self.assertGreaterEqual(len(layers), 2)
                    self.assertFalse(partial)

    def test_fallback_records_when_only_one_layer_clears_the_floor(self) -> None:
        scores = [0.01] * 11 + [0.90]
        layers, rule, above, partial = select_sink_heavy_layers(scores)
        self.assertEqual(layers, (11,))
        self.assertEqual(rule, "fallback_top_two_above_floor")
        self.assertEqual(above, 1)
        self.assertTrue(partial)

    def test_no_layer_above_floor_fails_preflight(self) -> None:
        with self.assertRaises(SinkPreflightError):
            select_sink_heavy_layers([0.149] * 12)

    def test_floor_is_inclusive_at_exactly_the_registered_value(self) -> None:
        scores = [0.0] * 10 + [REGISTERED_SINK_FLOOR, REGISTERED_SINK_FLOOR]
        layers, _rule, above, _partial = select_sink_heavy_layers(scores)
        self.assertEqual(layers, (10, 11))
        self.assertEqual(above, 2)

    def test_ties_break_by_ascending_layer_index(self) -> None:
        scores = [0.5] * 12
        layers, _rule, _above, _partial = select_sink_heavy_layers(scores)
        self.assertEqual(layers, (0, 1, 2))


class SinkHeavyHeadTests(unittest.TestCase):
    def test_top_quartile_of_heads_within_each_layer(self) -> None:
        layer_head = np.zeros((2, 12))
        layer_head[0] = np.arange(12) / 100.0     # heads 9, 10, 11 highest
        layer_head[1] = np.arange(12)[::-1] / 100.0  # heads 0, 1, 2 highest
        heads = select_sink_heavy_heads(layer_head, [0, 1])
        self.assertEqual(heads[0], (9, 10, 11))
        self.assertEqual(heads[1], (0, 1, 2))


class CausalOrderTests(unittest.TestCase):
    def test_eligible_layers_exclude_same_and_later_layers(self) -> None:
        # AGENTS.md required test 9: an MLP at layer l may only target attention j > l.
        self.assertEqual(eligible_mlp_layers([7, 9], 12), tuple(range(9)))
        self.assertNotIn(9, eligible_mlp_layers([7, 9], 12))
        self.assertNotIn(11, eligible_mlp_layers([7, 9], 12))

    def test_future_sink_layers_are_strictly_later(self) -> None:
        future = future_sink_layers_by_mlp_layer([5, 7, 9], 12)
        self.assertEqual(future[0], (5, 7, 9))
        self.assertEqual(future[5], (7, 9))
        self.assertEqual(future[7], (9,))
        self.assertNotIn(9, future)
        for layer, targets in future.items():
            self.assertTrue(all(target > layer for target in targets))
            self.assertTrue(targets)

    def test_last_layer_sink_leaves_all_earlier_layers_eligible(self) -> None:
        self.assertEqual(eligible_mlp_layers([11], 12), tuple(range(11)))

    def test_empty_sink_layers_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            eligible_mlp_layers([], 12)


class SinkScopeTests(unittest.TestCase):
    def test_scope_is_internally_consistent_and_serialisable(self) -> None:
        layer_head = np.full((12, 12), 0.01)
        layer_head[7] = 0.70
        layer_head[9] = 0.60
        layer_head[10] = 0.50
        scope = build_sink_scope(layer_head)
        self.assertEqual(scope.sink_layers, (7, 9, 10))
        self.assertEqual(scope.num_layers, 12)
        self.assertEqual(scope.num_heads, 12)
        self.assertEqual(scope.quartile_size, 3)
        self.assertEqual(scope.eligible_mlp_layers, tuple(range(10)))
        self.assertEqual(scope.floor, REGISTERED_SINK_FLOOR)

        document = scope.to_dict()
        self.assertEqual(document["sink_layers"], [7, 9, 10])
        self.assertEqual(document["layer_indexing"], "zero_indexed")
        self.assertEqual(document["target_position"], 0)
        self.assertEqual(sorted(document["future_sink_layers"]), sorted(map(str, range(10))))
        self.assertEqual(document["future_sink_layers"]["9"], [10])

    def test_scope_matches_layer_scores(self) -> None:
        layer_head = np.full((12, 12), 0.01)
        layer_head[7] = 0.70
        layer_head[9] = 0.60
        scores = layer_scores(layer_head)
        self.assertAlmostEqual(scores[7], 0.70, places=9)
        self.assertAlmostEqual(scores[0], 0.01, places=9)
        scope = build_sink_scope(layer_head)
        for layer in scope.sink_layers:
            self.assertGreaterEqual(scores[layer], REGISTERED_SINK_FLOOR)

    def test_preflight_failure_propagates_from_build(self) -> None:
        with self.assertRaises(SinkPreflightError):
            build_sink_scope(np.full((12, 12), 0.02))


if __name__ == "__main__":
    unittest.main()
