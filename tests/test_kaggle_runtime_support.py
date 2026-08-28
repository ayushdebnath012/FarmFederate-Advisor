import io
import inspect
import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import backend.FarmFederate_Colab_Complete as farmfederate


class KaggleRuntimeSupportTests(unittest.TestCase):
    def test_inter_model_ranking_uses_only_common_support_and_dense_ties(self):
        def common_row(macro_f1, micro_f1, predictions):
            return {
                "f1_macro": macro_f1,
                "f1_micro": micro_f1,
                "n": len(predictions),
                "predictions": predictions,
            }

        results = {
            # This intentionally higher, different-support family score must
            # never enter the controlled common-test ranking.
            "llm_models": {
                "unfair_cross_scope_high": {
                    "f1": 0.99, "f1_macro": 0.99, "params": 1_000_000,
                }
            },
            "vit_models": {},
            "vlm_models": {},
            "fusion_common_test": {
                "text_parent": common_row(0.70, 0.75, [0, 1, 2, 3]),
                "vision_parent": common_row(0.65, 0.75, [0, 1, 0, 3]),
                "concat_baseline": common_row(0.72, 0.75, [0, 1, 2, 4]),
                "validation_selected_early_fusion": common_row(
                    0.72, 0.75, [0, 1, 2, 4]
                ),
                "pretrained_late_fusion": common_row(
                    0.72, 0.75, [0, 1, 2, 3]
                ),
                "hard_prediction_equivalence_groups": [{
                    "members": [
                        "concat_baseline", "validation_selected_early_fusion"
                    ],
                    "n": 4,
                    "identical_hard_predictions": True,
                }],
            },
        }

        comparison = farmfederate.run_inter_model_comparison(results)
        rankings = comparison["rankings"]
        ranked_names = {row["name"] for row in rankings}

        self.assertNotIn("unfair_cross_scope_high", ranked_names)
        self.assertIn("concat_baseline", ranked_names)
        tied_rows = [row for row in rankings if row["f1_macro"] == 0.72]
        self.assertGreaterEqual(len(tied_rows), 3)
        self.assertEqual({row["dense_rank"] for row in tied_rows}, {1})
        self.assertEqual(
            comparison["ranking_scope"], "fusion_common_test_macro_f1"
        )

    def test_empty_measured_diagnostic_plot_run_returns_cleanly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            results = {}
            saved = farmfederate.generate_measured_diagnostic_plots(
                results, Path(temp_dir), bootstrap_replicates=2
            )

        self.assertEqual(saved, [])
        self.assertIn("multimodal_diagnostics", results)

    def test_comprehensive_comparison_handles_partial_results_without_fake_rank(self):
        partial_results = {
            "llm_models": {
                "only_text_model": {
                    "f1": 0.75,
                    "f1_macro": 0.61,
                    "precision": 0.62,
                    "recall": 0.60,
                    "params": 1_000_000,
                }
            },
            "vit_models": {},
            "vlm_models": {},
        }

        captured = io.StringIO()
        with mock.patch("sys.stdout", captured):
            comparison = farmfederate.print_comprehensive_model_comparison(
                partial_results
            )

        self.assertEqual(comparison["unified_ranking"], [])
        self.assertEqual(comparison["unified_ranking_scope"], "unavailable")
        self.assertIn("Group Acc/Micro", captured.getvalue())
        self.assertIn("No unified rank is assigned", captured.getvalue())
        self.assertNotIn("Top reported score", captured.getvalue())

    def test_prediction_audit_distinguishes_score_ties_from_identical_decisions(self):
        reference = {
            "labels": [0, 1, 2, 2],
            "predictions": [0, 1, 0, 2],
            "probabilities": [
                [0.8, 0.1, 0.1], [0.1, 0.8, 0.1],
                [0.6, 0.2, 0.2], [0.1, 0.2, 0.7],
            ],
        }
        same_decisions = {
            **reference,
            "probabilities": [
                [0.7, 0.2, 0.1], [0.2, 0.7, 0.1],
                [0.5, 0.3, 0.2], [0.1, 0.3, 0.6],
            ],
        }
        different_decisions_same_correct_count = {
            "labels": reference["labels"],
            "predictions": [0, 2, 2, 2],
            "probabilities": reference["probabilities"],
        }

        same_audit = farmfederate.paired_prediction_change_audit(
            reference, same_decisions
        )
        tied_audit = farmfederate.paired_prediction_change_audit(
            reference, different_decisions_same_correct_count
        )

        self.assertTrue(same_audit["identical_hard_predictions"])
        self.assertEqual(same_audit["changed_prediction_count"], 0)
        self.assertGreater(
            same_audit["probability_max_absolute_difference"], 0.0
        )
        self.assertFalse(tied_audit["identical_hard_predictions"])
        self.assertEqual(tied_audit["changed_prediction_count"], 2)
        self.assertEqual(tied_audit["net_correct_gain"], 0)

    def test_flexible_gate_supervision_uses_amp_safe_raw_logits(self):
        training_source = inspect.getsource(
            farmfederate.train_flexible_cross_attention_vlm
        )

        self.assertIn(
            "F.binary_cross_entropy_with_logits(", training_source
        )
        self.assertIn("output['fusion_gate_logits']", training_source)
        self.assertNotIn("F.binary_cross_entropy(", training_source)

    def test_both_flexible_vlms_expose_gate_logits_before_sigmoid(self):
        for model_class in (
            farmfederate.FlexibleCrossAttentionVLM,
            farmfederate.PretrainedMultimodalFusionVLM,
        ):
            with self.subTest(model=model_class.__name__):
                forward_source = inspect.getsource(model_class.forward)
                self.assertIn(
                    "'fusion_gate_logits': fusion_gate_logits",
                    forward_source,
                )
                self.assertIn(
                    "torch.sigmoid(fusion_gate_logits)", forward_source
                )

    def test_convnext_style_backbone_skips_unsupported_gradient_checkpointing(self):
        class FakeConvNextModel:
            supports_gradient_checkpointing = False

            def __init__(self):
                self.enable_calls = 0

            def gradient_checkpointing_enable(self):
                self.enable_calls += 1
                raise ValueError("ConvNextModel does not support gradient checkpointing.")

        backbone = FakeConvNextModel()

        self.assertFalse(
            farmfederate._enable_gradient_checkpointing_if_supported(backbone)
        )
        self.assertEqual(backbone.enable_calls, 0)

    @unittest.skipIf(
        farmfederate.torch is None,
        "constructor integration requires PyTorch",
    )
    def test_vision_wrapper_constructs_with_convnext_style_backbone(self):
        class FakeConvNextModel(farmfederate.nn.Module):
            supports_gradient_checkpointing = False

            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(hidden_size=8)
                self.enable_calls = 0

            def gradient_checkpointing_enable(self):
                self.enable_calls += 1
                raise ValueError("ConvNextModel does not support gradient checkpointing.")

        backbone = FakeConvNextModel()
        fake_transformers = ModuleType("transformers")
        fake_transformers.AutoModel = SimpleNamespace(
            from_pretrained=mock.Mock(return_value=backbone)
        )
        fake_transformers.AutoImageProcessor = SimpleNamespace(
            from_pretrained=mock.Mock(
                return_value=SimpleNamespace(
                    image_mean=[0.485, 0.456, 0.406],
                    image_std=[0.229, 0.224, 0.225],
                )
            )
        )

        with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
            model = farmfederate.HuggingFaceVisionClassifier(
                "fake/convnext", num_labels=5
            )

        self.assertIs(model.encoder, backbone)
        self.assertFalse(model.gradient_checkpointing_enabled)
        self.assertEqual(backbone.enable_calls, 0)
        self.assertEqual(model.feature_dim, 8)
        self.assertEqual(model.classifier[-1].out_features, 5)

    @unittest.skipIf(
        farmfederate.torch is None,
        "constructor integration requires PyTorch",
    )
    def test_text_wrapper_checkpointing_is_safe_by_default_and_opt_in(self):
        class FakeTextBackbone(farmfederate.nn.Module):
            supports_gradient_checkpointing = True

            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(hidden_size=8)
                self.embedding = farmfederate.nn.Embedding(32, 8)
                self.enable_calls = 0
                self.disable_calls = 0

            def forward(self, input_ids=None, attention_mask=None,
                        return_dict=True):
                return SimpleNamespace(
                    last_hidden_state=self.embedding(input_ids)
                )

            def gradient_checkpointing_enable(self):
                self.enable_calls += 1

            def gradient_checkpointing_disable(self):
                self.disable_calls += 1

        safe_backbone = FakeTextBackbone()
        opted_in_backbone = FakeTextBackbone()
        fake_transformers = ModuleType("transformers")
        fake_transformers.AutoModel = SimpleNamespace(
            from_pretrained=mock.Mock(
                side_effect=[safe_backbone, opted_in_backbone]
            )
        )

        with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
            safe_model = farmfederate.HuggingFaceTextClassifier(
                "fake/bert", num_labels=5
            )
            opted_in_model = farmfederate.HuggingFaceTextClassifier(
                "fake/bert", num_labels=5,
                enable_gradient_checkpointing=True,
            )

        self.assertFalse(safe_model.gradient_checkpointing_enabled)
        self.assertEqual(safe_backbone.enable_calls, 0)
        self.assertEqual(safe_backbone.disable_calls, 1)
        output = safe_model(
            input_ids=farmfederate.torch.tensor([[1, 2, 3]]),
            attention_mask=farmfederate.torch.tensor([[1, 1, 1]]),
            labels=farmfederate.torch.tensor([[1, 0, 0, 0, 0]]),
        )
        output["loss"].backward()
        self.assertIsNotNone(safe_backbone.embedding.weight.grad)
        self.assertTrue(opted_in_model.gradient_checkpointing_enabled)
        self.assertEqual(opted_in_backbone.enable_calls, 1)
        self.assertEqual(opted_in_backbone.disable_calls, 0)

    def test_supported_backbone_enables_gradient_checkpointing(self):
        class SupportedBackbone:
            supports_gradient_checkpointing = True

            def __init__(self):
                self.enable_calls = 0

            def gradient_checkpointing_enable(self):
                self.enable_calls += 1

        backbone = SupportedBackbone()

        self.assertTrue(
            farmfederate._enable_gradient_checkpointing_if_supported(backbone)
        )
        self.assertEqual(backbone.enable_calls, 1)

    def test_backbone_without_checkpointing_method_is_left_unchanged(self):
        class BackboneWithoutCheckpointing:
            supports_gradient_checkpointing = True

        self.assertFalse(
            farmfederate._enable_gradient_checkpointing_if_supported(
                BackboneWithoutCheckpointing()
            )
        )

    def test_optimizer_schedule_counts_updates_after_accumulation(self):
        updates_per_epoch, total_updates, warmup_updates = (
            farmfederate.optimizer_schedule_steps(
                num_batches=101,
                num_epochs=12,
                accumulation_steps=2,
                warmup_ratio=0.05,
            )
        )

        self.assertEqual(updates_per_epoch, 51)
        self.assertEqual(total_updates, 612)
        self.assertEqual(warmup_updates, 30)

    def test_stable_vision_transfer_defaults_are_conservative(self):
        config = farmfederate.Config()

        self.assertFalse(config.pretrained_text_gradient_checkpointing)

        self.assertEqual(config.pretrained_vision_epochs, 15)
        self.assertEqual(config.pretrained_vision_learning_rate, 2e-5)
        self.assertEqual(config.pretrained_vision_head_learning_rate, 1e-4)
        self.assertEqual(config.pretrained_vision_linear_probe_epochs, 1)
        self.assertEqual(config.pretrained_vision_tail_unfreeze_epochs, 2)
        self.assertTrue(config.pretrained_vision_freeze_batch_norm_stats)
        self.assertTrue(config.pretrained_vision_inference_tta)

    @unittest.skipIf(
        farmfederate.torch is None,
        "activation-aware scheduling requires PyTorch",
    )
    def test_activation_aware_scheduler_does_not_decay_frozen_groups(self):
        torch = farmfederate.torch
        first = torch.nn.Parameter(torch.tensor([1.0]))
        delayed = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.SGD([
            {"params": [first], "lr": 1.0},
            {"params": [delayed], "lr": 1.0},
        ])
        scheduler = farmfederate.get_activation_aware_warmup_scheduler(
            optimizer,
            activation_steps=[0, 2],
            total_steps=6,
            warmup_ratio=0.0,
        )

        self.assertEqual(optimizer.param_groups[1]["lr"], 0.0)
        optimizer.step()
        scheduler.step()
        self.assertEqual(optimizer.param_groups[1]["lr"], 0.0)
        optimizer.step()
        scheduler.step()
        self.assertGreater(optimizer.param_groups[1]["lr"], 0.0)

    @unittest.skipIf(
        farmfederate.torch is None,
        "vision transfer stages require PyTorch",
    )
    def test_vision_transfer_stages_freeze_tail_then_full(self):
        class TransformerEncoder(farmfederate.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = farmfederate.nn.ModuleList([
                    farmfederate.nn.Linear(2, 2),
                    farmfederate.nn.Linear(2, 2),
                ])

        class FakeVisionEncoder(farmfederate.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = TransformerEncoder()
                self.layernorm = farmfederate.nn.LayerNorm(2)

        encoder = FakeVisionEncoder()
        stage, tail = farmfederate.configure_vision_encoder_stage(
            encoder, "frozen"
        )
        self.assertEqual(stage, "frozen")
        self.assertEqual(tail, [])
        self.assertTrue(all(not p.requires_grad for p in encoder.parameters()))

        stage, tail = farmfederate.configure_vision_encoder_stage(
            encoder, "tail"
        )
        self.assertEqual(stage, "tail")
        self.assertEqual(tail, [encoder.encoder.layer[-1], encoder.layernorm])
        self.assertTrue(all(
            not p.requires_grad for p in encoder.encoder.layer[0].parameters()
        ))
        self.assertTrue(all(
            p.requires_grad for p in encoder.encoder.layer[-1].parameters()
        ))
        self.assertTrue(all(
            p.requires_grad for p in encoder.layernorm.parameters()
        ))

        stage, tail = farmfederate.configure_vision_encoder_stage(
            encoder, "full"
        )
        self.assertEqual(stage, "full")
        self.assertEqual(tail, [])
        self.assertTrue(all(p.requires_grad for p in encoder.parameters()))

    @unittest.skipIf(
        farmfederate.torch is None,
        "EfficientNet tail discovery requires PyTorch",
    )
    def test_efficientnet_tail_includes_top_projection_and_batchnorm(self):
        class EfficientNetInner(farmfederate.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = farmfederate.nn.ModuleList([
                    farmfederate.nn.Linear(2, 2),
                    farmfederate.nn.Linear(2, 2),
                ])
                self.top_conv = farmfederate.nn.Conv2d(2, 2, 1)
                self.top_batchnorm = farmfederate.nn.BatchNorm2d(2)

        class FakeEfficientNet(farmfederate.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = EfficientNetInner()

        encoder = FakeEfficientNet()
        tail = farmfederate.vision_encoder_tail_modules(encoder)

        self.assertEqual(tail, [
            encoder.encoder.blocks[-1],
            encoder.encoder.top_conv,
            encoder.encoder.top_batchnorm,
        ])

    @unittest.skipIf(
        farmfederate.torch is None,
        "BatchNorm transfer behavior requires PyTorch",
    )
    def test_batchnorm_stats_freeze_preserves_trainable_affine_parameters(self):
        encoder = farmfederate.nn.Sequential(
            farmfederate.nn.Conv2d(3, 3, 1),
            farmfederate.nn.BatchNorm2d(3),
        )

        frozen_count = farmfederate.set_vision_encoder_training_mode(
            encoder, "full", freeze_batch_norm_stats=True
        )
        self.assertEqual(frozen_count, 1)
        self.assertTrue(encoder.training)
        self.assertTrue(encoder[0].training)
        self.assertFalse(encoder[1].training)
        self.assertTrue(encoder[1].weight.requires_grad)
        self.assertTrue(encoder[1].bias.requires_grad)

        frozen_count = farmfederate.set_vision_encoder_training_mode(
            encoder, "full", freeze_batch_norm_stats=False
        )
        self.assertEqual(frozen_count, 0)
        self.assertTrue(encoder[1].training)

    @unittest.skipIf(
        farmfederate.torch is None,
        "TTA behavior requires PyTorch",
    )
    def test_hflip_tta_is_eval_only_and_averages_features(self):
        torch = farmfederate.torch

        class LocationEncoder(farmfederate.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(model_type="fake")
                self.calls = 0
                self.inputs = []

            def forward(self, pixel_values, return_dict=True):
                self.calls += 1
                self.inputs.append(pixel_values.detach().clone())
                return SimpleNamespace(
                    last_hidden_state=pixel_values[:, :1, 0, 0]
                )

        model = farmfederate.HuggingFaceVisionClassifier.__new__(
            farmfederate.HuggingFaceVisionClassifier
        )
        farmfederate.nn.Module.__init__(model)
        model.encoder = LocationEncoder()
        model.inference_tta = True
        model.register_buffer(
            "processor_mean", torch.zeros(1, 3, 1, 1)
        )
        model.register_buffer(
            "processor_std", torch.ones(1, 3, 1, 1)
        )
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        raw = torch.tensor([[
            [[0.0, 1.0]], [[0.0, 1.0]], [[0.0, 1.0]],
        ]])
        pixel_values = (raw - imagenet_mean) / imagenet_std

        model.eval()
        model.encoder.calls = 0
        model.encode_features(pixel_values)
        self.assertEqual(model.encoder.calls, 1)

        model.encoder.calls = 0
        model.encoder.inputs.clear()
        with torch.no_grad():
            features = model.encode_features(pixel_values)
        self.assertEqual(model.encoder.calls, 2)
        torch.testing.assert_close(features, torch.tensor([[0.5]]))
        torch.testing.assert_close(
            model.encoder.inputs[1],
            torch.flip(model.encoder.inputs[0], dims=[3]),
        )

        model.train()
        model.encoder.calls = 0
        with torch.no_grad():
            model.encode_features(pixel_values)
        self.assertEqual(model.encoder.calls, 1)

    def test_group_aggregation_averages_probabilities_per_independent_group(self):
        np = farmfederate.np
        metrics = {
            "labels": np.asarray([0, 0, 1, 1]),
            "probabilities": np.asarray([
                [0.9, 0.1], [0.1, 0.9], [0.2, 0.8], [0.8, 0.2],
            ]),
        }

        result = farmfederate.aggregate_classification_metrics_by_group(
            metrics, ["a", "a", "b", "c"]
        )

        self.assertEqual(result["n_samples"], 3)
        self.assertEqual(result["row_count"], 4)
        self.assertEqual(result["n_correct"], 2)
        self.assertEqual(result["labels"].tolist(), [0, 1, 1])
        self.assertEqual(result["predictions"].tolist(), [0, 1, 0])
        np.testing.assert_allclose(
            result["probabilities"],
            [[0.5, 0.5], [0.2, 0.8], [0.8, 0.2]],
        )
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertAlmostEqual(result["f1_micro"], 2 / 3)
        self.assertAlmostEqual(result["f1_macro"], 2 / 3)

    def test_group_aggregation_rejects_invalid_groups(self):
        np = farmfederate.np
        metrics = {
            "labels": np.asarray([0, 1]),
            "probabilities": np.asarray([[0.8, 0.2], [0.2, 0.8]]),
        }
        with self.assertRaisesRegex(ValueError, "align exactly"):
            farmfederate.aggregate_classification_metrics_by_group(
                metrics, ["only-one"]
            )
        with self.assertRaisesRegex(ValueError, "multiple labels"):
            farmfederate.aggregate_classification_metrics_by_group(
                metrics, ["shared", "shared"]
            )

    def test_text_split_reserves_multiple_independent_test_groups(self):
        texts = []
        labels = []
        for class_id in range(5):
            for group_id in range(11):
                texts.append(
                    f"Class {class_id} template {group_id}. Distinct detail."
                )
                labels.append([class_id])

        _, _, (test_texts, _), audit = farmfederate.grouped_text_split(
            texts,
            labels,
            train_ratio=0.70,
            val_ratio=0.15,
            seed=42,
        )

        self.assertEqual(len(test_texts), 15)
        self.assertEqual(audit["split_groups"]["test"], 15)
        self.assertTrue(
            all(
                allocation == {"train": 6, "val": 2, "test": 3}
                for allocation in audit["split_groups_per_class"].values()
            )
        )
        self.assertEqual(audit["group_overlap_train_test"], 0)

    def test_wilson_interval_exposes_small_image_test_uncertainty(self):
        low, high = farmfederate.wilson_score_interval(26, 31)

        self.assertAlmostEqual(low, 0.6737, places=3)
        self.assertAlmostEqual(high, 0.9291, places=3)
        self.assertLess(low, 26 / 31)
        self.assertGreater(high, 26 / 31)

    def test_accuracy_target_requires_confidence_bound_not_point_only(self):
        evidence = farmfederate.accuracy_target_evidence(27, 30, 0.90)

        self.assertTrue(evidence["point_target_met"])
        self.assertFalse(evidence["target_supported"])
        self.assertLess(evidence["accuracy_ci95_low"], 0.90)

    def test_macro_f1_is_primary_validation_selection_metric(self):
        balanced = {
            "f1_macro": 0.78,
            "f1_micro": 0.80,
            "labels": [0, 1],
            "probabilities": [[0.8, 0.2], [0.2, 0.8]],
        }
        majority_favoured = {
            "f1_macro": 0.70,
            "f1_micro": 0.84,
            "labels": [0, 1],
            "probabilities": [[0.9, 0.1], [0.4, 0.6]],
        }

        self.assertGreater(
            farmfederate.validation_model_selection_key(balanced, "ViT"),
            farmfederate.validation_model_selection_key(
                majority_favoured, "ConvNeXT"
            ),
        )

    def test_grouped_selection_uses_row_f1_before_confidence(self):
        group_tie_a = {"f1_macro": 1.0, "f1_micro": 1.0}
        group_tie_b = {"f1_macro": 1.0, "f1_micro": 1.0}
        stronger_rows = {"f1_macro": 0.91, "f1_micro": 0.92}
        weaker_rows = {"f1_macro": 0.86, "f1_micro": 0.90}

        self.assertGreater(
            farmfederate.grouped_validation_model_selection_key(
                group_tie_a, stronger_rows
            ),
            farmfederate.grouped_validation_model_selection_key(
                group_tie_b, weaker_rows
            ),
        )

    def test_checkpoint_tta_metadata_is_backward_compatible(self):
        self.assertTrue(farmfederate.checkpoint_inference_tta_enabled({
            "inference_tta": "horizontal_flip_feature_mean"
        }))
        self.assertFalse(farmfederate.checkpoint_inference_tta_enabled({
            "inference_tta_enabled": False,
            "inference_tta": "horizontal_flip_feature_mean",
        }))

    def test_proxy_text_subset_round_robins_template_groups(self):
        texts = [
            f"Template {group}. Independent opening. Variant {variant}."
            for group in range(3)
            for variant in range(4)
        ]

        selected = farmfederate.select_template_diverse_texts(
            texts, limit=3, seed=42
        )

        self.assertEqual(len(selected), 3)
        self.assertEqual(
            len({farmfederate._text_template_group_key(text) for text in selected}),
            3,
        )

    def test_balanced_sampler_scales_epoch_instead_of_over_replaying_minority(self):
        labels = (
            [[0]] * 42 + [[1]] * 5 + [[2]] * 26
            + [[3]] * 33 + [[4]] * 34
        )

        sampler = farmfederate.BalancedBatchSampler(
            labels, batch_size=8, num_classes=5
        )

        self.assertEqual(len(sampler), 27)
        sampled_indices = {
            index for batch in sampler for index in batch
        }
        self.assertTrue(set(range(42)).issubset(sampled_indices))

    def test_balanced_sampler_matches_v16_source_unique_support(self):
        class_counts = [41, 6, 25, 34, 33]
        labels = [
            [class_id]
            for class_id, count in enumerate(class_counts)
            for _ in range(count)
        ]
        sampler = farmfederate.BalancedBatchSampler(
            labels, batch_size=8, num_classes=5
        )
        batches = list(sampler)

        self.assertEqual(len(batches), 26)
        self.assertTrue(all(len(batch) == 8 for batch in batches))
        self.assertTrue(
            all({labels[index][0] for index in batch} == set(range(5))
                for batch in batches)
        )
        sampled_indices = {index for batch in batches for index in batch}
        self.assertTrue(set(range(41)).issubset(sampled_indices))

    def test_balanced_sampler_rejects_invalid_balancing_contracts(self):
        invalid_cases = (
            ([], 8, 5, "at least one label"),
            ([[0], [1], [2], [3], [4]], 4, 5, "batch_size >= num_classes"),
            ([[0], [1], [2], [3]], 8, 5, "missing classes"),
            ([[0], [1], [2], [3], [5]], 8, 5, "labels must be in"),
            ([[0], [1], [2], [3], []], 8, 5, "empty nested label"),
        )
        for labels, batch_size, num_classes, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    farmfederate.BalancedBatchSampler(
                        labels,
                        batch_size=batch_size,
                        num_classes=num_classes,
                    )

    def test_grouped_client_split_repairs_empty_client_from_eligible_donor(self):
        groups = ["large_source"] * 100 + ["small_b", "small_c", "small_d"]
        labels = [0] * len(groups)
        clients = farmfederate.split_data_non_iid(
            list(range(len(groups))),
            num_clients=3,
            alpha=0.1,
            labels=labels,
            groups=groups,
            seed=3,
        )

        self.assertTrue(all(clients))
        self.assertEqual(
            sorted(index for client in clients for index in client),
            list(range(len(groups))),
        )
        owners = {}
        for client_id, indices in enumerate(clients):
            for index in indices:
                previous = owners.setdefault(groups[index], client_id)
                self.assertEqual(previous, client_id)

    def test_client_split_rejects_ambiguous_group_inputs(self):
        with self.assertRaisesRegex(ValueError, "empty dataset"):
            farmfederate.split_data_non_iid(
                [], num_clients=3, labels=[], groups=[]
            )
        with self.assertRaisesRegex(ValueError, "groups require labels"):
            farmfederate.split_data_non_iid(
                [0, 1], num_clients=2, groups=["a", "b"]
            )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            farmfederate.split_data_non_iid(
                [0, 1],
                num_clients=2,
                labels=[0, -1],
                groups=["a", "b"],
            )

    def test_matched_centralized_and_federated_select_macro_first(self):
        for training_function in (
            farmfederate.federated_train,
            farmfederate.centralized_train_fedavg_matched,
        ):
            with self.subTest(function=training_function.__name__):
                source = inspect.getsource(training_function)
                self.assertIn("validation_model_selection_key", source)
                self.assertIn("selected_validation", source)
                self.assertNotIn(
                    "if metrics['f1_micro'] >", source
                )

    def test_federated_sweeps_forward_group_ownership(self):
        run_source = inspect.getsource(farmfederate.federated_train_run)
        sweep_source = inspect.getsource(farmfederate.run_federated_sweep)

        self.assertIn("groups=None", run_source)
        self.assertIn("groups=groups", run_source)
        self.assertIn("groups=None", sweep_source)
        self.assertGreaterEqual(sweep_source.count("groups=groups"), 4)

    def test_literature_output_is_context_not_cross_dataset_ranking(self):
        source = inspect.getsource(
            farmfederate.print_research_paper_comparison
        )

        self.assertIn("NOT A RANKING", source)
        self.assertNotIn('comparison = "BETTER"', source)
        self.assertNotIn("OVERALL RANKING", source)

    @unittest.skipIf(
        farmfederate.torch is None or farmfederate.np is None,
        "weights-only checkpoint test requires PyTorch and NumPy",
    )
    def test_checkpoint_metadata_round_trips_with_weights_only_loader(self):
        metadata = farmfederate.checkpoint_safe_metadata({
            "path": Path("results/example"),
            "score": farmfederate.np.float64(0.75),
            "history": [farmfederate.np.int64(3)],
        })

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "safe.pt"
            farmfederate.torch.save({
                "model_state_dict": {
                    "weight": farmfederate.torch.tensor([1.0])
                },
                "metadata": metadata,
            }, checkpoint_path)
            loaded = farmfederate.torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )

        self.assertEqual(
            loaded["metadata"]["path"], str(Path("results/example"))
        )
        self.assertEqual(loaded["metadata"]["score"], 0.75)
        self.assertEqual(loaded["metadata"]["history"], [3])

    def test_exact_mcnemar_uses_paired_disagreements(self):
        labels = [0, 1, 2, 3]
        model_a = [0, 1, 2, 3]
        model_b = [9, 9, 9, 3]

        self.assertAlmostEqual(
            farmfederate.exact_mcnemar_pvalue(labels, model_a, model_b),
            0.25,
        )

    def test_roboflow_variants_share_one_source_photo_group(self):
        first = "IMG_20230624_101122_jpg.rf.5eaec8974e8cc022.jpg"
        second = "IMG_20230624_101122_jpg.rf.8e32454ba74e53d7.jpg"
        distinct = "IMG_20230624_101123_jpg.rf.8e32454ba74e53d7.jpg"

        self.assertEqual(
            farmfederate._real_image_source_group(first),
            farmfederate._real_image_source_group(second),
        )
        self.assertNotEqual(
            farmfederate._real_image_source_group(first),
            farmfederate._real_image_source_group(distinct),
        )

    def test_grouped_image_split_keeps_export_variants_together(self):
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("grouped split integration requires scikit-learn")

        labels = []
        groups = []
        for class_id, file_count in enumerate((62, 8, 35, 48, 47)):
            for sample_id in range(file_count):
                labels.append([class_id])
                # The first two Blight exports are near-duplicate variants of
                # one source photograph, matching the real portable bundle.
                source_id = 0 if class_id == 0 and sample_id < 2 else sample_id
                groups.append(f"class_{class_id}_source_{source_id}")

        train_idx, val_idx, test_idx = farmfederate.grouped_multimodal_split(
            labels, groups, train_ratio=0.70, val_ratio=0.15, seed=42
        )
        split_group_sets = [
            {groups[index] for index in indices}
            for indices in (train_idx, val_idx, test_idx)
        ]

        self.assertFalse(split_group_sets[0] & split_group_sets[1])
        self.assertFalse(split_group_sets[0] & split_group_sets[2])
        self.assertFalse(split_group_sets[1] & split_group_sets[2])
        self.assertTrue(
            all(
                {labels[index][0] for index in indices} == set(range(5))
                for indices in (train_idx, val_idx, test_idx)
            )
        )
        duplicate_locations = [
            split_id
            for split_id, indices in enumerate((train_idx, val_idx, test_idx))
            if 0 in indices or 1 in indices
        ]
        self.assertEqual(len(duplicate_locations), 1)
        self.assertTrue(
            all(
                sum(labels[index][0] == 1 for index in indices) >= 1
                for indices in (val_idx, test_idx)
            )
        )

    def test_exact_grouped_image_split_allocates_scarce_class_6_1_1(self):
        labels = []
        groups = []
        for class_id, file_count in enumerate((62, 8, 35, 48, 47)):
            for sample_id in range(file_count):
                labels.append([class_id])
                source_id = 0 if class_id == 0 and sample_id < 2 else sample_id
                groups.append(f"class_{class_id}_source_{source_id}")

        train_idx, val_idx, test_idx = farmfederate.grouped_multimodal_split(
            labels, groups, train_ratio=0.70, val_ratio=0.15, seed=42
        )

        self.assertEqual(
            (len(train_idx), len(val_idx), len(test_idx)), (140, 30, 30)
        )
        self.assertEqual(
            tuple(
                sum(labels[index][0] == 1 for index in indices)
                for indices in (train_idx, val_idx, test_idx)
            ),
            (6, 1, 1),
        )
        split_group_sets = [
            {groups[index] for index in indices}
            for indices in (train_idx, val_idx, test_idx)
        ]
        self.assertFalse(split_group_sets[0] & split_group_sets[1])
        self.assertFalse(split_group_sets[0] & split_group_sets[2])
        self.assertFalse(split_group_sets[1] & split_group_sets[2])

    def test_grouped_split_rejects_classes_without_three_source_groups(self):
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("grouped split integration requires scikit-learn")

        labels = [[0], [0], [1], [1], [2], [2], [3], [3], [4], [4]]
        groups = [f"source_{index}" for index in range(len(labels))]
        with self.assertRaisesRegex(ValueError, "three independent"):
            farmfederate.grouped_multimodal_split(labels, groups)

    def test_vit_and_deit_use_pretrained_cls_feature_before_pooler(self):
        source = inspect.getsource(
            farmfederate.HuggingFaceVisionClassifier.encode_features
        )

        self.assertIn("model_type in {'vit', 'deit'}", source)
        self.assertIn("return hidden[:, 0, :]", source)
        self.assertLess(
            source.index("return hidden[:, 0, :]"),
            source.index("pooler_output"),
        )

    def test_flexible_fusion_can_follow_either_validation_parent(self):
        config = farmfederate.Config()
        training_source = inspect.getsource(
            farmfederate.train_flexible_cross_attention_vlm
        )
        pretrained_forward = inspect.getsource(
            farmfederate.PretrainedMultimodalFusionVLM.forward
        )

        self.assertEqual(config.flexible_vlm_min_text_weight, 0.05)
        self.assertEqual(config.flexible_vlm_max_text_weight, 0.95)
        self.assertIn("torch.minimum(", training_source)
        self.assertIn("self.log_temp_text.exp()", pretrained_forward)
        self.assertIn("self.log_temp_vision.exp()", pretrained_forward)

    def test_kaggle_runtime_takes_priority_over_colab_false_positive(self):
        with (
            mock.patch.dict(
                os.environ, {"KAGGLE_KERNEL_RUN_TYPE": "Interactive"}, clear=False
            ),
            mock.patch.object(
                farmfederate, "_detect_colab_runtime", return_value=True
            ),
        ):
            self.assertEqual(
                farmfederate._detect_cloud_runtime(), (False, True)
            )

    def test_colab_local_upload_is_not_discarded_when_drive_bundle_is_missing(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            uploaded_bundle = root / "data_final.zip"
            uploaded_bundle.write_bytes(b"uploaded bundle")

            selected, source = farmfederate.prefer_colab_bundle_candidate(
                uploaded_bundle,
                [root / "missing-drive" / "data_final.zip"],
            )

            self.assertEqual(selected, uploaded_bundle)
            self.assertEqual(source, "uploaded_or_local")

    def test_colab_mounted_drive_bundle_still_has_priority(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            uploaded_bundle = root / "data_final.zip"
            mounted_bundle = root / "drive" / "data_final.zip"
            uploaded_bundle.write_bytes(b"uploaded bundle")
            mounted_bundle.parent.mkdir()
            mounted_bundle.write_bytes(b"mounted bundle")

            selected, source = farmfederate.prefer_colab_bundle_candidate(
                uploaded_bundle, [mounted_bundle]
            )

            self.assertEqual(selected, mounted_bundle)
            self.assertEqual(source, "mounted_drive")

    def test_detects_kaggle_from_official_environment_marker(self):
        with mock.patch.dict(
            os.environ, {"KAGGLE_KERNEL_RUN_TYPE": "Interactive"}, clear=False
        ):
            self.assertTrue(farmfederate._detect_kaggle_runtime())

    def test_discovers_nested_uploaded_data_final_zip(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            bundle = root / "farmfederate-data" / "colab_upload" / "data_final.zip"
            bundle.parent.mkdir(parents=True)
            bundle.write_bytes(b"fixture")

            self.assertEqual(
                farmfederate.discover_kaggle_data_final_zip(root), bundle
            )

    def test_ambiguous_kaggle_bundles_require_explicit_override(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            for slug in ("first", "second"):
                bundle = root / slug / "data_final.zip"
                bundle.parent.mkdir(parents=True)
                bundle.write_bytes(slug.encode("utf-8"))

            with self.assertRaisesRegex(FileExistsError, "--data-final-zip"):
                farmfederate.discover_kaggle_data_final_zip(root)

    def test_discovers_versioned_expanded_kaggle_bundle(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            bundle_root = root / "farmfederate-data" / "data_final"
            image_root = bundle_root / "real_dataset_sorted"
            text_root = bundle_root / "text_data"
            image_root.mkdir(parents=True)
            text_root.mkdir(parents=True)
            (bundle_root / "label_schema.json").write_text("{}", encoding="utf-8")
            (text_root / "annotations.csv").write_text(
                "text,class_id,class_name\n", encoding="utf-8"
            )

            self.assertEqual(
                farmfederate.discover_expanded_kaggle_bundle(root),
                (image_root, text_root),
            )

    def test_unversioned_expanded_input_is_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / "real_dataset_sorted").mkdir()
            text_root = root / "text_data"
            text_root.mkdir()
            (text_root / "annotations.csv").write_text(
                "text,class_id\n", encoding="utf-8"
            )

            self.assertEqual(
                farmfederate.discover_expanded_kaggle_bundle(root), (None, None)
            )

    def test_ambiguous_expanded_bundles_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            for slug in ("first", "second"):
                bundle_root = root / slug / "data_final"
                (bundle_root / "real_dataset_sorted").mkdir(parents=True)
                (bundle_root / "text_data").mkdir()
                (bundle_root / "label_schema.json").write_text(
                    "{}", encoding="utf-8"
                )
                (bundle_root / "text_data" / "annotations.csv").write_text(
                    "text,class_id,class_name\n", encoding="utf-8"
                )

            with self.assertRaisesRegex(FileExistsError, "Multiple expanded"):
                farmfederate.discover_expanded_kaggle_bundle(root)

    def test_expanded_bundle_requires_canonical_manifest_layout(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            bundle_root = root / "data_final"
            nested_root = bundle_root / "nested"
            (nested_root / "real_dataset_sorted").mkdir(parents=True)
            (nested_root / "text_data").mkdir()
            (bundle_root / "label_schema.json").write_text("{}", encoding="utf-8")
            (nested_root / "text_data" / "annotations.csv").write_text(
                "text,class_id,class_name\n", encoding="utf-8"
            )

            self.assertEqual(
                farmfederate.discover_expanded_kaggle_bundle(root), (None, None)
            )

    def test_portable_manifest_requires_canonical_class_order(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            bundle_root = Path(temporary_dir) / "data_final"
            image_root = bundle_root / "real_dataset_sorted"
            image_root.mkdir(parents=True)
            manifest_path = bundle_root / "label_schema.json"
            expected_mapping = {
                str(key): value
                for key, value in farmfederate.RAW_YOLO_ID_TO_STRESS.items()
            }
            manifest = {
                "schema_version": farmfederate.STRESS_SCHEMA_VERSION,
                "raw_yolo_id_to_stress": expected_mapping,
                "class_order": list(farmfederate.STRESS_LABELS),
            }
            manifest_path.write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            validated = farmfederate._ensure_portable_bundle_label_schema(
                image_root, bundle_root
            )
            self.assertEqual(
                validated["class_order"], farmfederate.STRESS_LABELS
            )

            manifest["class_order"][0], manifest["class_order"][2] = (
                manifest["class_order"][2],
                manifest["class_order"][0],
            )
            manifest_path.write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "class order"):
                farmfederate._ensure_portable_bundle_label_schema(
                    image_root, bundle_root
                )

    def test_sorted_dataset_discovery_accepts_supported_image_extensions(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            for label in farmfederate.STRESS_LABELS:
                (root / label).mkdir()
            png_image = root / farmfederate.STRESS_LABELS[0] / "sample.PNG"
            png_image.write_bytes(b"fixture")

            self.assertTrue(farmfederate._is_usable_sorted_dataset(root))

            png_image.unlink()
            (root / farmfederate.STRESS_LABELS[0] / "sample.txt").write_text(
                "not an image", encoding="utf-8"
            )
            self.assertFalse(farmfederate._is_usable_sorted_dataset(root))

    def test_extracts_windows_style_members_into_working_directory(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            bundle = root / "data_final.zip"
            extract_root = root / "working" / "farmfederate_data"
            with zipfile.ZipFile(bundle, "w") as archive:
                archive.writestr(
                    "data_final\\label_schema.json", '{"schema_version": "test"}'
                )
                archive.writestr(
                    "data_final\\real_dataset_sorted\\LEAF_BLIGHT\\sample.jpg",
                    b"fixture",
                )
                archive.writestr(
                    "data_final\\text_data\\annotations.csv",
                    "text,class_id,class_name\n",
                )

            image_root, text_root = farmfederate.extract_portable_bundle(
                bundle, extract_root
            )

            self.assertEqual(
                image_root,
                extract_root / "data_final" / "real_dataset_sorted",
            )
            self.assertEqual(text_root, extract_root / "data_final" / "text_data")
            self.assertTrue((image_root / "LEAF_BLIGHT" / "sample.jpg").is_file())

    def test_legacy_bundle_repair_remains_idempotent_across_reruns(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            bundle = root / "data_final.zip"
            extract_root = root / "working" / "content-extract-v2"
            with zipfile.ZipFile(bundle, "w") as archive:
                for index in range(35):
                    archive.writestr(
                        "data_final/real_dataset_sorted/LEAF_BLIGHT/"
                        f"blight_{index:02d}.jpg",
                        b"legacy blight fixture",
                    )
                for index in range(62):
                    archive.writestr(
                        "data_final/real_dataset_sorted/LEAF_RUST/"
                        f"rust_{index:02d}.jpg",
                        b"legacy rust fixture",
                    )
                archive.writestr(
                    "data_final/text_data/annotations.csv",
                    "text,class_id,class_name\n",
                )

            first_image_root, first_text_root = (
                farmfederate.extract_portable_bundle(bundle, extract_root)
            )
            first_manifest = farmfederate._ensure_portable_bundle_label_schema(
                first_image_root, first_image_root.parent
            )
            self.assertTrue(first_manifest["legacy_blightrust_swap_repaired"])
            self.assertTrue(
                (extract_root / farmfederate.PORTABLE_EXTRACTION_MARKER).is_file()
            )

            second_image_root, second_text_root = (
                farmfederate.extract_portable_bundle(bundle, extract_root)
            )
            farmfederate._ensure_portable_bundle_label_schema(
                second_image_root, second_image_root.parent
            )

            self.assertEqual(second_image_root, first_image_root)
            self.assertEqual(second_text_root, first_text_root)
            blight_files = sorted(
                path.name
                for path in (second_image_root / "LEAF_BLIGHT").glob("*.jpg")
            )
            rust_files = sorted(
                path.name
                for path in (second_image_root / "LEAF_RUST").glob("*.jpg")
            )
            self.assertEqual(len(blight_files), 62)
            self.assertEqual(len(rust_files), 35)
            self.assertTrue(all(name.startswith("rust_") for name in blight_files))
            self.assertTrue(all(name.startswith("blight_") for name in rust_files))

    def test_rejects_zip_path_traversal(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            bundle = root / "data_final.zip"
            with zipfile.ZipFile(bundle, "w") as archive:
                archive.writestr("..\\escape.txt", "unsafe")

            with self.assertRaisesRegex(ValueError, "Unsafe path"):
                farmfederate.extract_portable_bundle(bundle, root / "working")
            self.assertFalse((root / "escape.txt").exists())

    def test_rejects_multiple_complete_bundles_inside_one_zip(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            bundle = root / "data_final.zip"
            with zipfile.ZipFile(bundle, "w") as archive:
                for slug in ("first", "second"):
                    archive.writestr(
                        f"{slug}/real_dataset_sorted/LEAF_BLIGHT/sample.jpg",
                        b"fixture",
                    )
                    archive.writestr(
                        f"{slug}/text_data/annotations.csv",
                        "text,class_id,class_name\n",
                    )

            with self.assertRaisesRegex(FileExistsError, "multiple complete"):
                farmfederate.extract_portable_bundle(bundle, root / "working")

    def test_kaggle_extract_and_output_paths_use_writable_working_root(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            with (
                mock.patch.object(farmfederate, "IN_COLAB", False),
                mock.patch.object(farmfederate, "IN_KAGGLE", True),
                mock.patch.dict(
                    os.environ, {"KAGGLE_WORKING_DIR": str(root)}, clear=False
                ),
            ):
                self.assertEqual(
                    farmfederate.portable_bundle_extract_root(),
                    root / "farmfederate_data",
                )

                config = farmfederate.Config()
                farmfederate.setup_gdrive_output_dirs(config)
                runtime_root = root / "farmfederate"
                self.assertEqual(config.output_dir, runtime_root / "results")
                self.assertEqual(
                    config.checkpoint_dir, runtime_root / "checkpoints"
                )
                self.assertEqual(config.plots_dir, runtime_root / "plots")
                self.assertTrue(
                    all(
                        Path(path).is_dir()
                        for path in (
                            config.output_dir,
                            config.checkpoint_dir,
                            config.plots_dir,
                        )
                    )
                )

    def test_bundle_extraction_directory_is_content_addressed(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            first = root / "first.zip"
            second = root / "second.zip"
            first.write_bytes(b"first bundle")
            second.write_bytes(b"second bundle")
            with (
                mock.patch.object(farmfederate, "IN_COLAB", False),
                mock.patch.object(farmfederate, "IN_KAGGLE", True),
                mock.patch.dict(
                    os.environ, {"KAGGLE_WORKING_DIR": str(root)}, clear=False
                ),
            ):
                first_root = farmfederate.portable_bundle_extract_root(first)
                self.assertEqual(
                    first_root,
                    farmfederate.portable_bundle_extract_root(first),
                )
                self.assertNotEqual(
                    first_root,
                    farmfederate.portable_bundle_extract_root(second),
                )
                self.assertEqual(first_root.parent, root / "farmfederate_data")

    def test_result_archive_is_written_to_kaggle_working_root(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            config = farmfederate.Config(
                output_dir=root / "results",
                checkpoint_dir=root / "checkpoints",
                plots_dir=root / "plots",
            )
            for directory in (
                config.output_dir,
                config.checkpoint_dir,
                config.plots_dir,
            ):
                directory.mkdir(parents=True)
            checkpoint_path = config.checkpoint_dir / "model.pt"
            checkpoint_path.write_bytes(b"checkpoint payload")

            with (
                mock.patch.object(farmfederate, "IN_KAGGLE", True),
                mock.patch.dict(
                    os.environ, {"KAGGLE_WORKING_DIR": str(root)}, clear=False
                ),
                mock.patch("sys.stdout", new=io.StringIO()),
            ):
                archive_path = Path(
                    farmfederate._download_results_with_config(config)
                )

            self.assertEqual(archive_path.parent, root)
            self.assertTrue(archive_path.is_file())
            self.assertFalse(
                archive_path.with_suffix(archive_path.suffix + ".partial").exists()
            )
            with zipfile.ZipFile(archive_path) as archive:
                self.assertIn("README.md", archive.namelist())
                self.assertIn("package_manifest.json", archive.namelist())
                self.assertEqual(
                    archive.getinfo("models/model.pt").compress_type,
                    zipfile.ZIP_STORED,
                )
                manifest = json.loads(
                    archive.read("package_manifest.json").decode("utf-8")
                )
                self.assertEqual(manifest["status"], "complete")
                self.assertEqual(manifest["model_checkpoints"], 1)


if __name__ == "__main__":
    unittest.main()
