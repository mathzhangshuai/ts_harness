import copy
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fixtures import DUMMY, REFERENCE
from gluonts.dataset.field_names import FieldName as FN
from torch.utils.data import DataLoader
from tszoo.data import MemmapWindows, collate_windows, write_store
from tszoo.models.chronos2 import (
    Chronos2Backbone,
    CoreConfig,
    FeatureSchema,
    SplitChronos2,
)


def entry(length=24, offset=0):
    return {
        FN.ITEM_ID: str(offset),
        FN.START: pd.Period("2024-01-01", freq="D"),
        FN.TARGET: np.arange(length, dtype=np.float32)[None] + offset,
        FN.FEAT_STATIC_CAT: np.array([1], dtype=np.int64),
        FN.FEAT_STATIC_REAL: np.array([3.0], dtype=np.float32),
        FN.FEAT_DYNAMIC_CAT: (np.arange(length + 4) % 3 + 1)[None],
        FN.FEAT_DYNAMIC_REAL: np.cos(np.arange(length + 4))[None].astype(np.float32),
        FN.PAST_FEAT_DYNAMIC_CAT: (np.arange(length) % 2 + 1)[None],
        FN.PAST_FEAT_DYNAMIC_REAL: np.sin(np.arange(length))[None].astype(np.float32),
    }


def schema():
    return FeatureSchema(
        feat_static_cat=(4,),
        feat_static_real=1,
        feat_dynamic_cat=(4,),
        feat_dynamic_real=1,
        past_feat_dynamic_cat=(3,),
        past_feat_dynamic_real=1,
    )


def core():
    return Chronos2Backbone(
        CoreConfig(
            d_model=12,
            d_kv=4,
            d_ff=24,
            num_layers=2,
            num_heads=2,
            context_length=32,
            patch_size=4,
            dropout_rate=0.0,
        )
    )


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "store"
        write_store([entry(), entry(offset=100)], self.root, "D")

    def tearDown(self):
        self.temp.cleanup()

    def test_selective_mapping_and_split_boundaries(self):
        dataset = MemmapWindows(
            self.root, 8, 4, fields=[FN.TARGET, FN.PAST_FEAT_DYNAMIC_REAL], end=16
        )
        self.assertEqual(len(dataset), 10)
        self.assertEqual(dataset._maps, {})
        value = dataset[4]
        torch.testing.assert_close(value[FN.TARGET], torch.arange(4, 12).float()[None])
        torch.testing.assert_close(
            value["future_target"], torch.arange(12, 16).float()[None]
        )
        self.assertEqual(value[FN.PAST_FEAT_DYNAMIC_REAL].shape, (1, 8))
        self.assertEqual(set(dataset._maps), {FN.TARGET, FN.PAST_FEAT_DYNAMIC_REAL})
        self.assertTrue(
            all(isinstance(mapping, np.memmap) for mapping in dataset._maps.values())
        )
        restored = pickle.loads(pickle.dumps(dataset))
        self.assertEqual(restored._maps, {})
        torch.testing.assert_close(restored[4][FN.TARGET], value[FN.TARGET])

    def test_predict_does_not_read_labels_or_past_future(self):
        dataset = MemmapWindows(self.root, 8, 4, mode="predict")
        window = dataset[0]
        self.assertNotIn("future_target", window)
        self.assertEqual(window[FN.FEAT_DYNAMIC_CAT].shape, (1, 12))
        self.assertEqual(window[FN.PAST_FEAT_DYNAMIC_CAT].shape, (1, 8))
        self.assertEqual(window[FN.FORECAST_START], pd.Period("2024-01-25", freq="D"))

    def test_worker_reopen(self):
        dataset = MemmapWindows(self.root, 8, 4, fields=[FN.TARGET], end=16)
        batches = list(
            DataLoader(dataset, batch_size=3, num_workers=2, collate_fn=collate_windows)
        )
        self.assertEqual(sum(len(batch) for batch in batches), len(dataset))

    def test_short_history_padding_and_missing_covariates(self):
        dataset = MemmapWindows(self.root, 32, 4, mode="predict")
        self.assertTrue(torch.isnan(dataset[0][FN.TARGET][..., :8]).all())
        self.assertTrue((dataset[0][FN.FEAT_DYNAMIC_CAT][..., :8] == 0).all())
        with self.assertRaises(ValueError):
            MemmapWindows(self.root, 8, 8, mode="predict")

    def test_missing_store_field_not_opened(self):
        (self.root / f"{FN.FEAT_STATIC_REAL}.bin").unlink()
        dataset = MemmapWindows(self.root, 8, 4, fields=[FN.TARGET])
        self.assertEqual(dataset[0][FN.TARGET].shape, (1, 8))

    def test_malformed_entries_and_overwrite(self):
        with self.assertRaises(FileExistsError):
            write_store([entry()], self.root, "D")
        bad = entry()
        bad[FN.FEAT_STATIC_CAT] = np.array([1.5])
        with self.assertRaises(ValueError):
            write_store([bad], Path(self.temp.name) / "bad", "D")


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        write_store([entry(), entry(offset=50)], self.root / "store", "D")
        self.dataset = MemmapWindows(self.root / "store", 8, 3, end=16)
        self.batch = [self.dataset[0], self.dataset[len(self.dataset) // 2]]
        self.model = SplitChronos2(core(), schema())

    def tearDown(self):
        self.dataset._maps.clear()
        self.temp.cleanup()

    def test_all_seven_fields_train_and_roundtrip(self):
        before = self.model.embeddings[FN.FEAT_STATIC_CAT][0].weight.detach().clone()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        output = self.model(self.batch, 3)
        self.assertEqual(output["quantile_preds"].shape, (2, 1, 3, 3))
        self.assertTrue(torch.isfinite(output["loss"]))
        output["loss"].backward()
        for name in self.model.embeddings:
            for layer in self.model.embeddings[name]:
                self.assertTrue(torch.isfinite(layer.weight.grad).all())
                self.assertGreater(layer.weight.grad.abs().sum().item(), 0)
        self.assertGreater(self.model.static_real[0].weight.grad.abs().sum().item(), 0)
        optimizer.step()
        self.assertFalse(
            torch.equal(before, self.model.embeddings[FN.FEAT_STATIC_CAT][0].weight)
        )
        self.model.eval()
        self.model.save_local(self.root / "checkpoint")
        restored = SplitChronos2.from_local(self.root / "checkpoint").eval()
        torch.testing.assert_close(
            self.model(self.batch, 3)["quantile_preds"],
            restored(self.batch, 3)["quantile_preds"],
        )
        forecasts = restored.predict(self.batch, 3)
        self.assertEqual(forecasts[0].quantile(0.5).shape, (3,))

    def test_group_isolation_and_label_leakage(self):
        self.model.eval()
        baseline = self.model(self.batch, 3)["quantile_preds"].detach()
        changed = copy.deepcopy(self.batch)
        changed[0]["future_target"].fill_(1e6)
        changed[1][FN.TARGET].fill_(1234)
        changed[1][FN.FEAT_STATIC_CAT].fill_(2)
        actual = self.model(changed, 3)["quantile_preds"].detach()
        torch.testing.assert_close(actual[0], baseline[0])

    def test_selected_fields_skip_encoders(self):
        model = SplitChronos2(core(), schema(), fields=[FN.TARGET])
        self.assertEqual(len(model.embeddings), 0)
        small = [
            {
                key: value
                for key, value in item.items()
                if key in (FN.TARGET, "future_target")
            }
            for item in self.batch
        ]
        torch.testing.assert_close(
            model(small, 3)["quantile_preds"], model(self.batch, 3)["quantile_preds"]
        )

    def test_static_identity_and_future_covariates_affect_prediction(self):
        self.model.eval()
        baseline = self.model(self.batch, 3)["quantile_preds"].detach()
        changed = copy.deepcopy(self.batch)
        changed[0][FN.FEAT_STATIC_CAT].fill_(2)
        self.assertFalse(
            torch.equal(baseline[0], self.model(changed, 3)["quantile_preds"][0])
        )
        changed = copy.deepcopy(self.batch)
        changed[0][FN.FEAT_DYNAMIC_REAL][:, -3:] += 100
        self.assertFalse(
            torch.equal(baseline[0], self.model(changed, 3)["quantile_preds"][0])
        )

    def test_nan_targets_and_category_validation(self):
        self.batch[0][FN.TARGET].fill_(float("nan"))
        self.batch[0]["future_target"].fill_(float("nan"))
        self.assertTrue(torch.isfinite(self.model(self.batch, 3)["loss"]))
        self.batch[0][FN.FEAT_STATIC_CAT].fill_(99)
        with self.assertRaises(ValueError):
            self.model(self.batch, 3)

    def test_multivariate_and_no_reg_token(self):
        config = CoreConfig(
            d_model=8,
            d_ff=16,
            d_kv=4,
            num_heads=2,
            num_layers=1,
            patch_size=4,
            use_reg_token=False,
            dropout_rate=0.0,
        )
        model = SplitChronos2(Chronos2Backbone(config), FeatureSchema(target_dim=2))
        batch = [
            {
                FN.TARGET: torch.randn(2, 7),
                FN.FORECAST_START: pd.Period("2024-01-01", freq="D"),
            }
        ]
        self.assertEqual(model(batch, 5)["quantile_preds"].shape, (1, 2, 3, 5))
        self.assertEqual(model.predict(batch, 5)[0].quantile(0.5).shape, (5, 2))

    def test_original_checkpoint_loads_strictly(self):
        if not DUMMY.exists():
            self.skipTest("Reference test weights not present")
        backbone = Chronos2Backbone.from_local(DUMMY).eval()
        self.assertEqual(len(backbone.state_dict()), 40)
        target = torch.arange(25).float()[None]
        model = SplitChronos2(backbone, FeatureSchema()).eval()
        torch.testing.assert_close(
            backbone(target, prediction_length=7),
            model([{FN.TARGET: target}], 7)["quantile_preds"][0],
        )

    def test_numerical_parity_with_upstream_source(self):
        from upstream_reference import load_upstream

        if not REFERENCE.exists():
            self.skipTest("Reference repository not present")
        reference = load_upstream(REFERENCE, DUMMY)
        local = Chronos2Backbone.from_local(DUMMY).eval()
        for history, horizon in ((25, 7), (32, 19)):
            past = torch.randn(4, history)
            past[0, :3] = float("nan")
            past[3] = float("nan")
            future = torch.full((4, horizon), float("nan"))
            future[1] = torch.randn(horizon)
            groups = torch.tensor([0, 0, 1, 1])
            expected = reference(
                context=past,
                future_covariates=future,
                group_ids=groups,
                num_output_patches=(horizon + 15) // 16,
            ).quantile_preds[..., :horizon]
            actual = local(past, future, groups)
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
