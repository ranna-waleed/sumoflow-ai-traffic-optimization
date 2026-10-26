from __future__ import annotations

import io
import math
import unittest

import numpy as np
import torch

from DeepQN.forecasting.contracts import ForecastContractError
from DeepQN.forecasting.inference import ResearchBiLSTM
from DeepQN.forecasting.research_contract import (
    PoolIdentity, fit_historical_mean, forecast_baselines, poisson_count_loss,
    reporting_sample_time, score_by_realization, score_counts, validate_scientific_pools,
)


class CausalTimeContractTests(unittest.TestCase):
    def test_exact_and_adjacent_boundaries_use_post_step_right_closed_bins(self):
        first, last = 0.0, 100.0
        self.assertEqual(reporting_sample_time(0.0, first, last), 0.0)
        self.assertEqual(reporting_sample_time(59.999999, first, last), 60.0)
        self.assertEqual(reporting_sample_time(60.0, first, last), 60.0)
        self.assertEqual(reporting_sample_time(60.000001, first, last), 61.0)
        self.assertEqual(reporting_sample_time(89.999999, first, last), 90.0)
        self.assertEqual(reporting_sample_time(90.0, first, last), 90.0)
        self.assertEqual(reporting_sample_time(90.000001, first, last), 91.0)

    def test_nonzero_begin_and_outside_or_nonfinite_events_fail(self):
        self.assertEqual(reporting_sample_time(27000.0, 27001.0, 27100.0), 27001.0)
        self.assertEqual(reporting_sample_time(27001.0, 27001.0, 27100.0), 27001.0)
        self.assertEqual(reporting_sample_time(27001.000001, 27001.0, 27100.0), 27002.0)
        for value in (26999.999, 27100.001, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ForecastContractError):
                reporting_sample_time(value, 27001.0, 27100.0)


class ResearchCountModelTests(unittest.TestCase):
    def test_shape_nonnegative_output_and_finite_gradients(self):
        torch.manual_seed(7)
        model = ResearchBiLSTM(hidden_size=8, num_layers=1, dropout=0)
        values = torch.randn(3, 60, 7)
        prediction = model(values)
        self.assertEqual(tuple(prediction.shape), (3, 4))
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertTrue(torch.all(prediction >= 0))
        for targets in (torch.zeros(3, 4), torch.full((3, 4), 10000.0)):
            model.zero_grad(set_to_none=True)
            loss = poisson_count_loss(model(values), targets)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                                for parameter in model.parameters()))

    def test_cpu_state_round_trip_is_equivalent_and_old_shape_rejected(self):
        torch.manual_seed(11)
        model = ResearchBiLSTM(6, 1, 0).eval()
        values = torch.randn(2, 60, 7)
        expected = model(values).detach()
        buffer = io.BytesIO(); torch.save(model.state_dict(), buffer); buffer.seek(0)
        restored = ResearchBiLSTM(6, 1, 0).eval()
        restored.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True), strict=True)
        torch.testing.assert_close(restored(values), expected, rtol=0, atol=0)
        incompatible = dict(model.state_dict())
        incompatible["lstm.weight_ih_l0"] = torch.zeros(24, 6)
        with self.assertRaises(RuntimeError):
            restored.load_state_dict(incompatible, strict=True)

    def test_invalid_count_loss_inputs_fail_closed(self):
        valid = torch.ones(2, 4)
        for prediction, target in (
            (torch.full((2, 4), -1.0), valid),
            (torch.full((2, 4), float("nan")), valid),
            (valid, torch.full((2, 4), -1.0)),
            (torch.ones(2, 3), torch.ones(2, 3)),
        ):
            with self.assertRaises(ForecastContractError):
                poisson_count_loss(prediction, target)


class ForecastBaselineAndLeakageTests(unittest.TestCase):
    def test_baselines_use_history_only_and_convert_rates_to_30_second_counts(self):
        history = np.zeros((60, 7), dtype=np.float64)
        history[-30:, 0] = 1
        history[-10:, 1] = 2
        history[:, 4] = history[:, :4].sum(axis=1)
        mean = fit_historical_mean(np.asarray([[2, 4, 6, 8], [4, 6, 8, 10]]))
        result = forecast_baselines(history, historical_mean=mean,
                                    moving_average_seconds=10)
        np.testing.assert_array_equal(result["zero"], [0, 0, 0, 0])
        np.testing.assert_array_equal(result["persistence"], [30, 20, 0, 0])
        np.testing.assert_array_equal(result["moving_average_rate"], [30, 60, 0, 0])
        np.testing.assert_array_equal(result["historical_mean"], [3, 5, 7, 9])
        changed_future = np.full((30, 4), 999.0)
        self.assertNotIn(id(changed_future), [id(value) for value in result.values()])
        with self.assertRaisesRegex(ForecastContractError, "training pool"):
            fit_historical_mean(np.ones((2, 4)), fit_pool="forecast_test")

    def test_scores_preserve_zero_count_cases_without_percentage_fabrication(self):
        score = score_counts(np.asarray([[0, 1, 2, 3]], dtype=float),
                             np.asarray([[0, 0, 2, 4]], dtype=float))
        self.assertTrue(math.isfinite(score["mae"]))
        self.assertTrue(math.isfinite(score["mean_poisson_deviance"]))
        self.assertIsNone(score["percentage_error"])
        self.assertEqual(score["directional_bias"], [0.0, 1.0, 0.0, -1.0])
        grouped = score_by_realization(
            np.asarray([[0, 1, 2, 3], [1, 1, 1, 1]], dtype=float),
            np.asarray([[0, 0, 2, 4], [0, 1, 1, 2]], dtype=float),
            ["demand-a", "demand-b"],
        )
        self.assertEqual(set(grouped), {"demand-a", "demand-b"})

    def test_all_scientific_pools_are_physically_disjoint(self):
        rows = tuple(PoolIdentity(f"r-{index}", pool,
                                  f"{index + 1:064x}", f"{index + 101:064x}")
                     for index, pool in enumerate((
                         "training", "calibration", "validation", "forecast_test",
                         "controller_evaluation")))
        self.assertEqual(validate_scientific_pools(rows), rows)
        for changed in (
            rows[:-1] + (PoolIdentity("r-0", "controller_evaluation",
                                     rows[-1].source_sha256,
                                     rows[-1].physical_fingerprint_sha256),),
            rows[:-1] + (PoolIdentity("new", "controller_evaluation",
                                     rows[0].source_sha256,
                                     rows[-1].physical_fingerprint_sha256),),
            rows[:-1] + (PoolIdentity("new", "controller_evaluation",
                                     rows[-1].source_sha256,
                                     rows[0].physical_fingerprint_sha256),),
        ):
            with self.assertRaisesRegex(ForecastContractError, "reused"):
                validate_scientific_pools(changed)


if __name__ == "__main__":
    unittest.main()
