import importlib.util
from pathlib import Path
import unittest


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ChunkReportingTest(unittest.TestCase):
    def test_matrix_size_unique_outputs_and_smoke_flag(self):
        launcher = load_script("run_chunk_matrix")
        jobs = list(launcher.matrix_jobs("/test"))
        self.assertEqual(len(jobs), 32)
        self.assertEqual(len({job["output"] for job in jobs}), 32)
        smoke = list(launcher.matrix_jobs("/test", smoke=True))
        self.assertEqual(len(smoke), 16)
        self.assertTrue(all("--smoke" in job["command"] for job in smoke))

    def test_auc_and_paired_differences(self):
        report = load_script("summarize_chunk_results")
        self.assertEqual(report.auc([(0, 10), (5, 20), (10, 0)]), 125)
        runs = []
        for horizon in (1, 5):
            for seed in (1, 2):
                for method in ("b0", "n"):
                    value = seed + (horizon * 2 if method == "n" else 0)
                    runs.append(dict(env_name="door", method=method, chunk_size=horizon, seed=seed,
                                     **{key: value for key in ("offline_endpoint", "online_start", "online_endpoint", "online_auc", "online_auc_mean")}))
        _, paired = report.aggregate(runs)
        h5 = next(row for row in paired if row["chunk_size"] == 5)
        self.assertEqual(h5["delta_online_endpoint"]["mean"], 10)
        self.assertEqual(h5["delta_minus_H1_online_endpoint"]["mean"], 8)
        with self.assertRaisesRegex(ValueError, "Duplicate seed"):
            report.aggregate(runs + [runs[0]])


if __name__ == "__main__":
    unittest.main()
