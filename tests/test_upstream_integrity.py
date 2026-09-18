import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]

# This branch (`audit-fixes`) is *audit-patched* upstream, not pristine
# upstream.  Three files carry opt-in patches that default to the upstream
# behaviour (see docs/AUDIT_PATCHES.md):
#   * main_meanflowql.py   - adds the `strict_norm_stats` flag (default False)
#   * agents/meanflowql.py - makes the critic learning rate configurable
#                            (`critic_lr`, default 3e-4 = upstream value)
#   * utils/datasets.py    - `compute_normalization_stats(valid_only=...)`
#                            (default False = upstream behaviour)
# The hashes below therefore describe the patched files.  The original upstream
# hashes are recorded in docs/AUDIT_PATCHES.md; runs produced from this branch
# must be labelled `audit-patched` and must not be mixed with pristine-upstream
# results.
UPSTREAM_SHA256 = {
    "main_meanflowql.py": "ba6d9adc063041914b0b9966f0071672ee112285a6e9d270ee60cd46ca31b5ff",
    "agents/meanflowql.py": "ef8202d6041246fd458fad1c043b9e8df28cc0649c51de8b0380aff05afedcfd",
    "utils/evaluation.py": "139762c6a941534b4d6139842aacead9f4063f3fa2784b70749ed37d03fbce68",
    "utils/datasets.py": "f04914c4886252e48890ae53ac5526c98b6a291a8fee8cc9df6480ad047bac8e",
    "utils/networks.py": "dca1e7aa4645fcaf343d73566cab0cd3854ff2da1b9326c553b86702c9ac9cf2",
    "utils/dit_jax.py": "686ffbc68f135cca26ea4d79b28e2ac0889225036dea07cf597c734f802a5fa5",
    "utils/flax_utils.py": "fa4f68a9f12b9b983560ced7e45cd2a99ae761752d8fd5a6c2f6611a13a4d4f8",
    "envs/env_utils.py": "e40b47d8b5bdfaecd43940f7a0b85b1a3eee554d41fa9737da501d2f71d7f032",
}


class UpstreamIntegrityTest(unittest.TestCase):
    def test_frozen_training_and_evaluation_files(self):
        for relative, expected in UPSTREAM_SHA256.items():
            path = ROOT / relative
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(actual, expected, relative)

    def test_no_unified_runner_is_present(self):
        self.assertFalse((ROOT / "utils/experiment.py").exists())
        source = (ROOT / "main_meanflowql.py").read_text()
        self.assertNotIn("unified_v1", source)
        self.assertNotIn("am_meanflow_note", source)

    def test_audit_patches_are_present_and_opt_in(self):
        main = (ROOT / "main_meanflowql.py").read_text()
        compact_main = main.replace(" ", "")
        self.assertIn("'strict_norm_stats',False", compact_main)
        self.assertIn("valid_only=FLAGS.strict_norm_stats", compact_main)

        agent = (ROOT / "agents/meanflowql.py").read_text()
        self.assertIn("config.get('critic_lr', 3e-4)", agent)

        datasets = (ROOT / "utils/datasets.py").read_text()
        self.assertIn("def compute_normalization_stats(self, valid_only=False)", datasets)
        self.assertIn("observations[: self.size]", datasets)


if __name__ == "__main__":
    unittest.main()
