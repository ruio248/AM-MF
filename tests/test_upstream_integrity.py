import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_SHA256 = {
    "main_meanflowql.py": "026a0ced1e5346d40b729827e0d1921cdc55d73049118a737ecfa0bae76a6fd7",
    "agents/meanflowql.py": "50525c5ada2462ce47981390f89fefa5c71afad67891e61579a4e94ac0307b8c",
    "utils/evaluation.py": "139762c6a941534b4d6139842aacead9f4063f3fa2784b70749ed37d03fbce68",
    "utils/datasets.py": "0a39d5c9da3b3ba3c37200abf451f3bbcb14a54d7b07872e25d8f9e7c28b40fa",
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


if __name__ == "__main__":
    unittest.main()
