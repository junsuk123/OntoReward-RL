import tempfile
import unittest
from pathlib import Path

from simlab.config import load_config
from simlab.ml.lineage import (
    active_model_for_signature,
    drone_model_signature,
    latest_model_for_signature,
    lineage_dir,
    write_model_pointer,
)


class ContinualTrainingLineageTest(unittest.TestCase):
    def test_signature_tracks_airframe_not_fleet_count(self):
        first = load_config()
        second = load_config()
        second.drones.friendly.count += 3
        self.assertEqual(drone_model_signature(first), drone_model_signature(second))
        second.drones.friendly.asset_scale += 0.1
        self.assertNotEqual(drone_model_signature(first), drone_model_signature(second))

    def test_model_pointers_are_scoped_by_signature(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory)
            model = artifacts / "best.pt"
            model.touch()
            signature = "abc123"
            write_model_pointer(lineage_dir(artifacts, signature) / "latest_model.txt", model)
            self.assertEqual(str(model), latest_model_for_signature(artifacts, signature))
            write_model_pointer(artifacts / "active_model.txt", model)
            (artifacts / "active_signature.txt").write_text(signature + "\n", encoding="utf-8")
            self.assertEqual(str(model), active_model_for_signature(artifacts, signature))
            self.assertIsNone(active_model_for_signature(artifacts, "different"))


if __name__ == "__main__":
    unittest.main()
