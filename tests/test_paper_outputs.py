import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from paper_outputs import HEATMAP_ARTIFACTS, PaperPackBuilder, _summarize_integrator


class PaperOutputSummaryTests(unittest.TestCase):
    def test_integrator_summary_uses_explicit_spherical_height_columns(self):
        rows = [
            {
                "rtol": "1e-6",
                "atol": "1e-9",
                "final_spherical_height_m": "419800.0",
                "final_velocity_m_s": "7657.0",
                "spherical_height_error_m": "-200.0",
                "velocity_error_m_s": "1.0",
                "radial_velocity_m_s": "0.9",
                "raw_path_ok": "1",
                "max_q_pa": "34890.0",
                "max_g": "3.98",
            },
            {
                "rtol": "1e-8",
                "atol": "1e-11",
                "final_spherical_height_m": "419990.0",
                "final_velocity_m_s": "7657.25",
                "spherical_height_error_m": "-10.0",
                "velocity_error_m_s": "0.05",
                "radial_velocity_m_s": "0.04",
                "raw_path_ok": "1",
                "max_q_pa": "34852.0",
                "max_g": "3.976",
            },
            {
                "rtol": "1e-10",
                "atol": "1e-13",
                "final_spherical_height_m": "420000.0",
                "final_velocity_m_s": "7657.30",
                "spherical_height_error_m": "0.0",
                "velocity_error_m_s": "0.0",
                "radial_velocity_m_s": "0.0",
                "raw_path_ok": "1",
                "max_q_pa": "34850.0",
                "max_g": "3.975",
            },
        ]

        summary = _summarize_integrator(rows)

        self.assertEqual(summary["selected_idx"], 1)
        self.assertEqual(summary["thresholds"]["altitude_m"], 50.0)

    def test_skip_heatmap_keeps_existing_artifacts_visible_to_manifest(self):
        with TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                output_dir=tmpdir,
                skip_heatmap=True,
            )
            builder = PaperPackBuilder(args)
            builder._ensure_dirs()

            for rel_path in HEATMAP_ARTIFACTS:
                path = Path(tmpdir) / rel_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("placeholder", encoding="utf-8")

            self.assertTrue(builder._heatmap_artifacts_exist())

            builder.has_heatmap_outputs = builder._heatmap_artifacts_exist()
            builder._write_notes()

            manifest = (Path(tmpdir) / "notes" / "paper_pack_manifest.md").read_text(encoding="utf-8")
            self.assertIn("fig_app_05_global_launch_cost", manifest)


if __name__ == "__main__":
    unittest.main()
