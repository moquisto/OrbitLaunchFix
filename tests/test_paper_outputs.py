import copy
import csv
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from analysis_tools.paper_outputs import (
    HEATMAP_ARTIFACTS,
    MULTISTART_ARTIFACTS,
    PaperPackBuilder,
    _refine_peak_time_value,
    _setup_plot_style,
    _summarize_integrator,
)
from analysis_tools.relabilityanalysis import ReliabilitySuite
from orbit_launch.config import EARTH_CONFIG, StarshipBlock2


class PaperOutputSummaryTests(unittest.TestCase):
    def test_setup_plot_style_enables_editable_vector_fonts(self):
        _setup_plot_style()

        self.assertEqual(plt.rcParams["pdf.fonttype"], 42)
        self.assertEqual(plt.rcParams["ps.fonttype"], 42)
        self.assertFalse(plt.rcParams["axes.grid"])

    def test_finalize_figure_allows_constrained_layout_with_colorbar(self):
        fake_suite = SimpleNamespace(
            save_outputs=False,
            show_plots=False,
            fig_dir=Path("tmp/test_figs"),
        )
        fig = plt.figure(figsize=(4.0, 3.0), layout="constrained")
        ax = fig.add_subplot(111)
        image = ax.imshow(np.arange(4, dtype=float).reshape(2, 2))
        fig.colorbar(image, ax=ax)

        ReliabilitySuite._finalize_figure(fake_suite, fig, "dummy")

    def test_global_launch_cost_render_uses_3d_heatmap_figure(self):
        captured = {}
        fake_suite = SimpleNamespace(
            save_outputs=False,
            show_plots=False,
            fig_dir=Path("tmp/test_figs"),
        )

        def finalize(fig, stem):
            captured["fig"] = fig
            captured["stem"] = stem

        fake_suite._finalize_figure = finalize
        rows = [
            {
                "latitude_deg": -90.0,
                "fuel_consumed_kg": 4_697_820.9,
                "final_mass_kg": 277_179.1,
                "earth_rotation_speed_m_s": 0.0,
                "solver_success": 1,
            },
            {
                "latitude_deg": 0.0,
                "fuel_consumed_kg": 4_653_433.1,
                "final_mass_kg": 321_566.9,
                "earth_rotation_speed_m_s": 465.1,
                "solver_success": 1,
            },
            {
                "latitude_deg": 45.0,
                "fuel_consumed_kg": np.nan,
                "final_mass_kg": np.nan,
                "earth_rotation_speed_m_s": np.nan,
                "solver_success": 0,
            },
            {
                "latitude_deg": 90.0,
                "fuel_consumed_kg": 4_697_820.9,
                "final_mass_kg": 277_179.1,
                "earth_rotation_speed_m_s": 0.0,
                "solver_success": 1,
            },
        ]

        rendered = ReliabilitySuite._render_global_launch_cost_heatmap(fake_suite, rows, "global_launch_cost_heatmap")

        self.assertTrue(rendered)
        self.assertEqual(captured["stem"], "global_launch_cost_heatmap")
        fig = captured["fig"]
        self.assertGreaterEqual(len(fig.axes), 2)
        self.assertEqual(getattr(fig.axes[0], "name", ""), "3d")
        self.assertEqual(fig.axes[0].get_title(), "")
        self.assertEqual(fig.axes[1].get_ylabel(), "Fuel consumed (kg)")
        plt.close(fig)

    def test_prepare_reliability_reenables_multistart_for_paper_runs(self):
        with TemporaryDirectory() as tmpdir:
            opt_res = {"success": True, "payload": {"x": 1}}
            args = SimpleNamespace(
                output_dir=tmpdir,
                reliability_dir="",
                max_workers=2,
                skip_heatmap=True,
            )
            builder = PaperPackBuilder(args)

            suite_mock = SimpleNamespace(
                test_toggles=SimpleNamespace(
                    randomized_multistart=False,
                    grid_independence=True,
                    interval_replay_audit=True,
                ),
                _baseline_opt_res=None,
            )
            suite_mock._save_table = Mock()
            suite_mock.run_all = Mock()

            with patch("analysis_tools.paper_outputs.ReliabilitySuite", return_value=suite_mock) as suite_cls:
                out_dir = builder._prepare_reliability(opt_res)

            suite_cls.assert_called_once()
            self.assertTrue(suite_mock.test_toggles.randomized_multistart)
            self.assertEqual(suite_mock._baseline_opt_res, opt_res)
            self.assertIsNot(suite_mock._baseline_opt_res, opt_res)
            self.assertIs(builder._reliability_suite, suite_mock)
            suite_mock._save_table.assert_called_once()
            self.assertEqual(suite_mock._save_table.call_args.args[0], "enabled_test_toggles")
            suite_mock.run_all.assert_called_once_with(max_workers=2)
            self.assertEqual(out_dir, Path(tmpdir) / "reliability_raw")

    def test_script_entry_point_runs_help_from_repo_root(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "analysis_tools" / "paper_outputs.py"

        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Generate a curated output pack for the final paper.", result.stdout)

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
            self.assertIn("fig_main_06_warm_start_multiseed", manifest)
            self.assertNotIn("fig_main_05_replay_drift", manifest)
            self.assertIn("table_04_replay_drift_by_phase", manifest)
            self.assertIn("table_05_multistart_consistency", manifest)
            self.assertNotIn("fig_app_06_warm_start_multiseed", manifest)

    def test_multistart_artifacts_exist_helper_checks_top_level_outputs(self):
        with TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                output_dir=tmpdir,
                skip_heatmap=True,
            )
            builder = PaperPackBuilder(args)
            builder._ensure_dirs()

            for rel_path in MULTISTART_ARTIFACTS:
                path = Path(tmpdir) / rel_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("placeholder", encoding="utf-8")

            self.assertTrue(builder._multistart_artifacts_exist())

    def test_build_launch_cost_heatmap_copies_suite_artifacts(self):
        with TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                output_dir=tmpdir,
                skip_heatmap=False,
                force_heatmap=False,
                reliability_dir="",
                heatmap_lat_step=5.0,
                heatmap_solver_cpu_time=120.0,
            )
            builder = PaperPackBuilder(args)
            builder._ensure_dirs()

            suite_dir = Path(tmpdir) / "reliability_raw"
            (suite_dir / "data").mkdir(parents=True, exist_ok=True)
            (suite_dir / "figures").mkdir(parents=True, exist_ok=True)
            (suite_dir / "data" / "global_launch_cost_latitude_sweep.csv").write_text(
                "latitude_deg,fuel_consumed_kg,final_mass_kg,earth_rotation_speed_m_s,solver_success\n"
                "-90,4697820.9,277179.1,0.0,1\n"
                "0,4653433.1,321566.9,465.1,1\n"
                "90,4697820.9,277179.1,0.0,1\n",
                encoding="utf-8",
            )
            (suite_dir / "figures" / "global_launch_cost_heatmap.png").write_text("png", encoding="utf-8")
            (suite_dir / "figures" / "global_launch_cost_heatmap.pdf").write_text("pdf", encoding="utf-8")
            builder._reliability_suite = SimpleNamespace(
                output_dir=suite_dir,
                analyze_global_launch_cost=Mock(side_effect=AssertionError("should not recompute when suite artifacts exist")),
            )

            copied = builder._build_launch_cost_heatmap()

            self.assertTrue(copied)
            self.assertTrue(builder._heatmap_artifacts_exist())
            builder._reliability_suite.analyze_global_launch_cost.assert_not_called()
            self.assertIn("fuel_consumed_kg", (Path(tmpdir) / "data" / "global_launch_cost_latitude_sweep.csv").read_text(encoding="utf-8"))
            self.assertGreater((Path(tmpdir) / "figures" / "fig_app_05_global_launch_cost.png").stat().st_size, 0)
            self.assertGreater((Path(tmpdir) / "figures" / "fig_app_05_global_launch_cost.pdf").stat().st_size, 0)

    def test_build_launch_cost_heatmap_recovers_from_cached_csv_without_recompute(self):
        with TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                output_dir=tmpdir,
                skip_heatmap=False,
                force_heatmap=False,
                reliability_dir="",
                heatmap_lat_step=5.0,
                heatmap_solver_cpu_time=120.0,
            )
            builder = PaperPackBuilder(args)
            builder._ensure_dirs()

            suite_dir = Path(tmpdir) / "reliability_raw"
            (suite_dir / "data").mkdir(parents=True, exist_ok=True)
            (suite_dir / "figures").mkdir(parents=True, exist_ok=True)
            (suite_dir / "data" / "global_launch_cost_latitude_sweep.csv").write_text(
                "latitude_deg,fuel_consumed_kg,final_mass_kg,earth_rotation_speed_m_s,solver_success\n"
                "-90,4697820.9,277179.1,0.0,1\n"
                "0,4630000.0,345000.0,465.0,1\n"
                "90,4697820.9,277179.1,0.0,1\n",
                encoding="utf-8",
            )
            builder._reliability_suite = SimpleNamespace(
                output_dir=suite_dir,
                analyze_global_launch_cost=Mock(side_effect=AssertionError("should not recompute when cached CSV exists")),
            )

            recovered = builder._build_launch_cost_heatmap()

            self.assertTrue(recovered)
            self.assertTrue((suite_dir / "figures" / "global_launch_cost_heatmap.png").exists())
            self.assertTrue((suite_dir / "figures" / "global_launch_cost_heatmap.pdf").exists())
            self.assertTrue((Path(tmpdir) / "figures" / "fig_app_05_global_launch_cost.png").exists())
            self.assertTrue((Path(tmpdir) / "figures" / "fig_app_05_global_launch_cost.pdf").exists())
            self.assertTrue((Path(tmpdir) / "data" / "global_launch_cost_latitude_sweep.csv").exists())

    def test_build_launch_cost_heatmap_regenerates_stale_output_figure_from_cached_csv(self):
        with TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                output_dir=tmpdir,
                skip_heatmap=False,
                force_heatmap=False,
                reliability_dir="",
                heatmap_lat_step=5.0,
                heatmap_solver_cpu_time=120.0,
            )
            builder = PaperPackBuilder(args)
            builder._ensure_dirs()

            output_csv = Path(tmpdir) / "data" / "global_launch_cost_latitude_sweep.csv"
            output_csv.write_text(
                "latitude_deg,fuel_consumed_kg,final_mass_kg,earth_rotation_speed_m_s,solver_success\n"
                "-90,4697820.9,277179.1,0.0,1\n"
                "0,4653433.1,321566.9,465.1,1\n"
                "90,4697820.9,277179.1,0.0,1\n",
                encoding="utf-8",
            )
            stale_png = Path(tmpdir) / "figures" / "fig_app_05_global_launch_cost.png"
            stale_pdf = Path(tmpdir) / "figures" / "fig_app_05_global_launch_cost.pdf"
            stale_png.write_text("stale", encoding="utf-8")
            stale_pdf.write_text("stale", encoding="utf-8")

            regenerated = builder._build_launch_cost_heatmap()

            self.assertTrue(regenerated)
            self.assertGreater(stale_png.stat().st_size, len("stale"))
            self.assertGreater(stale_pdf.stat().st_size, len("stale"))

    def test_copy_multistart_artifacts_promotes_outputs_into_paper_pack(self):
        with TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                output_dir=tmpdir,
                skip_heatmap=True,
            )
            builder = PaperPackBuilder(args)
            builder._ensure_dirs()

            reliability_dir = Path(tmpdir) / "reliability_raw"
            (reliability_dir / "figures").mkdir(parents=True, exist_ok=True)
            (reliability_dir / "data").mkdir(parents=True, exist_ok=True)
            (reliability_dir / "figures" / "randomized_multistart.png").write_text("png", encoding="utf-8")
            (reliability_dir / "figures" / "randomized_multistart.pdf").write_text("pdf", encoding="utf-8")
            (reliability_dir / "data" / "randomized_multistart.csv").write_text(
                "trial,label,status,solver_success,mission_success,final_mass_gap_to_best_kg,max_q_pa,max_g,alt_err_m,vel_err_m_s\n"
                "1,nominal,PASS_RAW,1,1,1.5e-05,34850.0,3.975,0.0,0.0\n"
                "2,time_minus,PASS_RAW,1,1,6.1e-08,34890.0,3.980,0.4,0.03\n",
                encoding="utf-8",
            )
            (reliability_dir / "data" / "randomized_multistart_deltas.csv").write_text(
                "trial,label,status,position_separation_km,delta_velocity_m_s\n"
                "1,nominal,PASS_RAW,0.0,0.0\n"
                "2,time_minus,PASS_RAW,0.012,0.03\n",
                encoding="utf-8",
            )
            (reliability_dir / "data" / "randomized_multistart_trajectory_summary.csv").write_text(
                "metric,value\nsolver_rate,1.0\nmission_success_rate_raw,1.0\nmedian_success_runtime_s,89.9\nmax_path_utilization_ratio,0.9953\nmax_terminal_criteria_ratio,0.00034\nmass_spread_kg,1.0e-5\n",
                encoding="utf-8",
            )

            copied = builder._copy_multistart_artifacts(reliability_dir)

            self.assertTrue(copied)
            self.assertTrue(builder._multistart_artifacts_exist())
            self.assertGreater((Path(tmpdir) / "figures" / "fig_main_06_warm_start_multiseed.png").stat().st_size, 0)
            self.assertGreater((Path(tmpdir) / "figures" / "fig_main_06_warm_start_multiseed.pdf").stat().st_size, 0)
            self.assertIn("nominal", (Path(tmpdir) / "data" / "randomized_multistart.csv").read_text(encoding="utf-8"))
            self.assertIn("time_minus", (Path(tmpdir) / "data" / "randomized_multistart_deltas.csv").read_text(encoding="utf-8"))

    def test_multistart_overview_uses_local_minimum_metrics(self):
        with TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                output_dir=tmpdir,
                skip_heatmap=True,
            )
            builder = PaperPackBuilder(args)
            builder._ensure_dirs()
            rows = [
                {
                    "trial": "1",
                    "label": "nominal",
                    "status": "PASS_RAW",
                    "solver_success": "1",
                    "mission_success": "1",
                    "final_mass_gap_to_best_kg": "1.5e-05",
                },
                {
                    "trial": "2",
                    "label": "state_plus",
                    "status": "PASS_RAW",
                    "solver_success": "1",
                    "mission_success": "1",
                    "final_mass_gap_to_best_kg": "6.1e-08",
                },
            ]
            summary = {
                "mission_success_rate_raw": "1.0",
            }
            delta_summary = {
                "nominal": {
                    "max_position_separation_m": 0.0,
                    "max_velocity_delta_m_s": 0.0,
                },
                "state_plus": {
                    "max_position_separation_m": 11.7,
                    "max_velocity_delta_m_s": 0.032,
                },
            }
            captured = {}

            def capture_figure(fig, stem):
                captured["fig"] = fig
                captured["stem"] = stem

            with patch("analysis_tools.paper_outputs._save_figure", side_effect=capture_figure):
                rendered = builder._render_multistart_overview(
                    rows,
                    summary,
                    delta_summary,
                    Path(tmpdir) / "figures" / "fig_main_06_warm_start_multiseed",
                )

            self.assertTrue(rendered)
            fig = captured["fig"]
            xlabels = [ax.get_xlabel() for ax in fig.axes]
            self.assertEqual(
                xlabels,
                [
                    "Final-mass gap to best (mg)",
                    "Max position separation (m)",
                    "Max velocity delta (m/s)",
                ],
            )
            self.assertEqual(len(fig.axes), 3)
            self.assertEqual(fig.axes[0].get_xscale(), "symlog")
            runtime_axis = fig.axes[0]
            runtime_tick_labels = [tick.get_text() for tick in runtime_axis.get_yticklabels()]
            self.assertIn("Nominal", runtime_tick_labels)
            self.assertIn("Lateral +", runtime_tick_labels)
            self.assertGreaterEqual(runtime_axis.get_xlim()[0], 0.0)
            axis_text = [text.get_text() for text in runtime_axis.texts]
            self.assertIn("Raw-feasible: 2/2", axis_text)
            plt.close(fig)

    def test_multistart_consistency_rows_format_seed_metrics_for_table(self):
        args = SimpleNamespace(
            output_dir="unused",
            skip_heatmap=True,
        )
        builder = PaperPackBuilder(args)
        rows = [
            {
                "label": "nominal",
                "status": "PASS_RAW",
                "final_mass_gap_to_best_kg": "1.5e-05",
            },
            {
                "label": "rand_05",
                "status": "PASS_SLACK",
                "final_mass_gap_to_best_kg": "6.1e-08",
            },
        ]
        delta_summary = {
            "nominal": {
                "max_position_separation_m": 0.0,
                "max_velocity_delta_m_s": 0.0,
            },
            "rand_05": {
                "max_position_separation_m": 10.527,
                "max_velocity_delta_m_s": 0.0224,
            },
        }

        table_rows = builder._build_multistart_consistency_rows(rows, delta_summary)

        self.assertEqual(
            table_rows,
            [
                ("Nominal", "Raw pass", "15.000", "0.000", "0.0000"),
                ("Random 1", "Slack pass", "0.061", "10.527", "0.0224"),
            ],
        )

    def test_nominal_profile_overlays_optimizer_and_replay_when_histories_available(self):
        with TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                output_dir=tmpdir,
                nominal_rtol=1e-9,
                nominal_atol=1e-12,
                skip_heatmap=True,
            )
            builder = PaperPackBuilder(args)
            builder._ensure_dirs()

            cfg = copy.deepcopy(StarshipBlock2)
            env_cfg = copy.deepcopy(EARTH_CONFIG)
            r_eq = float(env_cfg.earth_radius_equator)
            opt_res = {
                "T1": 150.0,
                "T3": 150.0,
                "X1": np.array(
                    [
                        [r_eq + 0.0, r_eq + 3.0e4, r_eq + 1.5e5],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [5.1e6, 4.0e6, 1.45e6],
                    ],
                    dtype=float,
                ),
                "X3": np.array(
                    [
                        [r_eq + 1.5e5, r_eq + 2.8e5, r_eq + 4.19e5],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [1.45e6, 7.2e5, 3.4e5],
                    ],
                    dtype=float,
                ),
            }
            timeseries = {
                "t_s": np.array([0.0, 50.0, 150.0, 300.0], dtype=float),
                "ellipsoidal_altitude_km": np.array([0.0, 25.0, 140.0, 420.0], dtype=float),
                "velocity_m_s": np.array([400.0, 900.0, 3100.0, 7657.0], dtype=float),
                "mass_kg": np.array([5.0e6, 4.1e6, 1.3e6, 3.2e5], dtype=float),
                "dynamic_pressure_kpa": np.array([0.0, 34.8, 2.0, 0.0], dtype=float),
                "downrange_km": np.array([0.0, 120.0, 420.0, 2100.0], dtype=float),
                "g_load": np.array([1.0, 1.8, 3.9, 1.8], dtype=float),
                "throttle": np.array([1.0, 0.9, 0.8, 0.4], dtype=float),
            }
            reliability_dir = Path(tmpdir) / "reliability_raw"
            (reliability_dir / "data").mkdir(parents=True, exist_ok=True)
            for name in ("grid_independence.csv", "integrator_tolerance.csv", "drift_phase_metrics.csv"):
                (reliability_dir / "data" / name).write_text("dummy\n", encoding="utf-8")

            captured = {}

            def capture_figure(fig, stem):
                if Path(stem).name == "fig_main_01_nominal_profile":
                    captured["fig"] = fig

            with patch.object(PaperPackBuilder, "_render_grid_independence"), patch.object(PaperPackBuilder, "_render_integrator_convergence"), patch.object(PaperPackBuilder, "_render_drift"), patch(
                "analysis_tools.paper_outputs._save_figure",
                side_effect=capture_figure,
            ):
                builder._render_main_figures(cfg, env_cfg, opt_res, None, timeseries, reliability_dir)

            fig = captured["fig"]
            axes = fig.axes
            altitude_curves = [line for line in axes[0].lines if len(np.unique(np.asarray(line.get_xdata()))) > 1]
            mass_curves = [line for line in axes[2].lines if len(np.unique(np.asarray(line.get_xdata()))) > 1]
            self.assertEqual(len(altitude_curves), 2)
            self.assertEqual(len(mass_curves), 2)
            self.assertIsNone(axes[0].get_legend())
            self.assertIsNone(axes[1].get_legend())
            self.assertIsNone(axes[2].get_legend())
            self.assertEqual(len(fig.legends), 1)
            legend = fig.legends[0]
            self.assertEqual(
                [text.get_text() for text in legend.get_texts()],
                ["Optimizer profile", "Forward replay", "Circular-orbit speed"],
            )
            self.assertEqual(altitude_curves[0].get_color(), mass_curves[0].get_color())
            self.assertEqual(altitude_curves[1].get_color(), mass_curves[1].get_color())
            plt.close(fig)

    def test_trajectory_appendix_right_panel_uses_orbit_plane_cross_section(self):
        with TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                output_dir=tmpdir,
                skip_heatmap=True,
            )
            builder = PaperPackBuilder(args)
            builder._ensure_dirs()

            r_eq = float(EARTH_CONFIG.earth_radius_equator)
            state = np.array(
                [
                    [r_eq, r_eq + 5000.0, r_eq + 20000.0],
                    [0.0, 1000.0, 5000.0],
                    [0.0, 50.0, 200.0],
                    [0.0, 0.0, 0.0],
                    [50.0, 150.0, 7600.0],
                    [0.0, 10.0, 100.0],
                    [10.0, 9.0, 8.0],
                ],
                dtype=float,
            )
            timeseries = {
                "state": state,
            }

            captured = {}

            def capture_figure(fig, stem):
                captured["fig"] = fig
                captured["stem"] = stem

            with patch("analysis_tools.paper_outputs._save_figure", side_effect=capture_figure):
                builder._render_3d_trajectory(EARTH_CONFIG, timeseries, Path(tmpdir) / "figures" / "fig_app_04_3d_trajectory")

            fig = captured["fig"]
            ax1 = fig.axes[0]
            ax2 = fig.axes[1]
            self.assertEqual(ax1.get_xlabel(), "ECI X (km)")
            self.assertEqual(ax1.get_ylabel(), "ECI Y (km)")
            self.assertEqual(ax1.get_zlabel(), "ECI Z (km)")
            self.assertEqual(ax2.get_xlabel(), "Orbit-plane x (km)")
            self.assertEqual(ax2.get_ylabel(), "Orbit-plane y (km)")
            self.assertEqual(ax2.get_aspect(), 1.0)
            self.assertGreaterEqual(len(ax2.lines), 2)
            plt.close(fig)

    def test_grid_independence_render_uses_two_panels(self):
        with TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                output_dir=tmpdir,
                skip_heatmap=True,
            )
            builder = PaperPackBuilder(args)
            builder._ensure_dirs()

            csv_path = Path(tmpdir) / "data" / "grid_independence.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "nodes",
                        "final_mass_kg",
                        "runtime_s",
                        "solver_success",
                        "terminal_valid",
                        "strict_terminal_ok",
                        "strict_path_ok",
                        "raw_path_ok",
                        "raw_q_ok",
                        "raw_g_ok",
                        "q_ok",
                        "g_ok",
                        "max_q_pa",
                        "max_g",
                        "status",
                    ]
                )
                writer.writerow([20, 315800.0, 7.0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 35390.0, 3.97, "PASS_SLACK"])
                writer.writerow([40, 316120.0, 11.0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 35150.0, 3.97, "PASS_SLACK"])
                writer.writerow([80, 316230.0, 20.0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 34860.0, 3.97, "PASS_RAW"])
                writer.writerow([140, 316255.0, 30.0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 34830.0, 3.97, "PASS_RAW"])

            captured = {}

            def capture_figure(fig, stem):
                captured["fig"] = fig
                captured["stem"] = stem

            with patch("analysis_tools.paper_outputs._save_figure", side_effect=capture_figure):
                builder._render_grid_independence(csv_path, Path(tmpdir) / "figures" / "fig_main_03_grid_independence")

            fig = captured["fig"]
            self.assertEqual(len(fig.axes), 2)
            self.assertEqual(fig.axes[0].get_ylabel(), "Final mass (kg)")
            self.assertEqual(fig.axes[1].get_ylabel(), "|m(N)-m_ref| (kg)")
            plt.close(fig)

    def test_refine_peak_time_value_falls_back_when_quadratic_peak_overshoots_samples(self):
        t = np.array([0.0, 1.0, 2.0], dtype=float)
        y = -((t - 1.2) ** 2) + 5.0

        t_peak, y_peak, idx = _refine_peak_time_value(t, y)

        self.assertEqual(idx, 1)
        self.assertAlmostEqual(t_peak, 1.0, places=6)
        self.assertAlmostEqual(y_peak, float(y[1]), places=6)
        self.assertLessEqual(y_peak, float(np.max(y)))

    def test_time_history_csv_deduplicates_stage_time_and_keeps_post_stage_optimizer_mass(self):
        with TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                output_dir=tmpdir,
                skip_heatmap=True,
            )
            builder = PaperPackBuilder(args)
            builder._ensure_dirs()

            env_cfg = copy.deepcopy(EARTH_CONFIG)
            r0 = env_cfg.earth_radius_equator + 10.0
            r1 = env_cfg.earth_radius_equator + 20.0
            r2 = env_cfg.earth_radius_equator + 30.0
            zeros = np.zeros(2, dtype=float)

            opt_res = {
                "T1": 1.0,
                "T2": 0.0,
                "T3": 1.0,
                "X1": np.vstack(
                    [
                        np.array([r0, r1], dtype=float),
                        zeros,
                        zeros,
                        zeros,
                        np.array([1.0, 1.0], dtype=float),
                        zeros,
                        np.array([10.0, 5.0], dtype=float),
                    ]
                ),
                "X3": np.vstack(
                    [
                        np.array([r1, r2], dtype=float),
                        zeros,
                        zeros,
                        zeros,
                        np.array([2.0, 2.0], dtype=float),
                        zeros,
                        np.array([3.0, 2.0], dtype=float),
                    ]
                ),
            }
            timeseries = {
                "t_s": np.array([0.0, 1.0, 1.0, 2.0], dtype=float),
                "spherical_height_km": np.array([0.010, 0.020, 0.020, 0.030], dtype=float),
                "ellipsoidal_altitude_km": np.array([0.010, 0.020, 0.020, 0.030], dtype=float),
                "velocity_m_s": np.array([1.0, 1.0, 2.0, 2.0], dtype=float),
                "mass_kg": np.array([10.0, 5.0, 3.0, 2.0], dtype=float),
                "dynamic_pressure_kpa": np.array([0.1, 0.2, 0.2, 0.1], dtype=float),
                "g_load": np.array([1.0, 2.0, 2.0, 1.0], dtype=float),
                "downrange_km": np.array([0.0, 1.0, 1.0, 2.0], dtype=float),
                "throttle": np.array([0.8, 0.7, 1.0, 0.9], dtype=float),
            }

            builder._write_time_history_csv(opt_res, env_cfg, timeseries)

            out_path = Path(tmpdir) / "data" / "nominal_time_history.csv"
            with open(out_path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            self.assertEqual([row["time_s"] for row in rows], ["0.000000", "1.000000", "2.000000"])
            self.assertAlmostEqual(float(rows[1]["optimizer_mass_kg"]), 3.0, places=6)
            self.assertAlmostEqual(float(rows[1]["simulation_mass_kg"]), 3.0, places=6)

    def test_interval_replay_render_keeps_all_phases(self):
        with TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                output_dir=tmpdir,
                skip_heatmap=True,
            )
            builder = PaperPackBuilder(args)
            builder._ensure_dirs()

            csv_path = Path(tmpdir) / "data" / "interval_replay_audit.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["phase", "interval_index", "solver_ok", "end_position_error_m", "end_velocity_error_m_s", "end_mass_error_kg", "max_relative_error", "sampled_max_q_pa", "sampled_max_g", "q_margin_pa", "g_margin"])
                writer.writerow(["boost", 0, 1, 1e-6, 1e-8, 0.0, 1e-9, 100.0, 1.0, -100.0, -1.0])
                writer.writerow(["coast", 0, 1, 2e-6, 2e-8, 0.0, 2e-9, 50.0, 0.5, -200.0, -2.0])
                writer.writerow(["ship", 0, 1, 3e-6, 3e-8, 0.0, 3e-9, 25.0, 0.3, -300.0, -3.0])

            captured = {}

            def capture_figure(fig, stem):
                captured["top_labels"] = [line.get_label() for line in fig.axes[0].lines]
                captured["bottom_labels"] = [line.get_label() for line in fig.axes[1].lines]
                plt.close(fig)

            with patch("analysis_tools.paper_outputs._save_figure", side_effect=capture_figure):
                builder._render_interval_replay_audit(csv_path, Path(tmpdir) / "figures" / "interval_replay_audit")

            self.assertIn("Boost", captured["top_labels"])
            self.assertIn("Coast", captured["top_labels"])
            self.assertIn("Ship", captured["top_labels"])
            self.assertIn("Boost", captured["bottom_labels"])
            self.assertIn("Coast", captured["bottom_labels"])
            self.assertIn("Ship", captured["bottom_labels"])
            self.assertIn("Reference threshold", captured["bottom_labels"])


if __name__ == "__main__":
    unittest.main()
