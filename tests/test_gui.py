import copy
import json

import pytest
from PyQt5 import QtCore, QtWidgets

from supervario.gui import MainWindow, ModelPanel
from supervario.pipeline import run


def test_optional_drilling_mappings_round_trip_and_can_be_cleared(qt_app, project):
    window = MainWindow()
    window.populate(project)
    assert not any(key in window.collect()["samples"] for key in ("hole_id", "from", "to"))
    for key, name in (("hole_id", "HOLE"), ("from", "FROM"), ("to", "TO")):
        window.samples.drilling[key].setCurrentText(name)
    saved = window.collect()
    window.populate(saved)
    assert window.collect() == saved
    window.samples.inspect()
    # Missing mappings are removed when reading a different table.
    assert all(widget.currentText() == "" for widget in window.samples.drilling.values())
    assert not any(key in window.collect()["samples"] for key in ("hole_id", "from", "to"))
    window.close()

def test_project_only_settings_are_preserved_without_json_tab(qt_app, project, tmp_path, monkeypatch):
    project["variogram"].update(bandwidth=15, seed=42)
    project["domains"].update(block_size=[10, 20, 30], max_voxels=123456)
    window = MainWindow()
    window.populate(project)
    destination = tmp_path / "advanced-settings.json"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *args: (str(destination), ""))
    window.save_project()
    saved = json.loads(destination.read_text())
    assert saved["variogram"]["bandwidth"] == 15
    assert saved["variogram"]["seed"] == 42
    assert saved["domains"]["block_size"] == [10, 20, 30]
    assert saved["domains"]["max_voxels"] == 123456
    window.close()

def test_reviewed_downhole_nugget_is_explicit_and_preserves_structure_geometry(qt_app, project, monkeypatch):
    from supervario.variography import downhole_diagnostic
    import numpy as np

    result = run(project, lambda _: None)["results"][0]
    original = copy.deepcopy(result["model"])
    result["downhole"] = downhole_diagnostic(
        np.array([0., 2., 4., 0., 2., 4.]),
        {"hole_id": np.array(["A"] * 3 + ["B"] * 3),
         "intervals": np.array([[0, 1], [1, 2], [2, 3]] * 2)}, min_pairs=1)
    from supervario.variography import downhole_review
    result["downhole"]["review"] = downhole_review(result["downhole"], original["nugget"])
    panel = ModelPanel()
    panel.bind(result)
    assert result["model"] == original
    assert panel.downhole_button.isEnabled()
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

    def inspect_dialog(dialog):
        canvas = dialog.findChild(FigureCanvasQTAgg)
        assert len(canvas.figure.axes) == 1
        assert len(canvas.figure.axes[0].lines) == 3
        assert "depth" in canvas.figure.axes[0].get_xlabel()
        assert any(b.text() == "Apply reviewed nugget" for b in dialog.findChildren(QtWidgets.QPushButton))
        return QtWidgets.QDialog.Rejected

    monkeypatch.setattr(QtWidgets.QDialog, "exec_", inspect_dialog)
    panel.review_downhole()
    assert result["model"] == original
    panel.apply_downhole_nugget(0.02)
    assert result["model"]["nugget"] == 0.02
    assert result["model"]["structures"] == original["structures"]
    assert result["model"]["sill"] == pytest.approx(
        0.02 + sum(s["contribution"] for s in original["structures"]))
    assert "weighted_rmse_normalised" not in result["fit"]
    assert result["fit"]["downhole_nugget_reviews"][0]["previous_nugget"] == original["nugget"]
    assert result["estimator_recommendations"]["warnings"]
    with pytest.raises(ValueError, match="finite"):
        panel.apply_downhole_nugget(float("nan"))
    panel.structures.item(0, 1).setText("123")
    with pytest.raises(ValueError, match="pending"):
        panel.apply_downhole_nugget(0.03)
    panel.close()

@pytest.mark.parametrize("normalised", [False, True])
def test_downhole_units_and_live_nugget_preview_apply_absolute_value(qt_app, monkeypatch, normalised):
    from pathlib import Path
    import numpy as np
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from supervario import lfv
    from supervario.variography import downhole_diagnostic

    model = lfv.load(Path(__file__).resolve().parents[1] / "Reference" / "Variogram Model RQDpc UGB01.lfv")
    diagnostic = downhole_diagnostic(
        np.array([0., 2., 4., 0., 2., 4.]),
        {"hole_id": np.array(["A"] * 3 + ["B"] * 3),
         "intervals": np.array([[0, 1], [1, 2], [2, 3]] * 2)}, min_pairs=1)
    result = {"domain": "Reference", "model": model, "downhole": diagnostic}
    original = copy.deepcopy(model)
    panel = ModelPanel()
    panel.normalised.setChecked(normalised)
    panel.bind(result)

    def inspect_dialog(dialog):
        canvas = dialog.findChild(FigureCanvasQTAgg)
        toggle = dialog.findChild(QtWidgets.QCheckBox)
        editor = dialog.findChild(QtWidgets.QDoubleSpinBox)
        variance = original["data_variance"]
        scale = variance if normalised else 1
        assert toggle.isChecked() == normalised
        assert editor.value() == pytest.approx(original["nugget"] / scale)
        editor.setValue(0.25)
        expected_absolute = 0.25 * scale

        def check_plot(divisor):
            axis = canvas.figure.axes[0]
            nugget = expected_absolute / divisor
            assert axis.lines[1].get_ydata()[0] == pytest.approx(nugget)
            assert axis.lines[2].get_ydata()[0] == pytest.approx(nugget)
            assert any(t.get_text() == f"Nugget = {nugget:.6g}" for t in axis.texts)
            assert axis.collections[0].get_offsets()[0, 1] == pytest.approx(2 / divisor)
            assert axis.lines[0].get_ydata()[0] == pytest.approx(
                (expected_absolute + sum(s["contribution"] for s in original["structures"])) / divisor)
            assert ("Normalised" in axis.get_ylabel()) == toggle.isChecked()

        check_plot(scale)
        toggle.setChecked(not normalised)
        other_scale = 1 if normalised else variance
        assert editor.value() == pytest.approx(expected_absolute / other_scale)
        check_plot(other_scale)
        toggle.setChecked(normalised)
        check_plot(scale)
        assert result["model"] == original
        next(b for b in dialog.findChildren(QtWidgets.QPushButton)
             if b.text() == "Apply reviewed nugget").click()
        assert result["model"]["nugget"] == pytest.approx(expected_absolute)
        assert result["model"]["structures"] == original["structures"]
        return QtWidgets.QDialog.Accepted

    monkeypatch.setattr(QtWidgets.QDialog, "exec_", inspect_dialog)
    panel.review_downhole()
    assert result["fit"]["manually_edited"]
    panel.close()

def test_fixed_nugget_refit_updates_model_and_preserves_data_and_history(qt_app, project, monkeypatch):
    result = run(project, lambda _: None)["results"][0]
    result["downhole"] = {"status": "available", "gamma": [0.1], "supported": [True], "review": "old"}
    original = copy.deepcopy(result)
    panel = ModelPanel()
    panel.bind(result)
    panel.refit_downhole_nugget(0.02)
    assert result["model"]["nugget"] == 0.02
    assert result["model"]["structures"] != original["model"]["structures"]
    assert result["experiments"] == original["experiments"]
    assert result["radial"] == original["radial"]
    assert result["orientation_diagnostics"] == original["orientation_diagnostics"]
    for s in result["model"]["structures"]:
        assert s["anisotropy"]["rotation"] == original["model"]["structures"][0]["anisotropy"]["rotation"]
    assert result["fit"]["fixed_nugget"] == 0.02
    assert result["fit"]["converged"]
    assert result["fit"]["orientation_reassessed"] is False
    assert result["fit"]["downhole_nugget_reviews"][-1]["previous_model"] == original["model"]
    assert result["estimator_recommendations"]["reference"]
    assert not panel.dirty
    unchanged = copy.deepcopy(result)

    def fail(*args, **kwargs):
        raise ValueError("test convergence failure")

    monkeypatch.setattr("supervario.gui.fit", fail)
    with pytest.raises(ValueError, match="convergence"):
        panel.refit_downhole_nugget(0.1)
    assert result == unchanged
    panel.close()

def test_downhole_dialog_shows_automatic_extrapolation_and_keeps_manual_override(qt_app, monkeypatch):
    from test_variography import supported_linear_downhole
    from test_recommendations import result_with_ranges
    from supervario.variography import estimate_downhole_nugget
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    result = result_with_ranges()
    diagnostic = supported_linear_downhole()
    diagnostic["automatic_nugget"] = estimate_downhole_nugget(diagnostic, 1)
    diagnostic["automatic_nugget"]["status"] = "applied"
    result["model"]["nugget"] = diagnostic["automatic_nugget"]["nugget"]
    result["model"]["sill"] = result["model"]["nugget"] + 0.9
    result["downhole"] = diagnostic
    panel = ModelPanel()
    panel.bind(result)

    def inspect_dialog(dialog):
        canvas = dialog.findChild(FigureCanvasQTAgg)
        axis = canvas.figure.axes[0]
        line = next(line for line in axis.lines if line.get_label() == "Automatic short-lag extrapolation")
        assert line.get_xdata()[0] == 0
        assert line.get_ydata()[0] == pytest.approx(0.2)
        assert any("sensitivity range" in label.text() for label in dialog.findChildren(QtWidgets.QLabel))
        dialog.findChild(QtWidgets.QDoubleSpinBox).setValue(0.15)
        next(b for b in dialog.findChildren(QtWidgets.QPushButton)
             if b.text() == "Apply reviewed nugget").click()
        return QtWidgets.QDialog.Accepted

    monkeypatch.setattr(QtWidgets.QDialog, "exec_", inspect_dialog)
    panel.review_downhole()
    assert result["model"]["nugget"] == 0.15
    assert diagnostic["automatic_nugget"]["superseded_by_manual_review"]
    panel.close()

def test_axis_toggles_reallocate_height_and_preserve_model(qt_app):
    from pathlib import Path
    from supervario import lfv
    model = lfv.load(Path(__file__).resolve().parents[1] / "Reference" / "Variogram Model RQDpc UGB01.lfv")
    original = copy.deepcopy(model)
    panel = ModelPanel()
    panel.bind({"domain": "Reference", "model": model})
    assert all(button.isChecked() for button in panel.axis_toggles.values())
    original_height = panel.figure.axes[1].get_subplotspec().get_position(panel.figure).height
    panel.axis_toggles["semi_major"].click()
    assert len(panel.figure.axes) == 3
    assert [axis.get_title().split(" (")[0] for axis in panel.figure.axes[1:]] == ["major", "minor"]
    height = panel.figure.axes[1].get_subplotspec().get_position(panel.figure).height
    assert height > original_height
    panel.axis_toggles["minor"].click()
    assert len(panel.figure.axes) == 2
    assert panel.figure.axes[1].get_subplotspec().get_position(panel.figure).height > height
    assert panel.figure.axes[1].get_xlabel() == "Distance (input coordinate units)"
    assert panel.figure.axes[1].lines[0].get_color() == "red"
    panel.axis_toggles["major"].click()
    assert len(panel.figure.axes) == 1
    assert panel.figure.axes[0].name == "polar"
    panel.bind({"domain": "Other", "model": model})
    assert len(panel.figure.axes) == 1
    for button in panel.axis_toggles.values():
        button.click()
    assert len(panel.figure.axes) == 4
    assert [axis.lines[0].get_color() for axis in panel.figure.axes[1:]] == ["red", "green", "blue"]
    assert model == original
    assert not panel.dirty
    panel.close()


def test_supplied_application_icon_has_windows_sizes_and_is_used_by_main_window(qt_app, monkeypatch, tmp_path):
    from supervario.gui import application_icon
    monkeypatch.chdir(tmp_path)
    icon = application_icon()
    assert not icon.isNull()
    sizes = {(size.width(), size.height()) for size in icon.availableSizes()}
    assert {(16, 16), (32, 32), (48, 48), (256, 256)} <= sizes
    window = MainWindow()
    assert not window.windowIcon().isNull()
    window.close()


def test_radial_local_angles_are_distinguished_from_actual_compass_bearings(qt_app):
    from pathlib import Path
    from supervario import lfv
    model = lfv.load(Path(__file__).resolve().parents[1] / "Reference" / "Variogram Model RQDpc UGB01.lfv")
    panel = ModelPanel()
    panel.bind({"domain": "Reference", "model": model,
                "radial": {"angles": list(range(0, 360, 15)), "gamma": [[1] * 3] * 24},
                "lag": 10})
    assert "Lag distance: 10" in panel.summary.text()
    assert "major 115.8 / 57.1" in panel.summary.text()
    assert "semi-major 18." in panel.summary.text() or "semi-major 19." in panel.summary.text()
    radial = panel.figure.axes[0]
    assert "pitch in dipping plane" in radial.get_title()
    assert "not compass azimuth" in radial.get_xlabel()
    assert radial.get_theta_direction() == -1
    from matplotlib.text import Annotation
    arrows = [t for t in radial.texts if isinstance(t, Annotation)]
    import numpy as np
    assert arrows[0].xy[0] == pytest.approx(np.deg2rad(95.58), abs=0.001)
    panel.bind({"domain": "Reference", "model": model})
    assert "Lag distance: Not available (LFV only)" in panel.summary.text()
    panel.close()


@pytest.mark.parametrize("normalised", [False, True])
@pytest.mark.parametrize("nugget", [0, 354.6331])
def test_axis_charts_align_zero_and_show_nugget_intercept(qt_app, normalised, nugget):
    import numpy as np
    from pathlib import Path
    from supervario import lfv
    from supervario.variography import evaluate

    model = lfv.load(Path(__file__).resolve().parents[1] / "Reference" / "Variogram Model RQDpc UGB01.lfv")
    model["nugget"] = nugget
    model["sill"] = nugget + sum(s["contribution"] for s in model["structures"])
    original = copy.deepcopy(model)
    panel = ModelPanel()
    panel.normalised.setChecked(normalised)
    panel.bind({"domain": "A", "model": model})
    divisor = model["data_variance"] if normalised else 1
    for axis, name in zip(panel.figure.axes[-3:], ("major", "semi_major", "minor")):
        assert axis.get_xlim()[0] == 0
        assert axis.get_ylim()[0] == 0
        np.testing.assert_allclose(axis.transData.transform((0, 0)), axis.transAxes.transform((0, 0)))
        curve, sill_line, marker = axis.lines
        assert curve.get_xdata()[0] == 0
        assert curve.get_ydata()[0] == pytest.approx(nugget / divisor)
        np.testing.assert_allclose(curve.get_ydata()[1:],
                                   evaluate(model, curve.get_xdata()[1:], name) / divisor)
        assert marker.get_xdata()[0] == 0
        assert marker.get_ydata()[0] == pytest.approx(nugget / divisor)
        assert not marker.get_clip_on()
        assert any(t.get_text() == f"Nugget = {nugget / divisor:.4g}" for t in axis.texts)
        assert evaluate(model, [0], name)[0] == 0
    assert model == original
    panel.close()


def test_plane_search_step_control_round_trip_and_legacy_default(qt_app, project, tmp_path, monkeypatch):
    from supervario.pipeline import load_config

    window = MainWindow()
    window.populate(project)
    step = window.settings["in_plane_step"]
    assert step.value() == 15
    assert step.singleStep() == 5
    assert "cone half-angle" in step.toolTip()
    step.setValue(5)
    assert window.collect()["variogram"]["in_plane_step"] == 5
    destination = tmp_path / "fine.json"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *args: (str(destination), ""))
    window.save_project()
    config = load_config(destination)
    assert config["variogram"]["in_plane_step"] == 5
    window.populate(config)
    assert step.value() == 5
    legacy = copy.deepcopy(project)
    legacy["variogram"].pop("in_plane_step", None)
    window.populate(legacy)
    assert step.value() == 15
    step.setValue(7)
    with pytest.raises(ValueError, match="multiple of 5"):
        window.collect()
    window.close()


def test_3d_angular_controls_save_restore_and_reset_legacy_defaults(qt_app, project, tmp_path, monkeypatch):
    from supervario.pipeline import load_config
    window = MainWindow()
    window.populate(project)
    for key, default in (("azimuth_step", 15), ("plunge_step", 30)):
        widget = window.settings[key]
        assert widget.value() == default
        assert widget.singleStep() == 5
        assert "alternative plane normals" in widget.toolTip()
        assert widget.width() < window.settings["max_distance"].width()
        widget.setValue(5)
    destination = tmp_path / "3d.json"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *args: (str(destination), ""))
    window.save_project()
    config = load_config(destination)
    for key in ("azimuth_step", "plunge_step"):
        assert config["variogram"][key] == 5
    window.populate(config)
    assert window.collect()["variogram"]["plunge_step"] == 5
    legacy = copy.deepcopy(project)
    legacy["variogram"].pop("azimuth_step", None)
    legacy["variogram"].pop("plunge_step", None)
    window.populate(legacy)
    assert window.settings["azimuth_step"].value() == 15
    assert window.settings["plunge_step"].value() == 30
    window.settings["azimuth_step"].setValue(7)
    with pytest.raises(ValueError, match="multiple of 5"):
        window.collect()
    window.close()


def test_legacy_string_theory_project_shows_notice_and_preserves_provenance(qt_app, project):
    window = MainWindow()
    project["variogram"]["trend_file"] = "legacy_missing_statistics.csv"
    window.populate(project)
    assert not hasattr(window, "trend_notice")
    assert not any("String Theory" in label.text() for label in window.findChildren(QtWidgets.QLabel))
    assert window.collect()["variogram"]["trend_file"] == "legacy_missing_statistics.csv"
    assert not hasattr(window, "trend_file")
    window.close()


def test_project_can_be_saved_twice_and_updated_after_reloading(qt_app, project, tmp_path, monkeypatch):
    from supervario.pipeline import load_config

    destination = tmp_path / "saved.json"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *args: (str(destination), ""))
    errors = []
    monkeypatch.setattr("supervario.gui.error", lambda parent, message: errors.append(message))
    window = MainWindow()
    window.populate(project)
    window.save_project()
    window.settings["bins"].setValue(15)
    window.save_project()
    assert errors == []
    assert json.loads(destination.read_text())["variogram"]["bins"] == 15
    window.populate(load_config(destination))
    window.settings["bins"].setValue(18)
    window.save_project()
    assert errors == []
    assert json.loads(destination.read_text())["variogram"]["bins"] == 18
    assert window.config["_protected_inputs"].count(str(destination)) == 1
    window.close()


def test_project_save_still_protects_source_data(qt_app, project, monkeypatch):
    from pathlib import Path

    source = Path(project["samples"]["path"])
    original = source.read_bytes()
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *args: (str(source), ""))
    errors = []
    monkeypatch.setattr("supervario.gui.error", lambda parent, message: errors.append(message))
    window = MainWindow()
    window.populate(project)
    window.save_project()
    assert len(errors) == 1
    assert "Refusing to overwrite input" in errors[0]
    assert source.read_bytes() == original
    window.close()


def test_invalid_settings_cannot_be_entered_in_numeric_controls(qt_app):
    window = MainWindow()
    for key, minimum in (("bins", 3), ("min_pairs", 1), ("max_samples", 12),
                         ("max_pairs", 1), ("structures", 1)):
        window.settings[key].setValue(0)
        assert window.settings[key].value() == minimum
    window.settings["angle_tolerance"].setValue(0)
    assert window.settings["angle_tolerance"].value() > 0
    window.contact_minimum.setValue(0)
    assert window.contact_minimum.value() == 2
    window.contact_distance.setValue(0)
    assert window.contact_distance.value() > 0
    assert window.settings["lag"].value() == 0
    assert window.settings["max_distance"].value() == 0
    window.close()


def test_automatic_labels_belong_only_to_distance_controls(qt_app):
    window = MainWindow()
    for key in ("lag", "max_distance"):
        assert window.settings[key].text() == "Automatic"
    assert window.settings["bins"].text() == "12"
    assert not window.settings["bins"].specialValueText()
    form = window.settings["bins"].parentWidget().layout()
    assert form.labelForField(window.settings["bins"]).text() == "Number of lag bins (minimum 3)"
    assert form.labelForField(window.settings["lag"]).text() == "Lag distance"
    window.close()


def test_control_widths_follow_content_and_labels_stay_close(qt_app):
    window = MainWindow()
    window.log_dock.hide()
    window.show()
    widths = []
    for width in (1100, 1440):
        window.resize(width, 1000)
        qt_app.processEvents()
        bins = window.settings["bins"]
        distance = window.settings["max_distance"]
        assert bins.width() < distance.width() < window.samples.path.width()
        assert window.settings["structures"].width() < window.settings["max_samples"].width()
        assert window.samples.sheet.width() < window.samples.path.width() / 2
        form = bins.parentWidget().layout()
        for key in ("bins", "lag", "max_distance", "angle_tolerance", "min_pairs"):
            widget = window.settings[key]
            label = form.labelForField(widget)
            assert 0 <= widget.geometry().left() - label.geometry().right() <= 10
        for widget in window.samples.xyz + window.blocks.size_cols + list(window.rotation.values()):
            label = widget.parentWidget().findChild(QtWidgets.QLabel)
            assert label.buddy() is widget
            assert 0 <= widget.geometry().left() - label.geometry().right() <= 8
        scroll = window.tabs.widget(0)
        assert scroll.horizontalScrollBar().maximum() == 0
        widths.append((window.samples.path.width(), bins.width(), distance.width()))
    assert widths[1][0] > widths[0][0]
    assert widths[1][1:] == widths[0][1:]
    window.close()


def test_invalid_project_is_rejected_without_changing_form(qt_app, project):
    window = MainWindow()
    window.populate(project)
    invalid = copy.deepcopy(project)
    invalid["variogram"]["bins"] = 0
    with pytest.raises(ValueError, match="Lag bins"):
        window.populate(invalid)
    assert window.collect() == project
    window.close()


def test_modelling_and_contact_controls_have_explanatory_tooltips(qt_app):
    window = MainWindow()
    for key in ("max_samples", "max_pairs", "structures"):
        widget = window.settings[key]
        assert len(widget.toolTip()) > 50
        assert widget.parentWidget().findChild(QtWidgets.QLabel).toolTip() == widget.toolTip()
    for widget in window.rotation.values():
        assert "Manual direction" in widget.toolTip()
        assert widget.parentWidget().findChild(QtWidgets.QLabel).toolTip() == widget.toolTip()
    assert "Recalculate" in window.manual_rotation.toolTip()
    assert "explicit approval" in window.contact_distance.toolTip()
    assert "each side" in window.contact_minimum.toolTip()
    form = window.contact_distance.parentWidget().layout()
    for widget in (window.contact_distance, window.contact_minimum):
        assert form.labelForField(widget).toolTip() == widget.toolTip()
    assert not any("Browse String Theory" in b.text() for b in window.findChildren(QtWidgets.QPushButton))
    window.close()

def test_window_configuration_and_charts(qt_app, project):
    window = MainWindow()
    assert [window.tabs.tabText(i) for i in range(window.tabs.count())] == [
        "1. Workflow", "2. Variogram review", "3. Contact review / approvals"]
    assert not hasattr(window, "json_editor")
    window.populate(project)
    assert window.collect() == project
    report = run(project, lambda _: None)
    window.completed(report)
    window.show()
    qt_app.processEvents()
    assert window.result_picker.count() == 2
    assert len(window.model_panel.figure.axes) == 5
    assert window.tabs.currentWidget().isAncestorOf(window.model_panel)
    assert f"Lag distance: {report['results'][0]['lag']:.8g}" in window.model_panel.summary.text()
    assert "Plane comparison:" in window.model_panel.summary.text()
    assert "alternative plane normals" in window.model_panel.summary.toolTip()
    assert window.contacts.rowCount() >= 1
    assert all(window.contacts.item(i, 0).checkState() == QtCore.Qt.Unchecked for i in range(window.contacts.rowCount()))
    window.close()

def test_structure_table_height_matches_populated_rows(qt_app, project):
    result = run(project, lambda _: None)["results"][0]
    result["model"]["structures"].append(copy.deepcopy(result["model"]["structures"][0]))
    panel = ModelPanel()
    panel.bind(result)
    table = panel.structures
    content_height = (table.horizontalHeader().sizeHint().height() + table.verticalHeader().length()
                      + 2 * table.frameWidth())
    assert table.height() == content_height

    result["model"]["structures"].pop()
    panel.bind(result)
    content_height = (table.horizontalHeader().sizeHint().height() + table.verticalHeader().length()
                      + 2 * table.frameWidth())
    assert table.height() == content_height
    panel.close()

def test_plot_details_are_side_by_side_scrollable_and_resizable(qt_app):
    panel = ModelPanel()
    panel.resize(1200, 900)
    panel.warnings.setText("\n".join(["A long fit warning for review."] * 100))
    panel.advice.setPlainText("\n".join(["Kriging recommendation"] * 100))
    panel.show()
    qt_app.processEvents()

    assert panel.plot_splitter.orientation() == QtCore.Qt.Vertical
    assert panel.plot_splitter.widget(0) is panel.canvas
    assert panel.plot_splitter.widget(1) is panel.details_splitter
    assert panel.details_splitter.orientation() == QtCore.Qt.Horizontal
    assert panel.details_splitter.widget(0) is panel.reading_scroll
    assert panel.details_splitter.widget(1).isAncestorOf(panel.advice)
    assert panel.reading_scroll.verticalScrollBar().maximum() > 0
    assert panel.advice.verticalScrollBar().maximum() > 0
    assert panel.advice.isReadOnly()

    original_height = panel.canvas.height()
    panel.plot_splitter.setSizes([1000, 0])
    qt_app.processEvents()
    assert panel.plot_splitter.sizes()[1] == 0
    assert panel.canvas.height() > original_height

    panel.plot_splitter.setSizes([400, 300])
    qt_app.processEvents()
    assert panel.plot_splitter.sizes()[1] > 0
    assert panel.canvas.height() < original_height
    panel.details_splitter.setSizes([300, 800])
    qt_app.processEvents()
    assert panel.details_splitter.sizes()[1] > panel.details_splitter.sizes()[0]
    panel.close()

def test_manual_edits_update_lfv_but_preserve_experiments(qt_app, project):
    result = run(project, lambda _: None)["results"][0]
    panel = ModelPanel()
    panel.bind(result)
    original = copy.deepcopy(result["experiments"])
    panel.structures.item(0, 1).setText("123")
    panel.apply_edits()
    assert result["model"]["structures"][0]["anisotropy"]["ranges"]["major"] == 123
    assert result["experiments"] == original
    assert result["fit"]["manually_edited"]
    assert "weighted_rmse_normalised" not in result["fit"]
    assert "Ordinary kriging" in panel.advice.toPlainText()
    assert result["estimator_recommendations"]["reference"]["base_semi_axes"]["major"] == min(123, result["max_distance"])
    panel.close()


def test_structure_arrows_toggle_and_identify_perpendicular_axes(qt_app, project):
    from matplotlib.text import Annotation
    result = run(project, lambda _: None)["results"][0]
    result["model"]["structures"].append(copy.deepcopy(result["model"]["structures"][0]))
    panel = ModelPanel()
    panel.bind(result)
    radial = panel.figure.axes[0]
    assert sum(isinstance(text, Annotation) for text in radial.texts) == 4
    assert any("S1/S2 minor" in text.get_text() for text in radial.texts)
    assert {text.get_text() for text in radial.get_legend().get_texts()} == {
        "S1/S2 major", "S1/S2 semi-major"}
    panel.resize(1100, 950)
    panel.show()
    qt_app.processEvents()
    panel.canvas.draw()
    renderer = panel.canvas.get_renderer()
    legend_bounds = radial.get_legend().get_window_extent(renderer)
    assert legend_bounds.y1 < radial.get_window_extent(renderer).y0
    note = next(text for text in radial.texts if "Out of plane" in text.get_text())
    assert legend_bounds.y1 < note.get_window_extent(renderer).y0
    assert legend_bounds.y0 >= panel.figure.bbox.y0
    panel.show_directions.setChecked(False)
    assert not panel.figure.axes[0].texts
    assert panel.figure.axes[0].get_legend() is None
    panel.close()


def test_radial_palette_selection_changes_colours_not_data_or_model(qt_app, project):
    result = run(project, lambda _: None)["results"][0]
    original = copy.deepcopy(result)
    panel = ModelPanel()
    panel.bind(result)
    assert panel.colour_scale.currentText() == "Rainbow"
    assert panel.figure.axes[0].collections[0].cmap.name == "rainbow"
    for index in range(panel.colour_scale.count()):
        panel.colour_scale.setCurrentIndex(index)
        assert panel.figure.axes[0].collections[0].cmap.name == panel.colour_scale.currentData()
    assert result == original
    assert not panel.dirty
    assert "same distance" in panel.plot_hint.text()
    assert "greater continuity" in panel.plot_hint.text()
    assert "no pairs" in panel.plot_hint.text()
    panel.close()


def test_independent_structure_rotations_are_projected_in_fixed_plot_plane():
    from supervario.gui import projected_structure_axes
    rotation = {"dip_direction": 0, "dip": 0, "pitch": 0}
    model = {"structures": [
        {"anisotropy": {"rotation": rotation}},
        {"anisotropy": {"rotation": {"dip_direction": 0, "dip": 0, "pitch": 90}}},
    ]}
    projections = projected_structure_axes(model, rotation)
    import numpy as np
    major = next(p for p in projections if p["structure"] == 2 and p["axis"] == "major")
    assert abs(major["angle"]) == pytest.approx(np.pi / 2)
    assert major["length"] == pytest.approx(1)
    assert all(p["length"] < 1e-6 for p in projections if p["axis"] == "minor")


def test_background_worker_completes_without_blocking_ui(qt_app, project):
    window = MainWindow()
    window.populate(project)
    window.calculate()
    assert window.worker is not None
    assert not window.run_button.isEnabled()
    assert window.log.isEnabled()
    assert window.log_toggle.isEnabled()
    assert window.activity.maximum() == 0
    loop = QtCore.QEventLoop()
    window.worker.finished.connect(loop.quit)
    timeout = QtCore.QTimer()
    timeout.setSingleShot(True)
    timeout.timeout.connect(loop.quit)
    timeout.start(30000)
    loop.exec_()
    assert window.worker is None, "Background worker timed out"
    assert window.report is not None
    assert window.run_button.isEnabled()
    assert window.export_button.isEnabled()
    assert not window.feedback_timer.isActive()
    assert window.activity.maximum() == 1
    assert "fitting" in window.log.toPlainText()
    window.close()


def test_right_log_panel_can_be_collapsed_while_calculating(qt_app):
    window = MainWindow()
    window.show()
    qt_app.processEvents()
    assert window.dockWidgetArea(window.log_dock) == QtCore.Qt.RightDockWidgetArea
    assert window.tabs.indexOf(window.log) == -1
    assert window.log_dock.isVisible()
    window.tabs.setEnabled(False)
    window.log_toggle.click()
    assert not window.log_dock.isVisible()
    assert window.log_toggle.text() == "Show log"
    window.log_toggle.click()
    assert window.log_dock.isVisible()
    assert window.log_toggle.text() == "Hide log"
    window.log_dock.close()
    assert window.log_toggle.text() == "Show log"
    window.close()


def test_progress_reports_current_stage_and_elapsed_time(qt_app):
    from supervario.gui import Worker
    window = MainWindow()
    window.worker = Worker({})
    window.elapsed.start()
    window.progress_message("Reading blocks (25%): 100,000 rows")
    assert "Reading blocks (25%)" in window.status.text()
    assert "00:00" in window.status.text()
    assert "100,000 rows" in window.log.toPlainText()
    window.worker = None
    window.close()

def test_cancellation_is_cooperative_and_does_not_publish_partial_report(qt_app, project):
    from supervario.gui import Worker
    worker = Worker(project)
    worker.isInterruptionRequested = lambda: True
    cancelled = []
    completed = []
    worker.cancelled.connect(lambda: cancelled.append(True))
    worker.completed.connect(completed.append)
    worker.run()
    assert cancelled == [True]
    assert completed == []
