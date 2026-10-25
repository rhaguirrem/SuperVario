import copy
import traceback
from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from . import lfv
from .data import header, table_format, xyz_columns
from .local_tools import local_module
from .pipeline import default_config, export_report, input_paths, load_config, run
from .variography import AXES, evaluate, validate_settings, rotation_basis, downhole_review, fit
from .recommendations import format_recommendations, refresh


def application_icon():
    icon = QtGui.QIcon(str(Path(__file__).parent / "assets" / "supervario.ico"))
    if icon.isNull():
        raise RuntimeError("Cannot load the SuperVario application icon.")
    return icon


def error(parent, message):
    QtWidgets.QMessageBox.critical(parent, "SuperVario", str(message))


def spin(value, maximum=1e12, decimals=5):
    widget = QtWidgets.QDoubleSpinBox()
    widget.setDecimals(decimals)
    widget.setRange(0, maximum)
    widget.setValue(value)
    return widget


def compact(widget, characters):
    width = widget.fontMetrics().horizontalAdvance("0" * characters) + 32
    widget.setFixedWidth(width)
    return widget


def labelled(label, widget):
    group = QtWidgets.QWidget()
    layout = QtWidgets.QHBoxLayout(group)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(5)
    caption = QtWidgets.QLabel(label)
    caption.setBuddy(widget)
    caption.setToolTip(widget.toolTip())
    layout.addWidget(caption)
    layout.addWidget(widget)
    group.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
    return group


def field_row():
    layout = QtWidgets.QHBoxLayout()
    layout.setSpacing(18)
    return layout


def projected_structure_axes(model, plane_rotation):
    plane = rotation_basis(plane_rotation)
    projections = []
    for number, structure in enumerate(model["structures"], 1):
        local = plane.T @ rotation_basis(structure["anisotropy"]["rotation"])
        for index, axis in enumerate(AXES):
            x, y, z = local[:, index]
            length = float(np.hypot(x, y))
            projections.append({"structure": number, "axis": axis, "length": length,
                                "angle": float(np.arctan2(y, x)), "out_of_plane": float(abs(z))})
    return projections


def combo(items=()):
    widget = QtWidgets.QComboBox()
    widget.setEditable(True)
    widget.addItems(list(items))
    return widget


class CalculationCancelled(RuntimeError):
    pass


class Worker(QtCore.QThread):
    message = QtCore.pyqtSignal(str)
    completed = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)
    cancelled = QtCore.pyqtSignal()

    def __init__(self, config):
        super().__init__()
        self.config = config

    def run(self):
        try:
            report = run(self.config, self.publish)
            self.publish("Calculation complete")
            self.completed.emit(report)
        except CalculationCancelled:
            self.cancelled.emit()
        except Exception:
            self.failed.emit(traceback.format_exc())

    def publish(self, message):
        if self.isInterruptionRequested():
            raise CalculationCancelled()
        self.message.emit(message)


class TableInput(QtWidgets.QGroupBox):
    def __init__(self, title, samples=False):
        super().__init__(title)
        self.samples = samples
        layout = QtWidgets.QFormLayout(self)
        layout.setLabelAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        layout.setHorizontalSpacing(8)
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        row = QtWidgets.QHBoxLayout()
        self.path = QtWidgets.QLineEdit()
        row.addWidget(self.path, 1)
        browse = QtWidgets.QPushButton("Browse...")
        browse.clicked.connect(self.browse)
        row.addWidget(browse)
        layout.addRow("File", row)
        fmt = field_row()
        self.sheet = compact(QtWidgets.QLineEdit(), 20)
        self.sheet.setPlaceholderText("first sheet")
        self.sep = combo(["", ",", ";", r"\t", "|"])
        self.decimal = combo(["", ".", ","])
        self.header_line = QtWidgets.QSpinBox()
        self.header_line.setRange(0, 100000)
        self.header_line.setSpecialValueText("Auto")
        compact(self.sep, 4)
        compact(self.decimal, 4)
        compact(self.header_line, 6)
        for label, widget in (("Sheet", self.sheet), ("Separator", self.sep),
                              ("Decimal", self.decimal), ("Header row", self.header_line)):
            fmt.addWidget(labelled(label, widget))
        fmt.addStretch()
        layout.addRow(fmt)
        coordinates = field_row()
        self.xyz = [compact(combo([""]), 20) for _ in range(3)]
        for name, widget in zip(("X", "Y", "Z"), self.xyz):
            widget.setToolTip("Blank means automatic detection")
            coordinates.addWidget(labelled(name, widget))
        coordinates.addStretch()
        layout.addRow("Coordinates", coordinates)
        self.attribute = compact(combo([""]), 30)
        layout.addRow("Value" if samples else "Domain field", self.attribute)
        if samples:
            drilling = field_row()
            self.drilling = {}
            for key, label in (("hole_id", "Hole ID"), ("from", "From"), ("to", "To")):
                widget = compact(combo([""]), 20)
                widget.setToolTip(
                    "Optional, recommended for spacing-informed automatic lag distance. "
                    "Map Hole ID; From and To must be supplied together, in coordinate distance units. "
                    "Coordinates must already be reliable desurveyed sample midpoints.")
                self.drilling[key] = widget
                drilling.addWidget(labelled(label, widget))
            drilling.addStretch()
            layout.addRow("Drilling (optional, recommended)", drilling)
        if not samples:
            sizes = field_row()
            self.size_cols = [compact(combo([""]), 20) for _ in range(3)]
            for name, widget in zip(("DX", "DY", "DZ"), self.size_cols):
                sizes.addWidget(labelled(name, widget))
            sizes.addStretch()
            layout.addRow("Block dimensions", sizes)
        inspect = QtWidgets.QPushButton("Read columns / detect mappings")
        inspect.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        inspect.clicked.connect(self.inspect)
        layout.addRow(inspect)
        self.status = QtWidgets.QLabel("Blank format fields are detected. Header row is 1-based; 0 means auto.")
        self.status.setWordWrap(True)
        layout.addRow(self.status)

    def browse(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select table", self.path.text(),
            "Tables (*.csv *.txt *.dat *.xlsx *.xlsm *.bmf);;All files (*)")
        if filename:
            self.path.setText(filename)
            self.inspect()

    def spec(self):
        xyz = [c.currentText().strip() for c in self.xyz]
        if any(xyz) and not all(xyz):
            raise ValueError("Choose all three coordinate columns, or leave all three blank for auto.")
        result = {"path": self.path.text().strip(), "sheet": self.sheet.text().strip(), "xyz": xyz if all(xyz) else [],
                  "format": {"sep": self.sep.currentText().replace(r"\t", "\t"),
                             "decimal": self.decimal.currentText(), "header_line": self.header_line.value()}}
        result["value" if self.samples else "domain"] = self.attribute.currentText().strip()
        if self.samples:
            for key, widget in self.drilling.items():
                if widget.currentText().strip():
                    result[key] = widget.currentText().strip()
        if not self.samples:
            sizes = [c.currentText().strip() for c in self.size_cols]
            if any(sizes) and not all(sizes):
                raise ValueError("Choose all three dimension columns, or leave all blank.")
            result["size_cols"] = sizes if all(sizes) else []
        return result

    def populate(self, spec):
        self.path.setText(spec.get("path", ""))
        self.sheet.setText(spec.get("sheet", ""))
        fmt = spec.get("format", {})
        self.sep.setCurrentText(fmt.get("sep", "").replace("\t", r"\t"))
        self.decimal.setCurrentText(fmt.get("decimal", ""))
        self.header_line.setValue(fmt.get("header_line", 0))
        for i, widget in enumerate(self.xyz):
            widget.setCurrentText((spec.get("xyz") or ["", "", ""])[i])
        self.attribute.setCurrentText(spec.get("value" if self.samples else "domain", ""))
        if self.samples:
            for key, widget in self.drilling.items():
                widget.setCurrentText(spec.get(key, ""))
        if not self.samples:
            for i, widget in enumerate(self.size_cols):
                widget.setCurrentText((spec.get("size_cols") or ["", "", ""])[i])

    def inspect(self):
        try:
            spec = self.spec()
            columns = header(spec)
            all_combos = self.xyz + [self.attribute] + (list(self.drilling.values()) if self.samples else self.size_cols)
            for widget in all_combos:
                old = widget.currentText()
                widget.clear()
                widget.addItems([""] + columns)
                widget.setCurrentText(old if old in columns else "")
            chosen = [w.currentText() for w in self.xyz]
            spec["xyz"] = chosen if all(chosen) else []
            try:
                detected = xyz_columns(spec, columns, self.samples)
            except ValueError:
                detected = ["", "", ""]
            for widget, name in zip(self.xyz, detected):
                widget.setCurrentText(name)
            if not self.samples:
                sizes = local_module("srkreliability.blockstats", "SRKReliability").detect_size_columns(columns)
                if sizes and not all(w.currentText() for w in self.size_cols):
                    for widget, name in zip(self.size_cols, sizes):
                        widget.setCurrentText(name)
            details = f"{len(columns)} columns: " + ", ".join(columns[:15])
            if Path(spec["path"]).suffix.lower() not in (".xlsx", ".xlsm", ".bmf"):
                io = local_module("srkreliability.io_utils", "SRKReliability")
                details += "\nDetected: " + str(vars(io.csv_layout(spec["path"], table_format(spec))))
            self.status.setText(details)
        except Exception:
            error(self, traceback.format_exc())


class ModelPanel(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.result = None
        self.dirty = False
        self.loading = False
        layout = QtWidgets.QVBoxLayout(self)
        self.summary = QtWidgets.QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Nugget (absolute)"))
        self.nugget = spin(0, 1e18, 10)
        compact(self.nugget, 22)
        self.nugget.valueChanged.connect(self.mark_dirty)
        controls.addWidget(self.nugget)
        self.normalised = QtWidgets.QCheckBox("Normalised plots")
        self.normalised.setChecked(True)
        self.normalised.toggled.connect(self.plot)
        controls.addWidget(self.normalised)
        self.show_directions = QtWidgets.QCheckBox("Structure directions")
        self.show_directions.setChecked(True)
        self.show_directions.setToolTip(
            "Show projected major, semi-major and minor axes for each structure. Solid = S1, dashed = S2. "
            "Shared orientations use one set labelled S1/S2. Arrows show orientation, not range. "
            "Perpendicular axes have no in-plane arrow.")
        self.show_directions.toggled.connect(self.plot)
        controls.addWidget(self.show_directions)
        apply = QtWidgets.QPushButton("Apply model edits")
        apply.clicked.connect(self.apply_clicked)
        controls.addWidget(apply)
        self.downhole_button = QtWidgets.QPushButton("Downhole diagnostic...")
        self.downhole_button.setEnabled(False)
        self.downhole_button.clicked.connect(self.review_downhole)
        controls.addWidget(self.downhole_button)
        controls.addStretch()
        layout.addLayout(controls)
        display_controls = field_row()
        self.colour_scale = QtWidgets.QComboBox()
        for label, name in (("Rainbow", "rainbow"), ("Turbo", "turbo"),
                            ("Viridis", "viridis"), ("Cividis", "cividis")):
            self.colour_scale.addItem(label, name)
        compact(self.colour_scale, 12)
        self.colour_scale.setToolTip(
            "Choose the radial plot colour palette. Colours encode semivariance, not continuity directly. "
            "Use the colour bar to compare values; Viridis and Cividis offer more uniform perceptual scales.")
        self.colour_scale.currentIndexChanged.connect(self.plot)
        display_controls.addWidget(labelled("Radial colour scale", self.colour_scale))
        display_controls.addStretch()
        layout.addLayout(display_controls)
        self.structures = QtWidgets.QTableWidget(0, 5)
        self.structures.setHorizontalHeaderLabels(["Contribution (absolute)", "Major range", "Semi-major range", "Minor range", "Type"])
        self.structures.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.structures.itemChanged.connect(self.mark_dirty)
        layout.addWidget(self.structures)
        axis_controls = QtWidgets.QHBoxLayout()
        axis_controls.addStretch(1)
        axis_buttons = QtWidgets.QHBoxLayout()
        self.axis_toggles = {}
        for axis in AXES:
            button = QtWidgets.QPushButton(axis.replace("_", "-").capitalize())
            button.setCheckable(True)
            button.setChecked(True)
            button.setToolTip("Show or hide this variogram. Visible variograms share the available height.")
            button.toggled.connect(self.plot)
            self.axis_toggles[axis] = button
            axis_buttons.addWidget(button)
        axis_controls.addLayout(axis_buttons, 1)
        layout.addLayout(axis_controls)
        self.figure = Figure(figsize=(11, 6), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumSize(0, 0)
        self.plot_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.plot_splitter.setHandleWidth(7)
        self.plot_splitter.addWidget(self.canvas)
        layout.addWidget(self.plot_splitter, 1)
        self.details_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.details_splitter.setHandleWidth(7)
        self.details_splitter.setChildrenCollapsible(False)
        reading = QtWidgets.QWidget()
        reading_layout = QtWidgets.QVBoxLayout(reading)
        reading_layout.addWidget(QtWidgets.QLabel("Plot reading and analysis"))
        self.plot_hint = QtWidgets.QLabel(
            "Reading the radial plot: compare directions at the same distance from the centre. "
            "Lower semivariance generally means greater continuity; higher semivariance means less similarity. "
            "Use the colour bar, not colour alone. Blank cells have no pairs, not zero variability. "
            "Sparse pairs can be misleading; this plane does not show the perpendicular minor direction.")
        self.plot_hint.setWordWrap(True)
        self.plot_hint.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        reading_layout.addWidget(self.plot_hint)
        self.warnings = QtWidgets.QLabel()
        self.warnings.setWordWrap(True)
        self.warnings.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        reading_layout.addWidget(self.warnings)
        reading_layout.addStretch()
        self.reading_scroll = QtWidgets.QScrollArea()
        self.reading_scroll.setWidgetResizable(True)
        self.reading_scroll.setWidget(reading)
        self.reading_scroll.setMinimumSize(0, 0)
        self.details_splitter.addWidget(self.reading_scroll)
        recommendations = QtWidgets.QWidget()
        recommendations_layout = QtWidgets.QVBoxLayout(recommendations)
        recommendations_layout.addWidget(QtWidgets.QLabel("Kriging recommendations"))
        self.advice = QtWidgets.QPlainTextEdit()
        self.advice.setReadOnly(True)
        self.advice.setMinimumSize(0, 0)
        recommendations_layout.addWidget(self.advice)
        self.details_splitter.addWidget(recommendations)
        self.details_splitter.setStretchFactor(0, 1)
        self.details_splitter.setStretchFactor(1, 1)
        self.plot_splitter.addWidget(self.details_splitter)
        self.plot_splitter.setCollapsible(0, False)
        self.plot_splitter.setStretchFactor(0, 1)
        self.plot_splitter.setStretchFactor(1, 0)
        self.plot_splitter.setSizes([600, 140])
        self.plot_splitter.handle(1).setToolTip(
            "Drag down to enlarge the plots or collapse the text panels; drag up to show more text.")

    def mark_dirty(self, *args):
        if not self.loading:
            self.dirty = True

    def bind(self, result):
        self.result = result
        self.downhole_button.setEnabled(True)
        self.loading = True
        model = result["model"]
        self.nugget.setValue(model["nugget"])
        self.structures.setRowCount(len(model["structures"]))
        for row, structure in enumerate(model["structures"]):
            cells = [structure["contribution"]] + [structure["anisotropy"]["ranges"][k] for k in AXES]
            for col, number in enumerate(cells):
                self.structures.setItem(row, col, QtWidgets.QTableWidgetItem(format(number, ".12g")))
            self.structures.setItem(row, 4, QtWidgets.QTableWidgetItem(structure["structure_type"]))
        self.structures.setFixedHeight(
            self.structures.horizontalHeader().sizeHint().height()
            + self.structures.verticalHeader().length()
            + 2 * self.structures.frameWidth()
        )
        self.loading = False
        self.dirty = False
        rotation = model["structures"][0]["anisotropy"]["rotation"]
        bearings = []
        for name, vector in zip(AXES, rotation_basis(rotation).T):
            vector = vector.copy()
            azimuth = float(np.degrees(np.arctan2(vector[0], vector[1])) % 360)
            if vector[2] > 1e-10 or (abs(vector[2]) <= 1e-10 and azimuth >= 180):
                vector *= -1
            azimuth = float(np.degrees(np.arctan2(vector[0], vector[1])) % 360)
            plunge = float(np.degrees(np.arcsin(np.clip(-vector[2], -1, 1))))
            bearings.append(f"{name.replace('_', '-')} {azimuth:.1f} / {plunge:.1f}")
        refinement = result.get("fit", {}).get("plane_refinement")
        plane_status = f" | Plane comparison: {refinement['status']}" if refinement else ""
        lag_distance = f"{result['lag']:.8g}" if "lag" in result else "Not available (LFV only)"
        lag_info = result.get("lag_selection")
        lag_details = (f" | Lag bins: {lag_info['bins']} | Lag selection: {lag_info['method']}"
                       if lag_info else "")
        self.summary.setText(
            f"Domain: {result['domain']} | Samples: {result.get('sample_count', 'LFV only')} | "
            f"Variance: {model['data_variance']:.6g} | Lag distance: {lag_distance}{lag_details}\n"
            f"Direction assessment: {result.get('orientation_diagnostics', {}).get('status', 'Not assessed in this report')}"
            f"{plane_status}\n"
            "Axis azimuth / downward plunge (degrees): " + "; ".join(bearings) + "\n"
            f"Dip azimuth {rotation['dip_direction']:.2f}, dip {rotation['dip']:.2f}, pitch {rotation['pitch']:.2f}. "
            "Change direction in the workflow settings and rerun; model edits here keep experimental directions fixed.")
        self.summary.setToolTip(
            "Automatic comparison tests shortlisted 3D frames with alternative plane normals, plus in-plane "
            "rotations with the minor direction fixed. "
            "A longer fitted structured range is preferred only with rising-curve and sill support, "
            "comparable held-out prediction error and adequate pair counts. "
            "Full comparison scores and rejection reasons are saved in the analysis report.")
        self.warnings.setText("\n".join(result.get("fit", {}).get("warnings", [])))
        self.advice.setPlainText(format_recommendations(result))
        self.plot()

    def apply_edits(self):
        if self.result is None or not self.dirty:
            return
        model = copy.deepcopy(self.result["model"])
        model["nugget"] = self.nugget.value()
        for row, structure in enumerate(model["structures"]):
            text = self.structures.item(row, 0).text()
            if text != format(structure["contribution"], ".12g"):
                structure["contribution"] = float(text)
            for col, axis in enumerate(AXES, 1):
                ranges = structure["anisotropy"]["ranges"]
                text = self.structures.item(row, col).text()
                if text != format(ranges[axis], ".12g"):
                    ranges[axis] = float(text)
            kind = self.structures.item(row, 4).text().strip().lower()
            if kind not in ("spherical", "linear"):
                raise ValueError("Manual curve editing supports spherical or linear types only.")
            structure["structure_type"] = kind
        model["sill"] = model["nugget"] + sum(s["contribution"] for s in model["structures"])
        lfv.validate(model)
        self.result["model"] = model
        fit_info = self.result.setdefault("fit", {})
        fit_info["manually_edited"] = True
        fit_info.pop("weighted_rmse_normalised", None)
        diagnostic = self.result.get("downhole")
        if diagnostic and diagnostic["status"] != "unavailable":
            automatic = diagnostic.get("automatic_nugget")
            if automatic and automatic["status"] == "applied" and model["nugget"] != automatic["nugget"]:
                automatic["superseded_by_manual_review"] = True
            old_review = diagnostic.get("review")
            fit_info["warnings"] = [w for w in fit_info.get("warnings", []) if w != old_review]
            diagnostic["review"] = downhole_review(diagnostic, model["nugget"])
            fit_info["warnings"].append(diagnostic["review"])
        if "Manual model edits applied; automatic fit statistics no longer apply." not in fit_info.setdefault("warnings", []):
            fit_info["warnings"].append("Manual model edits applied; automatic fit statistics no longer apply.")
        self.dirty = False
        self.warnings.setText("\n".join(fit_info["warnings"]))
        refresh(self.result)
        self.advice.setPlainText(format_recommendations(self.result))
        self.plot()

    def apply_downhole_nugget(self, value):
        if self.dirty:
            raise ValueError("Apply or discard pending model edits before reviewing the downhole nugget.")
        diagnostic = self.result.get("downhole")
        if not diagnostic or diagnostic["status"] != "available":
            raise ValueError("Supported downhole evidence is required for a reviewed nugget override.")
        if not np.isfinite(value) or value < 0 or value > self.nugget.maximum():
            raise ValueError("Reviewed nugget must be finite and within the nugget control range.")
        previous = self.result["model"]["nugget"]
        self.nugget.setValue(value)
        self.mark_dirty()
        self.apply_edits()
        self.result["fit"].setdefault("downhole_nugget_reviews", []).append({
            "previous_nugget": previous, "reviewed_nugget": self.result["model"]["nugget"],
            "policy": "Explicit user review; contributions, ranges and orientations retained; total sill recomputed.",
        })

    def review_downhole(self):
        diagnostic = self.result.get("downhole")
        if not diagnostic or diagnostic["status"] == "unavailable":
            QtWidgets.QMessageBox.information(
                self, "Downhole diagnostic",
                (diagnostic or {}).get("reason", "Map Hole ID, From and To and recalculate this domain."))
            return
        if self.dirty:
            error(self, "Apply or discard pending model edits before reviewing the downhole nugget.")
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"Downhole diagnostic - {self.result['domain']}")
        dialog.resize(950, 700)
        layout = QtWidgets.QVBoxLayout(dialog)
        note = QtWidgets.QLabel()
        note.setWordWrap(True)
        layout.addWidget(note)
        normalised = QtWidgets.QCheckBox("Normalised values (nugget and Y-axis)")
        normalised.setChecked(self.normalised.isChecked())
        layout.addWidget(normalised)
        figure = Figure(constrained_layout=True)
        canvas = FigureCanvasQTAgg(figure)
        layout.addWidget(canvas, 1)
        row = QtWidgets.QHBoxLayout()
        model = self.result["model"]
        absolute_nugget = model["nugget"]
        reviewed = spin(0, 1e18, 10)
        nugget_field = labelled("Reviewed nugget", reviewed)
        nugget_label = nugget_field.findChild(QtWidgets.QLabel)
        row.addWidget(nugget_field)

        def divisor():
            return model["data_variance"] if normalised.isChecked() else 1.0

        def render():
            scale = divisor()
            figure.clear()
            axis = figure.add_subplot(111)
            for x, gamma, count, holes, supported in zip(
                    diagnostic["distance"], diagnostic["gamma"], diagnostic["count"],
                    diagnostic["hole_count"], diagnostic["supported"]):
                if gamma is None:
                    continue
                axis.scatter([x], [gamma / scale], s=np.sqrt(count) * 3 + 6,
                             color="black" if supported else "grey")
                axis.annotate(f"{count} pairs / {holes} holes", (x, gamma / scale),
                              xytext=(0, 8), textcoords="offset points", fontsize=7, ha="center")
            sill = absolute_nugget + sum(s["contribution"] for s in model["structures"])
            nugget = absolute_nugget / scale
            axis.axhline(sill / scale, color="grey", linestyle="--", label="Total sill (preview)")
            axis.axhline(nugget, color="red", linestyle=":", label="Reviewed nugget (preview)")
            axis.plot([0], [nugget], marker="o", markerfacecolor="white",
                      markeredgecolor="red", clip_on=False, zorder=5)
            automatic = diagnostic.get("automatic_nugget")
            automatic_note = ""
            if automatic:
                automatic_note = ("\nCalculation automatic nugget: " + automatic["status"] + ". " + automatic["reason"]
                                  + "\n" + automatic["limitations"])
                if automatic.get("superseded_by_manual_review"):
                    automatic_note += "\nThe automatic nugget has since been overridden by manual review."
                if "nugget" in automatic:
                    x = np.r_[0, np.asarray(diagnostic["distance"])[automatic["window_bins"]]]
                    axis.plot(x, (automatic["nugget"] + automatic["slope"] * x) / scale,
                              color="darkorange", linestyle="--", label="Automatic short-lag extrapolation")
                    bounds = automatic["sensitivity_nugget_min_max"]
                    automatic_note += (f"\nAutomatic intercept: {automatic['nugget'] / scale:.6g}; "
                                       f"sensitivity range: {bounds[0] / scale:.6g} to {bounds[1] / scale:.6g} "
                                       "(display units; not a confidence interval).")
            axis.annotate(f"Nugget = {nugget:.6g}", xy=(0, nugget), xytext=(8, 8),
                          textcoords="offset points", fontsize=8,
                          bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8})
            axis.set(xlabel="Downhole midpoint-depth separation", ylabel=(
                "Normalised semivariance" if normalised.isChecked() else "Semivariance"),
                title=f"Same-hole pairs; lag distance {diagnostic['lag']:.6g} (grey = insufficient support)")
            axis.set_xlim(left=0)
            axis.set_ylim(bottom=0)
            axis.legend()
            axis.grid(alpha=0.25)
            note.setText(
                downhole_review(diagnostic, absolute_nugget) + "\n" + "\n".join(diagnostic["limitations"])
                + automatic_note
                + "\nPair sampling uses the fitting sample cap; approved borrowed samples are included."
                + "\nPreview only until Apply reviewed nugget; total sill changes with the nugget.")
            canvas.draw_idle()

        def change_units():
            reviewed.blockSignals(True)
            reviewed.setMaximum(self.nugget.maximum() / divisor())
            reviewed.setValue(absolute_nugget / divisor())
            reviewed.blockSignals(False)
            nugget_label.setText("Reviewed nugget (normalised)" if normalised.isChecked()
                                 else "Reviewed nugget (absolute)")
            render()

        def preview_nugget(value):
            nonlocal absolute_nugget
            absolute_nugget = value * divisor()
            render()

        normalised.toggled.connect(change_units)
        reviewed.valueChanged.connect(preview_nugget)
        change_units()
        apply = QtWidgets.QPushButton("Apply reviewed nugget")
        apply.setEnabled(diagnostic["status"] == "available")
        apply.setToolTip("Keeps contributions and ranges unchanged; total sill changes. Automatic fit statistics are invalidated.")
        refit = QtWidgets.QPushButton("Refit with reviewed nugget fixed")
        refit.setEnabled(diagnostic["status"] == "available")
        refit.setToolTip(
            "Refit spherical contributions and ranges to existing directional data; "
            "keep the reviewed nugget and current orientation fixed. Total sill is free.")

        def accept_nugget():
            try:
                self.apply_downhole_nugget(absolute_nugget)
                dialog.accept()
            except Exception:
                error(dialog, traceback.format_exc())

        apply.clicked.connect(accept_nugget)
        row.addWidget(apply)
        def accept_refit():
            try:
                self.refit_downhole_nugget(absolute_nugget)
                dialog.accept()
            except Exception:
                error(dialog, traceback.format_exc())

        refit.clicked.connect(accept_refit)
        row.addWidget(refit)
        close = QtWidgets.QPushButton("Close without changes")
        close.clicked.connect(dialog.reject)
        row.addWidget(close)
        layout.addLayout(row)
        canvas.draw_idle()
        dialog.exec_()

    def refit_downhole_nugget(self, value):
        if self.dirty:
            raise ValueError("Apply or discard pending model edits before refitting.")
        if not np.isfinite(value) or value < 0 or value > self.nugget.maximum():
            raise ValueError("Reviewed nugget must be finite and within the nugget control range.")
        result = self.result
        if not result or result.get("downhole", {}).get("status") != "available":
            raise ValueError("Supported downhole evidence is required for a fixed-nugget refit.")
        settings = result.get("fitting_settings")
        if not settings or not result.get("experiments"):
            raise ValueError("Saved fitting settings and directional data are required; recalculate this domain.")
        original = result["model"]
        if any(s["structure_type"] != "spherical" for s in original["structures"]):
            raise ValueError("Fixed-nugget refitting supports spherical structures only.")
        rotation = original["structures"][0]["anisotropy"]["rotation"]
        if any(s["anisotropy"]["rotation"] != rotation for s in original["structures"]):
            raise ValueError("Refitting requires a shared structure orientation; recalculate this domain.")
        model, info = fit(
            [result["experiments"][axis] for axis in AXES],
            original["data_variance"], rotation, original["metadata"].get("attribute", ""),
            result["domain"], result["lag"], result["max_distance"], len(original["structures"]),
            settings["min_pairs"], fixed_nugget=value)
        model["metadata"] = copy.deepcopy(original["metadata"])
        prior = result.get("fit", {})
        old_review = result["downhole"].get("review")
        retained = [warning for warning in prior.get("warnings", [])
                    if warning not in prior.get("fitting_warnings", [])
                    and warning != old_review
                    and warning != "Manual model edits applied; automatic fit statistics no longer apply."]
        diagnostic = copy.deepcopy(result["downhole"])
        if "automatic_nugget" in diagnostic:
            diagnostic["automatic_nugget"]["superseded_by_manual_review"] = True
        diagnostic["review"] = downhole_review(diagnostic, model["nugget"])
        info["warnings"] += retained + [diagnostic["review"]]
        info["fixed_nugget"] = float(value)
        info["nugget_policy"] = "Explicit reviewed nugget fixed; spherical contributions/ranges refitted; orientation retained."
        info["orientation_reassessed"] = False
        info["downhole_nugget_reviews"] = copy.deepcopy(prior.get("downhole_nugget_reviews", []))
        info["downhole_nugget_reviews"].append({
            "previous_model": copy.deepcopy(original),
            "previous_fit": {key: copy.deepcopy(item) for key, item in prior.items()
                             if key != "downhole_nugget_reviews"},
            "reviewed_nugget": float(value), "policy": info["nugget_policy"],
        })
        updated = dict(result, model=model, fit=info, downhole=diagnostic)
        refresh(updated)
        result.update(updated)
        self.bind(result)

    def apply_clicked(self):
        try:
            self.apply_edits()
        except Exception:
            error(self, traceback.format_exc())

    def plot(self, *args):
        self.figure.clear()
        if self.result is None:
            self.canvas.draw()
            return
        result = self.result
        model = result["model"]
        variance = model["data_variance"] if self.normalised.isChecked() else 1.0
        visible_axes = [axis for axis in AXES if self.axis_toggles[axis].isChecked()]
        grid = self.figure.add_gridspec(max(1, len(visible_axes)), 2)
        radial = self.figure.add_subplot(grid[:, 0], projection="polar")
        data = result.get("radial")
        plane_rotation = (data or {}).get("rotation", model["structures"][0]["anisotropy"]["rotation"])
        pitch = np.deg2rad(plane_rotation["pitch"])
        radial.set_theta_direction(-1)
        radial.set_xlabel("Pitch in dipping plane (not compass azimuth)")
        if data:
            angles = np.deg2rad(np.r_[data["angles"], 360]) + pitch
            bins = np.arange(len(data["gamma"][0]) + 1) * result["lag"]
            values = np.array([[np.nan if v is None else v / variance for v in row] for row in data["gamma"]])
            image = radial.pcolormesh(angles, bins, values.T, shading="flat",
                                     cmap=self.colour_scale.currentData())
            self.figure.colorbar(image, ax=radial, shrink=0.6, label="Semivariance")
            radial.set_title("2D variogram - pitch in dipping plane\n"
                             f"Dip = {plane_rotation['dip']:.2f}, Dip Azimuth = {plane_rotation['dip_direction']:.2f}")
        else:
            radial.set_title("LFV model only\nNo experimental data loaded")
        if self.show_directions.isChecked():
            extent = float(bins[-1]) if data else 1.0
            if not data:
                radial.set_ylim(0, extent)
            perpendicular = []
            projections = projected_structure_axes(model, plane_rotation)
            shared = len(model["structures"]) == 2 and np.allclose(
                np.abs(rotation_basis(model["structures"][0]["anisotropy"]["rotation"]).T @
                       rotation_basis(model["structures"][1]["anisotropy"]["rotation"])),
                np.eye(3), atol=1e-8)
            for projection in projections:
                number, axis = projection["structure"], projection["axis"]
                if shared and number == 2:
                    continue
                color = dict(zip(AXES, ("red", "green", "blue")))[axis]
                structure_label = "S1/S2" if shared else f"S{number}"
                label = f"{structure_label} {axis.replace('_', '-')}"
                style = "-" if number == 1 else "--"
                if projection["length"] < 1e-6:
                    perpendicular.append(label)
                    continue
                radius = extent * (0.90 if number == 1 else 0.69) * projection["length"]
                for sign in (0, np.pi):
                    angle = projection["angle"] + pitch + sign
                    radial.annotate("", xy=(angle, radius), xytext=(angle, 0),
                                    arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.6,
                                                "linestyle": style})
                radial.plot([], [], color=color, linestyle=style, label=label)
            radial.legend(loc="upper center", bbox_to_anchor=(0.5, -0.23), ncol=2,
                          fontsize=7, title="Projected axes (not ranges)", title_fontsize=7,
                          frameon=False)
            if perpendicular:
                radial.text(0.5, -0.12, "Out of plane: " + ", ".join(perpendicular),
                            transform=radial.transAxes, ha="center", fontsize=8, color="blue")
        for row, axis in enumerate(visible_axes):
            i = AXES.index(axis)
            plot = self.figure.add_subplot(grid[row, 1])
            exp = result.get("experiments", {}).get(axis)
            maximum = result.get("max_distance", max(s["anisotropy"]["ranges"][axis] for s in model["structures"]) * 1.2)
            h = np.linspace(0, maximum, 300)
            unsupported = any(s["structure_type"] == "spheroidal" for s in model["structures"])
            if not unsupported:
                curve = evaluate(model, h, axis) / variance
                # Plot the right-hand limit at zero, not a spurious ramp
                # between gamma(0) = 0 and the first positive-distance point.
                curve[0] = model["nugget"] / variance
                plot.plot(h, curve, color=("red", "green", "blue")[i])
            else:
                plot.text(0.05, 0.7, "Spheroidal curve rendering unsupported", transform=plot.transAxes)
            if exp:
                y = np.array([np.nan if x is None else x / variance for x in exp["gamma"]])
                sizes = np.sqrt(np.array(exp["count"])) * 3 + 6
                plot.scatter(exp["distance"], y, s=sizes, color=("red", "green", "blue")[i], alpha=0.65)
            plot.axhline(model["sill"] / variance, color="grey", linestyle="--", linewidth=0.7)
            nugget = model["nugget"] / variance
            plot.plot([0], [nugget], marker="o", markersize=5, markerfacecolor="white",
                      markeredgecolor=("red", "green", "blue")[i], clip_on=False, zorder=5)
            plot.annotate(f"Nugget = {nugget:.4g}", xy=(0, nugget), xytext=(8, 8),
                          textcoords="offset points", fontsize=8,
                          bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
                          zorder=6)
            plot.set_xlim(left=0)
            plot.set_ylim(bottom=0)
            plot.set_title(axis.replace("_", "-") + " (marker size reflects pair count)", fontsize=9)
            plot.set_ylabel("Semivariance")
            plot.grid(alpha=0.25)
            if row == len(visible_axes) - 1:
                plot.set_xlabel("Distance (input coordinate units)")
        self.canvas.draw_idle()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SuperVario - Leapfrog Edge variogram candidates")
        self.setWindowIcon(application_icon())
        self.resize(1440, 960)
        self.config = default_config()
        self.report = None
        self.worker = None
        self.bound_result = None
        self.reference_windows = []
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        toolbar = QtWidgets.QHBoxLayout()
        for label, callback in (("Load project...", self.load_project), ("Save project...", self.save_project),
                                ("Open LFV...", self.open_lfv)):
            button = QtWidgets.QPushButton(label)
            button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            button.clicked.connect(callback)
            toolbar.addWidget(button)
        self.run_button = QtWidgets.QPushButton("Calculate / refit domains")
        self.run_button.clicked.connect(self.calculate)
        toolbar.addWidget(self.run_button)
        self.cancel_button = QtWidgets.QPushButton("Cancel after current step")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel)
        toolbar.addWidget(self.cancel_button)
        self.export_button = QtWidgets.QPushButton("Export reviewed LFVs + report...")
        self.export_button.clicked.connect(self.export)
        self.export_button.setEnabled(False)
        toolbar.addWidget(self.export_button)
        for button in (self.run_button, self.cancel_button, self.export_button):
            button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        toolbar.addStretch()
        layout.addLayout(toolbar)
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs, 1)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        workflow = QtWidgets.QWidget()
        flow = QtWidgets.QVBoxLayout(workflow)
        self.samples = TableInput("Samples: CSV / Excel", samples=True)
        flow.addWidget(self.samples)
        self.kind = QtWidgets.QComboBox()
        self.kind.addItems(["Flagged block model", "Domain solids"])
        compact(self.kind, 30)
        flow.addWidget(self.kind, alignment=QtCore.Qt.AlignLeft)
        self.domain_stack = QtWidgets.QStackedWidget()
        self.blocks = TableInput("Block model: streamed CSV / Excel, supported BMF variants")
        self.domain_stack.addWidget(self.blocks)
        solid_widget = QtWidgets.QWidget()
        solids_layout = QtWidgets.QVBoxLayout(solid_widget)
        self.solids = QtWidgets.QTableWidget(0, 2)
        self.solids.setHorizontalHeaderLabels(["Unique domain name (editable)", "Solid file"])
        self.solids.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Interactive)
        self.solids.setColumnWidth(0, 220)
        self.solids.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.solids.setMinimumHeight(170)
        solids_layout.addWidget(self.solids)
        row = QtWidgets.QHBoxLayout()
        add = QtWidgets.QPushButton("Add solids...")
        add.clicked.connect(self.add_solids)
        remove = QtWidgets.QPushButton("Remove selected solid")
        remove.clicked.connect(lambda: self.solids.removeRow(self.solids.currentRow()))
        row.addWidget(add)
        row.addWidget(remove)
        self.repair = QtWidgets.QComboBox()
        self.repair.addItems(["none", "meshops", "voxel"])
        compact(self.repair, 9)
        row.addWidget(labelled("Repair", self.repair))
        self.voxel_pitch = spin(5)
        self.voxel_pitch.setMinimum(0.00001)
        compact(self.voxel_pitch, 13)
        row.addWidget(labelled("Voxel pitch", self.voxel_pitch))
        row.addStretch()
        solids_layout.addLayout(row)
        note = QtWidgets.QLabel(
            "Voxel mode is approximate: one shared grid, first solid wins conflicting cells. "
            "Choose a suitable resolution and review geometry in MeshOps. No repair modifies your inputs.")
        note.setWordWrap(True)
        solids_layout.addWidget(note)
        self.domain_stack.addWidget(solid_widget)
        self.kind.currentIndexChanged.connect(self.domain_stack.setCurrentIndex)
        flow.addWidget(self.domain_stack)
        filter_row = QtWidgets.QHBoxLayout()
        filters_button = QtWidgets.QPushButton("Edit sample / flagged-block filters...")
        filters_button.clicked.connect(self.edit_filters)
        filter_row.addWidget(filters_button)
        self.filter_summary = QtWidgets.QLabel("No filters")
        self.filter_summary.setWordWrap(True)
        filter_row.addWidget(self.filter_summary, 1)
        flow.addLayout(filter_row)
        model_settings = QtWidgets.QGroupBox("Experimental variogram and model settings")
        model_settings.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        settings_form = QtWidgets.QFormLayout(model_settings)
        settings_form.setLabelAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        settings_form.setHorizontalSpacing(8)
        self.settings = {}
        for key, label, value, maximum in (
            ("bins", "Number of lag bins (minimum 3)", 12, 1000), ("lag", "Lag distance", 0, 1e9),
            ("max_distance", "Maximum distance", 0, 1e12),
            ("angle_tolerance", "Cone half-angle", 30, 90), ("min_pairs", "Min pairs/bin", 10, 1000000)):
            widget = spin(value, maximum, 0 if key in ("bins", "min_pairs") else 3)
            widget.setMinimum({"bins": 3, "min_pairs": 1, "angle_tolerance": 0.001}.get(key, 0))
            if key in ("lag", "max_distance"):
                widget.setSpecialValueText("Automatic")
                widget.setToolTip("Automatic is stored as 0. Enter a positive distance to override.")
                if key == "lag":
                    widget.setToolTip(
                        "Automatic uses drilling spacing when Hole ID is mapped, with optional From/To support; "
                        "otherwise maximum distance / bin count. Spacing selection may adjust the effective "
                        "bin count to preserve distance coverage. Enter a positive distance to override.")
            elif key == "bins":
                widget.setToolTip(
                    "Choose at least 3 bins. Default: 12. Used for manual lag distance and the geometric default. "
                    "Spacing-informed automatic lag selection may adjust the effective count; see the review summary.")
            self.settings[key] = widget
            compact(widget, 6 if key == "bins" else 9 if key == "min_pairs" else 10 if key == "angle_tolerance" else 16)
            settings_form.addRow(label, widget)
        row = field_row()
        for key, label, value, maximum in (
            ("max_samples", "Sample cap/domain", 5000, 10000000),
            ("max_pairs", "Pair cap/domain", 300000, 100000000),
            ("structures", "Spherical structures (1/2)", 1, 2)):
            widget = spin(value, maximum, 0)
            widget.setMinimum({"max_samples": 12, "max_pairs": 1, "structures": 1}[key])
            widget.setToolTip({
                "max_samples": "Maximum samples used for variogram pairs in each domain. Larger datasets are uniformly "
                               "subsampled with a reproducible seed. Increase for more detail at greater computation cost.",
                "max_pairs": "Maximum unordered sample pairs drawn per domain before distance and direction filtering. "
                             "Increase to improve lag-bin support, at greater memory and computation cost.",
                "structures": "Fit one or two nested spherical components in addition to the nugget. "
                              "Two components can represent different continuity scales but need stronger data support.",
            }[key])
            self.settings[key] = widget
            compact(widget, 3 if key == "structures" else 10)
            row.addWidget(labelled(label, widget))
        row.addStretch()
        settings_form.addRow(row)
        plane_step = QtWidgets.QSpinBox()
        plane_step.setRange(5, 90)
        plane_step.setSingleStep(5)
        plane_step.setValue(15)
        plane_step.setSuffix(" deg")
        plane_step.setToolTip(
            "Rotate the major and semi-major axes together within their plane during automatic fitting. "
            "Use multiples of 5 degrees (default 15). Smaller steps compare more frames and take longer; "
            "they do not add experimental support. Separate from radial display divisions and cone half-angle. "
            "Manual direction bypasses this search.")
        compact(plane_step, 9)
        self.settings["in_plane_step"] = plane_step
        settings_form.addRow("In-plane search step", plane_step)
        for key, label, default in (("azimuth_step", "Azimuth search step", 15),
                                    ("plunge_step", "Plunge search step", 30)):
            widget = QtWidgets.QSpinBox()
            widget.setRange(5, 90)
            widget.setSingleStep(5)
            widget.setValue(default)
            widget.setSuffix(" deg")
            widget.setToolTip(
                f"Angular increment for the automatic 3D direction grid (default {default} degrees). "
                "Use multiples of 5 degrees. Smaller steps screen more directions and cost more computation; "
                "a bounded shortlist of complete frames is then compared jointly, including alternative plane normals. "
                + ("Also controls perpendicular-direction screening. " if key == "azimuth_step" else "") +
                "Separate from in-plane refinement and radial display resolution. Manual direction bypasses this search.")
            compact(widget, 9)
            self.settings[key] = widget
            settings_form.addRow(label, widget)
        row = field_row()
        self.manual_rotation = QtWidgets.QCheckBox("Manual direction (otherwise coarse automatic search)")
        self.manual_rotation.setToolTip(
            "Use the dip azimuth, dip and pitch below instead of searching automatically for continuity directions. "
            "Bypasses automatic frame search. Recalculate to apply the direction.")
        row.addWidget(self.manual_rotation)
        self.rotation = {}
        for key, label, limit in (("dip_direction", "Dip azimuth", 360), ("dip", "Dip", 90), ("pitch", "Pitch", 180)):
            widget = spin(0, limit, 3)
            widget.setToolTip({
                "dip_direction": "Compass direction in which the major/semi-major plane dips: "
                                 "0 = north, 90 = east (0-360 degrees). Used only with Manual direction enabled.",
                "dip": "Inclination of the major/semi-major plane: 0 = horizontal, 90 = vertical. "
                       "Used only with Manual direction enabled.",
                "pitch": "Rotation of the major axis within the dipping plane: 0 = along strike, "
                         "90 = down dip (0-180 degrees). Used only with Manual direction enabled.",
            }[key])
            compact(widget, 8)
            self.rotation[key] = widget
            row.addWidget(labelled(label, widget))
        row.addStretch()
        settings_form.addRow(row)
        self.contact_distance = spin(25)
        self.contact_distance.setToolTip(
            "Distance in coordinate units to the nearest opposite-domain sample used for contact analysis. "
            "For an approved domain pair, it also limits borrowing to samples near target-domain samples. "
            "This is not distance to the actual boundary; sharing requires explicit approval and recalculation.")
        self.contact_distance.setMinimum(0.00001)
        compact(self.contact_distance, 16)
        settings_form.addRow("Contact / approved sharing distance", self.contact_distance)
        self.contact_minimum = spin(20, 1000000, 0)
        self.contact_minimum.setToolTip(
            "Minimum near-contact samples required on each side before suggesting a soft-boundary review. "
            "This does not approve sharing or set the minimum samples for variogram fitting or estimation.")
        self.contact_minimum.setMinimum(2)
        compact(self.contact_minimum, 9)
        settings_form.addRow("Contact minimum samples/side", self.contact_minimum)
        for widget in (self.contact_distance, self.contact_minimum):
            settings_form.labelForField(widget).setToolTip(widget.toolTip())
        flow.addWidget(model_settings)
        note = QtWidgets.QLabel(
            "Classical semivariograms; no implicit compositing, detrending or declustering. "
            "Same Cartesian coordinate system/units required. Automatic fits and contact suggestions require geological review.")
        note.setWordWrap(True)
        flow.addWidget(note)
        flow.addStretch()
        scroll.setWidget(workflow)
        self.tabs.addTab(scroll, "1. Workflow")
        results = QtWidgets.QWidget()
        result_layout = QtWidgets.QVBoxLayout(results)
        self.result_picker = QtWidgets.QComboBox()
        self.result_picker.currentIndexChanged.connect(self.select_result)
        result_layout.addWidget(self.result_picker)
        self.model_panel = ModelPanel()
        result_layout.addWidget(self.model_panel, 1)
        self.tabs.addTab(results, "2. Variogram review")
        contacts_widget = QtWidgets.QWidget()
        contacts_layout = QtWidgets.QVBoxLayout(contacts_widget)
        warning = QtWidgets.QLabel(
            "Nearest opposite-domain SAMPLE distance is a contact proxy, not geometric boundary distance. "
            "Similar near-contact distributions do not prove continuity. Sharing is OFF until you approve and rerun.")
        warning.setWordWrap(True)
        contacts_layout.addWidget(warning)
        self.contacts = QtWidgets.QTableWidget(0, 8)
        self.contacts.setHorizontalHeaderLabels(["Approve", "Domain A", "Domain B", "N A", "N B", "Mean effect", "KS statistic", "Review suggestion"])
        self.contacts.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.contacts.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.contacts.itemSelectionChanged.connect(self.plot_contact)
        contacts_layout.addWidget(self.contacts)
        approve = QtWidgets.QPushButton("Use checked approvals and recalculate (unchecked = hard boundary)")
        approve.clicked.connect(self.approve_boundaries)
        contacts_layout.addWidget(approve)
        self.contact_figure = Figure(figsize=(9, 3), constrained_layout=True)
        self.contact_canvas = FigureCanvasQTAgg(self.contact_figure)
        contacts_layout.addWidget(self.contact_canvas, 1)
        self.tabs.addTab(contacts_widget, "3. Contact review / approvals")
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumWidth(180)
        self.log_dock = QtWidgets.QDockWidget("Run log", self)
        self.log_dock.setAllowedAreas(QtCore.Qt.RightDockWidgetArea)
        self.log_dock.setFeatures(QtWidgets.QDockWidget.DockWidgetClosable)
        self.log_dock.setWidget(self.log)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.log_dock)
        self.resizeDocks([self.log_dock], [320], QtCore.Qt.Horizontal)
        self.status = QtWidgets.QLabel("Ready. Load a project or select samples and domains.")
        self.status.setWordWrap(True)
        feedback = QtWidgets.QHBoxLayout()
        feedback.addWidget(self.status, 1)
        self.activity = QtWidgets.QProgressBar()
        self.activity.setFixedWidth(110)
        self.activity.setTextVisible(False)
        self.activity.setRange(0, 1)
        self.activity.setValue(0)
        self.activity.setToolTip("Busy animation indicates activity, not a percentage of completed work.")
        feedback.addWidget(self.activity)
        self.log_toggle = QtWidgets.QPushButton("Hide log")
        self.log_toggle.setCheckable(True)
        self.log_toggle.setChecked(True)
        self.log_toggle.setToolTip("Show or hide the resizable log panel on the right, including during calculation.")
        self.log_toggle.toggled.connect(self.log_dock.setVisible)
        self.log_dock.visibilityChanged.connect(self.log_visibility_changed)
        feedback.addWidget(self.log_toggle)
        layout.addLayout(feedback)
        self.elapsed = QtCore.QElapsedTimer()
        self.feedback_timer = QtCore.QTimer(self)
        self.feedback_timer.setInterval(1000)
        self.feedback_timer.timeout.connect(self.update_feedback)
        self.current_step = ""
        self.populate(self.config)

    def log_visibility_changed(self, visible):
        self.log_toggle.blockSignals(True)
        self.log_toggle.setChecked(visible)
        self.log_toggle.setText("Hide log" if visible else "Show log")
        self.log_toggle.blockSignals(False)

    def progress_message(self, message):
        self.log.appendPlainText(message)
        self.current_step = message
        if self.worker is not None:
            self.update_feedback()

    def update_feedback(self):
        seconds = self.elapsed.elapsed() // 1000
        prefix = "Cancelling" if self.worker is not None and self.worker.isInterruptionRequested() else "Calculating"
        self.status.setText(f"{prefix} ({seconds // 60:02d}:{seconds % 60:02d}) - {self.current_step}")

    def collect(self):
        config = copy.deepcopy(self.config)
        for key in ("hole_id", "from", "to"):
            config["samples"].pop(key, None)
        config["samples"].update(self.samples.spec())
        config["domains"].update(self.blocks.spec())
        config["domains"]["kind"] = "blocks" if self.kind.currentIndex() == 0 else "solids"
        config["domains"]["solids"] = [
            {"name": self.solids.item(i, 0).text().strip(), "path": self.solids.item(i, 1).text()}
            for i in range(self.solids.rowCount())]
        config["domains"].update(repair=self.repair.currentText(), voxel_pitch=self.voxel_pitch.value())
        integer_keys = ("bins", "min_pairs", "max_samples", "max_pairs", "structures", "in_plane_step",
                        "azimuth_step", "plunge_step")
        for key, widget in self.settings.items():
            config["variogram"][key] = int(widget.value()) if key in integer_keys else widget.value()
        config["variogram"]["rotation"] = {k: w.value() for k, w in self.rotation.items()} if self.manual_rotation.isChecked() else None
        config["boundaries"]["distance"] = self.contact_distance.value()
        config["boundaries"]["min_samples"] = int(self.contact_minimum.value())
        validate_settings(config["variogram"])
        return config

    def populate(self, config):
        validate_settings(config.get("variogram", {}))
        self.config = copy.deepcopy(config)
        self.samples.populate(config["samples"])
        self.blocks.populate(config["domains"])
        self.kind.setCurrentIndex(0 if config["domains"]["kind"] == "blocks" else 1)
        specs = config["domains"].get("solids", [])
        self.solids.setRowCount(len(specs))
        for row, spec in enumerate(specs):
            for col, field in enumerate(("name", "path")):
                self.solids.setItem(row, col, QtWidgets.QTableWidgetItem(spec[field]))
        self.repair.setCurrentText(config["domains"].get("repair", "none"))
        self.voxel_pitch.setValue(config["domains"].get("voxel_pitch", 5))
        for key, widget in self.settings.items():
            default = {"in_plane_step": 15, "azimuth_step": 15, "plunge_step": 30}.get(key, widget.value())
            widget.setValue(config["variogram"].get(key, default))
        rotation = config["variogram"].get("rotation")
        self.manual_rotation.setChecked(rotation is not None)
        for key, widget in self.rotation.items():
            widget.setValue((rotation or {}).get(key, 0))
        self.contact_distance.setValue(config["boundaries"].get("distance", 25))
        self.contact_minimum.setValue(config["boundaries"].get("min_samples", 20))
        self.filter_summary.setText(f"{len(config.get('filters', []))} filter(s); all are combined with AND")

    def add_solids(self):
        filenames, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Add solid domains (table order is voxel priority)", "",
            "Meshes (*.dxf *.00t *.msh *.stl *.obj *.ply);;All files (*)")
        for filename in filenames:
            row = self.solids.rowCount()
            self.solids.insertRow(row)
            self.solids.setItem(row, 0, QtWidgets.QTableWidgetItem(Path(filename).stem))
            self.solids.setItem(row, 1, QtWidgets.QTableWidgetItem(filename))

    def edit_filters(self):
        sources = []
        try:
            config = self.collect()
            source_class = local_module("srkreliability.filter_source", "SRKReliability").FilterDataSource
            dialog_class = local_module("srkreliability.gui_filters", "SRKReliability").FiltersDialog
            virtual = {}
            if config["domains"]["kind"] == "blocks":
                block = config["domains"]
                if Path(block["path"]).suffix.lower() == ".bmf":
                    raise ValueError("For BMF filter previews, convert to CSV first, or define BM: filters in a saved project JSON file and load it.")
                block_source = source_class(block["path"], sheet=block.get("sheet") or None, fmt=table_format(block))
                sources.append(block_source)
                virtual = {"BM:" + c: (block_source, c) for c in block_source.columns}
            sample = config["samples"]
            source = source_class(sample["path"], sheet=sample.get("sheet") or None, fmt=table_format(sample), virtual=virtual)
            sources.append(source)
            dialog = dialog_class(source, config.get("filters", []), self)
            if dialog.exec_() == QtWidgets.QDialog.Accepted:
                self.config["filters"] = dialog.filters
                self.filter_summary.setText(f"{len(dialog.filters)} filter(s); all combined with AND")
        except Exception:
            error(self, traceback.format_exc())
        finally:
            for source in sources:
                source.close()

    def load_project(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load project", "", "JSON (*.json)")
        if filename:
            try:
                self.populate(load_config(filename))
                self.status.setText("Project settings loaded. Calculate to update results; any displayed results are from the prior calculation.")
            except Exception:
                error(self, traceback.format_exc())

    def save_project(self):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save project", "", "JSON (*.json)")
        if filename:
            try:
                config = self.collect()
                public = {k: v for k, v in config.items() if not k.startswith("_")}
                lfv.write_json(filename, public, input_paths(config, include_projects=False))
                self.config = config
                saved_path = str(Path(filename).resolve())
                protected = self.config.setdefault("_protected_inputs", [])
                if saved_path not in protected:
                    protected.append(saved_path)
                self.status.setText(f"Project saved: {saved_path}")
            except Exception:
                error(self, traceback.format_exc())

    def calculate(self):
        if self.worker is not None:
            return
        try:
            config = self.collect()
        except ValueError as exc:
            error(self, str(exc))
            return
        try:
            self.config = copy.deepcopy(config)
            self.log.clear()
            self.current_step = "Starting; inputs are read-only."
            self.elapsed.start()
            self.activity.setRange(0, 0)
            self.feedback_timer.start()
            self.tabs.setEnabled(False)
            self.run_button.setEnabled(False)
            self.export_button.setEnabled(False)
            # Disable project actions while the worker owns a configuration snapshot.
            for button in self.centralWidget().findChildren(QtWidgets.QPushButton):
                button.setEnabled(False)
            self.log_toggle.setEnabled(True)
            self.cancel_button.setEnabled(True)
            self.worker = Worker(config)
            self.worker.message.connect(self.progress_message)
            self.worker.completed.connect(self.completed)
            self.worker.failed.connect(self.failed)
            self.worker.cancelled.connect(self.cancelled)
            self.worker.finished.connect(self.finished)
            self.update_feedback()
            self.worker.start()
        except Exception:
            error(self, traceback.format_exc())

    def completed(self, report):
        self.feedback_timer.stop()
        self.report = report
        self.bound_result = None
        self.model_panel.result = None
        self.model_panel.dirty = False
        self.result_picker.blockSignals(True)
        self.result_picker.clear()
        self.result_picker.addItems([r["domain"] for r in report["results"]])
        self.result_picker.blockSignals(False)
        self.select_result(0)
        self.populate_contacts()
        failures = report["failures"]
        self.status.setText(f"Calculated {len(report['results'])} domain(s); {len(failures)} failed. Review models and contact suggestions before export.")
        if failures:
            error(self, "\n".join(f"{f['domain']}: {f['error']}" for f in failures))
        self.tabs.setCurrentIndex(self.tabs.indexOf(self.model_panel.parentWidget()))
        self.export_button.setEnabled(self.worker is None)

    def cancel(self):
        if self.worker is not None:
            self.worker.requestInterruption()
            self.cancel_button.setEnabled(False)
            self.status.setText("Cancellation requested; waiting for the current read, repair or fit step to finish.")

    def cancelled(self):
        self.feedback_timer.stop()
        self.log.appendPlainText("Calculation cancelled. No new files exported.")
        self.status.setText("Cancelled. Any previously completed report remains unchanged.")

    def failed(self, details):
        self.feedback_timer.stop()
        self.log.appendPlainText(details)
        self.status.setText("Calculation failed. No new report exported. See Run log.")
        error(self, details)

    def finished(self):
        self.feedback_timer.stop()
        self.activity.setRange(0, 1)
        self.activity.setValue(0)
        self.worker.deleteLater()
        self.worker = None
        self.tabs.setEnabled(True)
        for button in self.centralWidget().findChildren(QtWidgets.QPushButton):
            button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.export_button.setEnabled(self.report is not None)

    def select_result(self, index):
        if self.report is None or index < 0:
            return
        try:
            self.model_panel.apply_edits()
            self.bound_result = index
            self.model_panel.bind(self.report["results"][index])
        except Exception:
            self.result_picker.blockSignals(True)
            self.result_picker.setCurrentIndex(self.bound_result if self.bound_result is not None else 0)
            self.result_picker.blockSignals(False)
            error(self, traceback.format_exc())

    def populate_contacts(self):
        contacts = self.report["contacts"]
        approved = {frozenset(p) for p in self.report["config"]["boundaries"].get("approved_pairs", [])}
        self.contacts.setRowCount(len(contacts))
        for row, contact in enumerate(contacts):
            check = QtWidgets.QTableWidgetItem()
            check.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsSelectable)
            check.setCheckState(QtCore.Qt.Checked if frozenset((contact["a"], contact["b"])) in approved else QtCore.Qt.Unchecked)
            self.contacts.setItem(row, 0, check)
            effect = contact["standardised_mean_difference"]
            values = [contact["a"], contact["b"], contact["n_a"], contact["n_b"],
                      f"{effect:.3g}" if effect is not None else "undefined", f"{contact['ks_statistic']:.3g}",
                      "Candidate for REVIEW" if contact["recommend_review"] else "No suggestion"]
            for col, value in enumerate(values, 1):
                item = QtWidgets.QTableWidgetItem(str(value))
                item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
                self.contacts.setItem(row, col, item)
        self.contact_figure.clear()
        if contacts:
            self.contacts.selectRow(0)
            self.plot_contact()
        self.contact_canvas.draw_idle()

    def plot_contact(self):
        row = self.contacts.currentRow()
        if self.report is None or row < 0 or row >= len(self.report["contacts"]):
            return
        contact = self.report["contacts"][row]
        self.contact_figure.clear()
        plot = self.contact_figure.add_subplot(111)
        profile = contact["profile"]
        x = [p["distance"] for p in profile]
        y = [np.nan if p["mean"] is None else p["mean"] for p in profile]
        plot.scatter(x, y, s=[p["count"] + 5 for p in profile])
        plot.axvline(0, color="grey", linestyle="--")
        plot.set_xlabel(f"Signed nearest opposite-domain sample distance ({contact['a']} negative; {contact['b']} positive)")
        plot.set_ylabel("Mean sample value")
        plot.set_title("Contact proxy profile - not geometric distance to boundary")
        plot.grid(alpha=0.25)
        self.contact_canvas.draw_idle()

    def approve_boundaries(self):
        if self.report is None:
            return
        try:
            current = self.collect()
            prior = copy.deepcopy(self.report["config"])
            for config in (current, prior):
                config.pop("_protected_inputs", None)
                config.pop("output", None)
                config["boundaries"].pop("approved_pairs", None)
            if current != prior:
                raise ValueError("Inputs or settings changed since contact analysis. Recalculate before approving these contacts.")
        except Exception:
            error(self, traceback.format_exc())
            return
        approvals = [[c["a"], c["b"]] for i, c in enumerate(self.report["contacts"])
                     if self.contacts.item(i, 0).checkState() == QtCore.Qt.Checked]
        answer = QtWidgets.QMessageBox.question(
            self, "Confirm boundary policy",
            f"Use {len(approvals)} approved domain pair(s) for sample sharing and recalculate?\n"
            "Sharing uses distance to target-domain samples, not a solid surface. Other boundaries remain hard.")
        if answer == QtWidgets.QMessageBox.Yes:
            self.config["boundaries"]["approved_pairs"] = approvals
            self.calculate()

    def export(self):
        if self.report is None:
            return
        try:
            self.model_panel.apply_edits()
            directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Export reviewed LFVs and audit report")
            if not directory:
                return
            try:
                paths = export_report(self.report, directory)
            except FileExistsError:
                answer = QtWidgets.QMessageBox.question(self, "Replace prior outputs?", "Replace the existing SuperVario output files in this folder?")
                if answer != QtWidgets.QMessageBox.Yes:
                    return
                paths = export_report(self.report, directory, overwrite=True)
            self.log.appendPlainText("\n".join(paths))
            self.status.setText(f"Exported {len(self.report['results'])} LFV(s), Excel/JSON estimator recommendations and analysis.json to {directory}")
        except Exception:
            error(self, traceback.format_exc())

    def open_lfv(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open reference LFV", "", "Leapfrog variogram (*.lfv)")
        if not filename:
            return
        try:
            model = lfv.load(filename)
            dialog = QtWidgets.QDialog(self)
            dialog.setWindowTitle("Reference LFV - " + Path(filename).name)
            dialog.resize(1200, 850)
            layout = QtWidgets.QVBoxLayout(dialog)
            panel = ModelPanel()
            panel.bind({"domain": model["metadata"].get("domain", "Reference"), "model": model})
            layout.addWidget(panel)
            save = QtWidgets.QPushButton("Save edited LFV as...")
            layout.addWidget(save)

            def save_reference():
                try:
                    panel.apply_edits()
                    output, _ = QtWidgets.QFileDialog.getSaveFileName(dialog, "Save LFV", "", "Leapfrog variogram (*.lfv)")
                    if output:
                        lfv.save(output, panel.result["model"], [filename] + input_paths(self.collect()))
                except Exception:
                    error(dialog, traceback.format_exc())

            save.clicked.connect(save_reference)
            self.reference_windows.append(dialog)
            dialog.show()
        except Exception:
            error(self, traceback.format_exc())

    def closeEvent(self, event):
        if self.worker is not None:
            QtWidgets.QMessageBox.information(self, "Calculation running", "Wait for the active calculation to finish before closing.")
            event.ignore()
        else:
            event.accept()


def launch(config_path=None):
    import sys
    if sys.platform == "win32":
        import ctypes
        result = ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("SuperVario.Desktop")
        if result != 0:
            raise OSError(f"Cannot set SuperVario taskbar identity (HRESULT {result}).")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setApplicationName("SuperVario")
    app.setWindowIcon(application_icon())
    window = MainWindow()
    if config_path:
        window.populate(load_config(config_path))
    window.show()
    return app.exec_()
