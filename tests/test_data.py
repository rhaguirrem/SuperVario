import numpy as np
import pandas as pd
import pytest
import trimesh

from supervario import data
from supervario.data import DOMAIN, assign_blocks, assign_solids, chunks, coherent_voxels, header, prepare
from supervario.local_tools import mesh_tools


def test_spanish_csv_and_comment_header_detection(tmp_path):
    path = tmp_path / "spanish.csv"
    path.write_text('# Leapfrog metadata\nX;Y;Z;grade;category\n1,5;2,5;3,5;4,5;"NA"\n', encoding="utf-8")
    spec = {"path": str(path)}
    assert header(spec) == ["X", "Y", "Z", "grade", "category"]
    frame = next(chunks(spec, text_columns=["category"]))
    assert frame["grade"].iloc[0] == 4.5
    assert frame["category"].iloc[0] == "NA"


def test_explicit_header_row_and_quoted_delimiter(tmp_path):
    path = tmp_path / "title.csv"
    path.write_text('Project title\nUnrelated information\nX,Y,Z,"grade,ppm"\n1,2,3,4\n', encoding="utf-8")
    spec = {"path": str(path), "format": {"header_line": 3, "sep": ","}}
    assert header(spec) == ["X", "Y", "Z", "grade,ppm"]


def test_excel_sheet_and_header_row(tmp_path):
    path = tmp_path / "samples.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"X": [1], "Y": [2], "Z": [3], "grade": [4]}).to_excel(
            writer, sheet_name="Samples", index=False, startrow=2)
    spec = {"path": str(path), "sheet": "Samples", "format": {"header_line": 3}}
    assert header(spec) == ["X", "Y", "Z", "grade"]
    assert next(chunks(spec))["grade"].iloc[0] == 4


def block_spec(tmp_path, rows):
    path = tmp_path / "blocks.csv"
    pd.DataFrame(rows, columns=["X", "Y", "Z", "DX", "DY", "DZ", "domain"]).to_csv(path, index=False)
    return {"path": str(path), "domain": "domain", "xyz": ["X", "Y", "Z"], "size_cols": ["DX", "DY", "DZ"]}


def test_subblocks_and_shared_face_ownership(tmp_path):
    spec = block_spec(tmp_path, [[0.5, 0.5, 0.5, 1, 1, 1, "007"], [1.5, 0.5, 0.5, 1, 1, 1, "NA"],
                                 [3, 3, 3, 2, 2, 2, "large"]])
    samples = pd.DataFrame({"X": [0.9, 1, 1.1, 3, 4.1], "Y": [0.5, 0.5, 0.5, 3, 3], "Z": [0.5, 0.5, 0.5, 3, 3]})
    result = assign_blocks(samples, ["X", "Y", "Z"], spec, ["domain"], lambda _: None)
    assert result[DOMAIN].iloc[:4].tolist() == ["007", "NA", "NA", "large"]
    assert pd.isna(result[DOMAIN].iloc[4])


def test_overlapping_blocks_are_not_silently_resolved(tmp_path):
    spec = block_spec(tmp_path, [[0, 0, 0, 2, 2, 2, "A"], [0, 0, 0, 1, 1, 1, "B"]])
    with pytest.raises(ValueError, match="Overlapping"):
        assign_blocks(pd.DataFrame({"X": [0], "Y": [0], "Z": [0]}), ["X", "Y", "Z"], spec, ["domain"], print)


def test_block_size_is_not_guessed(tmp_path):
    path = tmp_path / "unknown.csv"
    pd.DataFrame({"X": [0], "Y": [0], "Z": [0], "domain": ["A"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="sizes are never guessed"):
        assign_blocks(pd.DataFrame({"X": [0], "Y": [0], "Z": [0]}), ["X", "Y", "Z"],
                      {"path": str(path), "domain": "domain"}, ["domain"], print)


def test_sample_and_flagged_expression_filters_use_numeric_types(project):
    project["filters"] = [{"type": "expression", "expression": "BM:active == 1 and grade > 5"}]
    samples, _, _, _ = prepare(project, lambda _: None)
    assert len(samples) > 0
    assert (samples["grade"] > 5).all()
    assert (samples["BM:active"] == 1).all()


def test_missing_filter_column_is_explicit(project):
    project["filters"] = [{"field": "absent", "type": "present"}]
    with pytest.raises((KeyError, ValueError), match="absent"):
        prepare(project, lambda _: None)

def test_drilling_columns_are_preserved_as_text_and_intervals_validated(project):
    path = project["samples"]["path"]
    frame = pd.read_csv(path)
    frame["HOLE"] = "001"
    frame["FROM"] = np.arange(len(frame))
    frame["TO"] = frame["FROM"] + 1
    frame.to_csv(path, index=False)
    project["samples"].update(hole_id="HOLE", **{"from": "FROM", "to": "TO"})
    samples, _, _, _ = prepare(project, lambda _: None)
    assert set(samples["HOLE"]) == {"001"}
    assert (samples["TO"] > samples["FROM"]).all()
    frame["TO"] = -1
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="From < To"):
        prepare(project, lambda _: None)


@pytest.mark.parametrize("mapping,message", [
    ({"from": "X"}, "both From and To"),
    ({"from": "X", "to": "Y"}, "Hole ID is required"),
    ({"hole_id": "absent"}, "missing"),
    ({"hole_id": "X", "from": "X", "to": "Y"}, "distinct"),
])
def test_incomplete_or_invalid_drilling_mappings_are_explicit(project, mapping, message):
    project["samples"].update(mapping)
    with pytest.raises(ValueError, match=message):
        prepare(project, lambda _: None)


def test_unsupported_bmf_cannot_be_mistaken_for_empty_valid_model(monkeypatch):
    class FakeReader:
        @staticmethod
        def load_bmf_table(path):
            return {"dataframe": pd.DataFrame(), "reader_mode": "basic-inspect", "row_count": 0}
    monkeypatch.setattr(data, "bmf_tools", lambda: FakeReader())
    with pytest.raises(ValueError, match="Export a flagged CSV"):
        data.load_bmf("unknown.bmf")


@pytest.mark.parametrize("extension", [".msh", ".dxf", ".stl"])
def test_local_mesh_readers_and_point_containment(tmp_path, extension):
    mesh = trimesh.creation.box(extents=[2, 2, 2])
    path = tmp_path / ("solid" + extension)
    tools = mesh_tools()
    if extension == ".msh":
        tools.export_leapfrog_msh(mesh, path)
    elif extension == ".dxf":
        tools.export_dxf_polyface(mesh, path)
    else:
        mesh.export(path)
    samples = pd.DataFrame({"X": [0, 3], "Y": [0, 0], "Z": [0, 0]})
    result, audit = assign_solids(samples, ["X", "Y", "Z"], [{"name": "A", "path": str(path)}], {}, print)
    assert result[DOMAIN].iloc[0] == "A"
    assert pd.isna(result[DOMAIN].iloc[1])
    assert audit["mode"] == "none"


def test_open_mesh_requires_explicit_repair(tmp_path):
    mesh = trimesh.creation.box()
    mesh.update_faces(np.arange(len(mesh.faces) - 1))
    path = tmp_path / "open.stl"
    mesh.export(path)
    with pytest.raises(ValueError, match="closed volume"):
        assign_solids(pd.DataFrame({"X": [0], "Y": [0], "Z": [0]}), ["X", "Y", "Z"],
                      [{"name": "A", "path": str(path)}], {}, print)


def test_voxel_grid_is_shared_exclusive_and_priority_is_explicit():
    a = trimesh.creation.box(extents=[2, 2, 2])
    b = a.copy()
    b.apply_translation([2.1, 0, 0])
    origin, owners, conflicts = coherent_voxels([a, b], 1, 10000, print)
    assert conflicts > 0  # Disjoint continuous boxes can touch the same coarse cell.
    assert set(owners.values()) == {0, 1}
    _, first, _ = coherent_voxels([a], 1, 10000, print)
    assert all(owners[key] == 0 for key in first)
    np.testing.assert_allclose(origin, [-1, -1, -1])


def test_voxel_budget_is_checked_before_allocating_grid():
    with pytest.raises(ValueError, match="budget"):
        coherent_voxels([trimesh.creation.box(extents=[100, 100, 100])], 0.01, 1000, print)
