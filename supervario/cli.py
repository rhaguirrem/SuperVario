import argparse
import json
import sys
import traceback
from pathlib import Path

from . import lfv
from .pipeline import default_config, export_report, load_config, run


def demo(directory):
    import numpy as np
    import pandas as pd

    root = Path(directory).resolve()
    config = default_config()
    config["samples"].update(path=str(root / "samples.csv"), xyz=["X", "Y", "Z"], value="grade")
    config["domains"].update(path=str(root / "blocks.csv"), domain="domain", xyz=["X", "Y", "Z"],
                             size_cols=["DX", "DY", "DZ"])
    config["output"] = str(root / "output")
    config["variogram"].update(max_distance=65, min_pairs=5)
    config_path = root / "project.json"
    for path in (config_path, Path(config["samples"]["path"]), Path(config["domains"]["path"])):
        if path.exists():
            raise FileExistsError(f"Demo would overwrite {path}")
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    points = rng.uniform(0, 100, (1200, 3))
    # A shared smooth field crossing two domains, plus short-range noise.
    grade = (5 + np.sin(points[:, 0] / 25) + np.cos(points[:, 1] / 12)
             + np.sin(points[:, 2] / 5) * 0.6 + rng.normal(0, 0.2, len(points)))
    samples = pd.DataFrame(points, columns=["X", "Y", "Z"])
    samples["grade"] = grade
    samples["quality"] = rng.choice(["good", "review"], len(points), p=[0.95, 0.05])
    samples.to_csv(config["samples"]["path"], index=False)
    pd.DataFrame([
        [25, 50, 50, 50, 100, 100, "A", 1],
        [75, 50, 50, 50, 100, 100, "B", 1],
    ], columns=["X", "Y", "Z", "DX", "DY", "DZ", "domain", "active"]).to_csv(
        config["domains"]["path"], index=False)
    config["filters"] = [{"field": "quality", "type": "categorical", "values": ["good"]},
                         {"field": "BM:active", "type": "numeric", "min": 1}]
    lfv.write_json(config_path, config)
    return config_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="SuperVario: local Leapfrog Edge variogram candidates")
    commands = parser.add_subparsers(dest="command", required=True)
    gui = commands.add_parser("gui", help="Launch the Windows desktop interface")
    gui.add_argument("config", nargs="?")
    init = commands.add_parser("init", help="Write an editable JSON project configuration")
    init.add_argument("path")
    init.add_argument("--force", action="store_true")
    inspect = commands.add_parser("header", help="Inspect CSV/Excel/BMF columns and file metadata")
    inspect.add_argument("path")
    inspect.add_argument("--sheet", default="")
    inspect.add_argument("--sep", default="")
    inspect.add_argument("--decimal", default="")
    inspect.add_argument("--header-line", type=int, default=0)
    execute = commands.add_parser("run", help="Fit and export one LFV per successful domain plus a JSON report")
    execute.add_argument("config")
    execute.add_argument("--overwrite", action="store_true")
    inspect_lfv = commands.add_parser("inspect-lfv", help="Validate and display a reference LFV")
    inspect_lfv.add_argument("path")
    example = commands.add_parser("demo", help="Generate reproducible synthetic inputs and a project")
    example.add_argument("directory")
    repair = commands.add_parser("repair-mesh", help="Repair a mesh with local MeshOps (no input overwrites)")
    repair.add_argument("input")
    repair.add_argument("output")
    args = parser.parse_args(argv)
    try:
        if args.command == "gui":
            from .gui import launch
            return launch(args.config)
        if args.command == "init":
            if Path(args.path).exists() and not args.force:
                raise FileExistsError(args.path)
            lfv.write_json(args.path, default_config())
        elif args.command == "header":
            from .data import header
            from .local_tools import readers
            spec = {"path": args.path, "sheet": args.sheet,
                    "format": {"sep": args.sep, "decimal": args.decimal, "header_line": args.header_line}}
            result = {"columns": header(spec)}
            if Path(args.path).suffix.lower() not in (".bmf", ".xlsx", ".xlsm", ".xls"):
                result["format"] = vars(readers().csv_layout(args.path, readers().TableFormat(**spec["format"])))
                result["metadata"] = readers().read_leapfrog_metadata(args.path)
            print(json.dumps(result, indent=2))
        elif args.command == "inspect-lfv":
            print(json.dumps(lfv.load(args.path), indent=2))
        elif args.command == "demo":
            print(demo(args.directory))
        elif args.command == "repair-mesh":
            from .local_tools import mesh_tools
            output = lfv.protect_output(args.output, [args.input])
            if output.exists():
                raise FileExistsError(output)
            tools = mesh_tools()
            mesh = tools.heal_mesh(tools.load_mesh(Path(args.input)), intended_mesh_type="solid",
                                   allow_residual_alpha_wrap_fallback=False, status_callback=print)
            if not mesh.is_volume:
                raise ValueError("Repair did not produce a valid closed volume; no mesh written.")
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.suffix.lower() == ".msh":
                tools.export_leapfrog_msh(mesh, output)
            elif output.suffix.lower() == ".dxf":
                tools.export_dxf_polyface(mesh, output)
            else:
                mesh.export(output)
        elif args.command == "run":
            config = load_config(args.config)
            report = run(config)
            for path in export_report(report, config["output"], overwrite=args.overwrite):
                print(path)
            if report["failures"]:
                print("Some domains failed. See analysis.json.", file=sys.stderr)
                return 2
        return 0
    except Exception:
        traceback.print_exc()
        return 1
