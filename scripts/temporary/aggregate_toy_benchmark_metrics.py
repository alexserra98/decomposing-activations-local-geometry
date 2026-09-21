"""Collect saved toy-benchmark JSON summaries without loading models or tensors.

Run with the repository's .venv/bin/python. Each evaluation setting contributes
one overall row and one row per planted manifold. KMeans thresholds are settings
of the same saved partition, not independent fits.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shlex
import sqlite3
import tempfile

import pandas as pd


REPO = Path(__file__).resolve().parents[2]
BENCHMARK = REPO / "dalg-cache/toy_manifolds_10types_1each_D128_30Keach_noise_sweep_seed0"
IDENTIFIERS = ("manifold_id", "type_id", "type_name", "intrinsic_dim", "embedding_dim")
CONTEXT_KEYS = {
    "schema_version", "evaluation", "model_kind", "run_id", "identity_hash",
    "K", "q_capacity", "q_max", "pca_capacity", "dataset", "artifacts",
    "partition_kind", "cattell",
}
FRONT_COLUMNS = [
    "evaluation_id", "run_id", "model_kind", "noise_ratio", "K", "q_capacity",
    "surgery_threshold", "seed", "split_seed", "dataset_seed", "training__surgery_threshold",
    "cattell_threshold", "scope", *IDENTIFIERS,
    "source_relative_path", "source_path", "source_sha256",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def scalar_leaves(value, path=(), *, skip_lists=False):
    """Keep source paths as tuples so flattening cannot silently merge keys."""
    if isinstance(value, dict):
        for key, item in value.items():
            require("__" not in key, f"Column separator collision at {path + (key,)}")
            yield from scalar_leaves(item, path + (key,), skip_lists=skip_lists)
    elif isinstance(value, list):
        if not skip_lists:
            for index, item in enumerate(value):
                yield from scalar_leaves(item, path + (index,))
    else:
        require(value is None or type(value) in (bool, int, float, str), f"Non-scalar at {path}")
        require(not isinstance(value, float) or math.isfinite(value), f"Non-finite value at {path}")
        yield path, value


def json_path(path):
    return "$" + "".join("[]" if isinstance(part, int) else f".{part}" for part in path)


class Inputs:
    def __init__(self):
        self.files = {}
        self.documents = {}

    def read(self, path, role, *, parse=True):
        path = Path(path).resolve()
        key = str(path)
        if key not in self.files:
            raw = path.read_bytes()
            self.files[key] = {"path": key, "sha256": hashlib.sha256(raw).hexdigest(), "roles": []}
            if parse:
                def unique_pairs(pairs):
                    obj = {}
                    for name, value in pairs:
                        require(name not in obj, f"Duplicate JSON key {name!r} in {path}")
                        obj[name] = value
                    return obj
                document = json.loads(raw, object_pairs_hook=unique_pairs)
                require(isinstance(document, dict), f"Expected JSON object in {path}")
                list(scalar_leaves(document))
                self.documents[key] = document
            else:
                self.documents[key] = raw.decode()
        if role not in self.files[key]["roles"]:
            self.files[key]["roles"].append(role)
        return self.documents[key]

    def verify(self):
        verify_inputs(self.files.values())


def verify_inputs(files):
    for item in files:
        actual = hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest()
        require(actual == item["sha256"], f"Input changed during aggregation: {item['path']}")


class Rows:
    """Track metric leaf coverage as cells are constructed and mapped to columns."""

    def __init__(self):
        self.records = []
        self.columns = {}
        self.covered = {}

    def put(self, row, column, value, *, role, path, scope):
        if column in row:
            require(row[column] == value, f"Conflicting values for {column}: {row[column]!r} / {value!r}")
        row[column] = value
        info = self.columns.setdefault(column, {"scopes": set(), "sources": set()})
        info["scopes"].add(scope)
        info["sources"].add((role, json_path(path)))
        if role == "metrics":
            self.covered[path] = value

    def tree(self, row, tree, *, prefix=(), source_prefix=(), role="metrics", scope):
        for path, value in scalar_leaves(tree, skip_lists=(role != "metrics")):
            require(not any(isinstance(p, int) for p in path), f"Unexpanded metric list: {source_prefix + path}")
            self.put(row, "__".join(prefix + path), value,
                     role=role, path=source_prefix + path, scope=scope)

    def derived(self, row, values, scope):
        for column, value in values.items():
            self.put(row, column, value, role="derived", path=(column,), scope=scope)


def manifold_index(items):
    result = {}
    for index, item in enumerate(items):
        mid = item["manifold_id"]
        require(type(mid) is int, "manifold_id must be an integer")
        require(mid not in result, f"Duplicate manifold_id {mid}")
        for name in IDENTIFIERS[:4]:
            require(name in item, f"Missing manifold identity field {name}")
        result[mid] = (index, item)
    return result


def report_directories(root):
    """A run/report directory is terminal; its checkpoint history is not input."""
    if (root / "metrics.json").is_file() or (root / "run_spec.json").is_file():
        yield root
        return
    for child in sorted(root.iterdir()):
        if child.is_dir() and not child.is_symlink() and child.name not in {"archive", "archived", "checkpoints", "snapshots"}:
            yield from report_directories(child)


def aggregate_benchmark(paths_file: Path | None, benchmark_root: Path):
    """Read the whole benchmark, or an optional paths-file subset. No writes."""
    paths_file = Path(paths_file).resolve() if paths_file is not None else None
    benchmark_root = Path(benchmark_root).resolve()
    require(benchmark_root.is_dir(), f"Benchmark directory does not exist: {benchmark_root}")
    inputs, builder = Inputs(), Rows()
    inputs.read(Path(__file__), "aggregator_source", parse=False)
    experiment_path = benchmark_root / "experiment.json"
    dataset_aliases = {}
    if experiment_path.is_file():
        experiment = inputs.read(experiment_path, "benchmark_config")
        for item in experiment["conditions"]:
            ratio = item["noise_ratio"]
            require(ratio not in dataset_aliases, f"Duplicate benchmark condition {ratio}")
            dataset_aliases[ratio] = Path(item["dataset"]).resolve()
    roots = [] if paths_file is not None else [benchmark_root]
    lines = inputs.read(paths_file, "paths_file", parse=False).splitlines() if paths_file is not None else []
    for line in lines:
        value = line.strip().removeprefix("- ").strip()
        if not value:
            continue
        path = Path(value)
        path = (paths_file.parent / path).resolve() if not path.is_absolute() else path.resolve()
        require(path.is_dir(), f"Source directory does not exist: {path}")
        require(path.is_relative_to(benchmark_root), f"Source outside benchmark: {path}")
        if path not in roots:
            roots.append(path)
    require(roots, "The paths file contains no source directories")
    directories = {p for root in roots for p in report_directories(root)}
    reports = sorted(p / "metrics.json" for p in directories if (p / "metrics.json").is_file())
    specs = sorted(p / "run_spec.json" for p in directories if (p / "run_spec.json").is_file())
    skipped = []
    for path in specs:
        if not (path.parent / "metrics.json").exists():
            spec = inputs.read(path, "run_spec")
            training = spec["training"]["arguments"]
            condition = benchmark_root / path.relative_to(benchmark_root).parts[0]
            dataset = inputs.read(condition / "dataset/config.json", "dataset_config")
            skipped.append({"run_id": spec["run_id"], "run_dir": str(path.parent), "reason": "missing_metrics",
                            "model_kind": spec["training"]["model_kind"], "K": training["K"],
                            "noise_ratio": dataset["generator_config"]["noise_ratio"],
                            "surgery_threshold": training.get("surgery_threshold")})
    for root in roots:
        if not any(p.is_relative_to(root) for p in reports + specs):
            skipped.append({"run_dir": str(root), "reason": "no_metrics_reports"})
    require(reports, "No current metrics.json reports in the approved sources")
    sources, catalogs = [], {}

    for source_path in reports:
        report = inputs.read(source_path, "metrics")
        require(report.get("schema_version") == 2, f"Unsupported metrics schema in {source_path}")
        evaluation = report["evaluation"]
        require(evaluation in ("toy_manifold_tiling", "toy_kmeans_initialization_cattell_sweep"),
                f"Unsupported evaluation {evaluation}")
        is_kmeans = evaluation == "toy_kmeans_initialization_cattell_sweep"
        relative = source_path.relative_to(benchmark_root)
        condition = benchmark_root / relative.parts[0]
        dataset_path = condition / "dataset/config.json"
        dataset = inputs.read(dataset_path, "dataset_config")
        generator = dataset["generator_config"]
        noise_ratio = generator["noise_ratio"]
        require(condition.name == f"noise_ratio_{noise_ratio:g}", f"Noise condition mismatch: {source_path}")
        configuration_paths = {"dataset_config": str(dataset_path)}

        def check_dataset(raw_path):
            # Historical assets/ directories are copies, not necessarily symlinks.
            directory = Path(raw_path).resolve()
            allowed = {(condition / "dataset").resolve()}
            if noise_ratio in dataset_aliases:
                allowed.add(dataset_aliases[noise_ratio])
            require(directory in allowed, f"Dataset path outside approved benchmark condition: {directory}")
            config = directory / "config.json"
            require(inputs.read(config, "dataset_config") == dataset,
                    f"Dataset provenance mismatch: {config} versus {dataset_path}")

        check_dataset(report["dataset"]["shard_dir"])
        require(report["dataset"]["layer"] in dataset["layers"], "Dataset layer mismatch")
        if "selected_rows" in report["dataset"]:
            require(report["dataset"]["selected_rows"] <= dataset["num_rows"], "Dataset row count mismatch")
        configurations = {"dataset_config": dataset}
        if is_kmeans:
            model_kind = "kmeans"
            run_id = "kmeans::" + str(relative.parent)
            init_path = Path(report["artifacts"]["centroids_path"]).parent / "config.json"
            initialization = inputs.read(init_path, "initialization_config")
            configurations["initialization"] = initialization
            configuration_paths["initialization"] = str(init_path.resolve())
            require(initialization["K"] == report["K"], "KMeans K mismatch")
            require(initialization["layer"] == report["dataset"]["layer"], "KMeans layer mismatch")
            require(0 < report["q_max"] <= report["pca_capacity"], "KMeans PCA capacity mismatch")
            check_dataset(initialization["source_shard_dir"])
            capacity, seed, split_seed = report["q_max"], initialization.get("seed"), None
        else:
            spec_path, config_path = source_path.parent / "run_spec.json", source_path.parent / "config.json"
            spec, saved_config = inputs.read(spec_path, "run_spec"), inputs.read(config_path, "model_config")
            training = spec["training"]["arguments"]
            model_kind, run_id = spec["training"]["model_kind"], spec["run_id"]
            for key, expected in {"run_id": run_id, "identity_hash": spec["identity_hash"],
                                  "model_kind": model_kind, "K": training["K"],
                                  "q_capacity": training["rank"]}.items():
                require(report[key] == expected, f"Report identity mismatch for {key}: {source_path}")
            require(saved_config["K"] == report["K"] and saved_config["rank"] == report["q_capacity"],
                    f"Saved model configuration mismatch: {source_path}")
            for raw_path in (training["shard_dir"], saved_config["shard_dir"], spec["dataset"]["shard_dir"]):
                check_dataset(raw_path)
            for key in training.keys() & saved_config.keys():
                if key not in ("shard_dir", "out_dir"):
                    require(training[key] == saved_config[key], f"Training/config mismatch for {key}: {source_path}")
            require(report["dataset"]["layer"] == training["layer"] == saved_config["layer"], "Layer mismatch")
            require(spec["dataset"]["layer"] == training["layer"], "Run-spec dataset layer mismatch")
            require(saved_config["d_model"] == dataset["d_model"], "Ambient dimension mismatch")
            if "identity" in spec:
                require(spec["identity"]["dataset"] == spec["dataset"], "Run-spec dataset identity mismatch")
                require(spec["identity"]["model_kind"] == model_kind, "Run-spec model identity mismatch")
                identity_args = dict(spec["identity"]["training_args"])
                # The planner hashes automatic initialization before resolving its run directory.
                if spec.get("initialization") is not None:
                    require(spec["identity"].get("initialization") == spec["initialization"],
                            f"Run-spec initialization identity mismatch: {source_path}")
                    require(identity_args.get("centroids_path") is None,
                            f"Automatic initialization has explicit identity centroids: {source_path}")
                    require(Path(training["centroids_path"]).resolve() == source_path.parent / "initialization/centroids.pt",
                            f"Automatic initialization path mismatch: {source_path}")
                    identity_args["centroids_path"] = training["centroids_path"]
                require(identity_args == {k: v for k, v in training.items() if k != "out_dir"},
                        f"Run-spec training identity mismatch: {source_path}")
            configurations.update(training=training, model_config=saved_config,
                                  evaluation_config=spec.get("evaluation", {}))
            configuration_paths.update(run_spec=str(spec_path), model_config=str(config_path))
            if training.get("centroids_path"):
                init_path = Path(training["centroids_path"]).parent / "config.json"
                initialization = inputs.read(init_path, "initialization_config")
                require(initialization["K"] == report["K"], "Initializer K mismatch")
                check_dataset(initialization["source_shard_dir"])
                configurations["initialization"] = initialization
                configuration_paths["initialization"] = str(init_path.resolve())
            capacity, seed, split_seed = report["q_capacity"], training.get("seed"), training.get("split_seed")

        surgery_threshold = None if is_kmeans else training.get("surgery_threshold")
        manifolds = manifold_index(report["per_manifold"])
        expected_types = Counter(generator["manifold_types"] * generator["manifolds_per_type"])
        require(Counter(item["type_name"] for _, item in manifolds.values()) == expected_types,
                f"Manifold coverage mismatch: {source_path}")
        catalog = {mid: {key: item[key] for key in IDENTIFIERS if key in item} for mid, (_, item) in manifolds.items()}
        if noise_ratio in catalogs:
            require(catalog.keys() == catalogs[noise_ratio].keys(), "Manifold IDs differ across reports")
            for mid, identity in catalog.items():
                previous = catalogs[noise_ratio][mid]
                require(all(identity[k] == previous[k] for k in identity.keys() & previous.keys()),
                        f"Manifold identity mismatch for {mid}: {source_path}")
                previous.update(identity)
        else:
            catalogs[noise_ratio] = catalog

        context = {key: value for key, value in report.items() if key in CONTEXT_KEYS}
        context.pop("cattell", None)
        if "cattell" in report:
            context["cattell"] = {k: v for k, v in report["cattell"].items() if k != "thresholds"}
        global_metrics = {k: v for k, v in report.items() if k not in CONTEXT_KEYS | {"per_manifold", "threshold_sweep"}}
        if is_kmeans:
            advertised = report["cattell"]["thresholds"]
            thresholds = [item["cattell_threshold"] for item in report["threshold_sweep"]]
            require(len(set(thresholds)) == len(thresholds) > 0, "Duplicate or empty KMeans threshold sweep")
            require(len(set(advertised)) == len(advertised) and set(advertised) == set(thresholds),
                    "Declared Cattell thresholds do not match sweep")
            settings = list(enumerate(report["threshold_sweep"]))
        else:
            settings = [(None, None)]

        builder.covered = {}
        first_row = len(builder.records)
        for threshold_index, setting in settings:
            threshold = setting["cattell_threshold"] if setting is not None else None
            evaluation_id = run_id + (f"::cattell={json.dumps(threshold)}" if is_kmeans else "")
            threshold_manifolds = manifold_index(setting["per_manifold"]) if is_kmeans else {}
            if is_kmeans:
                require(threshold_manifolds.keys() == manifolds.keys(), "Threshold manifold IDs do not match base report")
            for mid in [None, *sorted(manifolds)]:
                scope = "overall" if mid is None else "manifold"
                row = {}
                builder.derived(row, {
                    "evaluation_id": evaluation_id, "run_id": run_id, "model_kind": model_kind,
                    "noise_ratio": noise_ratio, "K": report["K"], "q_capacity": capacity,
                    "surgery_threshold": surgery_threshold,
                    "seed": seed, "split_seed": split_seed, "dataset_seed": generator["seed"],
                    "cattell_threshold": threshold, "scope": scope,
                    "source_relative_path": str(relative), "source_path": str(source_path),
                    "source_sha256": inputs.files[str(source_path)]["sha256"],
                }, scope)
                builder.tree(row, context, scope=scope)
                for name, config in configurations.items():
                    role = {"training": "run_spec", "evaluation_config": "run_spec", "initialization": "initialization_config"}.get(name, name)
                    source_prefix = {"training": ("training", "arguments"), "evaluation_config": ("evaluation",)}.get(name, ())
                    builder.tree(row, config, prefix=(name,), source_prefix=source_prefix, role=role, scope=scope)
                if is_kmeans:
                    builder.put(row, "cattell_threshold", threshold, role="metrics", scope=scope,
                                path=("cattell", "thresholds", advertised.index(threshold)))
                    builder.put(row, "cattell_threshold", threshold, role="metrics", scope=scope,
                                path=("threshold_sweep", threshold_index, "cattell_threshold"))
                if mid is None:
                    builder.tree(row, global_metrics, scope=scope)
                    if is_kmeans:
                        builder.tree(row, {k: v for k, v in setting.items() if k not in ("per_manifold", "cattell_threshold")},
                                     source_prefix=("threshold_sweep", threshold_index), scope=scope)
                else:
                    index, item = manifolds[mid]
                    builder.tree(row, item, source_prefix=("per_manifold", index), scope=scope)
                    if is_kmeans:
                        sweep_index, sweep_item = threshold_manifolds[mid]
                        builder.tree(row, sweep_item, source_prefix=("threshold_sweep", threshold_index, "per_manifold", sweep_index), scope=scope)
                builder.records.append(row)
        expected = dict(scalar_leaves(report))
        require(builder.covered == expected,
                f"Scalar metric coverage mismatch in {source_path}: {expected.keys() - builder.covered.keys()}")
        sources.append({"path": str(source_path), "relative_path": str(relative),
                        "sha256": inputs.files[str(source_path)]["sha256"], "run_id": run_id,
                        "model_kind": model_kind, "noise_ratio": noise_ratio,
                        "K": report["K"], "surgery_threshold": surgery_threshold,
                        "configuration_paths": configuration_paths, "settings": len(settings),
                        "rows": len(builder.records) - first_row, "scalar_leaves_verified": len(expected)})

    # Build each Series directly to avoid an intermediate float conversion of nullable integers.
    names = FRONT_COLUMNS + sorted(builder.columns.keys() - set(FRONT_COLUMNS))
    series = {}
    for name in names:
        values = [row.get(name) for row in builder.records]
        types = {type(value) for value in values if value is not None}
        if not types:
            dtype = "string" if name == "type_name" else "Int64" if name in IDENTIFIERS else "Float64"
        elif types == {str}:
            dtype = "string"
        elif types == {bool}:
            dtype = "boolean"
        elif types == {int}:
            dtype = "Int64"
        else:
            require(types <= {int, float}, f"Mixed scalar types in column {name}: {types}")
            dtype = "Float64"
        series[name] = pd.Series(values, dtype=dtype)
    df = pd.DataFrame(series)
    for index, record in enumerate(builder.records):
        for name, original in record.items():
            value = df.at[index, name]
            if hasattr(value, "item"):
                value = value.item()
            require(pd.isna(value) if original is None else value == original,
                    f"Value changed during DataFrame conversion: row {index}, {name}")
    require(not df.duplicated(["evaluation_id", "scope", "manifold_id"]).any(), "Duplicate evaluation row identity")
    require(df.loc[df.scope == "overall", "evaluation_id"].is_unique, "Duplicate evaluation_id")
    df = df.sort_values(["noise_ratio", "model_kind", "K", "surgery_threshold", "evaluation_id", "manifold_id"],
                        na_position="first", kind="stable").reset_index(drop=True)
    columns = {}
    for name in df.columns:
        info = builder.columns.get(name, {"scopes": set(), "sources": set()})
        columns[name] = {"dtype": str(df[name].dtype), "scopes": sorted(info["scopes"]),
                         "sources": [{"role": role, "json_path": path} for role, path in sorted(info["sources"])]}
    manifest = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "paths_file": str(paths_file) if paths_file is not None else None, "benchmark_root": str(benchmark_root),
        "source_roots": [str(p) for p in roots], "sources": sources, "skipped_runs": skipped,
        "coverage": {"reports": len(sources), "evaluation_settings": int((df.scope == "overall").sum()),
                     "rows": len(df), "columns": len(df.columns),
                     "scalar_leaves_verified": sum(item["scalar_leaves_verified"] for item in sources),
                     "reports_by_model_kind": dict(Counter(item["model_kind"] for item in sources))},
        "sweep_values": {name: sorted(df[name].dropna().unique().tolist())
                         for name in ("noise_ratio", "K", "surgery_threshold", "cattell_threshold")},
        "input_files": sorted(inputs.files.values(), key=lambda item: item["path"]),
        "configuration_metadata": {path: doc for path, doc in inputs.documents.items()
                                   if "metrics" not in inputs.files[path]["roles"] and isinstance(doc, dict)},
    }
    inputs.verify()
    return df, manifest, columns


def handoff_readme(manifest):
    arguments = [str(REPO / ".venv/bin/python"), str(Path(__file__).resolve()),
                 "--benchmark-root", manifest["benchmark_root"], "--output-dir", manifest["output_dir"]]
    if manifest["paths_file"] is not None:
        arguments.extend(["--paths-file", manifest["paths_file"]])
    command = shlex.join(arguments)
    output = Path(manifest["output_dir"])
    coverage = manifest["coverage"]
    return f'''# Toy noise benchmark metrics

Snapshot: {coverage['reports']} reports, {coverage['evaluation_settings']} evaluation settings,
{coverage['rows']} rows, {coverage['columns']} columns. {len(manifest['skipped_runs'])} runs/directories
without metric reports were skipped. Their identities and available sweep parameters
are recorded in `manifest.json` and the SQLite `skipped_runs` table.

## Load and query

`metrics.sqlite` contains the `metrics` table and the `overall_metrics` and
`manifold_metrics` views. Sweep columns are indexed. `metadata` stores the input
provenance manifest and column dictionary as JSON. `metrics.parquet` contains the
same metric rows with nullable pandas dtypes and exact integer/float values.

```python
import pandas as pd

df = pd.read_parquet({str(output / "metrics.parquet")!r})
overall = df.query("scope == 'overall'")
manifolds = df.query("scope == 'manifold'")

comparison = overall.query("model_kind == 'hddc' and noise_ratio == 100")[[
    "noise_ratio", "K", "surgery_threshold", "nll__validation", "bic__value",
    "clustering__adjusted_rand_index", "rank__mean_absolute_error",
]]

# Keep all sweep dimensions when preparing a threshold plot.
sweep = manifolds.query("model_kind == 'hddc'")
plot_data = sweep.pivot(index=["noise_ratio", "K", "surgery_threshold"],
                        columns="manifold_id", values="rank__mean_absolute_error")
```

```python
import sqlite3

with sqlite3.connect({str(output / "metrics.sqlite")!r}) as connection:
    comparison = pd.read_sql_query(\"\"\"
        SELECT noise_ratio, K, surgery_threshold, nll__validation,
               rank__mean_absolute_error, tangent_alignment__subspace_overlap__mean
        FROM overall_metrics
        WHERE model_kind = 'hddc'
        ORDER BY noise_ratio, K, surgery_threshold
    \"\"\", connection)
    missing = pd.read_sql_query("SELECT * FROM skipped_runs", connection)
```

## Row and column contract

`evaluation_id` identifies one trained run or one KMeans threshold setting. Each
setting has one `scope='overall'` row and one `scope='manifold'` row per planted
instance. Use `manifold_id` as the instance key; `type_name` is its readable label.
Overall rows have null manifold identifiers. Fields absent from a report remain null.

The training sweep axes are `noise_ratio`, `K`, and `surgery_threshold`.
`surgery_threshold` aliases `training__surgery_threshold`; both are null when the
setting does not use surgery. `cattell_threshold` is the separate post-fit KMeans
rank-selection threshold. The available values are recorded in `manifest.json`.

Metric names follow JSON paths with `__` separators. All scalar summary fields,
including rank histogram counts, are retained. Configuration is repeated under
`training__`, `model_config__`, `initialization__`, `evaluation_config__`, and
`dataset_config__`. Top-level dataset/artifact identifiers are shared context.
Global metrics occur only in overall rows; manifold rows contain the summaries
actually reported for that manifold. NLL, BIC, global clustering scores, and global
counts are not copied into manifold rows. Null means absent, inapplicable, or
explicitly undefined in the source, never an imputed zero.

`columns.json` gives dtypes, scopes, and source JSON paths (`[]` denotes an array
element). `source_path` and `source_sha256` identify each original `metrics.json`.
The manifest preserves complete configuration metadata, including lists.

## Interpretation

- KMeans thresholds reuse the same partition. Select a threshold or deduplicate
  by `run_id` when using threshold-independent KMeans metrics.
- PCA/rank capacities and initialization methods can vary; inspect `q_capacity`,
  `pca_capacity`, `initialization__*`, and `training__centroids_path`.
- Rank definitions and evaluated component populations can differ across model
  kinds. Consult `rank__definition`, `rank__population`, eligibility fields, and
  geometry validity counts before combining results.
- KMeans results are in sample. Model NLL has separate train/validation fields;
  dataset and training columns preserve the split context.
- `bic__value` is the higher-is-better active-BIC score:
  `-standard_bic / n + active_components`. Compare this within the same dataset,
  split, and nominal K because the maximum activity reward changes with K.
  `bic__standard_bic` retains the conventional lower-is-better diagnostic.
- Larger `noise_ratio` means less Gaussian noise.
- The pivot example assumes one run per noise/K/threshold combination. If there
  are multiple seeds or evaluation protocols, filter or include those dimensions.

## Reproduce and audit

```bash
{command}
```

By default, discovery includes all current reports under `--benchmark-root`.
An optional `--paths-file` restricts it to the listed directories. Rerunning picks
up newly available reports. Historical backups and checkpoint contents are excluded.
Source artifacts are read-only inputs. Missing reports are audited; no missing
metrics or parameter combinations are imputed and no model evaluations are launched.

Every scalar metric leaf is checked against the constructed cells, all inputs are
rehashed, and both saved formats are read back with exact value/dtype checks before
publishing the output files. `manifest.json` includes hashes of both saved formats.
'''


def write_sqlite(df, manifest, columns, path):
    """Save a queryable snapshot and verify exact scalar values, including int64."""
    with sqlite3.connect(path) as connection:
        df.to_sql("metrics", connection, index=False)
        connection.executescript("""
            CREATE UNIQUE INDEX overall_identity ON metrics (evaluation_id) WHERE scope = 'overall';
            CREATE UNIQUE INDEX manifold_identity ON metrics (evaluation_id, manifold_id) WHERE scope = 'manifold';
            CREATE INDEX sweep_parameters ON metrics (model_kind, noise_ratio, K, surgery_threshold, scope);
            CREATE VIEW overall_metrics AS SELECT * FROM metrics WHERE scope = 'overall';
            CREATE VIEW manifold_metrics AS SELECT * FROM metrics WHERE scope = 'manifold';
            CREATE TABLE metadata (name TEXT PRIMARY KEY, value_json TEXT NOT NULL);
        """)
        skipped = pd.DataFrame(manifest["skipped_runs"], columns=[
            "run_id", "run_dir", "reason", "model_kind", "noise_ratio", "K", "surgery_threshold",
        ])
        skipped.to_sql("skipped_runs", connection, index=False,
                       dtype={"noise_ratio": "REAL", "K": "INTEGER", "surgery_threshold": "REAL"})
        connection.executemany("INSERT INTO metadata VALUES (?, ?)", [
            (name, json.dumps(value, allow_nan=False))
            for name, value in (("manifest", manifest), ("columns", columns))
        ])
    with sqlite3.connect(path) as connection:
        require(connection.execute("PRAGMA integrity_check").fetchone() == ("ok",), "SQLite integrity check failed")
        records = connection.execute("SELECT * FROM metrics ORDER BY rowid").fetchall()
        # Construct typed Series directly: nullable large integers must never pass through float64.
        restored = pd.DataFrame({name: pd.Series([row[i] for row in records], dtype=df[name].dtype)
                                 for i, name in enumerate(df.columns)})
        pd.testing.assert_frame_equal(df, restored, check_exact=True)


def write_outputs(df, manifest, columns, output_dir: Path):
    output_dir = Path(output_dir).resolve()
    require(not output_dir.is_relative_to(Path(manifest["benchmark_root"])),
            "Output directory must be outside the benchmark source artifacts")
    require(output_dir not in {Path(item["path"]).parent for item in manifest["input_files"]},
            "Output directory must not be an input file's directory")
    manifest = {**manifest, "output_dir": str(output_dir)}
    verify_inputs(manifest["input_files"])
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".toy_metrics_", dir=output_dir.parent) as staging:
        staging = Path(staging)
        df.to_parquet(staging / "metrics.parquet", engine="pyarrow", index=False)
        pd.testing.assert_frame_equal(df, pd.read_parquet(staging / "metrics.parquet"), check_exact=True)
        manifest["parquet_sha256"] = hashlib.sha256((staging / "metrics.parquet").read_bytes()).hexdigest()
        write_sqlite(df, manifest, columns, staging / "metrics.sqlite")
        manifest["sqlite_sha256"] = hashlib.sha256((staging / "metrics.sqlite").read_bytes()).hexdigest()
        manifest["validation"] = {"scalar_coverage": "passed", "dataframe_values": "passed",
                                  "parquet_round_trip": "passed", "sqlite_round_trip": "passed", "input_hashes": "passed"}
        for name, data in (("manifest.json", manifest), ("columns.json", columns)):
            (staging / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
        (staging / "README.md").write_text(handoff_readme(manifest))
        verify_inputs(manifest["input_files"])
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in ("metrics.parquet", "metrics.sqlite", "columns.json", "README.md", "manifest.json"):
            (staging / name).replace(output_dir / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-file", type=Path, help="Optional directory allowlist; defaults to the whole benchmark")
    parser.add_argument("--benchmark-root", type=Path, default=BENCHMARK, help="Benchmark directory containing noise conditions")
    parser.add_argument("--output-dir", type=Path, default=REPO / "outputs/experiments/toy_noise_benchmark_metrics")
    args = parser.parse_args()
    df, manifest, columns = aggregate_benchmark(args.paths_file, args.benchmark_root)
    write_outputs(df, manifest, columns, args.output_dir)
    print(json.dumps({**manifest["coverage"], "skipped_runs": len(manifest["skipped_runs"]),
                      "output_dir": str(args.output_dir.resolve())}, indent=2))


if __name__ == "__main__":
    main()
