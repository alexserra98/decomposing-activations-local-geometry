"""Numerical and CLI checks for full soft empirical covariance spectra."""

import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import pytest
import torch

from dalg.analysis.empirical_spectra import compute_empirical_spectra, _spectra_from_statistics
from dalg.data.shard_activations import stratified_split
from dalg.models.mfa import MFA, save_mfa
from dalg.models.adaptive_q.mfa_hddc import MFA_HDDC, save_mfa_hddc

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/temporary/compute_empirical_spectra.py"
export_spectra = runpy.run_path(str(SCRIPT))["export_spectra"]


def direct_statistics(model, x):
    with torch.no_grad():
        W = model._W()
        fitted = W @ W.transpose(-1, -2) + torch.diag_embed(model._psi())
        normal = torch.distributions.MultivariateNormal(model.mu, covariance_matrix=fitted)
        joint = normal.log_prob(x.double()[:, None]) + model.pi_logits.log_softmax(0)
        r = joint.softmax(-1)
        counts = r.sum(0)
        means = r.T @ x.double() / counts[:, None]
        centered = x.double()[None] - means[:, None]
        covariance = (centered * r.T[:, :, None]).transpose(1, 2) @ centered / counts[:, None, None]
    return counts, means, covariance


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("per_component", [False, True])
def test_mfa_cached_likelihood_preserves_dtype(dtype, per_component):
    torch.manual_seed(12)
    model = MFA(torch.randn(3, 4, dtype=dtype), rank=2, psi_per_component=per_component)
    x = torch.randn(9, 4, dtype=dtype)
    with torch.no_grad(), model.inference_cache():
        W = model._W()
        covariance = W @ W.transpose(-1, -2) + torch.diag_embed(model._psi())
        expected = torch.distributions.MultivariateNormal(model.mu, covariance_matrix=covariance).log_prob(x[:, None])
        actual = model.log_prob_components(x)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("family", ["mfa", "hddc", "em"])
def test_streamed_matches_direct_and_batch_sizes(family):
    torch.manual_seed(7)
    mu = torch.randn(3, 4, dtype=torch.float64) + 2
    model = MFA(mu, rank=2) if family == "mfa" else MFA_HDDC(
        mu, rank=2, isotropic_psi=family == "hddc", shared_b=family == "em",
    )
    x = torch.randn(37, 4, dtype=torch.float64)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    counts, means, covariance = direct_statistics(model, x)
    results = [compute_empirical_spectra(model, x.split(batch), eig_batch_size=eig_batch)
               for batch, eig_batch in [(7, 1), (37, 3)]]
    for result in results:
        values = result["eigenvalues"]
        torch.testing.assert_close(result["effective_counts"], counts)
        torch.testing.assert_close(result["empirical_means"], means)
        torch.testing.assert_close(values, torch.linalg.eigvalsh(covariance).flip(-1))
        torch.testing.assert_close(values.sum(-1), covariance.diagonal(dim1=-2, dim2=-1).sum(-1))
        assert result["valid"].all() and result["n_activations"] == len(x)
        assert (values[:, :-1] >= values[:, 1:]).all()
        assert values.dtype == torch.float64 and values.device.type == "cpu"
    assert not torch.allclose(means, mu)  # The checkpoint means must not be the covariance centers.
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key])


def test_zero_mass_components_keep_ids():
    model = MFA_HDDC(torch.zeros(3, 4, dtype=torch.float64), rank=2, shared_b=True)
    with torch.no_grad():
        model.pi_logits[1] = -torch.inf
    result = compute_empirical_spectra(model, [torch.randn(10, 4)])
    assert result["valid"].tolist() == [True, False, True]
    assert result["effective_counts"][1] == 0
    assert result["eigenvalues"][1].isnan().all()
    assert result["empirical_means"][1].isnan().all()


def test_negative_roundoff_and_invalid_statistics():
    mu = torch.zeros(1, 2, dtype=torch.float64)
    counts = torch.ones(1, dtype=torch.float64)
    scatter = torch.diag_embed(torch.tensor([[1., -1e-15]], dtype=torch.float64))
    result = _spectra_from_statistics(mu, counts, mu, scatter, eig_batch_size=1)
    assert result["eigenvalues"].tolist() == [[1., 0.]]
    for bad in [-0.1, float("nan"), float("inf")]:
        scatter[0, 1, 1] = bad
        with pytest.raises(ValueError):
            _spectra_from_statistics(mu, counts, mu, scatter, eig_batch_size=1)


def test_empty_and_nonfinite_streams():
    model = MFA(torch.zeros(2, 3), rank=1)
    with pytest.raises(ValueError, match="non-empty"):
        compute_empirical_spectra(model, [])
    with pytest.raises(ValueError, match="finite activation"):
        compute_empirical_spectra(model, [torch.full((2, 3), float("nan"))])


@pytest.fixture
def run_files(tmp_path):
    shards = tmp_path / "activations"
    (shards / "layer00").mkdir(parents=True)
    (shards / "meta").mkdir()
    extraction = {"window": 3, "d_model": 4, "drop_prefix": 1}
    (shards / "config.json").write_text(json.dumps(extraction))
    rows = [{"subset": str(i % 2)} for i in range(12)]
    (shards / "meta/shard_00000.json").write_text(json.dumps({
        "row_indices": list(range(12)), "rows": rows,
    }))
    torch.manual_seed(8)
    x = torch.randn(12, 3, 4)
    positions, validation = stratified_split(rows, val_frac=0.25, seed=9)
    x[validation] += 100  # Detect accidental inclusion of validation or prefix tokens.
    x[:, 0] = -100
    torch.save(x, shards / "layer00/shard_00000.pt")
    cfg = {"K": 2, "rank": 2, **extraction, "shard_dir": str(shards),
           "layer": 0, "val_frac": 0.25, "split_seed": 9}
    config = tmp_path / "config.json"
    config.write_text(json.dumps(cfg))
    return tmp_path, config, cfg, x[positions, 1:].reshape(-1, 4)


@pytest.mark.parametrize("family", ["mfa", "hddc", "em"])
def test_cli_full_training_split_and_roundtrip(run_files, family):
    path, config, cfg, train = run_files
    mu = torch.randn(2, 4)
    if family == "mfa":
        model = MFA(mu, rank=2)
        save = save_mfa
    else:
        model = MFA_HDDC(mu, rank=2, shared_b=family == "em", isotropic_psi=family == "hddc")
        save = save_mfa_hddc
        cfg.update(model="MFA_HDDC", fit_method="em" if family == "em" else "adam")
        config.write_text(json.dumps(cfg))
    checkpoint = path / "mfa_model.pt"
    save(model, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    output = path / "results/spectra.pt"
    command = [sys.executable, str(SCRIPT), "--checkpoint", str(checkpoint),
               "--output", str(output), "--batch-size", "3", "--eig-batch-size", "1"]
    if family == "em":
        # A relocated checkpoint uses the original run config explicitly.
        relocated = path / "snapshot"
        relocated.mkdir()
        checkpoint = checkpoint.rename(relocated / "mfa_model.pt")
        command[3] = str(checkpoint)
        command += ["--config", str(config)]
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), OMP_NUM_THREADS="1")
    subprocess.run(command, cwd=path, env=env, check=True, capture_output=True, text=True)
    result = torch.load(output, weights_only=True)
    counts, means, covariance = direct_statistics(model.double(), train)
    torch.testing.assert_close(result["eigenvalues"], torch.linalg.eigvalsh(covariance).flip(-1))
    torch.testing.assert_close(result["effective_counts"], counts)
    torch.testing.assert_close(result["empirical_means"], means)
    assert result["n_activations"] == len(train)
    assert result["source"]["split"] == "train"
    assert result["source"]["config"] == str(config)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
    output_digest = hashlib.sha256(output.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        export_spectra(checkpoint, output, config=config)
    assert hashlib.sha256(output.read_bytes()).hexdigest() == output_digest


def test_resume_checkpoint_rejected(run_files):
    path, _, _, _ = run_files
    checkpoint = path / "checkpoint.pt"
    torch.save({"model": {}, "optimizer": {}}, checkpoint)
    with pytest.raises(ValueError, match="model export"):
        export_spectra(checkpoint, path / "spectra.pt")
    assert not (path / "spectra.pt").exists()
