# Orfeo HPC cluster operations

> **Kind:** Workflow · **Status:** Current · **Use when:** Inspecting, testing,
> debugging, submitting, monitoring, or recovering DALG work on AREA Science
> Park's Orfeo cluster.

This is the mandatory operating policy for this repository on Orfeo. Its goal
is to keep scientific work reproducible while strictly separating login-node
coordination from compute-node execution.

## Cluster identity

- SSH alias: `area`
- connection command: `ssh area`
- username: `malessi`
- Slurm account: `lade` (existing project launchers spell it `LADE`; verify the
  live association before creating or changing a launcher)
- default QoS: `normal`
- user scratch root as seen through the usual SSH environment:
  `/u/dssc/malessi/scratch/`

The same storage may appear under a different mounted path, such as
`/orfeo/cephfs/scratch/dssc/malessi/`, in another environment. Always establish
the current repository, data, environment, and output paths rather than
rewriting paths based on an assumed equivalence. In particular, this
repository's `dalg-cache/` can point to shared or legacy storage owned by a
different user; inspect the symlink and permissions before using it.

## Absolute login-node boundary

Start every command-executing task with this lightweight gate:

```bash
hostname
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "Slurm allocation: $SLURM_JOB_ID"
else
  echo "No Slurm allocation"
fi
```

On an Orfeo login node without a Slurm allocation, never run:

- Python, project CLIs, notebooks, `pytest`, or any test or smoke test;
- scientific computation, model training, evaluation, or data processing;
- package installation, compilation, parallel builds, or environment solving;
- large recursive scans, checksums, archive operations, or bulk copies.

Login nodes may be used only for connecting, Git synchronization, lightweight
text and configuration inspection, small edits or script copies, job
submission, scheduler queries, and log/accounting inspection. Run all
meaningful computation with `sbatch` or inside an interactive `srun`
allocation. If classification is uncertain, use Slurm.

Commands shown elsewhere in this repository under local development or testing
are not authorization to execute them on a login node.

## Project discovery before execution

Before proposing or executing cluster work, determine all of the following:

- repository path, remote, branch, commit, and working-tree status;
- whether the cluster checkout is the intended source checkout;
- intended input paths, output root, and scratch or temporary paths;
- correct Conda environment, required modules, and relevant package versions;
- existing Slurm launchers under `scripts/slurm/`;
- whether the work is inspection, validation, debugging, a smoke test, or
  production;
- CPU, GPU, memory, time, storage, and expected-output requirements;
- available storage and applicable quotas.

These are login-node-safe inspection examples:

```bash
pwd -P
git status --short
git branch --show-current
git rev-parse HEAD
git remote -v
find scripts/slurm -maxdepth 3 -type f -name '*.sh' -print
```

Use bounded searches. Do not recursively inventory large data trees from the
login node. `df` and the cluster's quota command are suitable for filesystem
capacity checks; run `du` or checksums through Slurm when the target is large or
its size is unknown.

Do not assume another checkout matches this one. Compare commits explicitly
before production. Prefer fast-forward-only synchronization and never use
destructive Git commands to force checkouts into agreement:

```bash
git pull --ff-only
```

Production must not run from uncommitted or unidentified source unless the user
explicitly approves that exact state.

## Inspect live Slurm resources

The resource relationships below are established Orfeo knowledge, but cluster
policy, limits, associations, and availability can change. Verify live state
before selecting resources:

```bash
sinfo -o "%P %.10l %.6D %.4c %.8G %.8m %N"
sacctmgr -n show assoc user=malessi format=account,partition,qos
squeue -u malessi -o "%.10i %.24j %.9P %.8T %.10M %b %R"
scontrol show partition
```

| Partition | Intended use | Accelerator |
| --- | --- | --- |
| `THIN` | General CPU workloads | None |
| `EPYC` | AMD CPU workloads | None |
| `GENOA` | AMD CPU workloads | None |
| `FAT` | High-memory CPU workloads | None |
| `GPU` | GPU workloads | NVIDIA V100 |
| `DGX` | GPU workloads | NVIDIA A100 |
| `H100` | GPU workloads | NVIDIA H100 |

Known approximate maximum times are 18 hours on `GPU` and 6 days on `DGX` and
`H100`. Verify the current limit before every submission. Select the smallest
reasonable allocation for the workload, not simply the newest hardware.

## GPU and CPU requests

Orfeo requires a typed GPU request matching the partition:

```text
GPU  -> --gres=gpu:V100:1
DGX  -> --gres=gpu:A100:1
H100 -> --gres=gpu:H100:1
```

Never use an untyped `--gres=gpu:1`. Do not request a GPU for CPU-only work.
Avoid `--nodelist` unless a specific node is genuinely required. CPU resources
requested with `--cpus-per-task` are separate from the GPU count and support
data loading, preprocessing, and CPU-side numerical work.

Request multiple GPUs only when the selected program explicitly implements the
corresponding distributed execution. Allocation alone does not parallelize a
single-GPU program. For this repository, remember that the maintained training
path uses component sharding over mixture components; it is not generic DDP
data parallelism.

Prefer an existing launcher under `scripts/slurm/` and verify all of its
directives and paths rather than copying a generic template blindly.

## Environment and CUDA

Initialize Conda and discover the project-specific environment:

```bash
source /u/dssc/malessi/miniconda3/etc/profile.d/conda.sh
conda env list
conda activate <project-environment>
```

Do not install or solve environments on a login node, and do not reuse an
environment merely because it exists for another project.

Known CUDA modules include `cuda/11.8`, `cuda/12.0`, `cuda/12.1`, `cuda/12.6`,
and `cuda/12.8`. Check `module avail` or `module spider cuda`; load a version
compatible with the framework build. Non-interactive SSH may require a login
shell, for example `ssh area 'bash -lc "module avail"'`.

Inside a GPU allocation, verify both the node and Python environment:

```bash
nvidia-smi
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("PyTorch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA is not available"
print("Device count:", torch.cuda.device_count())
print("Device:", torch.cuda.get_device_name(0))
PY
```

A loaded CUDA module alone does not prove that the environment contains a
CUDA-enabled framework build.

## Batch-job templates

Adapt these only after project discovery. Use explicit repository, environment,
input, output, and log paths.

CPU job:

```bash
#!/bin/bash
#SBATCH --job-name=<job-name>
#SBATCH --partition=THIN
#SBATCH --account=lade
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err

set -euo pipefail

source /u/dssc/malessi/miniconda3/etc/profile.d/conda.sh
conda activate <project-environment>
cd /u/dssc/malessi/scratch/<project-directory>

python <script> <arguments>
```

CPU jobs must not contain a GPU request. Choose `THIN`, `EPYC`, `GENOA`, or
`FAT` according to workload evidence and current policy.

GPU job:

```bash
#!/bin/bash
#SBATCH --job-name=<job-name>
#SBATCH --partition=DGX
#SBATCH --account=lade
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err

set -euo pipefail

source /u/dssc/malessi/miniconda3/etc/profile.d/conda.sh
conda activate <project-environment>
module load <compatible-cuda-module>

nvidia-smi
python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'

cd /u/dssc/malessi/scratch/<project-directory>
python <training-script> <arguments>
```

When changing GPU partitions, change the partition and typed GPU request
together.

## Interactive debugging

Use an interactive compute allocation for short debugging only:

```bash
srun \
  --partition=DGX \
  --account=lade \
  --qos=normal \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=8 \
  --gres=gpu:A100:1 \
  --mem=64G \
  --time=02:00:00 \
  --pty bash
```

Initialize the environment and verify hardware only after the shell is inside
the allocation. Do not leave allocations idle. Use batch jobs for long-running
or production workloads.

## Required validation sequence

For a new or changed workflow:

1. Read repository documentation and existing launchers.
2. Record the cluster checkout's path, remote, branch, commit, and dirty state.
3. Confirm exact inputs, outputs, environment, and storage availability.
4. Perform only lightweight text/configuration inspection on the login node.
5. Submit tests and validation through Slurm.
6. If required, run a GPU environment preflight in an allocation.
7. Run a small end-to-end smoke test.
8. Inspect scheduler state, exit code, stdout, stderr, artifacts, runtime, and
   memory use.
9. Estimate production resources from the smoke-test evidence.
10. Present the exact production plan and obtain explicit authorization.
11. Submit and record every job ID and dependency.
12. Monitor through Slurm and validate final artifacts before declaring
    completion.

Do not skip validation because similar code ran previously; the current commit,
environment, inputs, and configuration must match the validated state.

## Production authorization boundary

Never submit a production job until the user explicitly authorizes that
submission. A request to inspect, debug, prepare, validate, or plan is not
production authorization.

Before requesting authorization, present:

- scientific objective; repository path, branch, commit, and dirty state;
- exact scripts, configurations, command lines, inputs, and output root;
- partition, account, QoS, GPU type, CPUs, memory, wall time, nodes, and tasks;
- array range, concurrency, and dependency graph;
- validation results and expected runtime/storage;
- overwrite protection, expected artifacts, monitoring, and recovery plan.

Label preflights, tests, and smoke tests explicitly. Never reinterpret a
validation run as authorization for production.

## Submission, arrays, and dependencies

Capture IDs immediately:

```bash
job_id=$(sbatch --parsable <job-script>)
echo "$job_id"
```

Record each job's purpose, commit, configuration, inputs, outputs, resources,
and dependencies. Acceptance by `sbatch` is not success.

Use arrays for independent samples, repetitions, or configurations. Define a
deterministic, recorded mapping from array index to task and cap concurrency
unless unrestricted parallelism was deliberately approved:

```bash
#SBATCH --array=0-15%4
task_index="${SLURM_ARRAY_TASK_ID}"
```

For partial failures, preserve successful results and rerun only validated
failed indices, for example `sbatch --array=2,7,11 recovery.sbatch`.

Use `afterok` when all upstream work must succeed, `aftercorr` only for arrays
with matching index mappings, and `afterany` only when downstream code is
designed to handle upstream failures. Use `--kill-on-invalid-dep=yes` where an
invalid dependency must not leave work pending indefinitely. Report every job
ID and the dependency graph.

## Monitoring and completion

```bash
squeue -u malessi -o "%.10i %.24j %.9P %.8T %.10M %b %R"
sacct -j <job-id> \
  --format=JobID,JobName,Partition,State,ExitCode,Elapsed,MaxRSS,AllocTRES%50
scontrol show job <job-id>
```

Inspect both stdout and stderr. An empty queue is not evidence of success. A job
is complete only when its Slurm state is successful, its exit code is zero,
stderr has no unexplained failure, expected artifacts exist, and project
validation passes.

Cancel only a verified explicit job ID after confirming its name, owner, and
dependents and preserving useful diagnostics. Never use broad cancellation
commands without explicit authorization.

## Reproducibility and output protection

Before production, record a machine-readable manifest where practical with:

- repository path and remote, branch, commit, and working-tree status;
- launcher and configuration checksums;
- input paths and practical input checksums (compute large checksums via
  Slurm);
- output root and a snapshot of the executed configuration;
- Conda environment, relevant versions, and loaded modules;
- partition and hardware; CPUs, GPUs, memory, time, array mapping and cap;
- random seeds, submission time, job IDs, and dependency relationships.

Treat existing data, models, and results as read-only unless modification is
explicitly requested. Never overwrite an existing production result directory.
Use a new run directory identified by date, experiment/configuration, commit,
or job ID. Programs should reject an existing output path unless deliberate
resumption is supported and requested.

## Failure and recovery

A failed or interrupted run remains incomplete. Do not relabel partial outputs
as completed results.

After a failure:

1. inspect `sacct`, stdout, and stderr for the exact job or array indices;
2. classify the cause as code, configuration, environment, input, resources,
   or infrastructure;
3. preserve failure evidence and valid upstream artifacts;
4. determine whether partial outputs are scientifically safe to reuse;
5. validate a fix with a targeted Slurm test or smoke run;
6. propose a bounded recovery and rerun only necessary work.

Adjust memory or time requests using accounting evidence, not guesswork. Do not
delete or overwrite existing files to make recovery easier without explicit
authorization.
