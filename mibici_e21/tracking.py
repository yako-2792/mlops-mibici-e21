"""MLflow setup and lineage helpers shared by ingestion, training, and registration.

Nothing runs on import. Entry points call `configure_tracking` once before they log.
"""

from pathlib import Path
import shutil

# subprocess only runs git, with an absolute path and arguments fixed in this module.
import subprocess  # nosec B404

import mlflow
from mlflow import MlflowClient
from mlflow.environment_variables import MLFLOW_RUN_ID

from mibici_e21.config import PROJ_ROOT, TRACKING_SETTINGS


def configure_tracking() -> str:
    """Point MLflow at the tracking server and select the project experiment.

    Under `mlflow run`, the launched run must already live in `EXPERIMENT_NAME`, because
    `mlflow.start_run` resumes it. Raises `RuntimeError` when it does not, which happens
    when `MLFLOW_EXPERIMENT_NAME` was not exported before `mlflow run`.

    Returns the experiment id.
    """
    mlflow.set_tracking_uri(TRACKING_SETTINGS.TRACKING_URI)
    expected = TRACKING_SETTINGS.EXPERIMENT_NAME

    project_run_id = MLFLOW_RUN_ID.get()
    if project_run_id:
        client = MlflowClient()
        experiment_id = client.get_run(project_run_id).info.experiment_id
        actual = client.get_experiment(experiment_id).name
        if actual != expected:
            raise RuntimeError(
                f"mlflow run created this run in experiment '{actual}', not '{expected}'. "
                f"Run `export MLFLOW_EXPERIMENT_NAME={expected}` before `mlflow run`."
            )

    return mlflow.set_experiment(expected).experiment_id


def git_lineage() -> dict[str, str]:
    """Return the `git_commit` and `git_dirty` tags of the working tree.

    `git_dirty` is `true` when there are uncommitted changes, so a run that cannot be
    reproduced from its commit is visible in the UI.
    """
    commit = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    return {"git_commit": commit, "git_dirty": str(dirty).lower()}


def base_tags(stage_of_work: str) -> dict[str, str]:
    """Return the tags every run carries: team, stage of work, and git lineage.

    - `stage_of_work`: `ingest`, `baseline`, `search`, or `candidate`.
    """
    return {"team": TRACKING_SETTINGS.TEAM, "stage_of_work": stage_of_work, **git_lineage()}


def log_data_manifest(manifest_path: Path) -> None:
    """Log the data manifest written by the ingestion under the `data/` artifact folder."""
    mlflow.log_artifact(str(manifest_path), artifact_path="data")


def _git(*args: str) -> str:
    """Run a git command at the project root and return its stripped output.

    git is resolved to an absolute path, so a different `git` earlier on `PATH` cannot run
    in its place. Raises `RuntimeError` when git is not installed.
    """
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is required to record the commit of every run.")
    command = [git, *args]
    # No shell, an absolute executable, and arguments that never come from user input.
    output = subprocess.check_output(command, cwd=PROJ_ROOT, text=True)  # nosec B603
    return output.strip()
