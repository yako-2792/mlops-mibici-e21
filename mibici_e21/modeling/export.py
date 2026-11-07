"""Export a registered model into `models/`, so the API image ships it without MLflow.

The model is requested by alias and frozen by version: the alias is resolved to the version
it points at now, that version is downloaded, and both are recorded in `registry.json`.

    python -m mibici_e21.modeling.export --question p1
    python -m mibici_e21.modeling.export --question p1 --alias challenger
    python -m mibici_e21.modeling.export --question p1 --version 3
"""

import argparse
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil

from loguru import logger
import mlflow
from mlflow import MlflowClient

from mibici_e21.config import MODELS_DIR, REGISTRY_SETTINGS, TRACKING_SETTINGS
from mibici_e21.modeling.base import Question
from mibici_e21.modeling.questions import QUESTIONS

REGISTRY_FILE = "registry.json"


@dataclass(frozen=True)
class ExportedModel:
    """Registry lineage written next to the exported model as `registry.json`.

    - `name`: registered model name.
    - `version`: registered version that was downloaded.
    - `alias`: alias that resolved to `version`, or `None` when the version was pinned.
    - `run_id`: run that logged the model.
    - `exported_at`: UTC timestamp of the export, in ISO 8601.
    """

    name: str
    version: str
    alias: str | None
    run_id: str
    exported_at: str


def export_dir(question: Question) -> Path:
    """Return the directory a question's model is exported to, such as `models/p1-demanda`."""
    return MODELS_DIR / question.registered_model_suffix


def export_model(
    question: Question,
    alias: str | None = REGISTRY_SETTINGS.CHAMPION_ALIAS,
    version: str | None = None,
    target: Path | None = None,
) -> ExportedModel:
    """Download one version of a registered model and record its lineage.

    - `question`: question whose registered model is exported.
    - `alias`: alias to resolve. Ignored when `version` is given.
    - `version`: registered version to export instead of the alias.
    - `target`: destination directory. Defaults to `export_dir(question)`. Its previous
      contents are replaced, so no file of an older version survives.
    """
    client = MlflowClient()
    name = question.registered_model_name
    if version is None:
        model_version = client.get_model_version_by_alias(name, alias)
    else:
        model_version = client.get_model_version(name, version)
        alias = None

    target = target or export_dir(question)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{name}/{model_version.version}", dst_path=str(target)
    )

    exported = ExportedModel(
        name=name,
        version=str(model_version.version),
        alias=alias,
        run_id=model_version.run_id,
        exported_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    (target / REGISTRY_FILE).write_text(json.dumps(asdict(exported), indent=2) + "\n")
    return exported


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a registered model into models/.")
    parser.add_argument("--question", choices=sorted(QUESTIONS), required=True)
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--alias", help="Alias to resolve. Defaults to champion.")
    selector.add_argument("--version", help="Registered version to export instead.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    mlflow.set_tracking_uri(TRACKING_SETTINGS.TRACKING_URI)
    question = QUESTIONS[args.question]
    alias = args.alias or REGISTRY_SETTINGS.CHAMPION_ALIAS
    exported = export_model(question, alias=alias, version=args.version)
    source = f"@{exported.alias}" if exported.alias else "pinned version"
    logger.info(
        f"{exported.name} version {exported.version} ({source}) exported to {export_dir(question)}"
    )


if __name__ == "__main__":
    main()
