from dataclasses import dataclass
from pathlib import Path

from thesis.paths import PROJECT_ROOT


@dataclass(frozen=True)
class ProjectSettings:
    project_root: Path = PROJECT_ROOT
    random_seed: int = 42


settings = ProjectSettings()

