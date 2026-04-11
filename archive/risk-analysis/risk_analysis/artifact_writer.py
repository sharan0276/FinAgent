from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Tuple, Union

from .models import CompanyAnalysisArtifactV1


def write_company_analysis(artifact: CompanyAnalysisArtifactV1, output_path: Union[str, Path]) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact.to_dict(), indent=2), encoding="utf-8")
    return output_path


def write_company_analyses(artifacts: Iterable[Tuple[CompanyAnalysisArtifactV1, Union[str, Path]]]) -> List[Path]:
    written: List[Path] = []
    for artifact, output_path in artifacts:
        written.append(write_company_analysis(artifact, output_path))
    return written
