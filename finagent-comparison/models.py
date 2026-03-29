from pydantic import BaseModel, Field
from typing import List


class ComparisonResult(BaseModel):
    overall_resemblance: str = Field(
        description="Overall resemblance to historical risk cases: strong, moderate, weak, or superficial"
    )
    key_similarities: List[str] = Field(
        description="Important similarities between the target company and historical cases"
    )
    key_differences: List[str] = Field(
        description="Important differences that weaken or qualify the comparison"
    )
    risk_signals: List[str] = Field(
        description="Risk signals explicitly mentioned in the input"
    )
    missing_data_or_caveats: List[str] = Field(
        description="Missing information, uncertainty, or caveats"
    )