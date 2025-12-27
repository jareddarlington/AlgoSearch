from pydantic import BaseModel, Field
from typing import List, Optional


class Algorithm(BaseModel):
    name: str = Field(..., description="Name of algorithm")
    aliases: List[str] = Field(default_factory=list, description="Alternative or abbreviated names")
    description: str = Field(
        ...,
        description="3-5 sentence description optimized for semantic search: lead with problem/technique, include paradigm keywords (greedy, DP, etc.), data structures, problem domains, and use cases. Include complexity information using plain language (linear, quadratic, logarithmic, etc.) when easily describable. Avoid math notation and implementation details.",
    )
    latex: str = Field(..., description="Rewritten algorithm in canonical LaTeX")
    categories: List[str] = Field(
        default_factory=list,
        description="3-5 high-level categories: algorithmic paradigms, data structures, problem domains",
    )


class AlgorithmList(BaseModel):
    algorithms: List[Algorithm] = Field(default_factory=list, description="All algorithms found in the paper")
