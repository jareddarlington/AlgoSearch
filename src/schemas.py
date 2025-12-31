from pydantic import BaseModel, Field
from typing import List

# EXTRACTION STRUCTURE


class Algorithm(BaseModel):
    name: str = Field(
        ...,
        description="Descriptive, searchable name that captures what the algorithm does (e.g., 'Greedy Set Cover Approximation' instead of 'Deterministic Greedy').",
    )
    description: str = Field(
        ...,
        description="3-5 sentence description optimized for semantic search: lead with problem/technique, include paradigm keywords (greedy, DP, etc.), data structures, problem domains, and use cases. Include complexity information using plain language (linear, quadratic, logarithmic, etc.) when easily describable. Avoid math notation and implementation details.",
    )
    latex: str = Field(
        ...,
        description="Rewritten algorithm in canonical LaTeX.",
    )
    categories: List[str] = Field(
        ...,
        description="3-5 high-level categories: algorithmic paradigms, data structures, problem domains.",
    )


class AlgorithmList(BaseModel):
    algorithms: List[Algorithm] = Field(
        ...,
        description="All algorithms found in the paper.",
    )


# QUERY STRUCTURE


class Queries(BaseModel):
    queries: List[str] = Field(
        ...,
        description="2-5 rewritten algorithm-search queries.",
    )
