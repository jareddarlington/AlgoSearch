from pydantic import BaseModel, Field
from typing import List, Optional


class Variable(BaseModel):
    symbol: str = Field(..., description="Variable or constant symbol, e.g. G, k, ε")
    description: str = Field(
        ...,
        description="One sentence description, including what it represents and its type (graph, matrix, etc.)",
    )


class Algorithm(BaseModel):
    name: str = Field(..., description="Algorithm name or caption-derived name")
    aliases: Optional[List[str]] = Field(None, description="Alternative or shortened names if mentioned")

    description: str = Field(..., description="Three to six sentence natural language description")
    latex: str = Field(..., description="Rewritten algorithm in LaTeX")

    inputs: List[str] = Field(default_factory=list, description="Inputs to the algorithm")
    outputs: List[str] = Field(default_factory=list, description="Outputs produced by the algorithm")
    variables: List[Variable] = Field(default_factory=list, description="Variables/constants used in the algorithm")

    categories: Optional[str] = Field(
        None,
        description="Categories/attributes of the algorithm (problem type/task, data structures used, algorithmic paradigm (greedy, DP, randomized, etc.), etc.)",
    )

    time_complexity: Optional[str] = Field(None, description="Time complexity if stated or clearly implied")
    space_complexity: Optional[str] = Field(None, description="Space complexity if stated or clearly implied")


class AlgorithmList(BaseModel):
    algorithms: List[Algorithm] = Field(
        default_factory=list, description="All well-defined algorithms found in the paper"
    )
