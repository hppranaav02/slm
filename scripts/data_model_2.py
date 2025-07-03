from typing import List, Optional, Union
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
#  Single-label schema (kept for backwards compatibility, if you still need it)
# ---------------------------------------------------------------------------
class ResponseFormat(BaseModel):
    type: str = Field(..., description="json")
    vulnerability: bool = Field(
        ..., description="Whether the response includes a vulnerability"
    )
    vulnerability_type: Optional[str] = Field(
        None, description="Single CWE ID (e.g. 'CWE-178')"
    )
    reasoning: Optional[str] = Field(None, description="Reasoning for the response")
    source: Optional[str] = Field(
        None, description="Line of code where the vulnerability is found"
    )


# ---------------------------------------------------------------------------
#  Multi-label schema for Scenario 2
# ---------------------------------------------------------------------------
class ResponseFormatMulti(BaseModel):
    type: str = Field(..., description="json")
    vulnerability: bool = Field(
        ..., description="Whether the response includes a vulnerability"
    )
    vulnerability_type: Optional[List[str]] = Field(
        None,
        description="List of CWE IDs, e.g. ['CWE-178', 'CWE-863'] "
        "(empty or omitted if secure)",
    )
    reasoning: Optional[str] = Field(None, description="Reasoning for the response")
    source: Optional[Union[str, List[str]]] = Field(
        None, description="Line(s) of code where the vulnerability is found"
    )


# ---------------------------------------------------------------------------
#  Prompts
# ---------------------------------------------------------------------------
instruction_single = (
    "Classify the Go code as Vulnerable or Secure, provide the CWE ID if applicable, "
    "think step by step, and always return reasoning and the line of code as source.\n"
    + ResponseFormat.schema_json(indent=2)
)

instruction_multi = (
    "Classify the Go code as Vulnerable or Secure, provide *all* CWE IDs if applicable "
    "as a JSON array, think step by step, and always return reasoning and source "
    "line(s).\n"
    + ResponseFormatMulti.schema_json(indent=2)
)
