from pydantic import BaseModel,Field

class Role(BaseModel):

    role: str = Field(
        description="Short and specific name of the agent role"
    )

    input: str = Field(
        description=(
            "What information or requirements "
            "this agent receives"
        )
    )

    output: str = Field(
        description=(
            "The actual work product this agent "
            "must produce or modify"
        )
    )

    tools: list[str] = Field(
        description=(
            "Exact names of tools this agent needs "
            "to perform its work"
        )
    )

    evaluation_criteria: list[str] = Field(
        description=(
            "Specific, observable criteria that must "
            "be satisfied after execution"
        )
    )

    system_prompt: str = Field(
        description=(
            "Complete operational instructions "
            "for the worker agent"
        )
    )


class ListRoles(BaseModel):

    roles: list[str] = Field(
        description=(
            "List of minimum roles required "
            "to accomplish the task"
        )
    )


class CriterionResult(BaseModel):

    criterion: str = Field(
        description=(
            "The evaluation criterion being checked"
        )
    )

    passed: bool = Field(
        description=(
            "Whether this criterion is satisfied"
        )
    )

    evidence: str = Field(
        description=(
            "Concrete evidence supporting the decision"
        )
    )

    reason: str = Field(
        description=(
            "Explanation of why the criterion "
            "passed or failed"
        )
    )

    fix_instruction: str = Field(
        description=(
            "Specific instruction for fixing "
            "the criterion if it failed"
        )
    )


class EvaluationResult(BaseModel):

    results: list[CriterionResult] = Field(
        description=(
            "Evaluation result for every criterion"
        )
    )

    overall_passed: bool = Field(
        description=(
            "True only when every criterion passes"
        )
    )

