from pydantic import BaseModel, Field
from typing import Optional


class Task(BaseModel):
    task_id: str = Field(
        description="Unique identifier for the task, such as T1, T2"
    )

    title: str = Field(
        description="Short name of the task"
    )

    description: str = Field(
        description="Detailed explanation of what needs to be done"
    )

    required_skills: list[str] = Field(
        description="Skills required to complete this task"
    )

    dependencies: list[str] = Field(
        default_factory=list,
        description="Task IDs that must be completed before this task"
    )


class ProjectPlan(BaseModel):
    project_name: str = Field(
        description="Name of the software project"
    )

    objective: str = Field(
        description="Main purpose and goal of the project"
    )

    features: list[str] = Field(
        description="List of features the final product must contain"
    )

    technical_stack: dict[str, list[str]] = Field(
        description="Technologies grouped by category, such as frontend, backend, and database"
    )

    tasks: list[Task] = Field(
        description="List of development tasks required to complete the project"
    )

    dependencies: list[str] = Field(
        description="External libraries, packages, or services required"
    )

    deliverables: list[str] = Field(
        description="Expected final outputs of the project"
    )



