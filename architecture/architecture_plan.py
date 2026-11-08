from architecture.project_schema import ProjectPlan
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field, ValidationError
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json
import re


load_dotenv()


# ============================================================
# 1. MODEL
# ============================================================

planner = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


# ============================================================
# 2. FOLLOW-UP SCHEMA
# ============================================================

class FollowUpQuestions(BaseModel):
    follow_up: bool = Field(
        description="Whether follow-up questions are required"
    )

    question: list[str] = Field(
        description=(
            "Essential questions needed to clarify the project. "
            "Return an empty list if clarification is unnecessary."
        )
    )


followup_parser = JsonOutputParser(
    pydantic_object=FollowUpQuestions
)

plan_parser = JsonOutputParser(
    pydantic_object=ProjectPlan
)


# ============================================================
# 3. PROMPT 1: FOLLOW-UP QUESTIONS
# ============================================================

prompt1 = PromptTemplate(
    template="""
You are an experienced software product planner.
Your job is to identify only the important uncertainties that
could change WHAT the user is asking to build.

Do not conduct a technical interview. The development team can
choose ordinary implementation details later.

============================================================
QUESTION-ASKING RULES
============================================================

1. ASK ZERO QUESTIONS FOR CLEAR, SMALL PROJECTS.

   Examples that normally need no follow-up questions:
   - A simple portfolio website.
   - A landing page or homepage.
   - A calculator or unit converter.
   - A basic to-do list.
   - A simple informational website.
   - A small UI/demo using a technology the user already named.

   For these, make reasonable assumptions and create the plan.

2. ASK ONLY ABOUT IMPORTANT PRODUCT-SCOPE AMBIGUITY.

   Ask a question only when two or more plausible interpretations
   would lead to meaningfully different features or architecture.
   For example, "food delivery app" may mean browsing restaurants
   only or a complete ordering workflow. Ask which scope is intended.

3. DO NOT ASK ABOUT ROUTINE TECHNICAL DECISIONS.

   Never ask the user to choose:
   - Frontend/backend frameworks or programming languages.
   - A database, unless the user's requirement makes a data
     persistence choice itself important to the product.
   - Hosting, deployment, Docker, CI/CD, folder structure,
     APIs, libraries, styling tools, or testing frameworks.
   - Responsive design for a normal website; assume it should
     work on common screen sizes.

   If the user has not specified a technology, recommend one
   in the plan. Do not ask them to select a stack.

4. DO NOT INVENT FEATURES FROM A YES/NO ANSWER.

   - A request for a homepage means build the requested homepage;
     it does not automatically mean build the complete application.
   - Authentication does not imply shopping carts, orders,
     payments, admin dashboards, or other workflows.
   - A food/restaurant theme does not automatically require
     ordering, payment, user accounts, or a database.
   - Add those features only if the user requests them or confirms
     that the broader workflow is in scope.

5. KEEP QUESTIONS SHORT AND EASY TO ANSWER.

   - Ask at most TWO questions.
   - Ask one thing per question.
   - Prefer a concise choice between meaningful alternatives.
   - Avoid long compound questions and lists of technical options.
   - Do not ask the user to design the solution for you.

6. DO NOT ASK ABOUT EXISTING PROJECT INTEGRATION.

   This planner creates a NEW, SELF-CONTAINED project. Never ask
   whether an existing backend, API, codebase, database, or project
   must be integrated, or ask the user to provide old project files.

7. IF A DETAIL IS NOT ESSENTIAL, ASSUME A SIMPLE DEFAULT.

   Return follow_up=false and question=[] when the description
   is sufficient to produce a useful first plan.

Project description:
{project_description}

Output format:
{format_instructions}
""",
    input_variables=["project_description"],
    partial_variables={
        "format_instructions": followup_parser.get_format_instructions()
    }
)


# ============================================================
# 4. PROMPT 2: GENERATE / REVISE PROJECT PLAN
# ============================================================

prompt2 = PromptTemplate(
    template="""
You are a senior software architect and project planner.

Your responsibility is to transform the user's software
requirement into a structured and actionable project plan.

============================================================
PROJECT INFORMATION
============================================================

Original project description:
{project_description}

Follow-up questions and answers:
{follow_up_qa}

Previous project plan:
{previous_plan}

User's requested changes:
{change_request}

Previous generation attempt (if this is a correction):
{previous_attempt}

Validation feedback (if this is a correction):
{validation_feedback}


============================================================
PLANNING INSTRUCTIONS
============================================================

1. UNDERSTAND THE REQUIREMENT

Understand the actual problem the user wants to solve.

Do not introduce features that are unrelated to the requirement.

2. CREATE A SELF-CONTAINED PROJECT

This is a NEW project that must work independently.

- Plan the frontend, backend, database, and other components
  that are actually required by the user's requirements.
- Do not assume that an existing backend, API, codebase,
  database, or external project will be supplied.
- Do not create an integration-only plan that depends on
  an unavailable existing project.
- If the project does not need a backend or database,
  do not create one merely for the sake of completeness.
- If mock/static data is sufficient for the requested scope,
  do not introduce a backend and database unnecessarily.
- A database is justified when the confirmed features need data
  to persist across sessions or users; do not add one just because
  the project is related to a business domain.
- Do not infer a complete workflow (such as cart, order placement,
  checkout, and payment) from a request for a homepage or a single
  feature such as login.

3. TECHNOLOGY SELECTION AND ASSUMPTIONS

If the user explicitly specifies a technology, respect that choice.

If the user does not specify a technology:

- Select a sensible and minimal technology stack.
- Prefer simple and commonly used technologies.
- Choose technologies appropriate for the project's actual needs.
- Avoid introducing technologies without a clear reason.

IMPORTANT:

- Your technology choices for unspecified areas are
  PLANNER RECOMMENDATIONS, not user-provided requirements.
- Never claim or imply that the user selected a technology
  when they did not.
- Do not ask the user to decide every implementation detail.
- Do not add unnecessary frameworks, services, or tools.
- Keep the stack consistent across the entire plan.

4. FEATURE SELECTION AND SCOPE CONTROL

Include the features explicitly requested by the user.

You may include only the supporting features genuinely required
for those requested features to work.

Treat the wording of the request carefully:
- "Homepage" or "landing page" means that page, not the entire
  product or its complete business workflow.
- "Website" does not automatically mean a full-stack application.
- A request for login/authentication does not imply carts, orders,
  checkout, payments, admin panels, or unrelated features.
- A food delivery homepage does not automatically include ordering.
- Do not add authentication, payments, persistent storage, or an
  admin dashboard unless requested or essential to the confirmed scope.

If the user confirms a narrow scope, do not expand it. If the
scope is still uncertain and materially changes the product,
use the follow-up answer; otherwise choose the simplest reasonable
interpretation and state it as an assumption in the plan narrative
where the existing schema allows.

Do not turn common optional features into mandatory features
unless they are needed to satisfy the user's requirement.

5. TASK DECOMPOSITION

Break the project into clear, independent and achievable tasks.

Each task must:

- Have one clear responsibility.
- Describe the actual work to be performed.
- Specify only the skills needed for that task.
- Avoid duplicating another task's responsibility.
- Have logical dependencies.

Do not create separate agents or tasks for work that can
reasonably be handled together.

6. TASK IDENTIFICATION

Assign task IDs sequentially:

T1, T2, T3, T4, ...

Every dependency inside a task must refer to an existing
task ID.

Do not create circular dependencies.

7. DEPENDENCIES

Distinguish between:

- Task dependencies: other tasks that must be completed first.
- Project dependencies: libraries, packages, frameworks,
  or external services required by the project.

Do not mix these two types of dependencies.

8. DELIVERABLES

Include only the actual outputs needed to satisfy the request.
Every deliverable must correspond to a requested feature or
an implementation output genuinely needed to complete the project.
Keep deliverables consistent with the features, tasks, and
technical stack.

Do not automatically add items such as:

- Design mockups or wireframes.
- API integration guides.
- Production deployment bundles.
- CI/CD pipelines.
- Docker configuration.
- Extensive documentation.

Include these only when the user requests them or when they
are genuinely necessary for the project's scope.

A concise README or setup instructions may be included when
needed to run and understand the generated project.

8.1 REPEATED OR USER-SUPPLIED ITEMS

When a requirement involves a collection of user-specific items
(such as portfolio projects, products, services, courses, or
articles):

- Do not invent a fixed number of items unless the user specifies
  that number.
- Do not create arbitrary names such as project1, project2, or
  project3 as though those are confirmed user content.
- Describe the deliverable in terms of each item the user supplies
  or confirms. For example, "an individual detail page for each
  project provided by the user".
- Where the requirement calls for navigation, explicitly include
  the links between the collection/overview page and each item's
  corresponding detail page.
- Specify the useful content each detail page should display only
  when supported by the requirement or available information.
  For a project detail page, this may include its description,
  technologies, features, screenshots, and relevant links, where
  those details are available.
- If item details have not been provided, do not fabricate them.
  Treat them as content to be supplied by the user or represented
  with clearly identified editable placeholders.

Do not turn this rule into unnecessary follow-up questions.
The planner should distinguish missing content from missing
product scope and use sensible, clearly stated assumptions.

9. AVOID OVER-ENGINEERING

Prefer the simplest architecture that fully satisfies the
requirements.

Do not add technologies, features, tasks, testing layers,
infrastructure, or deliverables just to make the plan look
more comprehensive.

The plan should be complete, but not unnecessarily large.

10. PLAN REVISION

If a previous plan exists:

- Treat the previous plan as the current source of truth.
- Preserve existing requirements unless the user requests changes.
- Apply the requested changes.
- Do not unnecessarily regenerate unrelated parts.
- Maintain consistency between features, tasks, technologies,
  dependencies and deliverables.
- Remove obsolete tasks, technologies, or deliverables when
  they are no longer needed after a requested change.

11. SCHEMA COMPATIBILITY

Return the project plan using the provided schema.

Do not introduce additional fields.

Do not omit required fields.

Do not change the expected data types.

12. CORRECTING A PREVIOUS INVALID OUTPUT

If validation feedback is provided, use it to correct the previous
attempt. You will also receive the original user requirement,
follow-up answers, previous approved/current plan, and requested
change. Consider ALL of that context together.

- Preserve the user's original requirements and requested change.
- Correct every reported schema or task-dependency problem.
- Return the COMPLETE project plan, not only the changed fields.
- Do not discard valid parts of the previous attempt unnecessarily.
- Do not invent new features while correcting structural errors.
- Follow the exact schema and output format again.


============================================================
FINAL REQUIREMENT
============================================================

Generate a complete internal execution plan.

The plan will be consumed by another agent responsible for
creating the development workforce.

Therefore, task descriptions and responsibilities must be
clear enough for that agent to understand what work needs
to be assigned.

Output format:
{format_instructions}
""",
    input_variables=[
        "project_description",
        "follow_up_qa",
        "previous_plan",
        "change_request",
        "previous_attempt",
        "validation_feedback"
    ],
    partial_variables={
        "format_instructions": plan_parser.get_format_instructions()
    }
)


# ============================================================
# 5. CHAINS
# ============================================================

# The full planning prompt is used only for the first generation.
# Corrections use a smaller prompt so the previous plan, schema,
# and validation feedback do not get repeated in a very large request.

correction_prompt = PromptTemplate(
    template="""
You are correcting an invalid software project plan.
Return the COMPLETE corrected plan as JSON matching the schema.

Original project request:
{project_description}

Clarification answers:
{follow_up_qa}

Requested revision:
{change_request}

Previous generated plan (correct this; preserve valid content):
{previous_attempt}

Validation errors to fix:
{validation_feedback}

Rules:
- Fix every listed validation error.
- Preserve the original requirements and requested revision.
- Do not add unrelated features.
- Keep task IDs unique and dependencies valid.
- Return only the complete JSON object; no Markdown or explanation.

Output format:
{format_instructions}
""",
    input_variables=[
        "project_description",
        "follow_up_qa",
        "change_request",
        "previous_attempt",
        "validation_feedback"
    ],
    partial_variables={
        "format_instructions": plan_parser.get_format_instructions()
    }
)


followup_chain = prompt1 | planner | followup_parser
plan_chain = prompt2 | planner | plan_parser
correction_chain = correction_prompt | planner | plan_parser


def get_follow_up_questions(description):
    """Return no more than two concise, scope-critical questions."""
    raw_result = followup_chain.invoke({
        "project_description": description
    })

    result = FollowUpQuestions.model_validate(raw_result)

    # Enforce the limit in code as well as in the prompt. This prevents
    # an overenthusiastic model response from burdening the user.
    questions = [
        question.strip()
        for question in result.question
        if isinstance(question, str) and question.strip()
    ][:2]

    # An empty question list means there is nothing to ask.
    if not result.follow_up or not questions:
        return []

    return questions


# ============================================================
# 6. PLAN GENERATION
# ============================================================

def generate_plan(
    description,
    qa,
    previous_plan=None,
    change_request="None",
    max_attempts=3
):
    """Generate a plan; use a compact prompt for validation corrections."""
    previous_attempt = "No previous generation attempt."
    validation_feedback = "No validation errors. Generate the plan normally."

    # Compact JSON avoids wasting tokens on indentation and whitespace.
    follow_up_qa = json.dumps(qa, separators=(",", ":"), default=str)

    previous_plan_json = (
        json.dumps(previous_plan, separators=(",", ":"), default=str)
        if previous_plan
        else "No previous plan"
    )

    for attempt in range(1, max_attempts + 1):
        try:
            if attempt == 1:
                # First attempt retains the existing, detailed planning prompt.
                result = plan_chain.invoke({
                    "project_description": description,
                    "follow_up_qa": follow_up_qa,
                    "previous_plan": previous_plan_json,
                    "change_request": change_request,
                    "previous_attempt": "No previous generation attempt.",
                    "validation_feedback": (
                        "No validation errors. Generate the plan normally."
                    )
                })

            else:
                # Do not resend the full planning prompt and previous plan.
                # The previous generated plan already contains the current
                # project details, including the earlier plan during revisions.
                result = correction_chain.invoke({
                    "project_description": description,
                    "follow_up_qa": follow_up_qa,
                    "change_request": change_request,
                    "previous_attempt": previous_attempt,
                    "validation_feedback": validation_feedback
                })

            previous_attempt = json.dumps(
                result,
                separators=(",", ":"),
                default=str
            )

            plan = ProjectPlan.model_validate(result)
            validate_task_dependencies(plan)

            return plan

        except ValidationError as error:
            # Keep only the field path and error message. Pydantic's full
            # error output may repeat the entire invalid input and inflate
            # the next Groq request substantially.
            concise_errors = [
                {
                    "field": ".".join(
                        str(part) for part in item.get("loc", ())
                    ),
                    "message": item.get("msg", "Invalid value"),
                    "type": item.get("type", "validation_error")
                } 
                for item in error.errors()
            ]

            validation_feedback = json.dumps(
                concise_errors,
                separators=(",", ":"),
                default=str
            )

        except OutputParserException as error:
            # Preserve the raw model output for correction, but do not copy
            # it again into the validation feedback.
            previous_attempt = str(
                getattr(error, "llm_output", "") or ""
            ) or "The previous response could not be parsed as JSON."

            validation_feedback = (
                "The response was not valid JSON matching the ProjectPlan "
                "schema. Return one complete valid JSON object."
            )

        except ValueError as error:
            # Includes custom task-dependency validation errors.
            validation_feedback = str(error)

        if attempt < max_attempts:
            print(
                f"Plan validation failed (attempt {attempt}/{max_attempts}). "
                "Retrying with a shorter correction prompt..."
            )

    raise RuntimeError(
        f"Could not generate a valid project plan after {max_attempts} attempts. "
        f"Last validation feedback: {validation_feedback}"
    )


# ============================================================
# 7. TASK DEPENDENCY VALIDATION
# ============================================================

def validate_task_dependencies(plan):
    tasks = plan.tasks

    task_ids = {
        task.task_id for task in tasks
    }

    # Check duplicate task IDs.
    if len(task_ids) != len(tasks):
        raise ValueError(
            "Invalid plan: Duplicate task IDs found."
        )

    # Check whether all dependencies exist and build the graph.
    graph = {}

    for task in tasks:
        for dependency in task.dependencies:
            if dependency not in task_ids:
                raise ValueError(
                    f"Invalid dependency: {task.task_id} "
                    f"depends on non-existing task {dependency}."
                )

            if dependency == task.task_id:
                raise ValueError(
                    f"Invalid dependency: {task.task_id} "
                    "cannot depend on itself."
                )

        graph[task.task_id] = task.dependencies

    # Check for circular dependencies.
    visited = set()
    visiting = set()

    def detect_cycle(task_id):
        if task_id in visiting:
            return True

        if task_id in visited:
            return False

        visiting.add(task_id)

        for dependency in graph[task_id]:
            if detect_cycle(dependency):
                return True

        visiting.remove(task_id)
        visited.add(task_id)

        return False

    for task_id in graph:
        if detect_cycle(task_id):
            raise ValueError(
                "Invalid plan: Circular task dependencies found."
            )


# ============================================================
# 8. PLAN VERSION HISTORY
# ============================================================

def save_plan_version(
    version_history,
    version_order,
    version_number,
    plan,
    based_on=None,
    change_description="Initial plan"
):
    """
    Save an independent snapshot of a plan.

    Old snapshots are never modified when the current plan changes.
    """
    version_history[version_number] = {
        "plan": plan.model_copy(deep=True),
        "based_on": based_on,
        "change_description": change_description
    }

    version_order.append(version_number)


def display_version_history(
    version_history,
    version_order,
    current_version
):
    print("\n========== PLAN VERSION HISTORY ==========")

    for version_number in version_order:
        details = version_history[version_number]

        marker = (
            "  <-- current version"
            if version_number == current_version
            else ""
        )

        print(
            f"v{version_number}: "
            f"{details['change_description']}{marker}"
        )

    print("===========================================")


def find_requested_version(user_input):
    """
    Recognize commands such as:
      restore v2
      go to v2
      revert to version 2
      revert back to v1
    """
    match = re.search(
        r"\b(?:restore|go\s+to|switch\s+to|revert(?:\s+back)?\s+to)"
        r"\s+(?:version\s*)?v?(\d+)\b",
        user_input.strip(),
        re.IGNORECASE
    )

    if match:
        return int(match.group(1))

    return None


# ============================================================
# 9. DISPLAY USER-FACING PROJECT OVERVIEW
# ============================================================

def display_plan(plan):
    print("\n========== PROJECT OVERVIEW ==========")

    print("\nProject:", plan.project_name)
    print("\nObjective:", plan.objective)

    print("\nMain Features:")
    for feature in plan.features:
        print("-", feature)

    print("\nTechnology Stack:")
    for category, technologies in plan.technical_stack.items():
        print(f"{category}: {', '.join(technologies)}")

    print("(Any technology you did not specify is a planner recommendation.)")

    print("\nDeliverables:")
    for item in plan.deliverables:
        print("-", item)

    print("\n======================================")


# ============================================================
# 10. MAIN WORKFLOW
# ============================================================

def main(description=None):
    if description is None:
        description = input(
            "Enter your project description: "
        ).strip()
    else:
        description = description.strip()

    if not description:
        print("Project description cannot be empty.")
        return None

    # --------------------------------------------------------
    # STEP 1: CHECK WHETHER CLARIFICATION IS NEEDED
    # --------------------------------------------------------

    questions = get_follow_up_questions(description)

    qa = {}

    # --------------------------------------------------------
    # STEP 2: ASK FOLLOW-UP QUESTIONS
    # --------------------------------------------------------

    if questions:
        print(
            "\nA couple of quick questions "
            "(only to clarify project scope):\n"
        )

        for question in questions:
            answer = input(
                f"{question}\nYour answer (or skip): "
            ).strip()

            if not answer or answer.lower() == "skip":
                answer = (
                    "Not specified. "
                    "Choose a reasonable default and treat it "
                    "as a planner recommendation."
                )

            qa[question] = answer

    # --------------------------------------------------------
    # STEP 3: GENERATE INITIAL PLAN
    # --------------------------------------------------------

    print("\nGenerating project plan...")

    current_plan = generate_plan(
        description=description,
        qa=qa
    )

    # Version IDs are never reused.
    version_history = {}
    version_order = []
    current_version = 1
    next_version = 2

    save_plan_version(
        version_history=version_history,
        version_order=version_order,
        version_number=current_version,
        plan=current_plan,
        based_on=None,
        change_description="Initial plan"
    )

    # --------------------------------------------------------
    # STEP 4: APPROVAL / REVISION / RESTORE LOOP
    # --------------------------------------------------------

    while True:
        print(
            f"\n{'=' * 20} PROJECT PLAN v{current_version} "
            f"{'=' * 20}"
        )

        display_plan(current_plan)

        print("\nDo you approve this plan?")
        print("Enter 'approve' to continue.")
        print("Describe changes to revise the plan.")
        print("Enter 'history' to view previous versions.")
        print(
            "Enter 'restore vN' to restore a version "
            "(e.g. restore v1)."
        )
        print("Enter 'cancel' to stop planning.")

        user_input = input(
            "\nYour response: "
        ).strip()

        normalized_input = user_input.lower()

        # ----------------------------------------------------
        # APPROVAL
        # ----------------------------------------------------

        if normalized_input in [
            "approve",
            "approved",
            "yes"
        ]:
            print("\nProject plan approved!")
            print("Ready to generate the workforce.")

            # Return the complete structured plan for Master Agent.
            return current_plan

        # ----------------------------------------------------
        # CANCELLATION
        # ----------------------------------------------------

        if normalized_input in [
            "exit",
            "quit",
            "cancel"
        ]:
            print("\nPlanning cancelled.")
            return None

        # ----------------------------------------------------
        # SHOW VERSION HISTORY
        # ----------------------------------------------------

        if normalized_input in [
            "history",
            "versions",
            "show history",
            "show versions"
        ]:
            display_version_history(
                version_history,
                version_order,
                current_version
            )
            continue

        # ----------------------------------------------------
        # RESTORE A PREVIOUS VERSION
        # ----------------------------------------------------

        requested_version = find_requested_version(user_input)

        if requested_version is not None:
            if requested_version not in version_history:
                print(
                    f"Version v{requested_version} does not exist. "
                    "Enter 'history' to see available versions."
                )
                continue

            current_version = requested_version

            current_plan = version_history[
                current_version
            ]["plan"].model_copy(deep=True)

            print(
                f"\nRestored project plan v{current_version}. "
                "Please review it and approve or make further changes."
            )
            continue

        # ----------------------------------------------------
        # EMPTY INPUT
        # ----------------------------------------------------

        if not user_input:
            print(
                "Please approve, describe the changes, "
                "or enter a version-history command."
            )
            continue

        # ----------------------------------------------------
        # REVISE THE CURRENT PLAN
        # ----------------------------------------------------

        print("\nRevising your project plan...")

        previous_version = current_version

        revised_plan = generate_plan(
            description=description,
            qa=qa,
            previous_plan=current_plan.model_dump(),
            change_request=user_input
        )

        # Save the new revision as a new version.
        # Do not overwrite any previous snapshot.
        current_version = next_version
        next_version += 1

        current_plan = revised_plan

        save_plan_version(
            version_history=version_history,
            version_order=version_order,
            version_number=current_version,
            plan=current_plan,
            based_on=previous_version,
            change_description=user_input
        )


# ============================================================
# 11. RUN
# ============================================================

if __name__ == "__main__":
    approved_plan = main()

    if approved_plan is not None:
        # Full structured plan is available here.
        # Pass this to your Master Agent when integrating it.

        project_plan = approved_plan.model_dump()

        # Example:
        # master_agent.invoke({
        #     "project_plan": project_plan
        # })