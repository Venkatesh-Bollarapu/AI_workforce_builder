from typing import TypedDict, Annotated
import os
import json
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from groq import RateLimitError

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import (
    HumanMessage,
    SystemMessage
)

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from tools import tools
from architecture.architecture_plan import main as planner_main

from dotenv import load_dotenv


load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

MAX_REPAIR_ATTEMPTS = 3
MAX_GENERATION_ATTEMPTS = 3


# Rate-limit retries are separate from JSON/schema validation retries.
MAX_RATE_LIMIT_RETRIES = 5
RATE_LIMIT_FALLBACK_DELAY = 2
MAX_RATE_LIMIT_DELAY = 30


def get_rate_limit_wait_seconds(exc, attempt):
    """Use Groq's retry/reset headers when available; otherwise back off."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {}) or {}

    retry_after = headers.get("retry-after") or headers.get("Retry-After")
    if retry_after:
        try:
            return max(0.1, min(float(retry_after), MAX_RATE_LIMIT_DELAY))
        except (TypeError, ValueError):
            duration_match = re.fullmatch(
                r"\s*(\d+(?:\.\d+)?)\s*(ms|s|m)\s*",
                str(retry_after),
                re.I,
            )
            if duration_match:
                value = float(duration_match.group(1))
                unit = duration_match.group(2).lower()
                seconds = value / 1000 if unit == "ms" else value * 60 if unit == "m" else value
                return max(0.1, min(seconds, MAX_RATE_LIMIT_DELAY))

            try:
                reset_at = parsedate_to_datetime(retry_after)
                if reset_at.tzinfo is None:
                    reset_at = reset_at.replace(tzinfo=timezone.utc)
                seconds = (reset_at - datetime.now(timezone.utc)).total_seconds()
                return max(0.1, min(seconds, MAX_RATE_LIMIT_DELAY))
            except (TypeError, ValueError, OverflowError):
                pass

    reset_tokens = headers.get("x-ratelimit-reset-tokens")
    if reset_tokens:
        match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(ms|s|m)?\s*", str(reset_tokens), re.I)
        if match:
            value = float(match.group(1))
            unit = (match.group(2) or "s").lower()
            seconds = value / 1000 if unit == "ms" else value * 60 if unit == "m" else value
            return max(0.1, min(seconds, MAX_RATE_LIMIT_DELAY))

    # Fallback: 2, 4, 8, 16... seconds, capped at 30 seconds.
    return min(RATE_LIMIT_FALLBACK_DELAY * (2 ** (attempt - 1)), MAX_RATE_LIMIT_DELAY)


def invoke_with_rate_limit_retry(llm, messages, *, label="LLM request",
                                 max_retries=MAX_RATE_LIMIT_RETRIES):
    """Retry only Groq rate-limit errors; let other failures propagate."""
    for attempt in range(1, max_retries + 2):
        try:
            return llm.invoke(messages)
        except RateLimitError as exc:
            if attempt > max_retries:
                raise

            wait_seconds = get_rate_limit_wait_seconds(exc, attempt)
            print(
                f"\\n[{label}] Groq rate limit reached "
                f"(retry {attempt}/{max_retries}). "
                f"Waiting {wait_seconds:.2f} seconds...",
                flush=True,
            )
            time.sleep(wait_seconds)



# ============================================================
# 1. SCHEMAS
# ============================================================


class Role(BaseModel):

    model_config = ConfigDict(extra="forbid")

    input: str = Field(
        description="The exact requirements and responsibilities assigned to this worker"
    )

    output: str = Field(
        description=(
            "Concrete work products this worker must create or modify, such as files, "
            "code, configuration, APIs, tests, or other deliverables."
        )
    )

    tools: list[str] = Field(
        description=(
            "Exact names of tools this worker needs. Use only names from AVAILABLE TOOLS."
        )
    )

    evaluation_criteria: list[str] = Field(
        description=(
            "Specific, observable, independently verifiable criteria for this worker's work."
        )
    )


class ListRoles(BaseModel):

    model_config = ConfigDict(extra="forbid")

    roles: list[str] = Field(
        description="List of minimum roles required to accomplish the task"
    )


class CriterionResult(BaseModel):

    model_config = ConfigDict(extra="forbid")

    criterion: str = Field(
        description=(
            "The EXACT evaluation criterion being checked. "
            "It must match the provided criterion exactly."
        )
    )

    passed: bool = Field(
        description="Whether this criterion is satisfied"
    )

    evidence: str = Field(
        description="Concrete evidence used to evaluate the criterion"
    )

    reason: str = Field(
        description="Explanation of why the criterion passed or failed"
    )

    fix_instruction: str = Field(
        description=(
            "If the criterion failed, provide a concrete instruction "
            "describing what the worker should inspect or repair. "
            "If passed, return an empty string."
        )
    )


class EvaluationResult(BaseModel):

    model_config = ConfigDict(extra="forbid")

    results: list[CriterionResult] = Field(
        description="Exactly one result for every evaluation criterion"
    )

    overall_passed: bool = Field(
        description=(
            "This field is informational only. The application will "
            "calculate the final pass/fail decision itself."
        )
    )


# ============================================================
# 2. MODELS
# ============================================================


model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=4096
)


agent_model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=4096
)


# ============================================================
# 3. TOOL REGISTRY
# ============================================================


available_tools = tools


TOOL_REGISTRY = {
    tool.name: tool
    for tool in available_tools
}


def get_tool_descriptions():

    return [
        {
            "name": tool.name,
            "description": tool.description,
        }
        for tool in available_tools
    ]


# ============================================================
# 4. PROJECT PLAN HELPERS + REUSABLE GENERATION VALIDATION
# ============================================================


def normalize_project_plan(project_plan):
    """Accept a Pydantic ProjectPlan or dictionary and return a dict."""
    if hasattr(project_plan, "model_dump"):
        project_plan = project_plan.model_dump()
    if not isinstance(project_plan, dict):
        raise TypeError("The approved project plan must be a dict or Pydantic model.")
    return project_plan


def format_project_plan(project_plan):
    return json.dumps(normalize_project_plan(project_plan), indent=2, ensure_ascii=False, default=str)


def response_content_to_text(response):
    """Convert a LangChain model response content value into text for JSON parsing."""
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
            else:
                text_parts.append(str(part))
        return "\n".join(text_parts)
    return str(content)


def generate_validated_output(
    prompt,
    input_values,
    schema,
    *,
    validator=None,
    max_attempts=MAX_GENERATION_ATTEMPTS,
    label="LLM output",
):
    """Generate JSON, parse it, validate its schema and optional business rules.

    Model/API invocation errors are allowed to propagate. Only output/validation
    failures trigger a corrective generation attempt.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1.")

    parser = JsonOutputParser(pydantic_object=schema)
    prompt_values = dict(input_values)
    prompt_values["format_instructions"] = parser.get_format_instructions()
    base_prompt = prompt.format(**prompt_values)
    correction_feedback = None
    last_error = None

    for attempt in range(1, max_attempts + 1):
        messages = [HumanMessage(content=base_prompt)]

        if correction_feedback:
            messages.append(HumanMessage(content=correction_feedback))

        print(f"\n[{label}] Generation attempt {attempt}/{max_attempts}")

        # Keep model/API errors separate from schema errors: do not mislabel
        # an outage, authentication failure, or rate limit as invalid JSON.
        response = invoke_with_rate_limit_retry(
            model, messages, label=label
        )
        raw_output = response_content_to_text(response)

        try:
            parsed_output = parser.parse(raw_output)
            validated_output = schema.model_validate(parsed_output)

            if validator is not None:
                validator(validated_output)

            print(f"[{label}] Output passed validation.")
            return validated_output

        except (OutputParserException, ValidationError, ValueError, TypeError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            print(f"[{label}] Validation failed: {last_error}")

            if attempt == max_attempts:
                break

            correction_feedback = f"""Your previous response did not satisfy the required output format or validation rules.

VALIDATION ERROR:
{last_error}

YOUR PREVIOUS OUTPUT:
{raw_output}

Correct the previous output. Preserve information that was already valid, fix every reported issue,
and return ONLY the complete corrected JSON object. Do not return explanations or Markdown fences.
Follow the original task and schema instructions exactly."""

    raise RuntimeError(
        f"{label} failed validation after {max_attempts} attempts. "
        f"Last validation error: {last_error}"
    )


def validate_role_list(role_list):
    """Validate role-list quality beyond basic Pydantic field types."""
    if not role_list.roles:
        raise ValueError("The roles list cannot be empty for a non-empty approved project plan.")

    cleaned = [name.strip() for name in role_list.roles]
    if any(not name for name in cleaned):
        raise ValueError("Role names cannot be empty or whitespace-only.")

    if len(cleaned) != len(set(name.casefold() for name in cleaned)):
        raise ValueError("The roles list contains duplicate role names.")


def roles_generator(project_plan, available_tools):
    prompt = PromptTemplate(
        template="""
You are the workforce architect of a software development agency.

Your task is to decide the MINIMUM distinct developer roles required to
complete the APPROVED PROJECT PLAN below.

APPROVED PROJECT PLAN (SOURCE OF TRUTH):
{project_plan}

AVAILABLE TOOLS:
{available_tools}

Rules:
1. Derive responsibilities only from the approved plan.
2. Create only roles that are genuinely necessary.
3. Prefer fewer capable roles over unnecessary specialization.
4. Every role must have a distinct responsibility.
5. Cover the planned tasks and deliverables; do not invent features.
6. Do not use the original user description as an alternative source of requirements.
7. Return role names only, with no explanations.

{format_instructions}
""",
        input_variables=["project_plan", "available_tools", "format_instructions"],
    )

    result = generate_validated_output(
        prompt,
        {
            "project_plan": format_project_plan(project_plan),
            "available_tools": json.dumps(available_tools, indent=2),
        },
        ListRoles,
        validator=validate_role_list,
        label="Workforce role selection",
    )

    # Return a plain dictionary to preserve the interface used by full_community.
    return result.model_dump()



# 5. AGENT SPECIFICATION GENERATOR
# ============================================================


def validate_agent_stack(stack):
    """Validate tool references and required content in a worker specification."""
    for field_name in ("input", "output"):
        value = getattr(stack, field_name)
        if not value.strip():
            raise ValueError(f"The '{field_name}' field cannot be empty.")

    if not stack.evaluation_criteria:
        raise ValueError("At least one evaluation criterion is required.")

    if any(not criterion.strip() for criterion in stack.evaluation_criteria):
        raise ValueError("Evaluation criteria cannot contain empty strings.")

    if len(stack.tools) != len(set(stack.tools)):
        raise ValueError("The tools list contains duplicate tool names.")

    unknown_tools = [name for name in stack.tools if name not in TOOL_REGISTRY]
    if unknown_tools:
        available_names = sorted(TOOL_REGISTRY.keys())
        raise ValueError(
            f"Unknown tool name(s): {unknown_tools}. "
            f"Use only these exact available tool names: {available_names}"
        )


def agent_stack_generator(role, project_plan, available_tools):
    prompt = PromptTemplate(
        template="""
You are an expert AI workforce architect designing ONE worker agent.

APPROVED PROJECT PLAN (SOURCE OF TRUTH):
{project_plan}

AGENT ROLE:
{role}

AVAILABLE TOOLS:
{available_tools}

Create a practical specification for this role with exactly these fields:
- input
- output
- tools
- evaluation_criteria

RULES:
1. The agent must perform real work and produce the artifacts required by its responsibility.
2. The input field must describe exactly which parts of the approved plan this agent owns.
3. The output field must describe concrete deliverables, not merely a summary.
4. Do not assign the entire project to every agent.
5. Do not invent features, technologies, requirements, or deliverables absent from the plan.
6. Keep the agent's responsibility distinct from other roles.
7. Select only exact tool names from AVAILABLE TOOLS. Never invent tool names.
8. Prefer the minimum tools needed to complete and verify the work. An empty tools list is allowed if no tool is needed.
9. Create specific, observable, independently verifiable evaluation criteria for THIS agent only.
10. Do not include criteria belonging exclusively to another role.
11. Describe work in future/required-work language because the worker has not executed yet.
12. Preserve consistency with the approved plan, including its technology stack and deliverables.
13. Do not generate a role field; the role is supplied by the application.
14. Do not generate a system_prompt; the application constructs common worker instructions in Python.

{format_instructions}
""",
        input_variables=["project_plan", "role", "available_tools", "format_instructions"],
    )

    validated_stack = generate_validated_output(
        prompt,
        {
            "project_plan": format_project_plan(project_plan),
            "role": role,
            "available_tools": json.dumps(available_tools, indent=2),
        },
        Role,
        validator=validate_agent_stack,
        label=f"{role} specification",
    )

    # The role is trusted input from roles_generator, not something to ask the
    # second LLM call to repeat. Convert the validated model to the dict shape
    # expected by agent_builder and the evaluator.
    return {
        "role": role,
        **validated_stack.model_dump(),
    }



# 6. RESOLVE TOOLS
# ============================================================


def resolve_tools(tool_names):
    """Resolve validated tool names to actual LangChain tool objects."""
    resolved = []
    unknown_tools = []

    for name in tool_names:
        tool = TOOL_REGISTRY.get(name)
        if tool is None:
            unknown_tools.append(name)
        else:
            resolved.append(tool)

    if unknown_tools:
        raise ValueError(f"Cannot build worker. Unknown tool names: {unknown_tools}")

    return resolved


# ============================================================
# 7. AGENT STATE
# ============================================================


class AgentState(TypedDict):

    messages: Annotated[
        list,
        add_messages
    ]


# ============================================================
# 8. BUILD ACTUAL AGENT
# ============================================================


def agent_builder(agent_stack):

    generated_prompt = f"""You are the {agent_stack['role']}.

YOUR ASSIGNED RESPONSIBILITIES:
{agent_stack['input']}

EXPECTED OUTPUT:
{agent_stack['output']}

AVAILABLE TOOLS ASSIGNED TO YOU:
{json.dumps(agent_stack.get('tools', []), indent=2)}

EVALUATION CRITERIA:
{json.dumps(agent_stack.get('evaluation_criteria', []), indent=2)}

Perform only the responsibilities assigned to this role. Use the assigned tools
when needed, inspect existing files before modifying them, verify your work,
and report limitations honestly. Do not invent requirements or claim work you
have not performed.
"""

    worker_rules = """
You are a software developer responsible for actually completing
the assigned task.

IMPORTANT WORKING RULES:

1. Treat the approved project plan included in your task message as the single source of truth.

2. Understand your assigned role and assigned responsibilities before taking action.

3. Use available tools whenever the task requires creating,
   reading, modifying, searching, or inspecting files.

4. Actually perform the requested work.
   Do not merely describe what should be done.

5. Treat the evaluation criteria as your completion checklist.

6. Verify every criterion that can be verified with your tools.

7. Use actual tool results as evidence.

8. Never assume that something passed simply because
   you intended to implement it.

9. Never claim that something was created unless you actually
   created it.

10. Never claim that something was verified unless you actually
    verified it.

11. Inspect important files after modifying them.

12. Keep files, imports, paths, dependencies, and references consistent.

13. If you discover an error, fix it before finishing.

14. Do not create unnecessary functionality.

15. If a tool fails, investigate and retry when possible.

16. If a required capability is unavailable, report the limitation
    instead of pretending the task was completed.

17. Never invent tool results, file contents, test results,
    API responses, or runtime behavior.

18. Do not claim work belonging to another agent.

19. Continue working until the assigned task is actually complete.

20. The evaluation criteria are a completion checklist.
    They are NOT instructions to simply claim that the criteria passed.

21. Before finishing, inspect your work and verify the criteria
    using actual evidence whenever possible.
"""

    system_prompt = (
        generated_prompt
        + "\n\n"
        + worker_rules
    )

    tool_names = agent_stack.get(
        "tools",
        []
    )

    real_tools = resolve_tools(
        tool_names
    )

    if real_tools:

        worker_model = agent_model.bind_tools(
            real_tools
        )

    else:

        worker_model = agent_model


    def call_model(state):

        messages = state["messages"]

        messages_with_system = [
            SystemMessage(
                content=system_prompt
            )
        ] + messages

        response = invoke_with_rate_limit_retry(
            worker_model, messages_with_system, label="Worker model"
        )

        print("\n========== MODEL RESPONSE ==========")
        print(response)

        return {
            "messages": [response]
        }


    graph = StateGraph(
        AgentState
    )


    graph.add_node(
        "agent",
        call_model
    )


    if real_tools:

        tool_node = ToolNode(
            real_tools
        )

        graph.add_node(
            "tools",
            tool_node
        )

        graph.add_edge(
            START,
            "agent"
        )

        graph.add_conditional_edges(

            "agent",

            tools_condition,

            {
                "tools": "tools",
                END: END
            }
        )

        graph.add_edge(
            "tools",
            "agent"
        )

    else:

        graph.add_edge(
            START,
            "agent"
        )

        graph.add_edge(
            "agent",
            END
        )


    return graph.compile()


# ============================================================
# 9. WORKSPACE EVIDENCE
# ============================================================


def get_workspace_evidence(max_chars=30000, max_file_chars=5000):
    evidence = []
    total_chars = 0
    ignored_dirs = {".git", ".venv", "venv", "__pycache__", "node_modules", "dist", "build", ".next"}
    allowed_extensions = {
        ".py", ".js", ".jsx", ".ts", ".tsx", ".html", ".css", ".json",
        ".md", ".txt", ".yml", ".yaml", ".toml", ".ini", ".sql", ".sh"
    }

    if not os.path.isdir("./workspace"):
        return "[WORKSPACE DIRECTORY DOES NOT EXIST]"

    for root, dirs, files in os.walk("./workspace"):
        dirs[:] = [d for d in dirs if d not in ignored_dirs]
        for filename in sorted(files):
            path = os.path.join(root, filename)
            if os.path.splitext(filename)[1].lower() not in allowed_extensions:
                continue
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read(max_file_chars)
                block = f"\n===== FILE: {path} =====\n{content}\n"
            except Exception as exc:
                block = f"\n===== FILE: {path} =====\n[Could not read file: {exc}]\n"

            if total_chars + len(block) > max_chars:
                evidence.append("[Evidence truncated: workspace evidence character limit reached.]")
                return "\n".join(evidence)
            evidence.append(block)
            total_chars += len(block)

    return "\n".join(evidence) if evidence else "[NO RELEVANT WORKSPACE FILES FOUND]"



# 10. EVALUATOR
# ============================================================


def validate_evaluator_output(evaluation, expected_criteria):
    """Ensure evaluator returned each expected criterion exactly once."""
    if not evaluation.results:
        raise ValueError("Evaluator returned an empty results list.")

    returned = [item.criterion for item in evaluation.results]
    if len(returned) != len(set(returned)):
        raise ValueError("Evaluator returned duplicate criteria.")

    expected_set = set(expected_criteria)
    returned_set = set(returned)
    missing = expected_set - returned_set
    extra = returned_set - expected_set

    if missing or extra:
        raise ValueError(
            f"Evaluator criteria mismatch. Missing: {sorted(missing)}; "
            f"unexpected: {sorted(extra)}. Copy every expected criterion exactly."
        )


def evaluate_agent(specification, project_plan):
    stack = specification["stack"]
    criteria = stack.get("evaluation_criteria", [])
    if not criteria:
        print("[WARNING] No evaluation criteria found.")
        return None

    prompt = PromptTemplate(
        template="""
You are an independent software quality evaluator.
Evaluate the actual work in the workspace against this agent's responsibilities.

APPROVED PROJECT PLAN (SOURCE OF TRUTH):
{project_plan}

AGENT ROLE:
{role}

AGENT ASSIGNED INPUT:
{assigned_input}

EXPECTED OUTPUT:
{output}

EVALUATION CRITERIA (return exactly one result for each, copying each criterion exactly):
{criteria}

ACTUAL WORKSPACE EVIDENCE:
{workspace_evidence}

Rules:
1. Evaluate every criterion individually and return exactly one result per criterion.
2. Copy each criterion exactly; do not rewrite it.
3. Judge only concrete workspace evidence, not worker claims.
4. If evidence is missing or insufficient, mark the criterion failed.
5. Never invent files, code, test results, runtime behavior, or API responses.
6. For failed criteria, provide a concrete fix_instruction. For passed criteria, use an empty fix_instruction.
7. Evaluate only this agent's responsibilities.
8. overall_passed is informational; the application calculates the final decision.

{format_instructions}
""",
        input_variables=[
            "project_plan", "role", "assigned_input", "output",
            "criteria", "workspace_evidence", "format_instructions"
        ],
    )

    return generate_validated_output(
        prompt,
        {
            "project_plan": format_project_plan(project_plan),
            "role": specification["role"],
            "assigned_input": stack["input"],
            "output": stack["output"],
            "criteria": json.dumps(criteria, indent=2),
            "workspace_evidence": get_workspace_evidence(),
        },
        EvaluationResult,
        validator=lambda result: validate_evaluator_output(result, criteria),
        label=f"{specification['role']} evaluation",
    )



# 11. DETERMINISTIC EVALUATION VALIDATION
# ============================================================


def validate_evaluation(evaluation, expected_criteria):
    if evaluation is None:
        return {"passed": False, "reason": "Evaluator returned no result.", "failed_results": []}

    results = evaluation.results
    if len(results) != len(expected_criteria):
        return {
            "passed": False,
            "reason": f"Expected {len(expected_criteria)} results but evaluator returned {len(results)}.",
            "failed_results": []
        }

    returned = [result.criterion for result in results]
    if len(returned) != len(set(returned)):
        return {"passed": False, "reason": "Evaluator returned duplicate criteria.", "failed_results": []}

    expected_set = set(expected_criteria)
    returned_set = set(returned)
    missing = expected_set - returned_set
    extra = returned_set - expected_set
    if missing or extra:
        return {
            "passed": False,
            "reason": f"Criteria mismatch. Missing: {sorted(missing)}; unexpected: {sorted(extra)}",
            "failed_results": []
        }

    failed_results = [result for result in results if not result.passed]
    if failed_results:
        return {"passed": False, "reason": "One or more evaluation criteria failed.", "failed_results": failed_results}
    return {"passed": True, "reason": "All evaluation criteria passed.", "failed_results": []}



# 12. BUILD REPAIR FEEDBACK
# ============================================================


def build_repair_feedback(agent, project_plan, validation, attempt):
    feedback = [
        f"""REPAIR ATTEMPT {attempt}

The previous implementation did not satisfy all required criteria.
Repair YOUR existing implementation; do not restart the project.
Inspect the workspace first, preserve working functionality, and change only what is necessary.

APPROVED PROJECT PLAN (SOURCE OF TRUTH):
{format_project_plan(project_plan)}

YOUR ROLE:
{agent['role']}

YOUR ASSIGNED INPUT:
{agent['stack']['input']}

YOUR EXPECTED OUTPUT:
{agent['stack']['output']}

FAILED CRITERIA:
"""
    ]

    for index, result in enumerate(validation.get("failed_results", []), start=1):
        feedback.append(
            f"""
--- FAILURE {index} ---
CRITERION: {result.criterion}
EVIDENCE: {result.evidence}
REASON: {result.reason}
REPAIR DIRECTION: {result.fix_instruction}
"""
        )

    feedback.append("""
REPAIR INSTRUCTIONS:
1. Inspect the current workspace before changing anything.
2. Identify the actual cause of each failure.
3. Repair the existing implementation using your tools.
4. Do not remove functionality that already works.
5. Inspect modified files and verify the failed criteria again.
6. Stay within your assigned responsibilities and the approved plan.
7. Never claim a repair succeeded unless you performed and verified it.
""")
    return "\n".join(feedback)



# 13. RUN WORKER
# ============================================================


def run_worker(
    agent,
    message
):

    print(
        "\n========== WORKER EXECUTION =========="
    )


    result = agent["agent"].invoke({

        "messages": [

            HumanMessage(
                content=message
            )

        ]
    })


    agent["result"] = result


    return result


# ============================================================
# 14. COMPLETE EVALUATION + REPAIR LOOP
# ============================================================


def build_initial_worker_message(agent, project_plan):
    stack = agent["stack"]
    return f"""You are beginning your assigned work.

APPROVED PROJECT PLAN (SOURCE OF TRUTH):
{format_project_plan(project_plan)}

YOUR ROLE:
{agent['role']}

YOUR ASSIGNED RESPONSIBILITIES:
{stack['input']}

EXPECTED OUTPUT:
{stack['output']}

YOUR EVALUATION CRITERIA:
{json.dumps(stack.get('evaluation_criteria', []), indent=2)}

Complete only your assigned responsibilities. Follow the approved technology stack,
features, tasks, dependencies, and deliverables. Inspect the workspace before making
changes, create or modify the actual required artifacts, and verify your work with
your available tools. Do not invent requirements or claim unperformed work.
"""


def run_evaluation_loop(agent, project_plan, max_attempts=MAX_REPAIR_ATTEMPTS):
    project_plan = normalize_project_plan(project_plan)
    print("\n" + "=" * 70)
    print(f"STARTING WORKER LOOP: {agent['role']}")
    print("=" * 70)

    run_worker(agent, build_initial_worker_message(agent, project_plan))

    for attempt in range(1, max_attempts + 1):
        print("\n" + "=" * 70)
        print(f"EVALUATION ATTEMPT {attempt}/{max_attempts}")
        print("=" * 70)

        evaluation = evaluate_agent(agent, project_plan)
        agent["evaluation"] = evaluation
        validation = validate_evaluation(evaluation, agent["stack"].get("evaluation_criteria", []))
        agent["validation"] = validation

        print("\n========== EVALUATION RESULTS ==========")
        if evaluation:
            for result in evaluation.results:
                status = "PASS" if result.passed else "FAIL"
                print(f"\n[{status}] {result.criterion}")
                print("Evidence:", result.evidence)
                print("Reason:", result.reason)
                if not result.passed:
                    print("Fix:", result.fix_instruction)

        if validation["passed"]:
            print("\n[PASS] Agent passed all evaluation criteria.")
            agent["final_status"] = "passed"
            agent["attempts"] = attempt
            return agent

        print("\n[FAIL] Agent did not pass evaluation.")
        if attempt >= max_attempts:
            print("Maximum repair attempts reached.")
            agent["final_status"] = "failed"
            agent["attempts"] = attempt
            return agent

        repair_feedback = build_repair_feedback(agent, project_plan, validation, attempt + 1)
        agent["repair_feedback"] = repair_feedback
        print("\n========== REPAIR WORKER ==========")
        run_worker(agent, repair_feedback)

    agent["final_status"] = "failed"
    agent["attempts"] = max_attempts
    return agent



# 15. BUILD COMPLETE WORKFORCE
# ============================================================


def full_community(project_plan, available_tools):
    project_plan = normalize_project_plan(project_plan)
    tool_descriptions = get_tool_descriptions()

    roles_result = roles_generator(project_plan, tool_descriptions)
    role_names = roles_result["roles"]

    if not role_names:
        raise ValueError("The workforce architect returned no roles for the approved plan.")

    print("\nROLES REQUIRED:")
    for role in role_names:
        print(" -", role)

    agents = []
    for role in role_names:
        print(f"\nGenerating specification for: {role}")
        stack = agent_stack_generator(role, project_plan, tool_descriptions)
        agent_graph = agent_builder(stack)
        agents.append({"role": role, "stack": stack, "agent": agent_graph})

    return agents



# 16. RUN PLANNER, GET APPROVAL, THEN BUILD WORKFORCE
# ============================================================


if __name__ == "__main__":
    description = input("\nDescribe the software you want to build: ").strip()
    if not description:
        raise SystemExit("Project description cannot be empty.")

    # Planner handles clarification, planning, revisions, and approval.
    approved_plan = planner_main(description)
    if approved_plan is None:
        print("\nProject planning was cancelled. No workforce was created.")
        raise SystemExit(0)

    # Convert once; from this point onward the approved plan is the source of truth.
    project_plan = normalize_project_plan(approved_plan)

    print("\n" + "=" * 70)
    print("APPROVED PROJECT PLAN PASSED TO WORKFORCE")
    print("=" * 70)
    print(format_project_plan(project_plan))

    agents = full_community(project_plan, available_tools)

    print("\n" + "=" * 70)
    print("WORKFORCE GENERATED")
    print("=" * 70)

    for agent in agents:
        print("\n" + "=" * 70)
        print("ROLE:", agent["role"])
        print("TOOLS:", agent["stack"]["tools"])
        print("ASSIGNED INPUT:", agent["stack"]["input"])
        print("EXPECTED OUTPUT:", agent["stack"]["output"])
        print("EVALUATION CRITERIA:")
        for criterion in agent["stack"]["evaluation_criteria"]:
            print(" -", criterion)

        run_evaluation_loop(agent, project_plan, max_attempts=MAX_REPAIR_ATTEMPTS)

        print("\n" + "=" * 70)
        if agent.get("final_status") == "passed":
            print(f"[PASS] {agent['role']} completed its assigned work.")
        else:
            print(f"[FAIL] {agent['role']} could not pass evaluation.")
        print("Attempts:", agent.get("attempts", 0))
        print("=" * 70)