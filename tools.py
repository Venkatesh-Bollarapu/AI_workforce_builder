from langchain_core.tools import BaseToolkit, tool

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.agent_toolkits import FileManagementToolkit

import os
import subprocess


# ============================================================
# WORKSPACE
# ============================================================

WORKSPACE = "./workspace"

os.makedirs(WORKSPACE, exist_ok=True)


# ============================================================
# FILE TOOLS
# ============================================================

file_toolkit = FileManagementToolkit(
    root_dir=WORKSPACE
)

file_tools = file_toolkit.get_tools()


# ============================================================
# WEB SEARCH
# ============================================================

web_search_tool = DuckDuckGoSearchRun()


# ============================================================
# CALCULATOR TOOLS
# ============================================================

@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool
def sub(a: float, b: float) -> float:
    """Subtract the second number from the first number."""
    return a - b


@tool
def mult(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@tool
def div(a: float, b: float) -> float:
    """Divide the first number by the second number."""

    if b == 0:
        raise ValueError("Cannot divide by zero.")

    return a / b


class CalculatorToolkit(BaseToolkit):

    def get_tools(self):
        return [
            add,
            sub,
            mult,
            div
        ]


calculator_toolkit = CalculatorToolkit()

calculator_tools = calculator_toolkit.get_tools()


# ============================================================
# COMMAND EXECUTION TOOL
# ============================================================

@tool
def run_command(command: str) -> str:
    """
    Execute a development/testing shell command inside the workspace.

    Use this tool for commands such as:

    - npm install
    - npm run build
    - npm test
    - pytest
    - python file.py
    - node server.js
    - curl API endpoints

    The command runs with ./workspace as its working directory.

    The tool returns:
    - command
    - exit code
    - stdout
    - stderr

    Do NOT use dangerous commands.
    """

    # --------------------------------------------------------
    # Basic safety checks
    # --------------------------------------------------------

    dangerous_patterns = [
        "rm -rf /",
        "rm -rf /*",
        "sudo ",
        "shutdown ",
        "reboot ",
        "mkfs",
        ":(){:|:&};:",
        "chmod -R 777 /",
        "dd if=",
        "curl | sh",
        "curl|sh",
        "wget | sh",
        "wget|sh"
    ]

    normalized = command.lower().replace(" ", "")

    for pattern in dangerous_patterns:

        if pattern.replace(" ", "") in normalized:

            return (
                "COMMAND BLOCKED FOR SAFETY.\n"
                f"Blocked command: {command}"
            )


    # --------------------------------------------------------
    # Execute command
    # --------------------------------------------------------

    try:

        result = subprocess.run(

            command,

            shell=True,

            cwd=WORKSPACE,

            capture_output=True,

            text=True,

            timeout=60
        )


        stdout = result.stdout

        stderr = result.stderr


        # Prevent enormous model context
        MAX_OUTPUT = 6000


        if len(stdout) > MAX_OUTPUT:

            stdout = (
                stdout[:MAX_OUTPUT]
                + "\n...[stdout truncated]"
            )


        if len(stderr) > MAX_OUTPUT:

            stderr = (
                stderr[:MAX_OUTPUT]
                + "\n...[stderr truncated]"
            )


        return (
            f"COMMAND:\n{command}\n\n"

            f"RETURN CODE:\n{result.returncode}\n\n"

            f"STDOUT:\n{stdout}\n\n"

            f"STDERR:\n{stderr}"
        )


    except subprocess.TimeoutExpired:

        return (
            f"COMMAND:\n{command}\n\n"
            "RESULT:\nCOMMAND TIMED OUT AFTER 60 SECONDS."
        )


    except Exception as e:

        return (
            f"COMMAND:\n{command}\n\n"
            f"EXECUTION ERROR:\n{str(e)}"
        )


# ============================================================
# FINAL TOOL LIST
# ============================================================

tools = (
    file_tools
    + calculator_tools
    + [
        web_search_tool,
        run_command
    ]
)