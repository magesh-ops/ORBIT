"""
ORBIT Task Planner

Decomposes user requests into validated TaskPlan objects using local Ollama/Gemma model.
"""

import json
import httpx
from typing import Optional, Dict, Any, List
from config.models import TaskPlan, SubTask
from config.settings import settings


class PlannerError(Exception):
    """Raised when planning fails."""
    pass


class Planner:
    """
    Decomposes user requests into structured task plans using local Ollama.

    Uses the configured Gemma model to analyze requests and generate
    validated TaskPlan objects with proper subtask decomposition.
    """

    def __init__(self, timeout: int = 30):
        """
        Initialize the planner.

        Args:
            timeout: Request timeout in seconds (default: 30)
        """
        self.ollama_url = settings.ollama_base_url
        self.model = settings.ollama_model
        self.max_subtasks = settings.max_subtasks_per_request
        self.timeout = timeout

    def _build_planning_prompt(self, user_request: str) -> str:
        """Build the prompt for Gemma to generate a task plan."""
        return f"""You are a task planning assistant. Analyze the user's request and break it down into independent or dependent subtasks.

User Request: {user_request}

Generate a JSON task plan with this exact structure:
{{
  "summary": "Brief summary of the overall plan",
  "subtasks": [
    {{
      "description": "Clear description of what needs to be done",
      "task_type": "coding|research|reasoning|general",
      "dependencies": ["task_id1", "task_id2"],
      "context_requirements": ["requirement1", "requirement2"]
    }}
  ]
}}

Rules:
- Maximum {self.max_subtasks} subtasks
- task_type must be one of: coding, research, reasoning, general
- dependencies is a list of task IDs (use indices like "0", "1" for dependent tasks)
- context_requirements lists what information is needed
- Each subtask should be atomic and executable
- Preserve logical dependencies between tasks

Return ONLY the JSON object, no other text."""

    def _call_ollama(self, prompt: str) -> str:
        """
        Call Ollama API to generate a response.

        Args:
            prompt: The prompt to send to Ollama

        Returns:
            Generated response text

        Raises:
            PlannerError: If Ollama is unavailable or request fails
        """
        try:
            response = httpx.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json"
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")

        except httpx.TimeoutException:
            raise PlannerError(f"Ollama request timed out after {self.timeout}s")
        except httpx.ConnectError:
            raise PlannerError(f"Cannot connect to Ollama at {self.ollama_url}")
        except httpx.HTTPStatusError as e:
            raise PlannerError(f"Ollama returned error: {e.response.status_code}")
        except Exception as e:
            raise PlannerError(f"Ollama request failed: {str(e)}")

    def _extract_json(self, response: str) -> Dict[str, Any]:
        """
        Extract and parse JSON from model response.

        Args:
            response: Raw response from Ollama

        Returns:
            Parsed JSON dictionary

        Raises:
            PlannerError: If JSON extraction or parsing fails
        """
        # Try to parse the entire response as JSON first
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from code blocks
        if "```json" in response:
            try:
                start = response.index("```json") + 7
                end = response.index("```", start)
                json_str = response[start:end].strip()
                return json.loads(json_str)
            except (ValueError, json.JSONDecodeError):
                pass

        # Try to find JSON object boundaries
        try:
            start = response.index("{")
            # Find matching closing brace
            brace_count = 0
            for i in range(start, len(response)):
                if response[i] == "{":
                    brace_count += 1
                elif response[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = response[start:i+1]
                        return json.loads(json_str)
        except (ValueError, json.JSONDecodeError):
            pass

        raise PlannerError("Could not extract valid JSON from model response")

    def _validate_dependencies(self, subtasks: List[SubTask]) -> None:
        """
        Validate task dependencies.

        Args:
            subtasks: List of subtasks to validate

        Raises:
            PlannerError: If dependencies are invalid
        """
        task_ids = {task.task_id for task in subtasks}

        for task in subtasks:
            # Check self-dependency
            if task.task_id in task.dependencies:
                raise PlannerError(f"Task {task.task_id} has self-dependency")

            # Check invalid dependency references
            for dep_id in task.dependencies:
                if dep_id not in task_ids:
                    raise PlannerError(
                        f"Task {task.task_id} references unknown dependency {dep_id}"
                    )

        # Check for cycles using DFS
        def has_cycle(task_id: str, visited: set, rec_stack: set) -> bool:
            visited.add(task_id)
            rec_stack.add(task_id)

            # Find task dependencies
            task = next((t for t in subtasks if t.task_id == task_id), None)
            if task:
                for dep in task.dependencies:
                    if dep not in visited:
                        if has_cycle(dep, visited, rec_stack):
                            return True
                    elif dep in rec_stack:
                        return True

            rec_stack.remove(task_id)
            return False

        visited = set()
        for task in subtasks:
            if task.task_id not in visited:
                if has_cycle(task.task_id, visited, set()):
                    raise PlannerError("Cyclic dependencies detected in task plan")

    def _create_subtasks(self, raw_subtasks: List[Dict[str, Any]]) -> List[SubTask]:
        """
        Create validated SubTask objects from raw JSON data.

        Args:
            raw_subtasks: Raw subtask dictionaries from JSON

        Returns:
            List of validated SubTask objects

        Raises:
            PlannerError: If validation fails
        """
        subtasks = []
        task_id_map = {}  # Map from index to actual UUID

        # First pass: create subtasks and build ID map
        for idx, raw_task in enumerate(raw_subtasks):
            try:
                # Create subtask (generates UUID automatically)
                subtask = SubTask(
                    description=raw_task.get("description", ""),
                    task_type=raw_task.get("task_type", "general"),
                    dependencies=[],  # Will be populated in second pass
                    context_requirements=raw_task.get("context_requirements", [])
                )
                subtasks.append(subtask)
                task_id_map[str(idx)] = subtask.task_id
            except Exception as e:
                raise PlannerError(f"Invalid subtask at index {idx}: {str(e)}")

        # Second pass: resolve dependencies using ID map
        for idx, raw_task in enumerate(raw_subtasks):
            raw_deps = raw_task.get("dependencies", [])
            resolved_deps = []

            for dep in raw_deps:
                dep_str = str(dep)
                if dep_str in task_id_map:
                    resolved_deps.append(task_id_map[dep_str])
                else:
                    # Try as direct UUID reference
                    resolved_deps.append(dep_str)

            subtasks[idx].dependencies = resolved_deps

        return subtasks

    def plan(self, user_request: str) -> TaskPlan:
        """
        Generate a validated task plan from a user request.

        Args:
            user_request: Natural language user request

        Returns:
            Validated TaskPlan object

        Raises:
            PlannerError: If planning fails for any reason
        """
        if not user_request or not user_request.strip():
            raise PlannerError("User request cannot be empty")

        # Generate plan using Ollama
        prompt = self._build_planning_prompt(user_request)

        try:
            response = self._call_ollama(prompt)
        except PlannerError:
            raise
        except Exception as e:
            raise PlannerError(f"Unexpected error calling Ollama: {str(e)}")

        # Extract JSON from response
        try:
            plan_data = self._extract_json(response)
        except PlannerError:
            raise
        except Exception as e:
            raise PlannerError(f"Failed to parse response: {str(e)}")

        # Validate structure
        if "summary" not in plan_data:
            raise PlannerError("Missing required field: summary")
        if "subtasks" not in plan_data:
            raise PlannerError("Missing required field: subtasks")

        raw_subtasks = plan_data["subtasks"]
        if not isinstance(raw_subtasks, list):
            raise PlannerError("subtasks must be a list")

        # Check maximum subtasks limit
        if len(raw_subtasks) > self.max_subtasks:
            raise PlannerError(
                f"Plan exceeds maximum of {self.max_subtasks} subtasks "
                f"(got {len(raw_subtasks)})"
            )

        if len(raw_subtasks) == 0:
            raise PlannerError("Plan must contain at least one subtask")

        # Create and validate subtasks
        try:
            subtasks = self._create_subtasks(raw_subtasks)
        except PlannerError:
            raise
        except Exception as e:
            raise PlannerError(f"Failed to create subtasks: {str(e)}")

        # Validate dependencies
        self._validate_dependencies(subtasks)

        # Create and validate final TaskPlan
        try:
            task_plan = TaskPlan(
                request=user_request,
                summary=plan_data["summary"],
                subtasks=subtasks
            )
            return task_plan
        except Exception as e:
            raise PlannerError(f"Failed to create TaskPlan: {str(e)}")


def plan_task(user_request: str, timeout: int = 30) -> TaskPlan:
    """
    Convenience function to generate a task plan.

    Args:
        user_request: Natural language user request
        timeout: Request timeout in seconds (default: 30)

    Returns:
        Validated TaskPlan object

    Raises:
        PlannerError: If planning fails
    """
    planner = Planner(timeout=timeout)
    return planner.plan(user_request)
