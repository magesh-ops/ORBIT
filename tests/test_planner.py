"""
Tests for ORBIT Task Planner

Tests planner functionality with mocked Ollama responses.
"""

import pytest
from unittest.mock import Mock, patch
from config.models import TaskPlan, SubTask
from orchestrator.planner import Planner, PlannerError, plan_task


# Mock Ollama responses
VALID_PLAN_RESPONSE = """{
  "summary": "Research and implement user authentication system",
  "subtasks": [
    {
      "description": "Research OAuth2 best practices",
      "task_type": "research",
      "dependencies": [],
      "context_requirements": ["security standards", "OAuth2 documentation"]
    },
    {
      "description": "Design authentication architecture",
      "task_type": "reasoning",
      "dependencies": ["0"],
      "context_requirements": ["research findings"]
    },
    {
      "description": "Implement authentication module",
      "task_type": "coding",
      "dependencies": ["1"],
      "context_requirements": ["architecture design"]
    }
  ]
}"""

SIMPLE_PLAN_RESPONSE = """{
  "summary": "Write a hello world program",
  "subtasks": [
    {
      "description": "Create hello.py with print statement",
      "task_type": "coding",
      "dependencies": [],
      "context_requirements": []
    }
  ]
}"""

TOO_MANY_TASKS_RESPONSE = """{
  "summary": "Complex project with too many tasks",
  "subtasks": [%s]
}""" % ",".join([
    '{"description": "Task %d", "task_type": "general", "dependencies": [], "context_requirements": []}' % i
    for i in range(25)
])

INVALID_JSON_RESPONSE = "This is not valid JSON {incomplete"

MISSING_SUMMARY_RESPONSE = """{
  "subtasks": [
    {
      "description": "Some task",
      "task_type": "general",
      "dependencies": [],
      "context_requirements": []
    }
  ]
}"""

INVALID_TASK_TYPE_RESPONSE = """{
  "summary": "Test plan",
  "subtasks": [
    {
      "description": "Invalid task",
      "task_type": "invalid_type",
      "dependencies": [],
      "context_requirements": []
    }
  ]
}"""

CYCLIC_DEPENDENCY_RESPONSE = """{
  "summary": "Plan with cyclic dependencies",
  "subtasks": [
    {
      "description": "Task A depends on B",
      "task_type": "general",
      "dependencies": ["1"],
      "context_requirements": []
    },
    {
      "description": "Task B depends on A",
      "task_type": "general",
      "dependencies": ["0"],
      "context_requirements": []
    }
  ]
}"""

SELF_DEPENDENCY_RESPONSE = """{
  "summary": "Plan with self-dependency",
  "subtasks": [
    {
      "description": "Task depends on itself",
      "task_type": "general",
      "dependencies": ["0"],
      "context_requirements": []
    }
  ]
}"""

INVALID_DEPENDENCY_RESPONSE = """{
  "summary": "Plan with invalid dependency",
  "subtasks": [
    {
      "description": "Task with invalid dependency",
      "task_type": "general",
      "dependencies": ["999"],
      "context_requirements": []
    }
  ]
}"""


class TestPlanner:
    """Test suite for Planner class."""

    @pytest.fixture
    def planner(self):
        """Create a planner instance."""
        return Planner(timeout=10)

    @pytest.fixture
    def mock_ollama_success(self):
        """Mock successful Ollama response."""
        with patch('httpx.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "response": VALID_PLAN_RESPONSE
            }
            mock_post.return_value = mock_response
            yield mock_post

    def test_valid_plan(self, planner, mock_ollama_success):
        """Test generating a valid task plan."""
        plan = planner.plan("Implement user authentication")

        assert isinstance(plan, TaskPlan)
        assert plan.request == "Implement user authentication"
        assert len(plan.subtasks) == 3
        assert plan.summary == "Research and implement user authentication system"

        # Check first subtask
        assert plan.subtasks[0].task_type == "research"
        assert len(plan.subtasks[0].dependencies) == 0

        # Check dependency chain
        assert len(plan.subtasks[1].dependencies) == 1
        assert plan.subtasks[1].dependencies[0] == plan.subtasks[0].task_id

        assert len(plan.subtasks[2].dependencies) == 1
        assert plan.subtasks[2].dependencies[0] == plan.subtasks[1].task_id

    def test_simple_plan(self, planner):
        """Test generating a simple single-task plan."""
        with patch('httpx.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {"response": SIMPLE_PLAN_RESPONSE}
            mock_post.return_value = mock_response

            plan = planner.plan("Write hello world")

            assert isinstance(plan, TaskPlan)
            assert len(plan.subtasks) == 1
            assert plan.subtasks[0].task_type == "coding"

    def test_empty_request(self, planner):
        """Test that empty request raises error."""
        with pytest.raises(PlannerError, match="cannot be empty"):
            planner.plan("")

        with pytest.raises(PlannerError, match="cannot be empty"):
            planner.plan("   ")

    def test_too_many_subtasks(self, planner):
        """Test that exceeding max subtasks raises error."""
        with patch('httpx.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {"response": TOO_MANY_TASKS_RESPONSE}
            mock_post.return_value = mock_response

            with pytest.raises(PlannerError, match="exceeds maximum"):
                planner.plan("Complex project")

    def test_invalid_json(self, planner):
        """Test handling of invalid JSON response."""
        with patch('httpx.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {"response": INVALID_JSON_RESPONSE}
            mock_post.return_value = mock_response

            with pytest.raises(PlannerError, match="Could not extract valid JSON"):
                planner.plan("Test request")

    def test_missing_required_field(self, planner):
        """Test handling of missing required fields."""
        with patch('httpx.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {"response": MISSING_SUMMARY_RESPONSE}
            mock_post.return_value = mock_response

            with pytest.raises(PlannerError, match="Missing required field: summary"):
                planner.plan("Test request")

    def test_invalid_task_type(self, planner):
        """Test handling of invalid task type."""
        with patch('httpx.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {"response": INVALID_TASK_TYPE_RESPONSE}
            mock_post.return_value = mock_response

            with pytest.raises(PlannerError, match="Invalid subtask"):
                planner.plan("Test request")

    def test_cyclic_dependencies(self, planner):
        """Test detection of cyclic dependencies."""
        with patch('httpx.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {"response": CYCLIC_DEPENDENCY_RESPONSE}
            mock_post.return_value = mock_response

            with pytest.raises(PlannerError, match="Cyclic dependencies"):
                planner.plan("Test request")

    def test_self_dependency(self, planner):
        """Test detection of self-dependency."""
        with patch('httpx.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {"response": SELF_DEPENDENCY_RESPONSE}
            mock_post.return_value = mock_response

            with pytest.raises(PlannerError, match="self-dependency"):
                planner.plan("Test request")

    def test_invalid_dependency_reference(self, planner):
        """Test detection of invalid dependency references."""
        with patch('httpx.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {"response": INVALID_DEPENDENCY_RESPONSE}
            mock_post.return_value = mock_response

            with pytest.raises(PlannerError, match="unknown dependency"):
                planner.plan("Test request")

    def test_ollama_unavailable(self, planner):
        """Test handling of Ollama connection failure."""
        with patch('httpx.post', side_effect=Exception("Connection refused")):
            with pytest.raises(PlannerError, match="Ollama request failed"):
                planner.plan("Test request")

    def test_ollama_timeout(self, planner):
        """Test handling of Ollama timeout."""
        import httpx
        with patch('httpx.post', side_effect=httpx.TimeoutException("Timeout")):
            with pytest.raises(PlannerError, match="timed out"):
                planner.plan("Test request")

    def test_ollama_http_error(self, planner):
        """Test handling of Ollama HTTP error."""
        import httpx
        with patch('httpx.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Server error", request=Mock(), response=mock_response
            )
            mock_post.return_value = mock_response

            with pytest.raises(PlannerError, match="Ollama returned error"):
                planner.plan("Test request")

    def test_plan_task_convenience_function(self):
        """Test the convenience function."""
        with patch('httpx.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {"response": SIMPLE_PLAN_RESPONSE}
            mock_post.return_value = mock_response

            plan = plan_task("Write hello world")

            assert isinstance(plan, TaskPlan)
            assert len(plan.subtasks) == 1


class TestPlannerIntegration:
    """Integration tests that can optionally run against real Ollama."""

    @pytest.mark.skip(reason="Requires running Ollama instance")
    def test_real_ollama_simple_request(self):
        """Test with real Ollama (skipped by default)."""
        planner = Planner(timeout=60)
        plan = planner.plan("Create a Python function to add two numbers")

        assert isinstance(plan, TaskPlan)
        assert len(plan.subtasks) > 0
        assert all(isinstance(task, SubTask) for task in plan.subtasks)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
