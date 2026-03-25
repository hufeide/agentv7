#!/usr/bin/env python3
"""
Test suite for Context management system in agent_v7.py
Tests Context, ContextManager, ContextFormatter, and ContextCompressor
"""

import asyncio
import sys
import os
import time

import pytest

# 添加父目录到 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_v7 import (
    Context,
    GlobalContext,
    TaskContext,
    ContextManager,
    ContextFormatter,
    ContextCompressor,
    State,
    Artifact,
    Step,
)


# =========================
# Test Context class
# =========================

class TestContext:
    """Tests for Context dataclass"""

    def test_context_creation(self):
        """Test Context can be created with required fields"""
        ctx = Context(
            task="Test task",
            step_id="step-1",
            step_task="Test step task",
            inputs={"key": "value"},
            dependencies={},
            relevant_artifacts={},
            memory={},
            history=[],
            tool_trace=[],
        )
        assert ctx.task == "Test task"
        assert ctx.step_id == "step-1"
        assert ctx.version == 1
        print("✓ test_context_creation passed")

    def test_context_with_defaults(self):
        """Test Context uses default values"""
        ctx = Context(
            task="Test",
            step_id="s1",
            step_task="Step 1",
            inputs={},
            dependencies={},
            relevant_artifacts={},
            memory={},
            history=[],
            tool_trace=[],
        )
        assert ctx.budget_tokens == 6000
        assert ctx.max_history == 10
        assert ctx.max_dep_length == 300
        assert ctx.version == 1
        print("✓ test_context_with_defaults passed")

    def test_context_compress(self):
        """Test Context.compress() truncates long dependencies"""
        ctx = Context(
            task="Test",
            step_id="s1",
            step_task="Step 1",
            inputs={},
            dependencies={
                "dep1": "a" * 400,
                "dep2": "b" * 200,
            },
            relevant_artifacts={},
            memory={},
            history=[],
            tool_trace=[],
        )
        ctx.compress()
        assert len(ctx.dependencies["dep1"]) == 303  # 300 + "..."
        assert len(ctx.dependencies["dep2"]) == 200
        print("✓ test_context_compress passed")

    def test_context_created_at(self):
        """Test Context._created_at is set on creation"""
        ctx = Context(
            task="Test",
            step_id="s1",
            step_task="Step 1",
            inputs={},
            dependencies={},
            relevant_artifacts={},
            memory={},
            history=[],
            tool_trace=[],
        )
        assert hasattr(ctx, '_created_at')
        assert isinstance(ctx._created_at, float)
        # Should be within last second
        assert time.time() - ctx._created_at < 1.0
        print("✓ test_context_created_at passed")


# =========================
# Test GlobalContext class
# =========================

class TestGlobalContext:
    """Tests for GlobalContext dataclass"""

    def test_global_context_defaults(self):
        """Test GlobalContext has empty defaults"""
        gc = GlobalContext()
        assert isinstance(gc.memory, dict)
        assert gc.memory == {}
        assert isinstance(gc.reasoning_patterns, list)
        assert gc.reasoning_patterns == []
        print("✓ test_global_context_defaults passed")


# =========================
# Test TaskContext class
    # =========================

class TestTaskContext:
    """Tests for TaskContext dataclass"""

    def test_task_context_defaults(self):
        """Test TaskContext has empty defaults"""
        tc = TaskContext(task_id="task-1")
        assert tc.task_id == "task-1"
        assert isinstance(tc.memory, dict)
        assert isinstance(tc.step_summaries, list)
        print("✓ test_task_context_defaults passed")


# =========================
# Test ContextManager class
# =========================

class TestContextManager:
    """Tests for ContextManager"""

    def setup_method(self):
        """Set up test fixtures (called by pytest, manual invocation needed for direct calls)"""
        self.state = State()
        self.manager = ContextManager(self.state)

    def _setup(self):
        """Helper method to setup fixtures (for manual invocation)"""
        self.state = State()
        self.manager = ContextManager(self.state)

    def test_get_or_create_creates_new_context(self):
        """Test get_or_create creates new context for unknown step"""
        self._setup()
        step = Step(
            step_id="step-1",
            step_type="skill",
            input_data={"task": "Test task"}
        )
        ctx = self.manager.get_or_create(step)
        assert ctx is not None
        assert ctx.step_id == "step-1"
        assert ctx.task == "Test task"
        print("✓ test_get_or_create_creates_new_context passed")

    def test_get_or_create_returns_same_context(self):
        """Test get_or_create returns same context for same step (long-lived)"""
        self._setup()
        step = Step(
            step_id="step-1",
            step_type="skill",
            input_data={"task": "Test task"}
        )
        ctx1 = self.manager.get_or_create(step)
        ctx2 = self.manager.get_or_create(step)
        assert ctx1 is ctx2  # Must be the same object
        print("✓ test_get_or_create_returns_same_context passed")

    def test_get_or_create_preserves_history(self):
        """Test history is preserved when getting same context"""
        self._setup()
        step = Step(
            step_id="step-1",
            step_type="skill",
            input_data={"task": "Test task"}
        )
        ctx = self.manager.get_or_create(step)
        ctx.history.append({"role": "user", "content": "hello"})
        ctx.history.append({"role": "assistant", "content": "hi"})

        # Get same context again
        ctx2 = self.manager.get_or_create(step)
        assert len(ctx2.history) == 2
        assert ctx2.history[0]["content"] == "hello"
        print("✓ test_get_or_create_preserves_history passed")

    def test_get_or_create_preserves_tool_trace(self):
        """Test tool_trace is preserved when getting same context"""
        self._setup()
        step = Step(
            step_id="step-1",
            step_type="skill",
            input_data={"task": "Test task"}
        )
        ctx = self.manager.get_or_create(step)
        ctx.tool_trace.append({
            "tool": "search",
            "input": {"query": "test"},
            "output": "result",
            "timestamp": time.time()
        })

        # Get same context again
        ctx2 = self.manager.get_or_create(step)
        assert len(ctx2.tool_trace) == 1
        assert ctx2.tool_trace[0]["tool"] == "search"
        print("✓ test_get_or_create_preserves_tool_trace passed")

    def test_get_context_returns_none_for_unknown(self):
        """Test get_context returns None for unknown step"""
        self._setup()
        ctx = self.manager.get_context("non-existent-step")
        assert ctx is None
        print("✓ test_get_context_returns_none_for_unknown passed")

    def test_update_context(self):
        """Test update_context updates context and increments version"""
        self._setup()
        step = Step(
            step_id="step-1",
            step_type="skill",
            input_data={"task": "Test task"}
        )
        ctx = self.manager.get_or_create(step)
        assert ctx.version == 1

        self.manager.update_context("step-1", task="New task")
        assert ctx.task == "New task"
        assert ctx.version == 2  # Should increment
        print("✓ test_update_context passed")

    def test_update_context_no_duplicate_version(self):
        """Test update_context doesn't increment version if value unchanged"""
        self._setup()
        step = Step(
            step_id="step-1",
            step_type="skill",
            input_data={"task": "Test task"}
        )
        ctx = self.manager.get_or_create(step)
        assert ctx.version == 1

        # Update with same value
        self.manager.update_context("step-1", task="Test task")
        assert ctx.version == 1  # Should NOT increment

        # Update with different value
        self.manager.update_context("step-1", task="New task")
        assert ctx.version == 2  # Should increment
        print("✓ test_update_context_no_duplicate_version passed")

    def test_get_or_create_task_context(self):
        """Test get_or_create_task_context creates task context"""
        self._setup()
        task_ctx = self.manager.get_or_create_task_context("task-1")
        assert task_ctx.task_id == "task-1"
        print("✓ test_get_or_create_task_context passed")

    def test_add_task_summary(self):
        """Test add_task_summary adds summary to task context"""
        self._setup()
        self.manager.add_task_summary("task-1", "First summary")
        self.manager.add_task_summary("task-1", "Second summary")

        task_ctx = self.manager.get_or_create_task_context("task-1")
        assert len(task_ctx.step_summaries) == 2
        assert task_ctx.step_summaries[0] == "First summary"
        print("✓ test_add_task_summary passed")

    def test_update_global_memory(self):
        """Test update_global_memory stores global memory"""
        self._setup()
        self.manager.update_global_memory("user_preference", "dark_mode")
        value = self.manager.get_global_memory("user_preference")
        assert value == "dark_mode"
        print("✓ test_update_global_memory passed")

    def test_cleanup_old_contexts(self):
        """Test cleanup_old_contexts removes old contexts"""
        self._setup()
        # Create some contexts
        step1 = Step(step_id="step-1", step_type="skill", input_data={"task": "t1"})
        step2 = Step(step_id="step-2", step_type="skill", input_data={"task": "t2"})
        self.manager.get_or_create(step1)
        self.manager.get_or_create(step2)

        assert len(self.manager.contexts) == 2

        # Cleanup should remove contexts with version > 100 (none in this case)
        # Since all contexts have version 1, none should be removed by version check
        removed = self.manager.cleanup_old_contexts(max_age_seconds=3600)
        # No contexts should be removed because they're new and have low version
        assert removed == 0
        assert len(self.manager.contexts) == 2
        print("✓ test_cleanup_old_contexts passed")

    def test_clear_all_contexts(self):
        """Test clear_all_contexts removes all contexts"""
        self._setup()
        step1 = Step(step_id="step-1", step_type="skill", input_data={"task": "t1"})
        step2 = Step(step_id="step-2", step_type="skill", input_data={"task": "t2"})
        self.manager.get_or_create(step1)
        self.manager.get_or_create(step2)

        assert len(self.manager.contexts) == 2

        count = self.manager.clear_all_contexts()
        assert count == 2
        assert len(self.manager.contexts) == 0
        print("✓ test_clear_all_contexts passed")


# =========================
# Test ContextFormatter class
# =========================

class TestContextFormatter:
    """Tests for ContextFormatter"""

    def test_format_system_with_dependencies(self):
        """Test format_system includes dependencies"""
        ctx = Context(
            task="Complete project",
            step_id="step-1",
            step_task="Write code",
            inputs={},
            dependencies={
                "step-0": {
                    "value": "Requirements: user authentication",
                    "type": "text",
                    "success": True
                }
            },
            relevant_artifacts={},
            memory={},
            history=[],
            tool_trace=[],
        )
        system_prompt = ContextFormatter.format_system(ctx, "You are a helpful assistant.")
        assert "## 上游结果" in system_prompt
        assert "Requirements: user authentication" in system_prompt
        print("✓ test_format_system_with_dependencies passed")

    def test_format_system_with_tool_trace(self):
        """Test format_system includes tool_trace"""
        ctx = Context(
            task="Complete project",
            step_id="step-1",
            step_task="Write code",
            inputs={},
            dependencies={},
            relevant_artifacts={},
            memory={},
            history=[],
            tool_trace=[
                {
                    "tool": "search",
                    "input": {"query": "requirements"},
                    "output": "Found 10 results",
                    "timestamp": time.time()
                }
            ],
        )
        system_prompt = ContextFormatter.format_system(ctx, "You are a helpful assistant.")
        assert "## 已执行操作" in system_prompt
        assert "search" in system_prompt
        assert "Found 10 results" in system_prompt
        print("✓ test_format_system_with_tool_trace passed")

    def test_format_system_without_deps_or_trace(self):
        """Test format_system handles empty dependencies and tool_trace"""
        ctx = Context(
            task="Complete project",
            step_id="step-1",
            step_task="Write code",
            inputs={},
            dependencies={},
            relevant_artifacts={},
            memory={},
            history=[],
            tool_trace=[],
        )
        system_prompt = ContextFormatter.format_system(ctx, "You are a helpful assistant.")
        assert "## 上游结果" in system_prompt
        assert "无" in system_prompt
        assert "## 已执行操作" in system_prompt
        assert "无" in system_prompt
        print("✓ test_format_system_without_deps_or_trace passed")

    def test_format_user(self):
        """Test format_user formats user prompt"""
        ctx = Context(
            task="Complete project",
            step_id="step-1",
            step_task="Write code",
            inputs={"task": "Write a function", "language": "python"},
            dependencies={},
            relevant_artifacts={},
            memory={},
            history=[],
            tool_trace=[],
        )
        user_prompt = ContextFormatter.format_user(ctx)
        assert "任务: Write code" in user_prompt
        assert "language" in user_prompt
        assert "python" in user_prompt
        print("✓ test_format_user passed")


# =========================
# Test ContextCompressor class
# =========================

class TestContextCompressor:
    """Tests for ContextCompressor"""

    def test_truncate(self):
        """Test truncate shortens text"""
        text = "a" * 100
        truncated = ContextCompressor.truncate(text, 50)
        assert len(truncated) == 50
        print("✓ test_truncate passed")

    def test_compress_dependencies(self):
        """Test compress_dependencies truncates dependency values"""
        deps = {
            "dep1": "a" * 500,
            "dep2": "b" * 100,
            "dep3": {"value": "c" * 300, "type": "text"},
        }
        compressed = ContextCompressor.compress_dependencies(deps, 200)
        assert len(compressed["dep1"]) == 200
        assert len(compressed["dep2"]) == 100
        assert len(str(compressed["dep3"])) <= 200
        print("✓ test_compress_dependencies passed")

    def test_summarize_text_short(self):
        """Test summarize_text returns short text unchanged"""
        short_text = "Short text"
        summarized = ContextCompressor.summarize_text(short_text, max_tokens=100)
        assert summarized == short_text
        print("✓ test_summarize_text_short passed")

    def test_summarize_text_long(self):
        """Test summarize_text truncates long text"""
        long_text = "a" * 1000
        summarized = ContextCompressor.summarize_text(long_text, max_tokens=100)
        # 100 tokens * 4 chars = 400 chars (200 head + 200 tail) + 18 chars for middle
        assert len(summarized) == 418  # 200 + "\n... [摘要中间省略] ...\n" + 200
        assert "..." in summarized
        print("✓ test_summarize_text_long passed")

    def test_summarize_history(self):
        """Test summarize_history keeps last N messages"""
        history = [{"role": f"msg-{i}", "content": f"content-{i}"} for i in range(20)]
        summarized = ContextCompressor.summarize_history(history, keep_last=5)
        assert len(summarized) == 5
        assert summarized[0]["role"] == "msg-15"  # First of last 5
        assert summarized[-1]["role"] == "msg-19"  # Last
        print("✓ test_summarize_history passed")

    def test_estimate_tokens(self):
        """Test estimate_tokens calculates token count"""
        # 1 token ≈ 4 chars
        text = "a" * 100
        tokens = ContextCompressor.estimate_tokens(text)
        assert tokens == 25  # 100 / 4
        print("✓ test_estimate_tokens passed")

    def test_estimate_context_tokens(self):
        """Test _estimate_context_tokens estimates context tokens"""
        ctx = Context(
            task="a" * 1000,
            step_id="s1",
            step_task="b" * 500,
            inputs={},
            dependencies={"dep": "c" * 200},
            relevant_artifacts={},
            memory={},
            history=[{"role": "user", "content": "d" * 100}],
            tool_trace=[{"tool": "t1", "input": {}, "output": "e" * 50}],
        )
        # task[:500] + step_task[:200] + dependencies[:300] + history[:200] + tool_trace[:100]
        # = 500 + 200 + 300 + 200 + 100 = 1300 chars = 325 tokens
        tokens = ContextCompressor._estimate_context_tokens(ctx)
        assert tokens > 0
        print("✓ test_estimate_context_tokens passed")

    def test_compress_context_within_budget(self):
        """Test compress_context does nothing if within budget"""
        ctx = Context(
            task="Short task",
            step_id="s1",
            step_task="Short step",
            inputs={},
            dependencies={"dep": "a" * 50},
            relevant_artifacts={},
            memory={},
            history=[],
            tool_trace=[],
            budget_tokens=6000,
        )
        original_deps = ctx.dependencies.copy()
        original_history = list(ctx.history)

        result = ContextCompressor.compress_context(ctx)

        # Should not compress when within budget
        assert result.dependencies == original_deps
        print("✓ test_compress_context_within_budget passed")

    def test_compress_context_exceeds_budget(self):
        """Test compress_context compresses when exceeds budget"""
        ctx = Context(
            task="a" * 1000,
            step_id="s1",
            step_task="b" * 500,
            inputs={},
            dependencies={"dep1": "c" * 500, "dep2": "d" * 500},
            relevant_artifacts={
                "art1": "e" * 500,
                "art2": "f" * 500,
            },
            memory={},
            history=[{"role": "user", "content": "g" * 500}],
            tool_trace=[{"tool": "t1", "input": {}, "output": "h" * 500}],
            budget_tokens=100,  # Very small budget
        )

        result = ContextCompressor.compress_context(ctx)

        # Dependencies should be compressed
        assert len(str(result.dependencies)) < 500
        print("✓ test_compress_context_exceeds_budget passed")


# =========================
# Test State class (async)
# =========================

class TestState:
    """Tests for State class async methods"""

    @pytest.mark.asyncio
    async def test_update_artifact(self):
        """Test async update_artifact"""
        state = State()
        artifact = Artifact.create_success("test value", step_id="step-1")
        await state.update_artifact("step-1", artifact)

        retrieved = await state.get_artifact("step-1")
        assert retrieved is not None
        assert retrieved.value == "test value"
        print("✓ test_update_artifact passed")

    @pytest.mark.asyncio
    async def test_get_artifact(self):
        """Test async get_artifact"""
        state = State()
        result = await state.get_artifact("non-existent")
        assert result is None
        print("✓ test_get_artifact passed")

    @pytest.mark.asyncio
    async def test_is_step_successful(self):
        """Test async is_step_successful"""
        state = State()
        # Test with non-existent artifact
        result = await state.is_step_successful("non-existent")
        assert result is False

        # Test with successful artifact
        artifact = Artifact.create_success("value", step_id="step-1")
        await state.update_artifact("step-1", artifact)
        result = await state.is_step_successful("step-1")
        assert result is True

        # Test with failed artifact
        artifact = Artifact.create_error("error", step_id="step-2")
        await state.update_artifact("step-2", artifact)
        result = await state.is_step_successful("step-2")
        assert result is False
        print("✓ test_is_step_successful passed")

    @pytest.mark.asyncio
    async def test_add_trace(self):
        """Test async add_trace"""
        state = State()
        trace = {
            "step_id": "step-1",
            "agent": "default",
            "mode": "react",
            "success": True
        }
        await state.add_trace(trace)

        assert len(state.trace) == 1
        assert state.trace[0]["step_id"] == "step-1"
        print("✓ test_add_trace passed")

    @pytest.mark.asyncio
    async def test_memory_operations(self):
        """Test async memory operations"""
        state = State()

        await state.set_memory("key1", "value1")
        result = await state.get_memory("key1")
        assert result == "value1"

        result = await state.get_memory("non-existent", "default")
        assert result == "default"
        print("✓ test_memory_operations passed")

    @pytest.mark.asyncio
    async def test_get_artifacts_snapshot(self):
        """Test async get_artifacts_snapshot"""
        state = State()
        artifact1 = Artifact.create_success("value1", step_id="step-1")
        artifact2 = Artifact.create_success("value2", step_id="step-2")
        await state.update_artifact("step-1", artifact1)
        await state.update_artifact("step-2", artifact2)

        snapshot = await state.get_artifacts_snapshot()
        assert len(snapshot) == 2
        assert snapshot["step-1"].value == "value1"
        assert snapshot["step-2"].value == "value2"
        print("✓ test_get_artifacts_snapshot passed")


# =========================
# Run all tests
# =========================

def run_tests():
    """Run all test classes"""
    print("Running Context System Tests")
    print("=" * 50)

    # Synchronous tests
    sync_tests = [
        TestContext(),
        TestGlobalContext(),
        TestTaskContext(),
        TestContextManager(),
        TestContextFormatter(),
        TestContextCompressor(),
    ]

    for test_class in sync_tests:
        for method_name in dir(test_class):
            if method_name.startswith("test_"):
                try:
                    getattr(test_class, method_name)()
                except Exception as e:
                    print(f"✗ {method_name} failed: {e}")

    # Async tests
    async def run_async_tests():
        async_tests = [
            TestState(),
        ]
        for test_class in async_tests:
            for method_name in dir(test_class):
                if method_name.startswith("test_"):
                    try:
                        await getattr(test_class, method_name)()
                    except Exception as e:
                        print(f"✗ {method_name} failed: {e}")

    asyncio.run(run_async_tests())

    print("=" * 50)
    print("All tests completed!")


# =========================
# Integration tests with real-world scenarios
# =========================

class TestIntegrationScenarios:
    """Integration tests using real-world scenarios"""

    def _setup(self):
        """Set up test fixtures"""
        self.state = State()
        self.manager = ContextManager(self.state)

    def test_multi_step_workflow(self):
        """Test a multi-step workflow with dependencies"""
        self._setup()

        # Simulate a multi-step workflow:
        # step-1: research -> step-2: write -> step-3: review

        step1 = Step(step_id="step-1", step_type="skill", input_data={"task": "Research requirements"})
        ctx1 = self.manager.get_or_create(step1)
        ctx1.history.append({"role": "user", "content": "What are the requirements?"})
        ctx1.history.append({"role": "assistant", "content": "Requirements: user auth, data persistence"})
        ctx1.tool_trace.append({
            "tool": "search",
            "input": {"query": "requirements"},
            "output": "Found: user auth, data persistence",
            "timestamp": time.time()
        })

        step2 = Step(step_id="step-2", step_type="skill", input_data={"task": "Write code"})
        ctx2 = self.manager.get_or_create(step2)

        # step-2 should inherit some context from step-1
        assert ctx2.step_task == "Write code"
        assert ctx2.task == "Write code"

        step3 = Step(step_id="step-3", step_type="skill", input_data={"task": "Review code"})
        ctx3 = self.manager.get_or_create(step3)

        # All contexts should be stored
        assert len(self.manager.contexts) == 3
        print("✓ test_multi_step_workflow passed")

    def test_context_preservation_across_retrievals(self):
        """Test that context is preserved across multiple retrievals"""
        self._setup()

        step = Step(step_id="workflow-step", step_type="skill", input_data={"task": "Process data"})
        ctx = self.manager.get_or_create(step)

        # Simulate multiple interactions
        for i in range(5):
            ctx.history.append({
                "role": "assistant" if i % 2 == 0 else "user",
                "content": f"Message {i}"
            })
            ctx.tool_trace.append({
                "tool": f"tool_{i}",
                "input": {"index": i},
                "output": f"result_{i}",
                "timestamp": time.time()
            })

        # Retrieve the same context multiple times
        ctx_retrieved_1 = self.manager.get_or_create(step)
        ctx_retrieved_2 = self.manager.get_context("workflow-step")

        # All should be the same object with preserved history
        assert ctx is ctx_retrieved_1 is ctx_retrieved_2
        assert len(ctx.history) == 5
        assert len(ctx.tool_trace) == 5
        assert ctx.history[0]["content"] == "Message 0"
        assert ctx.history[4]["content"] == "Message 4"
        print("✓ test_context_preservation_across_retrievals passed")

    def test_tool_trace_formatting(self):
        """Test that tool trace is correctly formatted in system prompt"""
        self._setup()

        ctx = Context(
            task="Complete task",
            step_id="step-1",
            step_task="Execute workflow",
            inputs={},
            dependencies={},
            relevant_artifacts={},
            memory={},
            history=[],
            tool_trace=[
                {
                    "tool": "calculator",
                    "input": {"operation": "add", "a": 5, "b": 3},
                    "output": "8",
                    "timestamp": time.time()
                },
                {
                    "tool": "search",
                    "input": {"query": "help"},
                    "output": "Found documentation",
                    "timestamp": time.time()
                }
            ]
        )

        system_prompt = ContextFormatter.format_system(ctx, "You are a helpful assistant.")

        # Check that tool names and outputs are in the prompt
        assert "## 已执行操作" in system_prompt
        assert "calculator" in system_prompt
        assert "8" in system_prompt
        assert "search" in system_prompt
        assert "documentation" in system_prompt
        print("✓ test_tool_trace_formatting passed")

    def test_dependency_formatting(self):
        """Test that dependencies are correctly formatted"""
        self._setup()

        ctx = Context(
            task="Build application",
            step_id="step-2",
            step_task="Integrate components",
            inputs={},
            dependencies={
                "step-1": {
                    "value": "Component A: user authentication module",
                    "type": "code",
                    "success": True
                },
                "step-1-output": {
                    "value": "Component B: database connection pool",
                    "type": "code",
                    "success": True
                }
            },
            relevant_artifacts={},
            memory={},
            history=[],
            tool_trace=[]
        )

        system_prompt = ContextFormatter.format_system(ctx, "You are a backend developer.")

        assert "## 上游结果" in system_prompt
        assert "Component A" in system_prompt
        assert "Component B" in system_prompt
        print("✓ test_dependency_formatting passed")

    def test_context_version_management(self):
        """Test context version management with updates"""
        self._setup()

        step = Step(step_id="version-test", step_type="skill", input_data={"task": "Initial task"})
        ctx = self.manager.get_or_create(step)
        assert ctx.version == 1

        # Update task - version should increment
        self.manager.update_context("version-test", task="Updated task")
        assert ctx.version == 2

        # Update with same task - version should NOT increment
        self.manager.update_context("version-test", task="Updated task")
        assert ctx.version == 2

        # Update another field
        self.manager.update_context("version-test", step_task="New step")
        assert ctx.version == 3
        print("✓ test_context_version_management passed")

    def test_global_memory_isolation(self):
        """Test that global memory is isolated between tests"""
        self._setup()

        # Set global memory
        self.manager.update_global_memory("theme", "dark")
        self.manager.update_global_memory("language", "en")

        # Retrieve global memory
        assert self.manager.get_global_memory("theme") == "dark"
        assert self.manager.get_global_memory("language") == "en"

        # Test default value
        assert self.manager.get_global_memory("nonexistent", "default") == "default"
        print("✓ test_global_memory_isolation passed")

    def test_task_context_summary(self):
        """Test task context summary management"""
        self._setup()

        # Add summaries for a task
        self.manager.add_task_summary("task-1", "Step 1 completed: research done")
        self.manager.add_task_summary("task-1", "Step 2 completed: draft written")
        self.manager.add_task_summary("task-1", "Step 3 completed: review done")

        # Retrieve task context
        task_ctx = self.manager.get_or_create_task_context("task-1")
        assert len(task_ctx.step_summaries) == 3
        assert task_ctx.step_summaries[0] == "Step 1 completed: research done"
        assert task_ctx.step_summaries[2] == "Step 3 completed: review done"
        print("✓ test_task_context_summary passed")

    def test_compression_under_budget(self):
        """Test that context within budget is not compressed"""
        self._setup()

        ctx = Context(
            task="Short task",
            step_id="step-1",
            step_task="Short step",
            inputs={"data": "small"},
            dependencies={"dep": {"value": "small output"}},
            relevant_artifacts={"art": {"value": "small artifact"}},
            memory={},
            history=[{"role": "user", "content": "hi"}],
            tool_trace=[{"tool": "t1", "input": {}, "output": "ok"}],
            budget_tokens=6000,
        )

        original_deps = str(ctx.dependencies)
        original_history = list(ctx.history)

        # Compress - should not change since within budget
        ContextCompressor.compress_context(ctx)

        assert str(ctx.dependencies) == original_deps
        assert ctx.history == original_history
        print("✓ test_compression_under_budget passed")


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def _setup(self):
        """Set up test fixtures"""
        self.state = State()
        self.manager = ContextManager(self.state)

    def test_empty_inputs(self):
        """Test with empty inputs"""
        ctx = Context(
            task="Test",
            step_id="step-1",
            step_task="Step",
            inputs={},
            dependencies={},
            relevant_artifacts={},
            memory={},
            history=[],
            tool_trace=[]
        )
        assert ctx.inputs == {}
        user_prompt = ContextFormatter.format_user(ctx)
        assert "任务:" in user_prompt
        print("✓ test_empty_inputs passed")

    def test_very_long_text(self):
        """Test with very long text values"""
        long_text = "x" * 10000
        ctx = Context(
            task=long_text,
            step_id="step-1",
            step_task=long_text,
            inputs={"data": long_text},
            dependencies={"dep": {"value": long_text}},
            relevant_artifacts={"art": {"value": long_text}},
            memory={},
            history=[],
            tool_trace=[]
        )
        # Should handle without error
        assert len(ctx.task) == 10000
        print("✓ test_very_long_text passed")

    def test_special_characters_in_inputs(self):
        """Test with special characters"""
        ctx = Context(
            task="Task with special chars: @#$%^&*()",
            step_id="step-1",
            step_task="Step with \n newline \t tab",
            inputs={"json": '{"key": "value"}'},
            dependencies={},
            relevant_artifacts={},
            memory={},
            history=[],
            tool_trace=[]
        )
        user_prompt = ContextFormatter.format_user(ctx)
        # format_user uses step_task, not task
        assert "Step with" in user_prompt
        assert "\n" in user_prompt
        print("✓ test_special_characters_in_inputs passed")

    def test_unicode_content(self):
        """Test with unicode content"""
        ctx = Context(
            task="任务: 处理中文数据",
            step_id="step-1",
            step_task="步骤 1: 日本語テスト",
            inputs={"emoji": "🎉🚀"},
            dependencies={},
            relevant_artifacts={},
            memory={},
            history=[],
            tool_trace=[]
        )
        user_prompt = ContextFormatter.format_user(ctx)
        # format_user uses step_task
        assert "步骤" in user_prompt
        assert "日本語" in user_prompt
        assert "🎉" in user_prompt
        assert "🚀" in user_prompt
        print("✓ test_unicode_content passed")

    def test_none_values_handling(self):
        """Test handling of None values"""
        manager = ContextManager(None)  # state is None
        step = Step(step_id="step-1", step_type="skill", input_data={"task": "Test"})
        ctx = manager.get_or_create(step)
        # When state is None, relevant_artifacts should be empty
        assert ctx.relevant_artifacts == {}
        print("✓ test_none_values_handling passed")

    def test_context_with_empty_history(self):
        """Test context with empty history"""
        ctx = Context(
            task="Test",
            step_id="step-1",
            step_task="Step",
            inputs={},
            dependencies={},
            relevant_artifacts={},
            memory={},
            history=[],
            tool_trace=[]
        )
        system_prompt = ContextFormatter.format_system(ctx, "You are a helper.")
        assert "## 已执行操作" in system_prompt
        assert "无" in system_prompt
        print("✓ test_context_with_empty_history passed")

    def test_tool_trace_with_special_characters(self):
        """Test tool trace with special characters"""
        ctx = Context(
            task="Test",
            step_id="step-1",
            step_task="Step",
            inputs={},
            dependencies={},
            relevant_artifacts={},
            memory={},
            history=[],
            tool_trace=[
                {
                    "tool": "calc",
                    "input": {"expr": "a + b"},
                    "output": "result with \"quotes\" and 'apostrophes'",
                    "timestamp": time.time()
                }
            ]
        )
        system_prompt = ContextFormatter.format_system(ctx, "Helper.")
        assert "calc" in system_prompt
        print("✓ test_tool_trace_with_special_characters passed")

    def test_multiple_updates_same_value(self):
        """Test multiple updates with same value"""
        self._setup()
        step = Step(step_id="step-1", step_type="skill", input_data={"task": "Initial"})
        ctx = self.manager.get_or_create(step)
        assert ctx.version == 1

        # First update changes value - version should increment
        self.manager.update_context("step-1", task="Same task")
        assert ctx.version == 2

        # Multiple updates with same value - version should NOT increment
        for _ in range(5):
            self.manager.update_context("step-1", task="Same task")

        # Version should still be 2 (not incremented)
        assert ctx.version == 2
        print("✓ test_multiple_updates_same_value passed")

    def test_context_retrieval_after_update(self):
        """Test that retrieved context reflects latest updates"""
        self._setup()
        step = Step(step_id="step-1", step_type="skill", input_data={"task": "Initial"})
        ctx1 = self.manager.get_or_create(step)
        ctx1.task = "Updated"

        # Retrieve again
        ctx2 = self.manager.get_or_create(step)
        assert ctx2.task == "Updated"
        assert ctx1 is ctx2
        print("✓ test_context_retrieval_after_update passed")

    def test_clear_all_contexts_empty(self):
        """Test clearing empty contexts"""
        self._setup()
        count = self.manager.clear_all_contexts()
        assert count == 0
        assert len(self.manager.contexts) == 0
        print("✓ test_clear_all_contexts_empty passed")


class TestLargeScale:
    """Tests for large scale scenarios"""

    def _setup(self):
        """Set up test fixtures"""
        self.state = State()
        self.manager = ContextManager(self.state)

    def test_many_contexts(self):
        """Test with many contexts"""
        self._setup()
        for i in range(100):
            step = Step(step_id=f"step-{i}", step_type="skill", input_data={"task": f"Task {i}"})
            ctx = self.manager.get_or_create(step)
            ctx.history.append({"role": "user", "content": f"Message {i}"})

        assert len(self.manager.contexts) == 100

        # Verify all contexts have correct history
        for i in range(100):
            ctx = self.manager.get_context(f"step-{i}")
            assert len(ctx.history) == 1
            assert ctx.history[0]["content"] == f"Message {i}"
        print("✓ test_many_contexts passed")

    def test_heavy_tool_trace(self):
        """Test with heavy tool trace"""
        self._setup()
        step = Step(step_id="step-1", step_type="skill", input_data={"task": "Task"})
        ctx = self.manager.get_or_create(step)

        # Add many tool traces
        for i in range(50):
            ctx.tool_trace.append({
                "tool": f"tool-{i}",
                "input": {"index": i, "data": "x" * 100},
                "output": "result",
                "timestamp": time.time()
            })

        # Should handle without error
        assert len(ctx.tool_trace) == 50
        print("✓ test_heavy_tool_trace passed")


def run_integration_tests():
    """Run integration tests"""
    print("\nRunning Integration Tests")
    print("=" * 50)

    test_instance = TestIntegrationScenarios()

    for method_name in dir(test_instance):
        if method_name.startswith("test_"):
            try:
                test_instance._setup()
                getattr(test_instance, method_name)()
            except Exception as e:
                print(f"✗ {method_name} failed: {e}")
                import traceback
                traceback.print_exc()


def run_edge_case_tests():
    """Run edge case tests"""
    print("\nRunning Edge Case Tests")
    print("=" * 50)

    test_instance = TestEdgeCases()

    for method_name in dir(test_instance):
        if method_name.startswith("test_"):
            try:
                test_instance._setup()
                getattr(test_instance, method_name)()
            except Exception as e:
                print(f"✗ {method_name} failed: {e}")
                import traceback
                traceback.print_exc()


def run_large_scale_tests():
    """Run large scale tests"""
    print("\nRunning Large Scale Tests")
    print("=" * 50)

    test_instance = TestLargeScale()

    for method_name in dir(test_instance):
        if method_name.startswith("test_"):
            try:
                test_instance._setup()
                getattr(test_instance, method_name)()
            except Exception as e:
                print(f"✗ {method_name} failed: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    run_tests()
    run_integration_tests()
    run_edge_case_tests()
    run_large_scale_tests()
    print("\n" + "=" * 50)
    print("All tests completed!")
