"""
AgentOS v6 - 多智能体任务执行框架
修复版本：解决所有已发现的问题
"""

from __future__ import annotations

import asyncio
import uuid
import time
import logging
import json
import os
import sys
import copy
import re
import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Callable, Tuple, Set
from openai import AsyncOpenAI


# =========================
# 错误类型定义
# =========================
class AgentOSError(Exception):
    """框架基础异常"""
    pass


class ToolExecutionError(AgentOSError):
    """工具执行异常"""
    def __init__(self, tool_name: str, cause: Exception):
        self.tool_name = tool_name
        self.cause = cause
        super().__init__(f"Tool '{tool_name}' execution failed: {cause}")


class StepTimeoutError(AgentOSError):
    """步骤超时异常"""
    def __init__(self, step_id: str, timeout: float):
        self.step_id = step_id
        self.timeout = timeout
        super().__init__(f"Step '{step_id}' timeout after {timeout}s")


class DependencyError(AgentOSError):
    """依赖步骤失败异常"""
    pass


# 尝试导入 PyYAML
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# =========================
# 配置管理（预定义默认配置，用于初始化日志）
# =========================


def _get_env_or_default(key: str, value_type: type, default: Any) -> Any:
    """在 logger 创建前获取环境变量"""
    env_val = os.getenv(key)
    if env_val is not None and env_val != "":
        try:
            if value_type == int:
                return int(env_val)
            elif value_type == float:
                return float(env_val)
            elif value_type == bool:
                return env_val.lower() == "true"
            else:
                return env_val
        except (ValueError, AttributeError):
            pass
    return default


# =========================
# 日志配置（必须在 ConfigManager 之前）
# =========================
# 先获取 LOG_LEVEL 用于配置日志
_temp_log_level = _get_env_or_default("LOG_LEVEL", str, "INFO")
logging.basicConfig(
    level=getattr(logging, _temp_log_level, logging.INFO),
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("AgentOSV6")


# =========================
# 配置管理
# =========================
class ConfigManager:
    def __init__(self):
        self._config = None
        try:
            from config import config
            self._config = config
        except ImportError:
            pass

        self.MAX_STEP_RETRIES = self._get_config_or_env("MAX_STEP_RETRIES", int, 3)
        self.GLOBAL_TIMEOUT = self._get_config_or_env("GLOBAL_TIMEOUT", int, 300)
        self.STEP_EXECUTION_TIMEOUT = self._get_config_or_env("STEP_EXECUTION_TIMEOUT", int, 120)
        self.STEP_READY_TIMEOUT = self._get_config_or_env("STEP_READY_TIMEOUT", float, 5.0)
        self.QUEUE_TIMEOUT = self._get_config_or_env("QUEUE_TIMEOUT", float, 1.0)
        self.LLM_CALL_TIMEOUT = self._get_config_or_env("LLM_CALL_TIMEOUT", float, 60.0)
        self.LLM_MAX_TOKENS = self._get_config_or_env("LLM_MAX_TOKENS", int, 6000)
        self.LLM_TEMPERATURE = self._get_config_or_env("LLM_TEMPERATURE", float, 0.3)
        self.EVENT_BUS_MAX_QUEUE_SIZE = self._get_config_or_env("EVENT_BUS_MAX_QUEUE_SIZE", int, 1000)
        self.DEAD_LETTER_QUEUE_ENABLED = self._get_config_or_env("DEAD_LETTER_QUEUE_ENABLED", bool, False)
        self.STEP_OUTPUT_SUMMARY_LEN = self._get_config_or_env("STEP_OUTPUT_SUMMARY_LEN", int, 200)
        self.VLLM_URL = self._get_config_or_env("VLLM_URL", str, "http://192.168.1.159:19000")
        self.MODEL_NAME = self._get_config_or_env("MODEL_NAME", str, "Qwen3Coder")
        self.LOG_LEVEL = self._get_config_or_env("LOG_LEVEL", str, "INFO")
        self.TAVILY_API_KEY = self._get_config_or_env("TAVILY_API_KEY", str, "")
        self.WORKER_COUNT = self._get_config_or_env("WORKER_COUNT", int, 2)
        self.SKILLS_DIR = self._get_config_or_env("SKILLS_DIR", str, "skills")

    def _get_config_or_env(self, key: str, value_type: type, default: Any) -> Any:
        if self._config is not None:
            val = getattr(self._config, key, None)
            if val is not None:
                return val
        env_val = os.getenv(key)
        if env_val is not None and env_val != "":
            try:
                if value_type == int:
                    return int(env_val)
                elif value_type == float:
                    return float(env_val)
                elif value_type == bool:
                    return env_val.lower() == "true"
                else:
                    return env_val
            except (ValueError, AttributeError):
                pass
        return default


# =========================
# 创建全局配置管理器实例
# =========================
config_manager = ConfigManager()
VLLM_URL = config_manager.VLLM_URL
MODEL_NAME = config_manager.MODEL_NAME
LOG_LEVEL = config_manager.LOG_LEVEL


# =========================
# Context 管理系统 - 数据结构
# =========================
@dataclass
class Context:
    """
    Context: 执行上下文 - 长生命周期对象

    核心修复：
    1. 长生命周期：不是每次重建，而是 get_or_create()
    2. 摘要存储：不存储完整 Artifact，只存摘要
    3. tool_trace 参与 prompt：防止重复调用工具
    4. 压缩策略：三层（裁剪→摘要→语义保留）
    """
    # ====== 核心层 ======
    task: str                      # 原始任务
    step_id: str                  # 当前 step ID
    step_task: str                # 当前子任务描述

    # ====== 执行上下文 ======
    inputs: Dict[str, Any]        # Step.input_data
    dependencies: Dict[str, Any]  # 上游输出（结构化摘要）

    # ====== 状态层 ======
    relevant_artifacts: Dict[str, Any]  # 关键 artifacts 摘要
    memory: Dict[str, Any]          # 长期记忆

    # ====== 推理层 ======
    history: List[Dict]             # LLM 对话历史（压缩后）
    tool_trace: List[Dict]          # 工具调用轨迹

    # ====== 控制层 ======
    budget_tokens: int = 6000       # token 预算
    max_history: int = 10           # 历史消息上限
    max_dep_length: int = 300       # 依赖输出最大长度
    version: int = 1                # 版本号，用于防止使用过期上下文
    _created_at: float = field(default_factory=time.time)  # 【新增】创建时间，用于清理过期 Context

    # ====== 派生方法 ======
    def compress(self):
        """压缩以符合 budget"""
        # 依赖截断
        for k, v in self.dependencies.items():
            if isinstance(v, str) and len(v) > self.max_dep_length:
                self.dependencies[k] = v[:self.max_dep_length] + "..."


@dataclass
class GlobalContext:
    """全局上下文 - 跨任务共享"""
    memory: Dict[str, Any] = field(default_factory=dict)
    reasoning_patterns: List[Dict] = field(default_factory=list)


@dataclass
class TaskContext:
    """任务上下文 - 单个任务内共享"""
    task_id: str
    memory: Dict[str, Any] = field(default_factory=dict)
    step_summaries: List[str] = field(default_factory=list)


class ContextManager:
    """
    ContextManager: Context 生命周期管理器

    关键功能：
    1. get_or_create() - 长生命周期，避免重建丢失 history
    2. _extract_relevant_artifacts() - 只存摘要，避免内存泄漏
    3. 支持三层作用域：Global → Task → Step
    """

    def __init__(self, state: Optional[Any] = None):
        self.state = state
        self.contexts: Dict[str, Context] = {}  # step_id -> Context
        self.global_context = GlobalContext()
        self.task_contexts: Dict[str, TaskContext] = {}
        self.max_dep_length = 300

    def get_or_create(self, step: Any, task_id: Optional[str] = None) -> Context:
        """
        获取或创建 Context - 避免重建丢失 history

        Context 是长生命周期对象，重复获取返回同一对象。
        """
        if step.step_id not in self.contexts:
            self.contexts[step.step_id] = self._build_base_context(step, task_id)
        return self.contexts[step.step_id]

    def _build_base_context(self, step: Any, task_id: Optional[str] = None) -> Context:
        """构建基础上下文（只提取必要信息）"""
        # 【修复】直接从前置步骤的 Context 中获取 relevant_artifacts，
        # 因为 state.artifacts 可能在 Context 创建时尚未填充
        relevant = {}
        for dep_id in getattr(step, 'depends_on', []):
            if dep_id in self.contexts:
                # 从前置步骤的 Context 中获取 relevant_artifacts
                dep_ctx = self.contexts[dep_id]
                relevant.update(dep_ctx.relevant_artifacts)

        # 如果前置步骤的 Context 没有 relevant_artifacts，尝试从 state 获取
        if not relevant:
            relevant = self._extract_relevant_artifacts(step)

        deps = {}
        for dep_id in getattr(step, 'depends_on', []):
            if dep_id in relevant:
                deps[dep_id] = relevant[dep_id]

        inherited_memory = self._get_inherited_memory(task_id)

        # 【新增】继承前置步骤的 history（去重）
        history = []
        seen_messages = set()  # 防止重复消息
        for dep_id in getattr(step, 'depends_on', []):
            if dep_id in self.contexts:
                dep_ctx = self.contexts[dep_id]
                # 继承前置步骤的历史记录，去重
                for msg in dep_ctx.history:
                    msg_key = (msg.get('role'), msg.get('content', ''))
                    if msg_key not in seen_messages:
                        seen_messages.add(msg_key)
                        history.append(msg)

        return Context(
            task=step.input_data.get("task", ""),
            step_id=step.step_id,
            step_task=self._build_step_description(step),
            inputs=step.input_data,
            dependencies=deps,
            relevant_artifacts=relevant,
            memory=inherited_memory,
            history=history,  # 【修改】使用继承的 history
            tool_trace=[],
            budget_tokens=config_manager.LLM_MAX_TOKENS,
            max_history=10,
            max_dep_length=self.max_dep_length,
            _created_at=time.time()  # 【新增】用于清理过期 Context
        )

    def _extract_relevant_artifacts(self, step: Any) -> Dict[str, Any]:
        """提取相关 artifacts 的摘要"""
        if not self.state:
            return {}

        result = {}
        for dep_id in getattr(step, 'depends_on', []):
            artifact = self.state.get_artifact(dep_id)
            if artifact:
                result[dep_id] = {
                    "value": str(artifact.value)[:self.max_dep_length],
                    "type": artifact.type,
                    "success": artifact.is_success()
                }
        return result

    def _get_inherited_memory(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        """获取继承的 memory（Global + Task）"""
        inherited = dict(self.global_context.memory)

        if task_id and task_id in self.task_contexts:
            inherited.update(self.task_contexts[task_id].memory)

        return inherited

    def _build_step_description(self, step: Any) -> str:
        """构建步骤描述"""
        if hasattr(step, 'input_data') and step.input_data:
            return step.input_data.get("task", step.step_id)
        return step.task if hasattr(step, 'task') else step.step_id

    def get_context(self, step_id: str) -> Optional[Context]:
        """获取已保存的上下文"""
        return self.contexts.get(step_id)

    def update_context(self, step_id: str, **kwargs):
        """更新上下文并递增版本号"""
        if step_id in self.contexts:
            ctx = self.contexts[step_id]
            for k, v in kwargs.items():
                if hasattr(ctx, k) and getattr(ctx, k) != v:
                    setattr(ctx, k, v)
                    ctx.version += 1

    # ====== Task 级别操作 ======
    def get_or_create_task_context(self, task_id: str) -> TaskContext:
        """获取或创建任务上下文"""
        if task_id not in self.task_contexts:
            self.task_contexts[task_id] = TaskContext(task_id=task_id)
        return self.task_contexts[task_id]

    def add_task_summary(self, task_id: str, summary: str):
        """添加任务步骤摘要"""
        task_ctx = self.get_or_create_task_context(task_id)
        task_ctx.step_summaries.append(summary)

    # ====== Global 级别操作 ======
    def update_global_memory(self, key: str, value: Any):
        """更新全局 memory"""
        self.global_context.memory[key] = value

    def get_global_memory(self, key: str, default=None) -> Any:
        """获取全局 memory"""
        return self.global_context.memory.get(key, default)

    def cleanup_old_contexts(self, max_age_seconds: int = 3600) -> int:
        """清理过期的 Context（默认 1 小时）"""
        import time
        current_time = time.time()
        to_remove = [
            step_id for step_id, ctx in self.contexts.items()
            if ctx.version > 100 or current_time - getattr(ctx, '_created_at', current_time) > max_age_seconds
        ]
        for step_id in to_remove:
            del self.contexts[step_id]
        return len(to_remove)

    def clear_all_contexts(self) -> int:
        """清除所有 Context（用于任务结束）"""
        count = len(self.contexts)
        self.contexts.clear()
        return count


class ContextFormatter:
    """
    ContextFormatter: Context 格式化器

    功能：将 Context 转换为 LLM 友好的 Prompt

    关键修复：tool_trace 参与 prompt，防止重复调用工具
    """

    @staticmethod
    def format_system(ctx: Context, skill_instruction: str) -> str:
        """格式化系统提示词"""
        dep_text = ContextFormatter._format_dependencies(ctx.dependencies)
        tool_text = ContextFormatter._format_tool_trace(ctx.tool_trace)

        return f"""{skill_instruction}

## 当前任务
{ctx.step_task}

## 任务目标
{ctx.task}

## 上游结果
{dep_text if dep_text else "无"}

## 已执行操作
{tool_text if tool_text else "无"}

## 注意事项
- 基于已有信息推理，不要重复工作
- 每个工具调用后等待结果再继续
- 已执行操作已记录，不要重复调用
"""

    @staticmethod
    def format_user(ctx: Context) -> str:
        """格式化用户提示词"""
        return f"""任务: {ctx.step_task}

输入:
{json.dumps(ctx.inputs, ensure_ascii=False)}
"""

    @staticmethod
    def _format_dependencies(deps: Dict[str, Any], max_length: int = 300) -> str:
        """格式化依赖输出"""
        if not deps:
            return ""

        parts = []
        for k, v in deps.items():
            if isinstance(v, dict):
                content = str(v.get("value", ""))[:max_length]
            else:
                content = str(v)[:max_length]
            parts.append(f"[{k}]\n{content}")
        return "\n\n".join(parts)

    @staticmethod
    def _format_tool_trace(trace: List[Dict], max_items: int = 5) -> str:
        """格式化工具调用轨迹（关键修复：让 LLM 知道已执行操作）"""
        if not trace:
            return "无"

        parts = []
        for t in trace[-max_items:]:  # 只显示最近 N 条
            tool = t.get("tool", "unknown")
            input_data = t.get("input", {})
            output = t.get("output", "")
            summary = str(output)[:100]  # 截断输出
            parts.append(f"- {tool}({input_data}) → {summary}")

        return "\n".join(parts)


class ContextCompressor:
    """Context 压缩器，三层策略"""

    # Level 1: 裁剪
    @staticmethod
    def truncate(text: str, max_length: int) -> str:
        """简单截断"""
        return text[:max_length]

    @staticmethod
    def compress_dependencies(deps: Dict[str, Any], max_length: int = 200) -> Dict[str, str]:
        """压缩依赖输出"""
        result = {}
        for k, v in deps.items():
            if isinstance(v, dict):
                val = str(v.get("value", ""))[:max_length]
            else:
                val = str(v)[:max_length]
            result[k] = val
        return result

    # Level 2: 摘要
    @staticmethod
    def summarize_text(text: str, max_tokens: int = 100) -> str:
        """文本摘要（简单版本）"""
        if len(text) <= max_tokens * 4:  # 1 token ≈ 4 chars
            return text

        # 简单摘要：取首尾各 50%
        half = max_tokens * 2
        return text[:half] + "\n... [摘要中间省略] ...\n" + text[-half:]

    @staticmethod
    def summarize_history(history: List[Dict], keep_last: int = 5) -> List[Dict]:
        """只保留最后几条消息"""
        return history[-keep_last:]

    # Level 3: 语义保留（需要 LLM）
    @staticmethod
    async def summarize_with_llm(llm: AsyncOpenAI, text: str) -> str:
        """使用 LLM 生成语义摘要"""
        try:
            response = await llm.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "请将以下内容压缩成一句话摘要，保留关键信息："},
                    {"role": "user", "content": text[:3000]}
                ],
                temperature=0.3,
                max_tokens=100
            )
            return response.choices[0].message.content
        except Exception:
            return text[:200]  # 失败时简单截断

    # 综合压缩
    @classmethod
    def compress_context(cls, ctx: Context) -> Context:
        """压缩 Context 以符合预算"""
        # 关键修复：只估算一次，避免 str(ctx) 被调用两次
        current_tokens = cls._estimate_context_tokens(ctx)

        if current_tokens > ctx.budget_tokens:
            # Level 1: 裁剪依赖
            if len(ctx.dependencies) > 0:
                ctx.dependencies = cls.compress_dependencies(ctx.dependencies, 200)

            # Level 2: 裁剪历史
            if len(ctx.history) > ctx.max_history:
                ctx.history = cls.summarize_history(ctx.history, ctx.max_history)

            # Level 3: 摘要（递归压缩直到符合预算）
            if cls._estimate_context_tokens(ctx) > ctx.budget_tokens:
                ctx.relevant_artifacts = {
                    k: cls.summarize_text(str(v), 100)
                    for k, v in ctx.relevant_artifacts.items()
                }

        return ctx

    @staticmethod
    def estimate_tokens(prompt: str) -> int:
        """简单估算 token 数（1 token ≈ 4 字符）"""
        return len(prompt) // 4

    @staticmethod
    def _estimate_context_tokens(ctx: Context) -> int:
        """优化的 Context token 估算，避免完整序列化"""
        # 只估算关键字段，避免 str(ctx) 的性能问题
        text = (
            ctx.task[:500] +
            ctx.step_task[:200] +
            str(ctx.dependencies)[:300] +
            str(ctx.history)[:200] +
            str(ctx.tool_trace)[:100]
        )
        return len(text) // 4


# =========================
# 事件类型枚举
# =========================
class EventType(str, Enum):
    TASK_SUBMITTED = "task_submitted"
    STEP_READY = "step_ready"
    STEP_CLAIMED = "step_claimed"
    STEP_COMPLETED = "step_completed"
    STEP_FAILED = "step_failed"
    STEP_CANCELLED = "step_cancelled"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    SKILL_PROGRESS = "skill_progress"


# =========================
# 核心事件模型
# =========================
@dataclass(frozen=True)
class Event:
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""
    step_id: Optional[str] = None
    event_type: EventType = EventType.STEP_READY
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


# =========================
# 事件订阅管理器（集中管理事件订阅）
# =========================
class EventSubscriptionManager:
    """集中管理事件订阅，防止重复订阅和内存泄漏"""

    def __init__(self, bus: "EventBus"):
        self._bus = bus
        self._subscriptions: Dict[str, Dict[str, int]] = {}
        self._id_to_handler: Dict[int, Tuple[str, str]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(
        self,
        component: str,
        name: str,
        handler: Callable,
        event_filter: Optional[EventType] = None
    ) -> int:
        async with self._lock:
            # 确保移除旧订阅
            if component in self._subscriptions and name in self._subscriptions.get(component, {}):
                old_id = self._subscriptions[component][name]
                self._bus.unsubscribe(old_id)
                self._id_to_handler.pop(old_id, None)

            async def wrapped_handler(event: Event):
                # 【调试】记录每次调用
                logger.debug(f"[EventSubscriptionManager.wrapped_handler] {component}.{name} called for event {event.event_type} (filter={event_filter})")
                if event_filter and event.event_type != event_filter:
                    logger.debug(f"[EventSubscriptionManager.wrapped_handler] Skipping - event type mismatch")
                    return
                try:
                    await handler(event)
                except Exception as e:
                    logger.exception(f"[{component}.{name}] Handler failed: {e}")

            # 明确转换为整数，确保类型一致
            sid = int(self._bus.subscribe(wrapped_handler))

            if component not in self._subscriptions:
                self._subscriptions[component] = {}
            self._subscriptions[component][name] = sid
            self._id_to_handler[sid] = (component, name)

            logger.info(f"[EventSubscriptionManager] {component}.{name} subscribed (id={sid}), filter={event_filter}")
            return sid

    async def unsubscribe_component(self, component: str):
        async with self._lock:
            if component not in self._subscriptions:
                return
            handlers = list(self._subscriptions[component].items())
            for name, sid in handlers:
                self._bus.unsubscribe(sid)
                if sid in self._id_to_handler:
                    del self._id_to_handler[sid]
                logger.debug(f"[EventSubscriptionManager] {component}.{name} unsubscribed")
            del self._subscriptions[component]

    async def unsubscribe_all(self):
        async with self._lock:
            components = list(self._subscriptions.keys())
            for component in components:
                handlers = list(self._subscriptions[component].items())
                for name, sid in handlers:
                    self._bus.unsubscribe(sid)
                    if sid in self._id_to_handler:
                        del self._id_to_handler[sid]
                    logger.debug(f"[EventSubscriptionManager] {component}.{name} unsubscribed")
            self._subscriptions.clear()

    def get_subscription_info(self) -> Dict[str, List[str]]:
        return {
            component: list(handlers.keys())
            for component, handlers in self._subscriptions.items()
        }


# =========================
# 事件总线
# =========================
class EventBus:
    def __init__(self, max_queue_size: int = None):
        if max_queue_size is None:
            max_queue_size = config_manager.EVENT_BUS_MAX_QUEUE_SIZE
        self._subscribers: Dict[int, Callable] = {}
        self._next_subscription_id = 0
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._dead_letter_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._dead_letter_enabled = config_manager.DEAD_LETTER_QUEUE_ENABLED
        self._dead_letter_flush_task = None
        self._worker_task = asyncio.create_task(self._process_queue())
        self._stop_event = asyncio.Event()
        self._max_queue_size = max_queue_size

        if self._dead_letter_enabled:
            self._dead_letter_flush_task = asyncio.create_task(self._flush_dead_letter_loop())

    def subscribe(self, handler: Callable) -> int:
        subscription_id = self._next_subscription_id
        self._subscribers[subscription_id] = handler
        self._next_subscription_id += 1
        return subscription_id

    def unsubscribe(self, subscription_id: int) -> bool:
        if subscription_id in self._subscribers:
            del self._subscribers[subscription_id]
            return True
        return False

    async def publish(self, event: Event, block: bool = True, timeout: Optional[float] = None) -> bool:
        try:
            if block:
                if timeout is None:
                    await self._queue.put(event)
                else:
                    await asyncio.wait_for(self._queue.put(event), timeout=timeout)
            else:
                self._queue.put_nowait(event)
            return True
        except asyncio.QueueFull:
            if self._dead_letter_enabled:
                try:
                    await self._dead_letter_queue.put((event, "queue_full"))
                except asyncio.QueueFull:
                    pass
            return False
        except asyncio.TimeoutError:
            return False

    async def _flush_dead_letter_loop(self):
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(5.0)
                while not self._dead_letter_queue.empty():
                    try:
                        item = await asyncio.wait_for(self._dead_letter_queue.get(), timeout=1.0)
                        event, reason = item if isinstance(item, tuple) else (item, "unknown")
                        logger.error(f"[DeadLetter] Event {event.event_id} dropped ({reason}): {event}")
                    except asyncio.TimeoutError:
                        break
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("[DeadLetter] Flush loop error")

    async def _process_queue(self):
        while not self._stop_event.is_set():
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=config_manager.QUEUE_TIMEOUT)
            except asyncio.TimeoutError:
                continue
            if event is None:
                break

            for handler in list(self._subscribers.values()):
                if handler is None:
                    continue
                try:
                    # 支持同步和异步处理器
                    handler_name = handler.__name__ if hasattr(handler, '__name__') else str(type(handler))
                    result = handler(event)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.exception(f"Handler {handler_name} failed: {e}")
            self._queue.task_done()

    async def shutdown(self):
        logger.info("[EventBus] Shutdown initiated...")
        self._stop_event.set()

        # 等待队列处理完成
        try:
            await asyncio.wait_for(self._queue.join(), timeout=5.0)
            logger.info("[EventBus] Main queue drained")
        except asyncio.TimeoutError:
            logger.warning("[EventBus] Main queue drain timeout, forcing shutdown")

        # 清除订阅者（先于 worker 停止，避免新事件）
        self._subscribers.clear()

        if self._dead_letter_enabled and self._dead_letter_flush_task:
            try:
                await asyncio.sleep(1.0)
                self._dead_letter_flush_task.cancel()
                try:
                    await self._dead_letter_flush_task
                except asyncio.CancelledError:
                    pass
            except Exception:
                logger.exception("[EventBus] Error shutting down dead_letter flush task")

        # 处理死信队列
        try:
            if not self._dead_letter_queue.empty():
                count = 0
                while not self._dead_letter_queue.empty():
                    try:
                        item = self._dead_letter_queue.get_nowait()
                        event, reason = item if isinstance(item, tuple) else (item, "unknown")
                        logger.error(f"[DeadLetter] Unprocessed event {event.event_id} ({reason})")
                        count += 1
                    except asyncio.QueueEmpty:
                        break
                if count > 0:
                    logger.warning(f"[EventBus] {count} events lost in dead_letter queue")
        except Exception:
            pass

        # 关闭 worker task
        try:
            await asyncio.wait_for(self._worker_task, timeout=2.0)
            logger.info("[EventBus] Worker task completed")
        except asyncio.TimeoutError:
            logger.warning("[EventBus] Worker task timeout, cancelling...")
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("[EventBus] Shutdown completed")


# =========================
# 能力抽象基类
# =========================
class Capability(ABC):
    """能力基类：统一命名和元数据"""
    @property
    @abstractmethod
    def name(self) -> str: pass

    @property
    @abstractmethod
    def schema(self) -> Dict[str, Any]: pass

    @abstractmethod
    async def execute(self, **kwargs) -> Any: pass


class ExecutableCapability(Capability):
    """原子工具：直接执行，无额外方法"""
    pass


class InstructableCapability(Capability):
    """可指令型能力：提供使用规范"""
    @abstractmethod
    def get_system_instruction(self) -> str: pass

    def get_examples(self) -> List[Dict[str, Any]]:
        return []


class AgenticCapability(InstructableCapability):
    """智能技能能力：兼具指令规范和专属工具箱"""

    def get_skill_tools(self) -> List['ExecutableCapability']:
        """返回该技能专属的临时工具（子工具）"""
        return []

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        pass


# =========================
# Capability 注册中心（统一管理 Tool 和 Skill）
# =========================
class CapabilityRegistry:
    """统一的能力注册中心，支持按类型过滤"""

    def __init__(self):
        self._capabilities: Dict[str, Capability] = {}

    def register(self, capability: Capability):
        """注册能力"""
        self._capabilities[capability.name] = capability

    def get(self, name: str) -> Optional[Capability]:
        """获取能力"""
        return self._capabilities.get(name)

    def get_all_names(self) -> List[str]:
        """获取所有能力名称"""
        return list(self._capabilities.keys())

    def get_executable_schemas(self) -> List[Dict]:
        """获取所有可执行工具的 schemas"""
        executables = [c for c in self._capabilities.values()
                       if isinstance(c, ExecutableCapability)]
        schemas = [c.schema for c in executables]
        return schemas

    def get_instructable_schemas(self) -> List[Dict]:
        """获取所有可指令型能力的 schemas"""
        return [c.schema for c in self._capabilities.values()
                if isinstance(c, InstructableCapability)]

    async def execute(self, name: str, args: Dict) -> Any:
        """执行能力"""
        cap = self.get(name)
        if not cap:
            raise ValueError(f"Capability {name} not found")
        return await cap.execute(**args)


# =========================
# ToolRegistry（Schema 驱动的工具注册中心）
# =========================
class ToolRegistry:
    """支持 Schema 的工具注册中心"""

    def __init__(self, tools_dir: str = "tools"):
        self.tools_dir = tools_dir
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._tool_instances: Dict[str, Any] = {}
        self._stats: Dict[str, Dict[str, int]] = {}

    def load_tools_from_directory(self, tools_dir: str = None) -> Dict[str, Any]:
        """从指定目录加载所有工具模块"""
        loaded_tools = {}
        load_dir = tools_dir or self.tools_dir
        if not os.path.exists(load_dir):
            return loaded_tools
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            script_dir = os.getcwd()
        tools_path = os.path.join(script_dir, load_dir)
        if not os.path.exists(tools_path):
            return loaded_tools
        if tools_path not in sys.path:
            sys.path.insert(0, tools_path)
        try:
            import tools
            loaded_tools['tools_package'] = tools
            if hasattr(tools, '__all__'):
                for tool_name in tools.__all__:
                    if hasattr(tools, tool_name):
                        loaded_tools[tool_name] = getattr(tools, tool_name)
        except Exception as e:
            logger.warning(f"Failed to load tools from directory: {e}")
        return loaded_tools

    def register_tool_instance(self, name: str, instance: Any, schema: Dict[str, Any], description: str = ""):
        """注册工具实例

        支持两种 schema 格式：
        1. 完整 OpenAI 格式: {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
        2. 简洁格式: {"parameters": {...}} 或直接是 parameters 对象
        """
        # 提取 parameters，支持两种格式
        if "function" in schema and "parameters" in schema["function"]:
            # 完整 OpenAI 格式
            parameters = schema["function"]["parameters"]
            desc = schema["function"].get("description", description)
        elif "parameters" in schema:
            # 简洁格式
            parameters = schema["parameters"]
            desc = description
        else:
            # 默认空参数
            parameters = {"type": "object", "properties": {}}
            desc = description

        self._tools[name] = {
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": parameters
            }
        }
        self._tool_instances[name] = instance
        self._stats[name] = {"call_count": 0, "success_count": 0, "error_count": 0}

    def get_tool_schema(self, name: str) -> Optional[Dict]:
        return self._tools.get(name)

    def get_tool_instance(self, name: str) -> Optional[Any]:
        return self._tool_instances.get(name)

    def get_stats(self) -> Dict[str, Dict[str, int]]:
        return self._stats.copy()


# =========================
# 工具执行隔离（进程池）
# =========================
class ToolExecutorPool:
    """工具执行器池，使用进程池隔离可能阻塞的工具"""

    def __init__(self, max_workers: int = 2):
        from concurrent.futures import ProcessPoolExecutor
        self._executor = ProcessPoolExecutor(max_workers=max_workers)
        self._max_workers = max_workers
        logger.info(f"[ToolExecutorPool] Initialized with {max_workers} workers")

    async def execute_async(self, func: Callable, *args, timeout: float = 30.0, **kwargs) -> Any:
        """异步执行函数（通过进程池隔离）"""
        loop = asyncio.get_event_loop()
        try:
            # 使用 functools.partial 避免 lambda 闭包问题
            bound_func = functools.partial(func, *args, **kwargs)
            result = await asyncio.wait_for(
                loop.run_in_executor(self._executor, bound_func),
                timeout=timeout
            )
            return result
        except asyncio.TimeoutError:
            logger.warning(f"[ToolExecutorPool] Execution timeout after {timeout}s")
            return f"Error: Execution timeout (>{timeout}s)"
        except Exception as e:
            return f"Error: {e}"

    async def shutdown(self):
        """关闭执行器池"""
        logger.info("[ToolExecutorPool] Shutting down...")
        self._executor.shutdown(wait=True)
        logger.info("[ToolExecutorPool] Shutdown completed")


# =========================
# 工具执行异常类
# =========================
@dataclass
class ToolExecutionResult:
    """工具执行结果"""
    success: bool
    output: Any
    error: Optional[str] = None
    duration_ms: int = 0


# =========================
# ToolCapability（工具封装）
# =========================
class ToolCapability(ExecutableCapability):
    """将现有工具封装为 Capability"""

    def __init__(
        self,
        name: str,
        registry: ToolRegistry,
        method_name: str = None,
        executor_pool: Optional[ToolExecutorPool] = None,
        timeout: Optional[float] = None
    ):
        self._name = name
        self._registry = registry
        self._method_name = method_name or name
        self._executor_pool = executor_pool
        # 从配置获取超时时间，默认30秒
        self._timeout = timeout if timeout is not None else config_manager.LLM_CALL_TIMEOUT

    @property
    def name(self) -> str:
        return self._name

    @property
    def schema(self) -> Dict[str, Any]:
        return self._registry.get_tool_schema(self._name) or {"type": "tool", "name": self._name}

    async def execute(self, **kwargs) -> Any:
        instance = self._registry.get_tool_instance(self._name)
        if not instance:
            return f"Error: Tool '{self._name}' instance not found"

        async def _execute_with_timeout():
            try:
                result = instance.execute(**kwargs)
                if asyncio.iscoroutine(result):
                    return await result
                return result
            except Exception as e:
                return f"Error executing {self._name}: {e}"

        try:
            # 使用可配置的超时时间
            # 注意：不使用 executor_pool，因为工具实例包含不可序列化的对象（如锁）
            # 进程池只适用于纯函数，不适用于带状态的工具实例
            return await asyncio.wait_for(_execute_with_timeout(), timeout=self._timeout)
        except asyncio.TimeoutError:
            logger.warning(f"[ToolCapability] Tool {self._name} timeout after {self._timeout}s")
            return f"Error: Tool '{self._name}' execution timeout (>{self._timeout}s), please try a simpler approach"

    def execute_sync(self, **kwargs) -> Any:
        """同步执行工具（供进程池调用）"""
        instance = self._registry.get_tool_instance(self._name)
        if not instance:
            return f"Error: Tool '{self._name}' instance not found"
        try:
            result = instance.execute(**kwargs)
            if asyncio.iscoroutine(result):
                return f"Error: Tool '{self._name}' requires async execution"
            return result
        except Exception as e:
            return f"Error executing {self._name}: {e}"


# =========================
# SkillCapability（技能封装）- 渐进披露三层架构
# =========================
class SkillCapability(AgenticCapability):
    """
    将技能封装为 Capability，符合渐进披露的三层架构

    架构原则：
    - Layer 1 (Metadata): 启动时加载（name, description）
    - Layer 2 (Instructions): 首次 get_system_instruction() 时按需加载
    - Layer 3 (Resources): 显式调用时加载（references/, scripts/）

    目录结构：
    skills/
    └── my-skill/
        ├── SKILL.md        # Layer 2: 指令（必需）
        ├── references/     # Layer 3: 详细文档（可选）
        └── scripts/        # Layer 3: 可执行脚本（可选）
    """

    def __init__(self, name: str, description: str, markdown_path: str, skill_dir: str = None):
        self._name = name
        self.description = description  # 公开属性，用于访问
        self._description = description
        self._markdown_path = markdown_path
        self._skill_dir = skill_dir or os.path.dirname(markdown_path)
        self._instruction_cache: Optional[str] = None
        self._resources_cache: Dict[str, str] = {}

    @property
    def name(self) -> str:
        """Layer 1: 元数据，启动时加载"""
        return self._name

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "skill",
            "name": self._name,
            "description": self._description
        }

    def get_system_instruction(self) -> str:
        """Layer 2: 按需加载完整指令（同步版本）"""
        if self._instruction_cache is None:
            try:
                with open(self._markdown_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                if content.startswith('---'):
                    parts = content.split('---', 2)
                    if len(parts) >= 3:
                        content = parts[2].strip()
                self._instruction_cache = content
            except Exception as e:
                logger.error(f"Failed to load skill instruction from {self._markdown_path}: {e}")
                self._instruction_cache = f"Error loading skill: {e}"
        return self._instruction_cache

    async def get_system_instruction_async(self) -> str:
        """Layer 2: 按需加载完整指令（异步版本）"""
        if self._instruction_cache is None:
            try:
                content = await asyncio.to_thread(self._read_markdown_sync)
                self._instruction_cache = content
            except Exception as e:
                logger.error(f"Failed to load skill instruction from {self._markdown_path}: {e}")
                self._instruction_cache = f"Error loading skill: {e}"
        return self._instruction_cache

    def _read_markdown_sync(self) -> str:
        """同步读取 markdown 文件（供 asyncio.to_thread 调用）"""
        with open(self._markdown_path, 'r', encoding='utf-8') as f:
            content = f.read()
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                content = parts[2].strip()
        return content

    def load_reference(self, ref_name: str) -> Optional[str]:
        """
        Layer 3: 加载引用文件（按需加载，同步版本）

        注意：SKILL.md 是技能的主文档，不应该通过 load_reference 加载。
        SKILL.md 应该在技能初始化时通过 get_system_instruction() 加载。
        """
        # SKILL.md 是主文档，不通过 references 加载
        if ref_name == "SKILL.md":
            return None

        ref_path = os.path.join(self._skill_dir, "references", ref_name)
        if ref_name in self._resources_cache:
            return self._resources_cache[ref_name]
        try:
            if os.path.exists(ref_path):
                with open(ref_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                self._resources_cache[ref_name] = content
                return content
        except Exception as e:
            logger.warning(f"Failed to load reference {ref_name}: {e}")
        return None

    async def load_reference_async(self, ref_name: str) -> Optional[str]:
        """
        Layer 3: 加载引用文件（按需加载，异步版本）

        注意：SKILL.md 是技能的主文档，不应该通过 load_reference 加载。
        SKILL.md 应该在技能初始化时通过 get_system_instruction() 加载。
        """
        # SKILL.md 是主文档，不通过 references 加载
        if ref_name == "SKILL.md":
            return None

        ref_path = os.path.join(self._skill_dir, "references", ref_name)
        if ref_name in self._resources_cache:
            return self._resources_cache[ref_name]
        try:
            content = await asyncio.to_thread(self._read_reference_sync, ref_path)
            self._resources_cache[ref_name] = content
            return content
        except Exception as e:
            logger.warning(f"Failed to load reference {ref_name}: {e}")
        return None

    def _read_reference_sync(self, ref_path: str) -> str:
        """同步读取参考文件（供 asyncio.to_thread 调用）"""
        with open(ref_path, 'r', encoding='utf-8') as f:
            return f.read()

    def get_script_path(self, script_name: str) -> Optional[str]:
        """
        Layer 3: 获取脚本文件路径（不加载到上下文）
        返回路径供 Bash 工具调用
        """
        script_path = os.path.join(self._skill_dir, "scripts", script_name)
        if os.path.exists(script_path):
            return script_path
        return None

    def list_resources(self) -> List[str]:
        """列出技能目录下的所有资源文件（同步版本）"""
        resources = []
        if os.path.exists(self._skill_dir):
            for item in os.listdir(self._skill_dir):
                item_path = os.path.join(self._skill_dir, item)
                if os.path.isfile(item_path) and item.endswith('.md'):
                    resources.append(item)
        return resources

    async def list_resources_async(self) -> List[str]:
        """列出技能目录下的所有资源文件（异步版本）"""
        items = await asyncio.to_thread(os.listdir, self._skill_dir)
        resources = []
        for item in items:
            item_path = os.path.join(self._skill_dir, item)
            if os.path.isfile(item_path) and item.endswith('.md'):
                resources.append(item)
        return resources

    async def execute(self, **kwargs: Any) -> Any:
        """执行技能（kwargs 保留用于未来扩展）"""
        _ = kwargs  # 保留参数用于未来扩展
        raise NotImplementedError("Skill should not execute code directly - use SkillPolicy with LLMRuntime")


# =========================
# SkillPolicy（技能策略层）
# =========================
class SkillPolicy:
    """
    SkillPolicy: 技能策略层（不是执行器）

    职责：
    1. 提供 system prompt 模板
    2. 提供可用 tools 列表
    3. 定义 reasoning style
    4. 使用 ContextManager 和 ContextFormatter（关键修复）
    """

    def __init__(self, skill: SkillCapability, llm_runtime: "LLMRuntime", context_manager: Optional[Any] = None):
        self.skill = skill
        self.llm_runtime = llm_runtime
        self.context_manager = context_manager

    async def get_system_prompt(self, step: "Step", context: Optional[Any] = None) -> str:
        """获取系统提示词（使用 ContextFormatter）"""
        # 关键修复：使用 ContextFormatter 格式化系统提示词
        if context:
            return ContextFormatter.format_system(context, self.skill.get_system_instruction())

        # 兼容旧逻辑：如果没传 context，直接构建 prompt（不创建 TempContext）
        # 移除了 TempContext 临时类，直接构建依赖上下文
        deps = {}
        for dep_id in getattr(step, 'depends_on', []):
            artifact = getattr(step, 'input_data', {}).get(f"_dep_{dep_id}")
            if artifact is not None:
                deps[dep_id] = {"value": str(artifact)[:300], "type": "text", "success": True}

        # 构建简单的依赖文本
        dep_text = ""
        if deps:
            parts = []
            for k, v in deps.items():
                content = str(v.get("value", ""))[:300]
                parts.append(f"[{k}]\n{content}")
            dep_text = "\n\n".join(parts)

        tool_text = "无"

        task = step.input_data.get("task", "") if hasattr(step, 'input_data') else step.step_id
        step_task = step.input_data.get("task", "") if hasattr(step, 'input_data') else step.step_id

        skill_instruction = self.skill.get_system_instruction()

        return f"""{skill_instruction}

## 当前任务
{step_task}

## 任务目标
{task}

## 上游结果
{dep_text if dep_text else "无"}

## 已执行操作
{tool_text}

## 注意事项
- 基于已有信息推理，不要重复工作
- 每个工具调用后等待结果再继续
- 已执行操作已记录，不要重复调用
"""

    async def get_user_prompt(self, step: "Step", context: Optional[Any] = None) -> str:
        """获取用户提示词"""
        if context:
            return ContextFormatter.format_user(context)

        # 兼容旧逻辑
        task = step.input_data.get("task", "")
        input_data = {k: v for k, v in step.input_data.items() if k != "task"}
        prompt = f"任务: {task}"
        if input_data:
            prompt += f"\n输入数据: {json.dumps(input_data, ensure_ascii=False)}"
        return prompt

    async def execute_with_policy(self, step: "Step", context: Optional[Any] = None, caller: str = "unknown") -> Tuple[bool, Any]:
        """使用技能策略执行任务（通过 LLMRuntime 的 tool_call）"""
        # 关键修复：使用 get_or_create 获取 context
        if context is None and self.context_manager:
            context = self.context_manager.get_or_create(step)

        # 记录上下文信息（在生成 prompt 前）
        if context:
            context_info = {
                "step_id": getattr(context, 'step_id', 'unknown'),
                "step_task": getattr(context, 'step_task', 'unknown'),
                "dependencies": list(getattr(context, 'dependencies', {}).keys()),
                "relevant_artifacts": list(getattr(context, 'relevant_artifacts', {}).keys()),
                "history_length": len(getattr(context, 'history', [])),
                "tool_trace_length": len(getattr(context, 'tool_trace', []))
            }
            logger.info(f"[SkillPolicy] Context info: {json.dumps(context_info, ensure_ascii=False, indent=2)}")

        # 使用 ContextFormatter 生成 prompt
        system_prompt = await self.get_system_prompt(step, context)
        user_prompt = await self.get_user_prompt(step, context)
        tools = await self._build_tools_list()

        # 记录 prompt 摘要
        logger.info(f"[SkillPolicy] Prompt info: system_len={len(system_prompt)}, user_len={len(user_prompt)}, tools_count={len(tools)}")

        success, result = await self.llm_runtime.tool_call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tools=tools,
            history=None,
            context=context,  # 传递 context 用于追踪
            skill_policy=self,
            caller=f"SkillPolicy({self.skill.name})-{caller}"
        )

        # 关键修复：更新 context 的历史记录
        if context and isinstance(result, dict) and "history" in result:
            context.history.extend(result["history"])

        return (success, result)

    async def _build_tools_list(self) -> List[Dict]:
        """构建工具列表（基础工具 + Skill专属工具）"""
        base_tools = self.llm_runtime.capability_registry.get_executable_schemas()

        # 获取 Skill 专属工具
        skill_tools_cap = self.skill.get_skill_tools()
        skill_tool_schemas = []
        for tool_cap in skill_tools_cap:
            if isinstance(tool_cap, ExecutableCapability):
                skill_tool_schemas.append(tool_cap.schema)

        # Skill 专属管理工具
        skill_admin_tools = [
            {
                "type": "function",
                "function": {
                    "name": "load_reference",
                    "description": "Load a reference document from skill's references/ directory. Use this to get additional context when writing code. If no reference is found, use your own knowledge.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ref_name": {"type": "string", "description": "Name of the reference file"}
                        },
                        "required": ["ref_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_resources",
                    "description": "List all available resources in the skill directory",
                    "parameters": {"type": "object", "properties": {}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_script",
                    "description": "Execute a script from skill's scripts/ directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "script_name": {"type": "string", "description": "Name of the script to execute"},
                            "args": {"type": "array", "items": {"type": "string"}, "description": "Arguments", "default": []}
                        },
                        "required": ["script_name"]
                    }
                }
            }
        ]

        return base_tools + skill_tool_schemas + skill_admin_tools

    async def _execute_script_async(self, script_name: str, args: List[str]) -> str:
        """异步执行脚本"""
        script_path = self.skill.get_script_path(script_name)
        if not script_path:
            return f"Error: Script '{script_name}' not found"
        if not os.path.exists(script_path):
            return f"Error: Script '{script_name}' does not exist at {script_path}"
        try:
            import subprocess
            cmd = ["bash", script_path] + args
            result = await asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout
            return f"Error: {result.stderr}"
        except Exception as e:
            return f"Error executing script: {e}"


# =========================
# 辅助函数：提取实际输出
# =========================
def extract_output(result: Any) -> str:
    """统一提取实际输出字符串"""
    if isinstance(result, dict):
        if "output" in result:
            return str(result["output"])
        elif "error" in result:
            return f"Error: {result['error']}"
        return json.dumps(result, ensure_ascii=False)
    return str(result)


# =========================
# LLMRuntime - 统一大脑
# =========================
class LLMRuntime:
    """
    LLMRuntime: 唯一的智能入口，统一处理所有 LLM 调用

    关键修复：
    1. 接受 context_manager 参数，支持 Context 长生命周期
    2. tool_call 支持 context 参数，记录 tool_trace 和 history
    3. 实现 compress_history 方法，三层压缩策略
    """

    def __init__(
        self,
        llm: AsyncOpenAI,
        capability_registry: CapabilityRegistry,
        max_iterations: int = 10,
        context_manager: Optional[Any] = None  # 新增
    ):
        self.llm = llm
        self.capability_registry = capability_registry
        self.max_iterations = max_iterations
        self.context_manager = context_manager  # 新增
        logger.info("[LLMRuntime] Created - unified LLM brain")

    async def call(self, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None) -> str:
        try:
            params = {
                "model": MODEL_NAME,
                "messages": messages,
                "temperature": config_manager.LLM_TEMPERATURE,
                "max_tokens": config_manager.LLM_MAX_TOKENS
            }
            if tools:
                params["tools"] = tools
                params["tool_choice"] = "auto"
            response = await self.llm.chat.completions.create(**params)
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"[LLMRuntime] Call failed: {e}")
            return f"Error: {e}"

    async def reason(self, system_prompt: str, user_prompt: str, caller: str = "unknown") -> str:
        """基础推理 - 无工具调用的对话"""
        logger.info(f"[LLMRuntime.reason] Called by {caller}, prompt length: system={len(system_prompt)}, user={len(user_prompt)}")
        logger.debug(f"[LLMRuntime.reason] System prompt: {system_prompt[:500]}...")
        logger.debug(f"[LLMRuntime.reason] User prompt: {user_prompt[:500]}...")
        try:
            logger.info(f"[LLMRuntime.reason] About to call LLM API...")
            # 添加超时保护
            response = await asyncio.wait_for(
                self.llm.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=config_manager.LLM_TEMPERATURE,
                    max_tokens=config_manager.LLM_MAX_TOKENS
                ),
                timeout=config_manager.LLM_CALL_TIMEOUT
            )
            logger.info(f"[LLMRuntime.reason] LLM API call completed, processing response...")
            result = response.choices[0].message.content
            logger.info(f"[LLMRuntime.reason] Completed by {caller}, response length: {len(result)}")
            logger.debug(f"[LLMRuntime.reason] Response: {result[:500]}...")
            return result
        except asyncio.TimeoutError:
            logger.error(f"[LLMRuntime.reason] Timeout after {config_manager.LLM_CALL_TIMEOUT}s by {caller}")
            raise TimeoutError(f"Reasoning timeout after {config_manager.LLM_CALL_TIMEOUT}s")
        except Exception as e:
            logger.error(f"[LLMRuntime] Reason failed: {e}")
            raise

    async def tool_call(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: List[Dict],
        history: Optional[List[Dict]] = None,
        context: Optional[Any] = None,  # 新增：Context 对象
        max_iterations: Optional[int] = None,
        skill_policy: Optional[Any] = None,
        caller: str = "unknown"
    ) -> Tuple[bool, Dict]:
        """工具调用 - ReAct 循环，带早期停止和去重机制"""
        # 记录上下文信息（如果传入了 context）
        if context:
            context_info = {
                "step_id": getattr(context, 'step_id', 'unknown'),
                "step_task": getattr(context, 'step_task', 'unknown'),
                "dependencies": list(getattr(context, 'dependencies', {}).keys()),
                "relevant_artifacts": list(getattr(context, 'relevant_artifacts', {}).keys()),
                "history_length": len(getattr(context, 'history', [])),
                "tool_trace_length": len(getattr(context, 'tool_trace', []))
            }
            logger.info(f"[LLMRuntime.tool_call] Context from {caller}: {json.dumps(context_info, ensure_ascii=False, indent=2)}")

        logger.info(f"[LLMRuntime.tool_call] Called by {caller}, tools_count={len(tools)}, prompt lengths: system={len(system_prompt)}, user={len(user_prompt)}")
        logger.debug(f"[LLMRuntime.tool_call] System prompt: {system_prompt[:500]}...")
        logger.debug(f"[LLMRuntime.tool_call] User prompt: {user_prompt[:500]}...")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # 关键修复：使用 context 的历史记录（覆盖传入的 history）
        if context and context.history:
            messages.extend(context.history[-context.max_history:])
        elif history:
            messages.extend(history)

        total_len = sum(len(m.get('content', '')) for m in messages)
        logger.debug(f"[LLMRuntime.tool_call] Messages (with history): {len(messages)} messages, total length: {total_len}")
        # 记录每条消息的简要信息
        for i, m in enumerate(messages):
            role = m.get('role', 'unknown')
            content_len = len(m.get('content', ''))
            tool_calls = len(m.get('tool_calls', []))
            logger.debug(f"[LLMRuntime.tool_call] Message {i}: role={role}, content_len={content_len}, tool_calls={tool_calls}")
            if content_len > 0 and content_len <= 300:
                logger.debug(f"[LLMRuntime.tool_call] Message {i} content: {m.get('content', '')[:300]}")

        # 动态调整迭代上限
        max_iter = max_iterations if max_iterations is not None else self.max_iterations

        # 工具调用跟踪：防止重复调用相同工具
        tool_call_history: List[Tuple[str, Tuple]] = []  # (tool_name, normalized_args)
        consecutive_same_tool = 0
        last_tool_name = None

        for iteration in range(max_iter):
            try:
                logger.info(f"[LLMRuntime.tool_call] Iteration {iteration + 1}/{max_iter} by {caller}")
                logger.debug(f"[LLMRuntime.tool_call] Iteration {iteration + 1} messages count: {len(messages)}")
                # 记录最后几条消息用于调试
                if len(messages) > 2:
                    last_msg = messages[-1]
                    logger.debug(f"[LLMRuntime.tool_call] Last message role: {last_msg.get('role')}, content length: {len(last_msg.get('content', ''))}")
                # 记录完整请求用于调试
                logger.debug(f"[LLMRuntime.tool_call] Iteration {iteration + 1} request: model={MODEL_NAME}, temp={config_manager.LLM_TEMPERATURE}, max_tokens={config_manager.LLM_MAX_TOKENS}")
                # 添加超时保护
                response = await asyncio.wait_for(
                    self.llm.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        tools=tools,
                        tool_choice="auto",
                        temperature=config_manager.LLM_TEMPERATURE,
                        max_tokens=config_manager.LLM_MAX_TOKENS
                    ),
                    timeout=config_manager.LLM_CALL_TIMEOUT
                )
                # 记录响应内容
                response_content = response.choices[0].message.content or ""
                response_tool_calls = len(response.choices[0].message.tool_calls or [])
                logger.info(f"[LLMRuntime.tool_call] Iteration {iteration + 1} HTTP response: OK, tool_calls={response_tool_calls}, content_length={len(response_content)}")
                if response_content:
                    logger.debug(f"[LLMRuntime.tool_call] Iteration {iteration + 1} response_content: {response_content[:500]}...")
                if response_tool_calls > 0:
                    for i, tc in enumerate(response.choices[0].message.tool_calls or []):
                        logger.debug(f"[LLMRuntime.tool_call] Iteration {iteration + 1} tool_call {i}: {tc.function.name}, args={tc.function.arguments[:300]}...")
                # 记录完整的 messages 用于调试（查看历史）
                logger.debug(f"[LLMRuntime.tool_call] Iteration {iteration + 1} complete messages history:")
                for i, m in enumerate(messages):
                    role = m.get('role', 'unknown')
                    content = m.get('content', '')
                    tool_calls = m.get('tool_calls', [])
                    if content:
                        logger.debug(f"[LLMRuntime.tool_call]   Message {i} [{role}]: {content[:400]}...")
                    if tool_calls:
                        logger.debug(f"[LLMRuntime.tool_call]   Message {i} [{role}] has {len(tool_calls)} tool_calls")
            except asyncio.TimeoutError:
                logger.error(f"[LLMRuntime.tool_call] Iteration {iteration + 1} timeout after {config_manager.LLM_CALL_TIMEOUT}s")
                return (False, {"error": f"LLM call timeout after {config_manager.LLM_CALL_TIMEOUT}s"})
            except Exception as e:
                return (False, {"error": f"LLM call failed: {e}"})

            msg = response.choices[0].message
            logger.debug(f"[LLMRuntime.tool_call] Iteration {iteration + 1} response: tool_calls={len(msg.tool_calls)}, content={msg.content[:200] if msg.content else 'None'}...")

            if not msg.tool_calls:
                # 没有工具调用，返回内容
                output = msg.content or ""

                logger.debug(f"[LLMRuntime.tool_call] No tool calls, output: {output[:500]}...")

                # 关键修复：如果传入 context，记录 tool_trace（空）和 history
                if context:
                    context.history.extend(messages[2:])  # 排除 system 和初始 user
                    # 压缩历史如果超出预算
                    if len(context.history) > context.max_history:
                        context.history = self._compress_history(context.history)

                return (True, {
                    "success": True,
                    "output": output,
                    "iterations": iteration + 1,
                    "history": self._compress_history(messages[2:])  # 排除 system 和第一个 user
                })

            # 收集本轮所有工具调用的结果
            tool_results = []
            for tool_call in msg.tool_calls:
                tool_name = tool_call.function.name
                tool_args_str = tool_call.function.arguments or "{}"

                # 正确解析工具参数
                try:
                    tool_args = json.loads(tool_args_str)
                except json.JSONDecodeError:
                    tool_args = {}
                    tool_result = "Error: Invalid JSON arguments"
                else:
                    # 使用标准化的参数哈希（排序后的键）进行去重
                    try:
                        normalized_args = tuple(sorted(tool_args.items()))
                        current_call = (tool_name, normalized_args)

                        # 检测重复调用
                        if len(tool_call_history) >= 2:
                            if tool_call_history[-1] == current_call and tool_call_history[-2] == current_call:
                                logger.warning(f"[LLMRuntime] Detected repeated tool call: {tool_name}, triggering early stop")
                                return (False, {
                                    "success": False,
                                    "output": f"Detected repeated tool calls for {tool_name}. Stopping to avoid infinite loop.",
                                    "iterations": iteration + 1,
                                    "warning": "early_stopped_duplicate_tool"
                                })

                        tool_call_history.append(current_call)

                        # 限制连续调用相同工具的次数
                        if tool_name == last_tool_name:
                            consecutive_same_tool += 1
                            if consecutive_same_tool >= 10:
                                logger.warning(f"[LLMRuntime] Detected 3+ consecutive calls to {tool_name}, triggering early stop")
                                return (False, {
                                    "success": False,
                                    "output": f"Stopped: {tool_name} called {consecutive_same_tool}+ times consecutively.",
                                    "iterations": iteration + 1,
                                    "warning": "early_stopped_consecutive_tool"
                                })
                        else:
                            consecutive_same_tool = 1
                            last_tool_name = tool_name
                    except Exception:
                        # 如果参数无法标准化，使用原始字符串作为后备
                        tool_call_history.append((tool_name, tool_args_str))
                        consecutive_same_tool = 1
                        last_tool_name = tool_name

                    # 执行工具调用（传递 skill_policy 以便执行管理工具）
                    logger.debug(f"[LLMRuntime.tool_call] Executing tool: {tool_name}, args: {tool_args}")
                    tool_result = await self._execute_tool(tool_name, tool_args, skill_policy=skill_policy)
                    logger.debug(f"[LLMRuntime.tool_call] Tool {tool_name} result: {str(tool_result)[:300]}...")

                tool_results.append({
                    "tool_call_id": tool_call.id,
                    "tool_name": tool_name,
                    "tool_result": str(tool_result)
                })

            # 【修复】按正确顺序添加消息：
            # 1. 先添加 assistant 消息（包含 tool_calls）
            # 2. 再添加对应的 tool 结果
            for i, tool_call in enumerate(msg.tool_calls):
                # 添加 assistant 消息，包含 tool_calls
                assistant_msg = {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [{
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    }]
                }
                messages.append(assistant_msg)

                # 紧跟对应的 tool 结果
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": tool_results[i]["tool_result"]
                })

        # 关键修复：在工具调用后收集信息到 context
        if context:
            # 收集 tool_trace
            for i, tool_call in enumerate(msg.tool_calls):
                context.tool_trace.append({
                    "tool": tool_results[i]["tool_name"],
                    "input": tool_args if 'tool_args' in locals() else {},
                    "output": tool_results[i]["tool_result"],
                    "timestamp": time.time()
                })
            # 收集对话历史
            context.history.extend(messages[2:])  # 排除 system 和初始 user
            # 压缩历史如果超出预算
            if len(context.history) > context.max_history:
                context.history = self._compress_history(context.history)

        return (False, {
            "success": False,
            "error": f"Max iterations ({max_iter}) reached",
            "history": self._compress_history(messages[2:])
        })

    async def _execute_tool(self, tool_name: str, tool_args: Dict, skill_policy: Optional[Any] = None) -> str:
        """执行工具（通过 CapabilityRegistry.execute() 或 SkillPolicy 执行管理工具）"""
        # 优先处理 Skill 管理工具（这些工具不在 CapabilityRegistry 中）
        skill_admin_tools = ["load_reference", "list_resources", "execute_script"]
        if skill_policy and tool_name in skill_admin_tools:
            logger.info(f"[LLMRuntime] Calling skill_admin_tool: {tool_name}, args: {tool_args}")
            try:
                if tool_name == "load_reference":
                    ref_name = tool_args.get("ref_name", "")
                    result = await skill_policy.skill.load_reference_async(ref_name)
                    if result is None:
                        return f"Reference '{ref_name}' not found, using your own knowledge."
                    return result
                elif tool_name == "list_resources":
                    result = await skill_policy.skill.list_resources_async()
                    return str(result)
                elif tool_name == "execute_script":
                    script_name = tool_args.get("script_name", "")
                    args = tool_args.get("args", [])
                    result = await skill_policy._execute_script_async(script_name, args)
                    return str(result)
            except Exception as e:
                return f"Error executing {tool_name}: {e}"

        # 其他工具通过 CapabilityRegistry 执行
        tool_cap = self.capability_registry.get(tool_name)
        if not tool_cap:
            logger.warning(f"[LLMRuntime] Tool '{tool_name}' not found, args: {tool_args}")
            return f"Error: Tool '{tool_name}' not found"

        logger.info(f"[LLMRuntime] Calling tool: {tool_name}, args: {tool_args}")
        try:
            result = await tool_cap.execute(**tool_args)
            result_str = str(result)

            logger.info(f"[LLMRuntime] Tool {tool_name} executed, result: {result_str[:300]}...")

            # 检查工具执行结果中是否包含已知的降级标记
            # 对于 Tavily API 配额耗尽等已知问题，直接使用降级结果
            if "API quota exceeded" in result_str or "432" in result_str:
                logger.info(f"[LLMRuntime] Tool {tool_name} using降级结果 (API quota issue)")
                return self._get_degraded_result(tool_name, tool_args)

            return result_str
        except Exception as e:
            # 检查错误消息中是否包含已知的可降级错误
            error_msg = str(e)
            if "432" in error_msg or "quota" in error_msg.lower():
                logger.info(f"[LLMRuntime] Tool {tool_name} using降级结果 due to error: {error_msg}")
                return self._get_degraded_result(tool_name, tool_args)
            logger.error(f"[LLMRuntime] Tool {tool_name} execution error: {error_msg}")
            return f"Error executing {tool_name}: {e}"

    def _get_degraded_result(self, tool_name: str, tool_args: Dict) -> str:
        """获取工具的降级结果（用于API配额耗尽等情况）"""
        if tool_name == "tavily_search":
            query = tool_args.get("query", "")
            return json.dumps({
                "query": query,
                "results": [],
                "answer": None,
                "images": [],
                "detail": "API quota exceeded - using empty results"
            }, ensure_ascii=False)
        return "Tool temporarily unavailable due to API quota limits"

    def _compress_history(self, history: List[Dict]) -> List[Dict]:
        """压缩历史记录"""
        compressed = []
        for item in history:
            if item.get("role") == "tool":
                content = item.get("content", "")
                if len(content) > 200:
                    content = content[:200] + "... [truncated]"
                compressed.append({
                    "role": "tool",
                    "name": item.get("name"),
                    "content": content
                })
            else:
                compressed.append(item)
        return compressed

    def _summarize_early_history(self, history: List[Dict]) -> Dict:
        """生成早期历史的摘要（三层压缩策略 Level 2）"""
        content_parts = []
        for msg in history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if role == "tool":
                content_parts.append(f"[工具结果] {content[:200]}")
            else:
                content_parts.append(f"[{role}] {content[:200]}")

        return {
            "role": "system",
            "content": f"【历史摘要】({' | '.join(content_parts)})"
        }


# =========================
# 核心数据结构
# =========================
class StepState(str, Enum):
    PENDING = "PENDING"
    READY = "READY"
    RUNNING = "RUNNING"
    BLOCKED = "BLOCKED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class Step:
    """Step：执行步骤（使用 depends_on 字段）"""
    step_id: str
    step_type: str  # "tool" | "skill" | "llm" | "analyze" | "answer"
    depends_on: List[str] = field(default_factory=list)
    max_retries: int = 3
    timeout: float = 60.0
    parallel: bool = True
    input_data: Dict[str, Any] = field(default_factory=dict)
    output: Any = None
    error: Optional[str] = None
    status: StepState = StepState.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    tool_name: Optional[str] = None
    tool_args: Dict[str, Any] = field(default_factory=dict)
    skill_name: Optional[str] = None

    # 兼容字段
    task: str = ""
    skill_name_legacy: str = ""

    def __post_init__(self):
        if not self.step_id:
            self.step_id = f"step-{uuid.uuid4().hex[:8]}"
        # 兼容 task 字段
        if not self.task and "task" in self.input_data:
            self.task = self.input_data["task"]


@dataclass
class Artifact:
    """结构化输出 artifact"""
    value: Any
    type: str  # "text" | "json" | "code" | "image" | "dataframe"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_success(self) -> bool:
        """检查 artifact 是否表示成功执行"""
        return self.metadata.get("error") is None and self.metadata.get("timeout") is not True

    def get_error(self) -> Optional[str]:
        """获取错误信息"""
        return self.metadata.get("error")

    def is_timeout(self) -> bool:
        """检查是否超时"""
        return self.metadata.get("timeout") is True

    def to_dict(self) -> Dict[str, Any]:
        return {"value": self.value, "type": self.type, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Artifact":
        return cls(
            value=data.get("value"),
            type=data.get("type", "text"),
            metadata=data.get("metadata", {})
        )

    @classmethod
    def create_success(cls, value: Any, step_id: str = "") -> "Artifact":
        """创建成功的 artifact"""
        return cls(
            value=value,
            type="text",
            metadata={"step_id": step_id, "success": True}
        )

    @classmethod
    def create_error(cls, error: str, value: Any = None, step_id: str = "") -> "Artifact":
        """创建表示错误的 artifact"""
        return cls(
            value=value or str(error),
            type="text",
            metadata={"step_id": step_id, "error": error, "success": False}
        )


@dataclass
class StepTrace:
    """Step 执行轨迹（用于观测性）"""
    step_id: str
    agent: str
    mode: str  # "react" | "direct" | "chain"
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    duration_ms: int = 0
    success: bool = False
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class State:
    """
    State: 认知状态系统（线程安全）
    - artifacts: 结构化输出（Artifact 对象）
    - memory: 长期记忆（用户偏好、领域知识等）
    - trace: 执行轨迹（用于调试和优化）

    关键修复：使用 asyncio.Lock 保护并发访问
    """

    def __init__(self):
        self.artifacts: Dict[str, Artifact] = {}
        self.memory: Dict[str, Any] = {}
        self.trace: List[StepTrace] = []
        self._lock = asyncio.Lock()  # 【新增】并发保护锁

    async def update_artifact(self, step_id: str, artifact: Artifact):
        async with self._lock:
            self.artifacts[step_id] = artifact

    async def get_artifact(self, step_id: str) -> Optional[Artifact]:
        async with self._lock:
            return self.artifacts.get(step_id)

    async def is_step_successful(self, step_id: str) -> bool:
        """检查步骤是否成功完成"""
        async with self._lock:
            artifact = self.artifacts.get(step_id)
            if artifact is None:
                return False
            return artifact.is_success()

    async def add_trace(self, trace: StepTrace):
        async with self._lock:
            self.trace.append(trace)

    async def get_memory(self, key: str, default=None) -> Any:
        async with self._lock:
            return self.memory.get(key, default)

    async def set_memory(self, key: str, value: Any):
        async with self._lock:
            self.memory[key] = value

    async def delete_artifact(self, step_id: str):
        """异步删除 artifact（用于重试等场景）"""
        async with self._lock:
            self.artifacts.pop(step_id, None)

    async def get_artifacts_snapshot(self) -> Dict[str, Artifact]:
        """获取 artifacts 快照（用于 Critic 等需要读取所有 artifacts 的场景）"""
        async with self._lock:
            return dict(self.artifacts)

    def get_snapshot(self) -> Dict[str, Any]:
        """获取状态快照（用于 Context 的 relevant_artifacts）"""
        # 不使用锁，因为返回的是副本
        return {
            "artifacts": dict(self.artifacts),
            "memory": dict(self.memory),
            "trace": list(self.trace)
        }


# =========================
# Critic（执行评估器）
# =========================
@dataclass
class CritiqueResult:
    """评估结果"""
    quality_score: float  # 0-1
    need_replan: bool = False
    suggestions: List[str] = field(default_factory=list)


class Critic:
    """Critic: 执行评估器"""

    def __init__(self, llm_runtime: LLMRuntime):
        self.llm_runtime = llm_runtime

    async def evaluate(self, step: Step, output: Any, state: State, caller: str = "unknown") -> CritiqueResult:
        """评估 Step 执行结果"""
        # 【修复】异步获取 artifacts 快照
        artifacts_snapshot = await state.get_artifacts_snapshot()
        context_parts = []
        for dep_id, artifact in artifacts_snapshot.items():
            context_parts.append(f"[{dep_id}]: {artifact.value}")
        context = "\n\n".join(context_parts) if context_parts else "No previous steps."

        # 【修复】限制 output 长度并进行安全转义
        output_str = str(output)
        # 只取前 300 字符，避免 prompt 过长
        if len(output_str) > 300:
            output_str = output_str[:300] + "... [truncated]"
        # 对特殊字符进行转义
        output_escaped = output_str.replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')

        prompt = f"""你是一个执行评估专家。

任务: {step.input_data.get("task", "Unknown")}
当前步骤类型: {step.step_type}
输出: {output_escaped}

已完成的步骤:
{context}

请评估本次执行：
1. 输出是否满足任务要求？
2. 工具使用是否合理？
3. 是否需要调整后续计划？

请以 JSON 格式返回：
{{
  "quality_score": 0.0-1.0,
  "need_replan": true/false,
  "suggestions": ["建议1", "建议2"]
}}"""

        try:
            response = await self.llm_runtime.reason(prompt, "", caller="Critic.evaluate")
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                eval_data = json.loads(json_match.group())
                quality_score = eval_data.get("quality_score", 0.5)
                # 确保 quality_score 在有效范围内
                if not isinstance(quality_score, (int, float)) or quality_score < 0 or quality_score > 1:
                    quality_score = 0.5
                return CritiqueResult(
                    quality_score=quality_score,
                    need_replan=eval_data.get("need_replan", False),
                    suggestions=eval_data.get("suggestions", [])
                )
            else:
                return CritiqueResult(quality_score=0.5, need_replan=False, suggestions=[])
        except Exception as e:
            logger.error(f"[Critic] Evaluation failed: {e}")
            # 【修复】避免因为 Critic 失败导致步骤被标记为失败
            # 返回中等分数，让步骤继续
            return CritiqueResult(quality_score=0.5, need_replan=False, suggestions=[f"Evaluation error: {str(e)}"])

class Replanner:
    """Replanner: 动态重规划器"""

    def __init__(self, llm_runtime: LLMRuntime):
        self.llm_runtime = llm_runtime

    async def replan_from_failure(self, task: str, failed_step: Step, error: str, current_state: State) -> List[Step]:
        """从失败中重规划"""
        context_parts = []
        for dep_id, artifact in current_state.artifacts.items():
            context_parts.append(f"[{dep_id}]: {artifact.value}")
        context = "\n\n".join(context_parts) if context_parts else "No previous steps completed."

        prompt = f"""你是一个任务规划专家。

原始任务:
{task}

失败的步骤:
- Step ID: {failed_step.step_id}
- 目标: {failed_step.input_data.get("task", "Unknown")}
- 错误: {error}

已完成的步骤输出:
{context}

请分析失败原因，并生成新的执行计划（3步以内）：
1. 是否需要更多信息？
2. 是否需要换一种方法？
3. 如何绕过失败的步骤？

请以 JSON 格式返回：
{{
  "analysis": "失败原因分析",
  "new_steps": [
    {{"step_id": "new_x", "step_type": "tool|llm|analyze", "task": "任务描述", "depends_on": []}}
  ]
}}"""

        try:
            response = await self.llm_runtime.reason(prompt, "", caller="Replanner.replan_from_failure")
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                plan_data = json.loads(json_match.group())
                new_steps = plan_data.get("new_steps", [])
                result = []
                for i, step_data in enumerate(new_steps):
                    step = Step(
                        step_id=step_data.get("step_id", f"replan_{i}"),
                        step_type=step_data.get("step_type", "tool"),
                        input_data={"task": step_data.get("task", "")},
                        depends_on=step_data.get("depends_on", [])
                    )
                    result.append(step)
                return result
            else:
                logger.warning(f"[Replanner] Failed to parse JSON: {response}")
                return []
        except Exception as e:
            logger.error(f"[Replanner] Replanning failed: {e}")
            return []

    async def replan_from_insufficient(self, task: str, current_state: State) -> List[Step]:
        """从信息不足中重规划"""
        context_parts = []
        for dep_id, artifact in current_state.artifacts.items():
            context_parts.append(f"[{dep_id}]: {artifact.value}")
        context = "\n\n".join(context_parts) if context_parts else "No previous steps completed."

        prompt = f"""你是一个任务规划专家。

原始任务:
{task}

当前已完成的步骤输出:
{context}

分析：当前信息是否足够完成任务？如果不够，缺少什么信息？

请以 JSON 格式返回：
{{
  "analysis": "信息分析",
  "missing_info": ["缺少的信息1", "缺少的信息2"],
  "new_steps": [
    {{"step_id": "new_x", "step_type": "tool|llm", "task": "获取缺失信息", "depends_on": []}}
  ]
}}"""

        try:
            response = await self.llm_runtime.reason(prompt, "", caller="Replanner.replan_from_insufficient")
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                plan_data = json.loads(json_match.group())
                new_steps = plan_data.get("new_steps", [])
                result = []
                for i, step_data in enumerate(new_steps):
                    step = Step(
                        step_id=step_data.get("step_id", f"insufficient_{i}"),
                        step_type=step_data.get("step_type", "tool"),
                        input_data={"task": step_data.get("task", "")},
                        depends_on=step_data.get("depends_on", [])
                    )
                    result.append(step)
                return result
            else:
                return []
        except Exception as e:
            logger.error(f"[Replanner] Replanning from insufficient failed: {e}")
            return []


# =========================
# ErrorRecovery（错误恢复管理器）
# =========================
@dataclass
class RetryPolicy:
    """重试策略"""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_backoff: bool = True


@dataclass
class FallbackStrategy:
    """降级策略"""
    fallback_to: str  # "direct_answer" | "default_tool" | "human_in_loop"
    condition: str  # "error" | "timeout" | "low_quality"
    threshold: Optional[float] = None


@dataclass
class RecoveryAction:
    """恢复动作"""
    action: str  # "retry" | "fallback" | "replan" | "fail"
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)


class ErrorRecovery:
    """错误恢复管理器"""

    def __init__(self, retry_policy: RetryPolicy = None, fallback_strategies: List[FallbackStrategy] = None):
        self.retry_policy = retry_policy or RetryPolicy()
        self.fallback_strategies = fallback_strategies or []
        # 跟踪每个步骤的重试次数
        self._retry_counts: Dict[str, int] = {}

    def get_retry_count(self, step_id: str) -> int:
        """获取步骤的重试次数"""
        return self._retry_counts.get(step_id, 0)

    def increment_retry(self, step_id: str) -> int:
        """增加并返回重试次数"""
        count = self._retry_counts.get(step_id, 0) + 1
        self._retry_counts[step_id] = count
        return count

    def reset_retry(self, step_id: str):
        """重置重试计数"""
        self._retry_counts.pop(step_id, None)

    async def handle_error(self, step: Step, error: str, state: State, attempt: int) -> RecoveryAction:
        """处理错误，根据错误类型返回恢复动作"""
        _ = step  # 保留参数用于未来扩展
        _ = state  # 保留参数用于未来扩展
        policy = self.retry_policy

        # 检查是否超过最大重试次数
        if attempt >= policy.max_retries:
            for fallback in self.fallback_strategies:
                if fallback.condition == "error":
                    return RecoveryAction(
                        action="fallback",
                        reason="Error fallback triggered",
                        details={"fallback_to": fallback.fallback_to}
                    )
            return RecoveryAction(action="fail", reason="Max retries exceeded", details={"error": error})

        # 错误类型识别：优先处理规划错误
        error_lower = error.lower()

        # 1. 技能不存在/无效技能：触发重规划
        if "not a skill capability" in error_lower or "skill" in error_lower and ("not found" in error_lower or "unknown" in error_lower):
            return RecoveryAction(
                action="replan",
                reason="Invalid skill name, need replanning",
                details={"error": error, "error_type": "invalid_skill"}
            )

        # 2. 工具不存在：触发重规划或 fallback
        if "tool" in error_lower and ("not found" in error_lower or "not exist" in error_lower):
            return RecoveryAction(
                action="replan",
                reason="Tool not found, need replanning",
                details={"error": error, "error_type": "tool_not_found"}
            )

        # 3. 信息不足：触发重规划
        if "insufficient" in error_lower or "missing" in error_lower:
            return RecoveryAction(
                action="replan",
                reason="Insufficient information, need replanning",
                details={"error": error, "error_type": "insufficient"}
            )

        # 4. LLM 迭代次数超限：重试或 fallback
        if "max iterations" in error_lower or "reached" in error_lower:
            if attempt < policy.max_retries - 1:
                return RecoveryAction(
                    action="retry",
                    reason="Max iterations reached, will retry",
                    details={"error": error, "attempt": attempt, "retryable": True}
                )
            return RecoveryAction(
                action="fallback",
                reason="Max iterations reached, falling back",
                details={"error": error, "fallback_to": "direct_answer"}
            )

        # 5. 默认：重试
        delay = policy.base_delay
        if policy.exponential_backoff:
            delay = min(delay * (2 ** attempt), policy.max_delay)

        return RecoveryAction(
            action="retry",
            reason=f"Retry attempt {attempt + 1}/{policy.max_retries}",
            details={"delay": delay, "error": error, "retryable": True}
        )


# =========================
# Planner（Agent 策略生成器）
# =========================
@dataclass
class StepPlan:
    """Step 计划（包含 Agent 策略）"""
    step_id: str
    task: str
    target_agent: str
    mode: str
    tool_strategy: str
    depends_on: List[str] = field(default_factory=list)
    input_data: Dict[str, Any] = field(default_factory=dict)
    react_config: Optional[Dict] = None
    status: Optional[StepState] = None  # 步骤状态（由执行引擎更新）


@dataclass
class Plan:
    """Plan：包含 Agent 策略的计划"""
    plan_id: str
    task: str
    steps: Dict[str, StepPlan]
    dag: Dict[str, List[str]]


class Planner:
    """Planner: Agent 策略生成器"""

    def __init__(self, runtime: LLMRuntime):
        self.runtime = runtime
        self._available_skill_names: List[str] = []

    def set_available_skills(self, skill_names: List[str]):
        """设置可用技能白名单（用于动态注入 Prompt）"""
        self._available_skill_names = skill_names

    async def plan(self, task: str, available_tools: List[Dict], available_skills: List[Dict], caller: str = "unknown") -> Plan:
        """生成完整的执行计划，包含 target_agent 合法性验证"""
        logger.info(f"[Planner.plan] Called by {caller}, task={task[:50]}...")
        logger.info(f"[Planner.plan] Calling LLMRuntime.reason for planning...")
        tools_info = "\n".join([
            f"- {t.get('name', 'unknown')}: {t.get('description', '')}"
            for t in available_tools
        ])
        skills_info = "\n".join([
            f"- {s.get('name', 'unknown')}: {s.get('description', 'No description')}"
            for s in available_skills
        ])

        # 动态注入技能白名单到 Prompt（解决技能幻觉）
        available_skill_names = [s.get('name', '') for s in available_skills if s.get('name')]
        self._available_skill_names = available_skill_names

        # 构建清晰的技能白名单说明
        if available_skill_names:
            skill_whitelist_info = f"""
## 可用技能白名单（必须从以下列表中严格选择）
注意：target_agent 必须精确匹配以下名称之一，不允许生成白名单之外的技能名称。
可用技能: {', '.join(available_skill_names)}
"""
        else:
            skill_whitelist_info = """
## 可用技能白名单
无可用技能。如果必须使用技能，请使用 'llm' 作为 target_agent。
"""

        # 添加示例说明，减少 LLM 幻觉
        examples_info = """
## 示例输出格式（单步骤）
{
  "plan_id": "plan_001",
  "steps": {
    "step_1": {
      "task": "获取最新信息",
      "target_agent": "search",
      "mode": "react",
      "tool_strategy": "optional",
      "depends_on": []
    }
  },
  "dag": {
    "step_1": []
  }
}

## 示例输出格式（多步骤，有依赖）
{
  "plan_id": "plan_002",
  "steps": {
    "step_1": {
      "task": "编写傅里叶变换代码",
      "target_agent": "code",
      "mode": "react",
      "tool_strategy": "optional",
      "depends_on": []
    },
    "step_2": {
      "task": "保存代码到文件",
      "target_agent": "code",
      "mode": "react",
      "tool_strategy": "optional",
      "depends_on": ["step_1"]
    }
  },
  "dag": {
    "step_1": [],
    "step_2": ["step_1"]
  }
}

## 多步骤规划原则（必须遵守）
- **单一职责**：每个步骤只完成一个明确的任务
- **避免重复**：如果 step_1 已经完成了某个工作（如编写代码），step_2 不应重复相同工作
- **清晰分工**：
  - step_1：编写代码/实现逻辑
  - step_2：验证/测试代码
  - step_3：保存到文件/持久化
- 每个步骤完成后，检查是否需要下一步；如果已完成，不要添加多余步骤
"""

        prompt = f"""你是一个任务规划专家。

任务: {task}

可用工具:
{tools_info}

可用 Agent/Skill:
{skills_info}

{skill_whitelist_info}
{examples_info}

请生成执行计划：
1. 将任务分解为多个步骤
2. 为每个步骤选择合适的 Agent/Skill（必须从可用技能白名单中选择）
3. 指定执行模式（react/direct/chain）
4. 指定工具策略（optional/required/auto）

## 工具参数规范（必须遵守）
- **file_write 工具**：参数必须是 `filepath` 和 `content`，不要使用 `file_name` 或其他名称
- **file_read 工具**：参数必须是 `filepath`，不要使用 `file_name` 或其他名称
- **bash_execute 工具**：参数必须是 `command`，不要使用 `cmd` 或其他名称
- 严格按照工具定义的参数名调用，避免参数名不匹配导致错误

重要规则：
- target_agent 必须严格匹配可用技能白名单中的名称
- 如果没有合适的技能，使用 'llm' 作为默认值
- 不要创建白名单中不存在的技能名称

## 依赖关系规则（必须遵守）
- **step_1**：通常是第一步，不需要依赖其他步骤
- **step_2**：如果它需要使用 step_1 的输出，则必须设置 `depends_on: ["step_1"]`
- **step_3**：如果它需要使用 step_1 或 step_2 的输出，则必须设置 `depends_on: ["step_1", "step_2"]` 或 `depends_on: ["step_2"]`
- 依赖关系必须反映真实的执行顺序，后继步骤必须依赖其输入所依赖的所有前置步骤

请以 JSON 格式返回："""

        try:
            logger.info(f"[Planner.plan] Starting LLM planning call...")
            response = await self.runtime.reason(prompt, "", caller=f"Planner.plan-{caller}")
            logger.info(f"[Planner.plan] LLM planning completed, response length={len(response)}")
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                plan_data = json.loads(json_match.group())
                steps = {}
                for step_id, step_data in plan_data.get("steps", {}).items():
                    target_agent = step_data.get("target_agent", "llm")
                    # 合法性验证：如果 target_agent 不在白名单中，降级为 "llm"
                    if target_agent not in available_skill_names:
                        logger.warning(f"[Planner] Step {step_id} target_agent '{target_agent}' not in whitelist, falling back to 'llm'")
                        target_agent = "llm"
                    steps[step_id] = StepPlan(
                        step_id=step_id,
                        task=step_data.get("task", ""),
                        target_agent=target_agent,
                        mode=step_data.get("mode", "react"),
                        tool_strategy=step_data.get("tool_strategy", "optional"),
                        depends_on=step_data.get("depends_on", []),
                        input_data={"task": step_data.get("task", "")},
                        react_config={"max_iterations": 10}
                    )
                return Plan(
                    plan_id=plan_data.get("plan_id", "plan_unknown"),
                    task=task,
                    steps=steps,
                    dag=plan_data.get("dag", {})
                )
            else:
                logger.warning(f"[Planner] Failed to parse JSON: {response}")
                return Plan(
                    plan_id="plan_fallback",
                    task=task,
                    steps={},
                    dag={}
                )
        except Exception as e:
            logger.error(f"[Planner] Planning failed: {e}")
            return Plan(
                plan_id="plan_error",
                task=task,
                steps={},
                dag={}
            )


# =========================
# DynamicPlan（COW + Version）
# =========================
class DynamicPlan:
    def __init__(self):
        self.steps: Dict[str, Step] = {}
        self.dag: Dict[str, List[str]] = {}
        self.version: int = 0

    def insert_subplan_atomic(self, parent_id: str, sub_steps: Dict[str, Step], sub_dag: Dict[str, List[str]]):
        """原子性插入子计划"""
        new_steps = copy.deepcopy(self.steps)
        new_dag = copy.deepcopy(self.dag)

        children = [sid for sid, deps in new_dag.items() if parent_id in deps]
        new_steps.update(sub_steps)
        new_dag.update(sub_dag)

        all_nodes = set(sub_steps.keys())
        non_leaf = set(d for deps in sub_dag.values() for d in deps)
        leaf = list(all_nodes - non_leaf)

        for c in children:
            new_dag[c] = [d for d in new_dag[c] if d != parent_id] + leaf

        self.steps = new_steps
        self.dag = new_dag
        self.version += 1

    def get_ready_steps(self) -> List[Step]:
        """返回 READY 步骤列表"""
        ready = []
        for step in self.steps.values():
            if step.status == StepState.PENDING:
                all_deps_done = all(
                    d in self.steps and self.steps[d].status == StepState.COMPLETED
                    for d in step.depends_on
                )
                if all_deps_done:
                    ready.append(step)
        return ready

    def get_step(self, step_id: str) -> Optional[Step]:
        return self.steps.get(step_id)

    def update_step(self, step: Step):
        if step.step_id in self.steps:
            self.steps[step.step_id] = step

    def commit_step(self, step_id: str, result: str):
        """提交步骤结果"""
        if step_id in self.steps:
            step = self.steps[step_id]
            step.status = StepState.COMPLETED
            step.output = result
            step.completed_at = time.time()

    def is_complete(self) -> bool:
        """检查是否所有步骤都完成"""
        if not self.steps:
            return False
        return all(s.status == StepState.COMPLETED for s in self.steps.values())

    def has_failed(self) -> bool:
        """检查是否有步骤失败"""
        if not self.steps:
            return False
        return any(s.status == StepState.FAILED for s in self.steps.values())


# =========================
# ExecutionEngine（执行引擎）
# =========================
class ExecutionEngine:
    """
    ExecutionEngine: 执行引擎
    改进点：
    1. 使用 State（artifacts + memory + trace）
    2. 支持 Replanner 集成
    3. 支持 Critic 集成
    4. 支持 ErrorRecovery 集成
    5. 【修复】依赖检查时验证 artifact 成功状态
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        capability_registry: CapabilityRegistry,
        planner: Optional[Planner] = None,
        replanner: Optional[Replanner] = None,
        critic: Optional[Critic] = None,
        error_recovery: Optional[ErrorRecovery] = None
    ):
        self.tool_registry = tool_registry
        self.capability_registry = capability_registry
        self.planner = planner
        self.replanner = replanner
        self.critic = critic
        self.error_recovery = error_recovery
        self.plan: Optional[Plan] = None
        self.state = State()
        self.bus: Optional[EventBus] = None
        self._running = False
        self._claim_lock = asyncio.Lock()
        self._claiming_steps: Set[str] = set()
        self._published_ready_steps: Set[str] = set()
        self._step_execution_lock = asyncio.Lock()  # 共享锁，防止步骤重复执行
        self._executing_steps: Set[str] = set()  # 跟踪正在执行的步骤
        self._completed_steps: Set[str] = set()  # 跟踪已完成的步骤，防止重复处理 STEP_COMPLETED
        self._sub_manager: Optional[EventSubscriptionManager] = None
        # 【新增】依赖错误跟踪
        self._failed_dependencies: Dict[str, Set[str]] = {}  # step_id -> set of failed dep ids
        # 【新增】防止 _check_and_publish_completion 循环调用
        self._completion_check_count = 0
        self._max_completion_checks = 10  # 最大检查次数
        self._check_in_progress = False  # 防止并发检查

    async def initialize(self, bus: EventBus):
        self.bus = bus
        self._sub_manager = EventSubscriptionManager(bus)
        await self._setup_subscriptions()
        logger.info("[ExecutionEngine] Initialized")

    async def _setup_subscriptions(self):
        logger.info(f"[ExecutionEngine] _setup_subscriptions called, bus={id(self.bus)}, sub_manager={id(self._sub_manager)}")
        logger.info(f"[ExecutionEngine] _setup_subscriptions: bus._subscribers before={len(self.bus._subscribers)}")
        if not self._sub_manager:
            raise RuntimeError("EventSubscriptionManager not initialized.")
        await self._sub_manager.subscribe(
            component="Engine",
            name="process_completed",
            handler=self.process_completed,
            event_filter=EventType.STEP_COMPLETED
        )
        logger.info(f"[ExecutionEngine] After process_completed: bus._subscribers={len(self.bus._subscribers)}")
        await self._sub_manager.subscribe(
            component="Engine",
            name="process_failed",
            handler=self.process_failed,
            event_filter=EventType.STEP_FAILED
        )
        logger.info(f"[ExecutionEngine] After process_failed: bus._subscribers={len(self.bus._subscribers)}")

    async def claim_step(self, step_id: str) -> bool:
        async with self._claim_lock:
            # 【关键修复】检查是否已完成，防止已完成的任务被重新声明
            if step_id in self._completed_steps:
                return False
            if step_id in self._claiming_steps:
                return False
            self._claiming_steps.add(step_id)
            return True

    async def release_claim(self, step_id: str):
        async with self._claim_lock:
            self._claiming_steps.discard(step_id)

    async def _publish_ready_steps(self):
        """发布 READY 步骤（DAG 并行，关键路径优先）【修复】依赖检查"""
        if not self.bus or not self.plan:
            return
        ready_steps = []
        for step_id in self.plan.steps:
            # 【修复】检查是否已完成（防止重复执行）
            if step_id in self._completed_steps:
                continue
            if step_id in self._published_ready_steps:
                continue
            if step_id in self._claiming_steps:
                continue
            if step_id in self._executing_steps:
                continue

            step_deps = self.plan.dag.get(step_id, [])

            # 【修复】检查所有依赖是否成功完成
            all_deps_successful = True
            for dep_id in step_deps:
                # 检查依赖是否有错误
                if dep_id in self._failed_dependencies:
                    all_deps_successful = False
                    break
                # 检查依赖的 artifact 是否存在且成功
                artifact = self.state.artifacts.get(dep_id)
                if artifact is None:
                    all_deps_successful = False
                    break
                if not artifact.is_success():
                    all_deps_successful = False
                    # 记录失败的依赖
                    if step_id not in self._failed_dependencies:
                        self._failed_dependencies[step_id] = set()
                    self._failed_dependencies[step_id].add(dep_id)
                    break

            if all_deps_successful:
                ready_steps.append(step_id)

        # 优化：按关键路径优先排序（解锁下游最多的步骤优先）
        ready_steps = self._prioritize_ready_steps(ready_steps)

        # 去重：只发布未发布的步骤
        published_count = 0
        for step_id in ready_steps:
            if step_id in self._published_ready_steps:
                continue
            self._published_ready_steps.add(step_id)
            logger.info(f"[Engine] Publishing STEP_READY for {step_id}")
            await self.bus.publish(Event(
                event_type=EventType.STEP_READY,
                step_id=step_id,
                payload={"step_type": "skill", "step_data": {"input_data": {}}}
            ))
            published_count += 1

        if published_count > 0:
            logger.info(f"[Engine] Published {published_count} STEP_READY events, total published: {len(self._published_ready_steps)}, plan_steps={list(self.plan.steps.keys())}, plan_len={len(self.plan.steps)}")

    def _prioritize_ready_steps(self, ready_steps: List[str]) -> List[str]:
        """按关键路径优先排序（解锁下游最多的步骤优先）"""
        if not ready_steps:
            return ready_steps

        # 计算每个步骤的下游影响（解锁的节点数）
        def downstream_impact(step_id: str) -> int:
            affected = set()
            to_process = [step_id]
            while to_process:
                current = to_process.pop()
                # 找到所有依赖于 current 的步骤
                for sid, deps in self.plan.dag.items():
                    if current in deps and sid not in affected:
                        affected.add(sid)
                        to_process.append(sid)
            return len(affected)

        # 按下游影响降序排序
        return sorted(ready_steps, key=lambda s: -downstream_impact(s))

    async def _check_and_publish_completion(self):
        """检查所有步骤是否完成，并发布相应事件"""
        # 【修复】防止循环调用超过限制
        if not self.plan:
            return

        # 【修复】防止并发检查
        if self._check_in_progress:
            logger.debug(f"[Engine] _check_and_publish_completion already in progress, skipping")
            return

        # 【修复】检查调用次数限制
        self._completion_check_count += 1
        if self._completion_check_count > self._max_completion_checks:
            logger.warning(f"[Engine] _check_and_publish_completion max calls reached ({self._completion_check_count} > {self._max_completion_checks}), stopping")
            return

        steps = self.plan.steps
        if not steps:
            return

        # 统计各状态步骤
        completed_count = sum(1 for s in steps.values() if s.status == StepState.COMPLETED)
        failed_count = sum(1 for s in steps.values() if s.status == StepState.FAILED)
        total = len(steps)

        logger.info(f"[Engine] _check_and_publish_completion: completed={completed_count}, failed={failed_count}, total={total}")
        for step_id, step in steps.items():
            logger.info(f"[Engine] Step {step_id[:8]} status: {step.status}")

        # 【修复】标记开始检查
        self._check_in_progress = True

        try:
            # 检查是否所有步骤都处于终态
            all_done = all(
                s.status in [StepState.COMPLETED, StepState.FAILED, StepState.BLOCKED]
                for s in steps.values()
            ) if steps else False

            logger.info(f"[Engine] all_done={all_done}")

            if all_done and total > 0:
                # 【修复】使用非阻塞方式发布完成事件，避免死锁
                if failed_count == 0:
                    logger.info(f"[Engine] All steps completed ({completed_count}/{total}), publishing TASK_COMPLETED")
                    if self.bus:
                        asyncio.create_task(self.bus.publish(Event(
                            event_type=EventType.TASK_COMPLETED,
                            payload={"total_steps": total, "completed_steps": completed_count}
                        ), block=False))
                else:
                    logger.info(f"[Engine] Task failed ({failed_count}/{total} steps failed), publishing TASK_FAILED")
                    if self.bus:
                        asyncio.create_task(self.bus.publish(Event(
                            event_type=EventType.TASK_FAILED,
                            payload={"total_steps": total, "failed_steps": failed_count}
                        ), block=False))
                # 【修复】任务完成，重置计数器
                self._completion_check_count = 0
        finally:
            # 【修复】标记检查完成
            self._check_in_progress = False

    async def process_completed(self, event: Event):
        """处理 STEP_COMPLETED 事件"""
        if event.event_type != EventType.STEP_COMPLETED or not event.step_id:
            return
        # 使用锁防止重复处理
        is_duplicate = False
        async with self._step_execution_lock:
            if event.step_id in self._completed_steps:
                # 记录 duplicate 事件用于诊断
                logger.debug(f"[Engine] STEP_COMPLETED for {event.step_id[:8]} (duplicate, skipping artifact update)")
                is_duplicate = True
                # 【调试】打印堆栈追踪
                import traceback
                logger.debug(f"[Engine] Duplicate event stack trace:\n{''.join(traceback.format_stack())}")
            else:
                self._completed_steps.add(event.step_id)

        # 【修复】即使重复，也要检查任务是否完成并发布下游步骤（在锁外调用）
        if is_duplicate:
            # 即使是重复事件，也要尝试发布下游步骤
            # 这是因为可能之前的发布失败了
            asyncio.create_task(self._delayed_publish_ready_steps())
            await self._check_and_publish_completion()
            return

        logger.info(f"[Engine] STEP_COMPLETED for {event.step_id[:8]}")

        # 【修复】正确提取输出
        output = event.payload.get("output")
        output_str = extract_output(output) if output is not None else ""

        artifact = Artifact.create_success(output_str, step_id=event.step_id)
        await self.state.update_artifact(event.step_id, artifact)

        # 更新 StepPlan 状态
        if self.plan and event.step_id in self.plan.steps:
            self.plan.steps[event.step_id].status = StepState.COMPLETED

        # 同步更新 dynamic_plan 状态（如果存在）
        dynamic_plan = getattr(self, 'dynamic_plan', None)
        if dynamic_plan and event.step_id in dynamic_plan.steps:
            dynamic_plan.steps[event.step_id].status = StepState.COMPLETED

        trace = StepTrace(
            step_id=event.step_id,
            agent="default",
            mode="react",
            success=True
        )
        await self.state.add_trace(trace)

        # 【修复】延迟发布下游步骤，避免竞态条件
        # 使用 create_task 让事件循环先处理完当前事件
        asyncio.create_task(self._delayed_publish_ready_steps())

        # 检查任务是否完成
        await self._check_and_publish_completion()

    async def _delayed_publish_ready_steps(self):
        """延迟发布 READY 步骤，给事件处理留出时间"""
        await asyncio.sleep(0.01)  # 短暂延迟
        await self._publish_ready_steps()

    async def process_failed(self, event: Event):
        """处理 STEP_FAILED 事件"""
        if event.event_type != EventType.STEP_FAILED or not event.step_id:
            return
        # 使用锁防止重复处理
        async with self._step_execution_lock:
            if event.step_id in self._completed_steps:
                logger.warning(f"[Engine] STEP_FAILED for {event.step_id[:8]} (duplicate, skipped)")
                return
            self._completed_steps.add(event.step_id)

        error_msg = event.payload.get("error", "Unknown error")
        logger.warning(f"[Engine] STEP_FAILED for {event.step_id[:8]}: {error_msg}")

        # 【修复】创建表示错误的 artifact
        artifact = Artifact.create_error(error_msg, step_id=event.step_id)
        await self.state.update_artifact(event.step_id, artifact)

        # 记录失败的依赖
        self._failed_dependencies[event.step_id] = {event.step_id}

        # 更新 StepPlan 状态
        if self.plan and event.step_id in self.plan.steps:
            self.plan.steps[event.step_id].status = StepState.FAILED

        # 同步更新 dynamic_plan 状态（如果存在）
        dynamic_plan = getattr(self, 'dynamic_plan', None)
        if dynamic_plan and event.step_id in dynamic_plan.steps:
            dynamic_plan.steps[event.step_id].status = StepState.FAILED

        trace = StepTrace(
            step_id=event.step_id,
            agent="default",
            mode="react",
            success=False,
            error=error_msg
        )
        await self.state.add_trace(trace)

        # 熔断：级联取消所有下游步骤
        await self._cancel_downstream(event.step_id)

        if self.error_recovery:
            # 【修复】使用 error_recovery 的重试计数
            attempt = self.error_recovery.get_retry_count(event.step_id)
            action = await self.error_recovery.handle_error(
                step=Step(step_id=event.step_id, step_type="skill", input_data={}),
                error=error_msg,
                state=self.state,
                attempt=attempt
            )
            logger.info(f"[Engine] Recovery action for {event.step_id[:8]}: {action.action}")

            if action.action == "retry":
                # 重试逻辑：清除状态，重新发布 STEP_READY
                logger.info(f"[Engine] Retrying step {event.step_id[:8]}...")
                self.error_recovery.increment_retry(event.step_id)

                # 清除完成标记和 artifact，允许重新执行
                self._completed_steps.discard(event.step_id)
                await self.state.delete_artifact(event.step_id)
                # 清除失败依赖记录
                self._failed_dependencies.pop(event.step_id, None)

                # 重置步骤状态
                if self.plan and event.step_id in self.plan.steps:
                    self.plan.steps[event.step_id].status = StepState.PENDING

                # 重新发布 STEP_READY
                if self.bus and self.plan:
                    self._published_ready_steps.discard(event.step_id)
                    await self.bus.publish(Event(
                        event_type=EventType.STEP_READY,
                        step_id=event.step_id,
                        payload={"step_type": "skill", "step_data": {"input_data": {}}}
                    ))
                    logger.info(f"[Engine] Rescheduled step {event.step_id[:8]} for retry")

            elif action.action == "replan" and self.replanner:
                logger.info(f"[Engine] Triggering replanning for {event.step_id[:8]}")
                new_steps = await self.replanner.replan_from_failure(
                    task=self.plan.steps[event.step_id].task if self.plan and event.step_id in self.plan.steps else "Unknown",
                    failed_step=Step(step_id=event.step_id, step_type="skill", input_data={}),
                    error=error_msg,
                    current_state=self.state
                )
                if new_steps:
                    logger.info(f"[Engine] Replanner generated {len(new_steps)} new steps")
                    if self.plan:
                        for new_step in new_steps:
                            self.plan.steps[new_step.step_id] = StepPlan(
                                step_id=new_step.step_id,
                                task=new_step.input_data.get("task", ""),
                                target_agent="llm",
                                mode="react",
                                tool_strategy="optional",
                                depends_on=new_step.depends_on,
                                input_data=new_step.input_data,
                                react_config={"max_iterations": 10}
                            )
                            if event.step_id in self.plan.dag:
                                self.plan.dag[new_step.step_id] = self.plan.dag[event.step_id]
                                self.plan.dag[event.step_id] = []

        # 检查任务是否完成
        await self._check_and_publish_completion()

    async def _cancel_downstream(self, failed_step_id: str, visited: Optional[Set[str]] = None, depth: int = 0, max_depth: int = 100):
        """
        熔断机制：取消所有依赖于失败步骤的下游步骤（使用 BFS 防止无限递归）

        关键修复：
        1. 添加 visited 集合防止重复访问
        2. 添加 max_depth 限制防止栈溢出
        3. 使用迭代方式替代纯递归
        """
        if not self.plan:
            return

        # 初始化 visited 集合
        if visited is None:
            visited = set()

        # 关键修复：深度限制防止无限递归
        if depth > max_depth:
            logger.warning(f"[Engine] _cancel_downstream max depth reached ({max_depth})")
            return

        # 使用队列进行 BFS 遍历
        from collections import deque
        queue = deque([failed_step_id])

        while queue:
            current_failed = queue.popleft()

            # 找到所有以 current_failed 为依赖的步骤
            children = []
            for step_id, depends_on in self.plan.dag.items():
                if current_failed in depends_on and step_id not in visited:
                    children.append(step_id)

            for child_id in children:
                # 跳过已经完成/失败/取消的步骤
                if child_id in self._completed_steps:
                    continue

                # 标记为取消
                if child_id in self.plan.steps:
                    self.plan.steps[child_id].status = StepState.BLOCKED
                    logger.info(f"[Engine] CANCELLED downstream step {child_id} due to failure of {current_failed}")

                # 发布 STEP_CANCELLED 事件
                await self.bus.publish(Event(
                    event_type=EventType.STEP_CANCELLED,
                    step_id=child_id,
                    payload={"reason": f"Parent {current_failed} failed", "cancelled_by": current_failed}
                ))
                self._completed_steps.add(child_id)
                visited.add(child_id)

                # 添加到队列继续处理
                queue.append(child_id)

    def set_plan(self, plan: Plan):
        self.plan = plan
        self._published_ready_steps.clear()
        self._failed_dependencies.clear()

        # 验证 Plan 中的 target_agent 是否在可用技能白名单中
        self._validate_plan_skills(plan)

        logger.info(f"[Engine] set_plan called, plan_id={plan.plan_id}, steps={list(plan.steps.keys())}, plan_len={len(plan.steps)}")

    def _validate_plan_skills(self, plan: Plan):
        """验证 Plan 中的 target_agent 是否在可用技能白名单中"""
        available_skills = self.capability_registry.get_all_names()

        invalid_steps = []
        for step_id, step_plan in plan.steps.items():
            target_agent = step_plan.target_agent
            # 'llm' 是特殊值，允许通过验证
            if target_agent and target_agent not in available_skills and target_agent != "llm":
                invalid_steps.append((step_id, target_agent))
                logger.warning(f"[Engine] Step {step_id} uses unknown skill '{target_agent}' (available: {available_skills})")

        if invalid_steps:
            error_msg = f"Plan validation failed: steps use unknown skills: {invalid_steps}"
            logger.error(f"[Engine] {error_msg}")
            raise ValueError(error_msg)

    async def start(self):
        """启动引擎"""
        if not self.bus or not self.plan:
            return
        logger.info("[Engine] Starting execution")
        await self._publish_ready_steps()

    async def shutdown(self):
        """关闭引擎"""
        logger.info("[Engine] Shutdown initiated")
        if self._sub_manager:
            await self._sub_manager.unsubscribe_all()
        logger.info("[Engine] Shutdown completed")


# =========================
# Worker（结构化上下文执行器）
# =========================
class Worker:
    """
    Worker: 结构化上下文执行器
    改进点：
    1. 使用 StepContext 而非字符串拼接
    2. 纯执行器，不构造上下文
    3. 支持 StepMode（react/direct/chain）
    4. 使用 State 认知状态
    5. 【修复】缩小锁粒度，执行在锁外进行
    """

    def __init__(
        self,
        worker_id: str,
        capability_registry: CapabilityRegistry,
        tool_registry: ToolRegistry,
        llm_runtime: LLMRuntime,
        critic: Optional[Critic] = None
    ):
        self.worker_id = worker_id
        self.capability_registry = capability_registry
        self.tool_registry = tool_registry
        self.llm_runtime = llm_runtime
        self.critic = critic
        self.bus: Optional[EventBus] = None
        self.plan: Optional[Plan] = None
        self.engine: Optional[ExecutionEngine] = None

    def set_event_bus(self, bus: EventBus) -> None:
        self.bus = bus

    def set_engine(self, engine: ExecutionEngine) -> None:
        self.engine = engine
        logger.info(f"[Worker {self.worker_id}] set_engine: engine_id={id(engine)}, lock_id={id(engine._step_execution_lock)}")

    async def on_step_ready(self, event: Event):
        """事件处理入口"""
        if event.event_type != EventType.STEP_READY:
            return
        step_id = event.step_id
        if not step_id or not self.engine:
            return

        # 【关键修复】在获取锁之前先快速检查多个状态
        if step_id in self.engine._completed_steps:
            logger.debug(f"[Worker {self.worker_id}] Step {step_id} already completed, skipping")
            return

        # 【关键修复】在锁内同时完成所有检查和 claim，避免竞态
        async with self.engine._step_execution_lock:
            # 检查所有状态
            if step_id in self.engine._completed_steps:
                logger.debug(f"[Worker {self.worker_id}] Step {step_id} already completed (in lock), skipping")
                return
            if step_id in self.engine._claiming_steps:
                logger.debug(f"[Worker {self.worker_id}] Step {step_id} already claimed (in lock), skipping")
                return
            if step_id in self.engine._executing_steps:
                logger.debug(f"[Worker {self.worker_id}] Step {step_id} already executing, skipping")
                return
            # 标记为正在执行
            self.engine._executing_steps.add(step_id)
            # 同时 claim（在锁内，防止竞态）
            self.engine._claiming_steps.add(step_id)

        # 从 DynamicPlan 获取 step
        dynamic_plan = getattr(self.engine, 'dynamic_plan', None)
        if dynamic_plan:
            step = dynamic_plan.steps.get(step_id)
        else:
            step = getattr(self.engine, 'plan', None)
            if step:
                step = step.steps.get(step_id)

        if not step:
            await self.engine.release_claim(step_id)
            async with self.engine._step_execution_lock:
                self.engine._executing_steps.discard(step_id)
            return

        # 【修复】不在锁内更新状态
        # 更新 StepPlan 状态为 RUNNING
        if self.engine.plan and step_id in self.engine.plan.steps:
            self.engine.plan.steps[step_id].status = StepState.RUNNING

        payload = event.payload or {}
        step_data = payload.get("step_data", {})
        input_data = step_data.get("input_data", {})
        if not input_data and hasattr(step, 'input_data'):
            input_data = step.input_data
        if not input_data and hasattr(step, 'task'):
            input_data = {"task": step.task}

        # 构建 StepContext
        context = await self._build_step_context(step, input_data)

        # 【修复】执行在锁外进行
        start_time = time.time()
        event_type = None
        payload = {}
        try:
            logger.info(f"[Worker {self.worker_id}] Executing step {step_id}...")

            # 步骤级超时（从配置获取，默认60秒）
            timeout = getattr(step, 'timeout', 60.0)
            success, result = await asyncio.wait_for(
                self._execute_with_context(step, context),
                timeout=timeout
            )

            duration_ms = int((time.time() - start_time) * 1000)
            logger.info(f"[Worker {self.worker_id}] Step {step_id} executed, success={success}, duration={duration_ms}ms")

            # 【修复】处理结果不在锁内进行，避免死锁
            event_type, payload = await self._handle_execution_result(step_id, step, success, result, duration_ms)

        except asyncio.TimeoutError:
            duration_ms = int((time.time() - start_time) * 1000)
            error_result = f"Error: Step execution timeout (>{timeout}s)"
            logger.warning(f"[Worker {self.worker_id}] Step {step_id} timeout after {timeout}s")

            event_type, payload = await self._handle_timeout(step_id, step, error_result, duration_ms)

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.exception(f"[Worker {self.worker_id}] Execution failed: {e}")

            event_type, payload = await self._handle_exception(step_id, step, e, duration_ms)

        finally:
            # 【调试】记录事件类型
            logger.debug(f"[Worker {self.worker_id}] Step {step_id} event_type={event_type}, success={success}")

            # 【关键修复】先标记已完成，再释放 claim，防止其他 Worker 抢走
            # 在这里直接添加到 _completed_steps，避免竞态条件
            # 同时更新 StepPlan 状态，确保 _check_and_publish_completion 能正确检测
            # 【修复】根据最终的事件类型设置正确的步骤状态
            final_status = StepState.COMPLETED if event_type == EventType.STEP_COMPLETED else StepState.FAILED
            async with self.engine._step_execution_lock:
                self.engine._completed_steps.add(step_id)
                self.engine._executing_steps.discard(step_id)
                # 更新 StepPlan 状态
                if self.engine.plan and step_id in self.engine.plan.steps:
                    self.engine.plan.steps[step_id].status = final_status
                # 更新 dynamic_plan 状态（如果存在）
                dynamic_plan = getattr(self.engine, 'dynamic_plan', None)
                if dynamic_plan and step_id in dynamic_plan.steps:
                    dynamic_plan.steps[step_id].status = final_status
            # 释放步骤声明（在标记完成后）
            await self.engine.release_claim(step_id)

            # 【修复】在释放锁之后发布事件，避免死锁
            if event_type:
                logger.debug(f"[Worker {self.worker_id}] About to publish {event_type} for {step_id[:8]}, queue size before: {self.bus._queue.qsize() if hasattr(self.bus, '_queue') else 'N/A'}")
                await self._publish_event(event_type, step_id, payload)
                logger.debug(f"[Worker {self.worker_id}] Published {event_type} for {step_id[:8]}, queue size after: {self.bus._queue.qsize() if hasattr(self.bus, '_queue') else 'N/A'}")

    async def _handle_execution_result(self, step_id: str, step: Step, success: bool, result: Any, duration_ms: int):
        """【新增】统一处理执行结果，返回要发布的事件类型和 payload"""
        # 【修复】正确提取实际输出
        output_str = extract_output(result)
        event_type = None
        payload = {}

        if success:
            # 检查是否是早停（保护机制触发）
            is_early_stopped = isinstance(result, dict) and str(result.get("warning", "")).startswith("early_stopped")

            if is_early_stopped:
                # 【修复】早停不是真正的成功，标记为失败
                logger.warning(f"[Worker {self.worker_id}] Step {step_id} early stopped by protection, marking as failed")
                # 使用警告信息作为错误消息
                error_msg = output_str if output_str else "Step stopped by protection mechanism"
                artifact = Artifact.create_error(error_msg, step_id=step_id)
                await self.engine.state.update_artifact(step_id, artifact)

                trace = StepTrace(
                    step_id=step_id,
                    agent="default",
                    mode="react",
                    duration_ms=duration_ms,
                    success=False,
                    error=error_msg
                )
                await self.engine.state.add_trace(trace)
                event_type = EventType.STEP_FAILED
                payload = {"error": error_msg}
            elif self.critic:
                # 进行质量检查
                artifact = Artifact.create_success(output_str, step_id=step_id)
                await self.engine.state.update_artifact(step_id, artifact)

                critique = await self.critic.evaluate(step, output_str, self.engine.state, caller=f"Worker-{self.worker_id}")
                if critique.quality_score < 0.5:
                    success = False
                    result_str = f"Quality check failed: {critique.suggestions}"
                    # 【修复】重新创建 artifact 表示失败
                    artifact = Artifact.create_error(result_str, value=output_str, step_id=step_id)
                    await self.engine.state.update_artifact(step_id, artifact)

                    trace = StepTrace(
                        step_id=step_id,
                        agent="default",
                        mode="react",
                        duration_ms=duration_ms,
                        success=False,
                        error=result_str
                    )
                    await self.engine.state.add_trace(trace)
                    event_type = EventType.STEP_FAILED
                    payload = {"error": result_str}
                else:
                    trace = StepTrace(
                        step_id=step_id,
                        agent="default",
                        mode="react",
                        duration_ms=duration_ms,
                        success=True
                    )
                    await self.engine.state.add_trace(trace)
                    event_type = EventType.STEP_COMPLETED
                    payload = {"output": output_str}
            else:
                artifact = Artifact.create_success(output_str, step_id=step_id)
                await self.engine.state.update_artifact(step_id, artifact)

                trace = StepTrace(
                    step_id=step_id,
                    agent="default",
                    mode="react",
                    duration_ms=duration_ms,
                    success=True
                )
                await self.engine.state.add_trace(trace)
                event_type = EventType.STEP_COMPLETED
                payload = {"output": output_str}
        else:
            # 【修复】正确处理失败情况
            error_str = str(result) if not isinstance(result, dict) else json.dumps(result, ensure_ascii=False)
            artifact = Artifact.create_error(error_str, step_id=step_id)
            await self.engine.state.update_artifact(step_id, artifact)

            trace = StepTrace(
                step_id=step_id,
                agent="default",
                mode="react",
                duration_ms=duration_ms,
                success=False,
                error=error_str
            )
            await self.engine.state.add_trace(trace)
            event_type = EventType.STEP_FAILED
            payload = {"error": error_str}

        return event_type, payload

    async def _handle_timeout(self, step_id: str, step: Step, error_result: str, duration_ms: int):
        """处理超时，返回要发布的事件类型和 payload"""
        _ = step  # 保留参数用于未来扩展
        artifact = Artifact.create_error(error_result, step_id=step_id)
        await self.engine.state.update_artifact(step_id, artifact)

        trace = StepTrace(
            step_id=step_id,
            agent="default",
            mode="react",
            duration_ms=duration_ms,
            success=False,
            error=error_result
        )
        await self.engine.state.add_trace(trace)
        return EventType.STEP_FAILED, {"error": error_result}

    async def _handle_exception(self, step_id: str, step: Step, exception: Exception, duration_ms: int):
        """处理异常，返回要发布的事件类型和 payload"""
        _ = step  # 保留参数用于未来扩展
        error_str = str(exception)
        artifact = Artifact.create_error(error_str, step_id=step_id)
        await self.engine.state.update_artifact(step_id, artifact)

        trace = StepTrace(
            step_id=step_id,
            agent="default",
            mode="react",
            duration_ms=duration_ms,
            success=False,
            error=error_str
        )
        await self.engine.state.add_trace(trace)
        return EventType.STEP_FAILED, {"error": error_str}

    async def _build_step_context(self, step: Step, input_data: Dict) -> Dict:
        """构建 StepContext（简化版本）"""
        depends_on = getattr(step, 'depends_on', [])
        task = step.input_data.get("task", "") if hasattr(step, 'input_data') else getattr(step, 'task', "")
        mode = getattr(step, 'mode', "react")

        # 【修复】异步获取 artifacts
        artifacts_snapshot = await self.engine.state.get_artifacts_snapshot()
        deps = {}
        for dep_id in depends_on:
            artifact = artifacts_snapshot.get(dep_id)
            if artifact:
                deps[dep_id] = artifact

        available_tools = self.capability_registry.get_executable_schemas()
        return {
            "goal": task,
            "task": task,
            "context": input_data,
            "deps": deps,
            "available_tools": available_tools,
            "config": {"mode": mode}
        }

    async def _execute_with_context(self, step: Step, context: Dict) -> Tuple[bool, Any]:  # noqa: F841
        """使用 StepContext 执行"""
        _ = context
        if step.step_type == "tool":
            return await self._execute_tool(step)
        elif step.step_type == "skill":
            return await self._execute_skill(step, context)
        else:
            return await self._execute_llm(step, context)

    async def _execute_tool(self, step: Step) -> Tuple[bool, Any]:
        """执行原子工具"""
        if not step.tool_name:
            return (False, "No tool_name specified")
        cap = self.capability_registry.get(step.tool_name)
        if not cap:
            return (False, f"Tool not found: {step.tool_name}")
        result = await cap.execute(**step.tool_args)
        return (True, result)

    async def _execute_skill(self, step: Step, context: Dict) -> Tuple[bool, Any]:  # noqa: F841
        """执行技能步骤"""
        _ = context
        logger.info(f"[Worker] _execute_skill: step_id={step.step_id}, skill_name={step.skill_name or step.input_data.get('skill_name')}")

        # 获取 context 信息
        context_manager = self.llm_runtime.context_manager
        if not context_manager:
            raise RuntimeError("context_manager not initialized in llm_runtime")

        ctx = context_manager.get_or_create(step)

        # 记录上下文信息
        context_info = {
            "step_id": ctx.step_id,
            "step_task": ctx.step_task,
            "task": ctx.task,
            "dependencies": list(ctx.dependencies.keys()),
            "relevant_artifacts": list(ctx.relevant_artifacts.keys()) if hasattr(ctx, 'relevant_artifacts') else [],
            "history_length": len(ctx.history) if hasattr(ctx, 'history') else 0,
            "tool_trace_length": len(ctx.tool_trace) if hasattr(ctx, 'tool_trace') else 0
        }
        logger.info(f"[Worker] Context info: {json.dumps(context_info, ensure_ascii=False, indent=2)}")
        try:
            skill_name = step.skill_name or step.input_data.get("skill_name")
            if not skill_name:
                return (False, "No skill_name specified")
            skill_cap = self.capability_registry.get(skill_name)
            if not isinstance(skill_cap, SkillCapability):
                return (False, f"Not a skill capability: {skill_name}")

            # 获取或创建 context（长生命周期）
            # context_manager 在 Worker 初始化时已从 runtime 获取，这里直接使用
            ctx = context_manager.get_or_create(step)

            # 将 context 传递给 SkillPolicy
            policy = SkillPolicy(skill_cap, self.llm_runtime, context_manager)
            return await policy.execute_with_policy(step, context=ctx, caller=f"Worker-{self.worker_id}")
        except Exception as e:
            logger.exception(f"[Worker] _execute_skill failed: {e}")
            return (False, f"Execution error: {e}")

    async def _execute_llm(self, step: Step, context: Dict) -> Tuple[bool, Any]:  # noqa: F841
        """执行通用 LLM 调用"""
        _ = context
        try:
            # 关键修复：使用 ContextFormatter 生成 prompt
            task = step.input_data.get("task", "")
            user_prompt = f"任务: {task}"
            tools = self.capability_registry.get_executable_schemas()

            context_manager = self.llm_runtime.context_manager
            if not context_manager:
                raise RuntimeError("context_manager not initialized in llm_runtime")
            ctx = context_manager.get_or_create(step)

            # 使用 ContextFormatter 格式化 prompt
            system_prompt = ContextFormatter.format_system(ctx, "你是一个智能助手。请根据任务完成目标。")

            success, result = await self.llm_runtime.tool_call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                tools=tools,
                context=ctx,  # 传递 context
                caller=f"Worker-{self.worker_id}-llm"
            )
            # 【修复】正确提取输出
            if success:
                return (True, extract_output(result))
            else:
                return (False, result.get("error", "Unknown error"))

        except Exception as e:
            logger.exception(f"[Worker] _execute_llm failed: {e}")
            return (False, f"Execution error: {e}")

    async def _publish_event(self, event_type: EventType, step_id: str, payload: Dict):
        """发布事件"""
        if self.bus:
            await self.bus.publish(Event(event_type=event_type, step_id=step_id, payload=payload))


# =========================
# 技能加载和解析
# =========================
def load_skills_from_directory(skills_dir: str = "skills") -> Dict[str, SkillCapability]:
    """
    从指定目录加载所有 Skills（符合渐进披露架构）
    """
    loaded_skills = {}
    if not os.path.exists(skills_dir):
        return loaded_skills

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
    skills_path = os.path.join(script_dir, skills_dir)

    if not os.path.exists(skills_path):
        return loaded_skills

    for item in os.listdir(skills_path):
        skill_dir = os.path.join(skills_path, item)

        if not os.path.isdir(skill_dir):
            continue

        skill_md_path = os.path.join(skill_dir, "SKILL.md")

        if not os.path.exists(skill_md_path):
            continue

        # Layer 1: 只提取 frontmatter
        try:
            with open(skill_md_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.warning(f"Failed to read SKILL.md for {item}: {e}")
            continue

        meta = _parse_skill_frontmatter(content)
        skill_name = meta.get("name", item)
        description = meta.get("description", "")

        # 只创建 SkillCapability，不加载完整内容
        skill_cap = SkillCapability(
            name=skill_name,
            description=description,
            markdown_path=skill_md_path,
            skill_dir=skill_dir
        )

        loaded_skills[skill_name] = skill_cap
        logger.info(f"[AgentOS] Loaded skill (Layer 1 only): {skill_name}")

    return loaded_skills


def _parse_skill_frontmatter(content: str) -> Dict[str, Any]:
    """
    解析技能 frontmatter（Layer 1 元数据）
    """
    meta = {}
    if not content.startswith('---'):
        return meta

    try:
        parts = content.split('---', 2)
        if len(parts) < 3:
            return meta

        yaml_content = parts[1].strip()

        if HAS_YAML:
            try:
                result = yaml.safe_load(yaml_content)
                if result:
                    meta["name"] = result.get("name", "")
                    meta["description"] = result.get("description", "")
                return meta
            except yaml.YAMLError:
                pass

        for line in yaml_content.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ':' in line:
                key, _, value = line.partition(':')
                key = key.strip()
                value = value.strip().strip('"\'')
                if key == "name":
                    meta["name"] = value
                elif key == "description":
                    meta["description"] = value
    except Exception as e:
        logger.warning(f"Failed to parse frontmatter: {e}")

    return meta


# =========================
# ProductionAgentOS（统一入口）
# =========================
class ProductionAgentOS:
    def __init__(
        self,
        worker_count: int = 2,
        skills_dir: str = "skills",
        tools_dir: str = "tools",
        max_react_iterations: int = 10
    ):
        self.worker_count = worker_count
        self.skills_dir = skills_dir
        self.tools_dir = tools_dir
        self.max_react_iterations = max_react_iterations
        self.capability_registry = CapabilityRegistry()
        self.tool_registry = ToolRegistry(tools_dir=tools_dir)
        self.bus: Optional[EventBus] = None
        self.llm: Optional[AsyncOpenAI] = None
        self.runtime: Optional[LLMRuntime] = None
        self.planner: Optional[Planner] = None
        self.engine: Optional[ExecutionEngine] = None
        self.critic: Optional[Critic] = None
        self.replanner: Optional[Replanner] = None
        self.error_recovery: Optional[ErrorRecovery] = None
        self.workers: List[Worker] = []
        self._completion_event: Optional[asyncio.Event] = None
        self._initialized = False

    async def initialize(self):
        if self._initialized:
            return

        # 创建 LLM
        self.llm = AsyncOpenAI(
            base_url=f"{VLLM_URL}/v1",
            api_key="sk-ignore",
            timeout=config_manager.LLM_CALL_TIMEOUT
        )

        # 关键修复：创建 ContextManager（长生命周期）
        context_manager = ContextManager()

        # 创建 LLMRuntime（统一的智能入口）
        self.runtime = LLMRuntime(
            self.llm,
            capability_registry=self.capability_registry,
            max_iterations=self.max_react_iterations,
            context_manager=context_manager  # 新增
        )

        # 创建 Planner
        self.planner = Planner(self.runtime)

        # 创建反馈闭环组件
        self.critic = Critic(self.runtime)
        self.replanner = Replanner(self.runtime)
        self.error_recovery = ErrorRecovery()

        # 创建 Engine
        self.engine = ExecutionEngine(
            self.tool_registry,
            self.capability_registry,
            self.planner,
            self.replanner,
            self.critic,
            self.error_recovery
        )

        self.bus = EventBus(max_queue_size=config_manager.EVENT_BUS_MAX_QUEUE_SIZE)
        self.engine.bus = self.bus
        await self.engine.initialize(self.bus)

        # 关键修复：将 context_manager 传递给 engine.state
        if hasattr(self.engine.state, 'context_manager'):
            self.engine.state.context_manager = context_manager

        # 加载工具
        self._register_tools()

        # 加载技能
        self._register_skills()

        self._initialized = True
        logger.info("[ProductionAgentOS] Initialized successfully")

    def _register_tools(self):
        """从 tools 目录加载并注册工具"""
        try:
            loaded_tools = self.tool_registry.load_tools_from_directory()
            logger.info(f"[AgentOS] Loaded tools: {list(loaded_tools.keys())}")

            # 注册 TavilySearchTool
            if 'TavilySearchTool' in loaded_tools:
                tool_class = loaded_tools['TavilySearchTool']
                search_tool = tool_class()
                schema = {
                    "type": "function",
                    "function": {
                        "name": "tavily_search",
                        "description": "使用 Tavily 进行网络搜索，适合查询最新信息、新闻、实时数据等",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "搜索查询词"},
                                "max_results": {"type": "integer", "description": "最大结果数", "default": 5},
                                "search_depth": {"type": "string", "description": "搜索深度", "default": "advanced"}
                            }
                        }
                    }
                }
                self.tool_registry.register_tool_instance(
                    name="tavily_search",
                    instance=search_tool,
                    schema=schema,
                    description="使用 Tavily 进行网络搜索"
                )
                logger.info("[AgentOS] Registered tavily_search tool")

            # 注册 FileReadTool
            if 'FileReadTool' in loaded_tools:
                tool_class = loaded_tools['FileReadTool']
                tool_instance = tool_class()
                schema = {
                    "type": "function",
                    "function": {
                        "name": "file_read",
                        "description": "读取文件内容",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "filepath": {"type": "string", "description": "文件路径"}
                            }
                        }
                    }
                }
                self.tool_registry.register_tool_instance(
                    name="file_read",
                    instance=tool_instance,
                    schema=schema,
                    description="读取文件内容"
                )
                logger.info("[AgentOS] Registered file_read tool")

            # 注册 FileWriteTool
            if 'FileWriteTool' in loaded_tools:
                tool_class = loaded_tools['FileWriteTool']
                tool_instance = tool_class()
                schema = {
                    "type": "function",
                    "function": {
                        "name": "file_write",
                        "description": "写入文件内容",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "filepath": {"type": "string", "description": "文件路径"},
                                "content": {"type": "string", "description": "文件内容"}
                            }
                        }
                    }
                }
                self.tool_registry.register_tool_instance(
                    name="file_write",
                    instance=tool_instance,
                    schema=schema,
                    description="写入文件内容"
                )
                logger.info("[AgentOS] Registered file_write tool")

            # 注册 BashTool
            if 'BashTool' in loaded_tools:
                tool_class = loaded_tools['BashTool']
                tool_instance = tool_class()
                schema = {
                    "type": "function",
                    "function": {
                        "name": "bash_execute",
                        "description": "执行 Bash 命令",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "command": {"type": "string", "description": "要执行的命令"}
                            }
                        }
                    }
                }
                self.tool_registry.register_tool_instance(
                    name="bash_execute",
                    instance=tool_instance,
                    schema=schema,
                    description="执行 Bash 命令"
                )
                logger.info("[AgentOS] Registered bash_execute tool")

        except Exception as e:
            logger.error(f"[AgentOS] Failed to load tools: {e}")

        # 初始化工具执行器池
        self._tool_executor_pool = ToolExecutorPool(max_workers=2)

        # 将工具注册到 CapabilityRegistry
        for tool_name in self.tool_registry._tool_instances.keys():
            tool_cap = ToolCapability(
                tool_name,
                self.tool_registry,
                executor_pool=self._tool_executor_pool
            )
            self.capability_registry.register(tool_cap)
            logger.info(f"[AgentOS] Registered capability: {tool_name}")

    def _register_skills(self):
        """从 skills 目录加载并注册技能到 CapabilityRegistry"""
        try:
            skills = load_skills_from_directory(self.skills_dir)
            for skill_name, skill_cap in skills.items():
                if isinstance(skill_cap, SkillCapability):
                    self.capability_registry.register(skill_cap)
                    logger.info(f"[AgentOS] Registered skill capability: {skill_name}")
                else:
                    logger.warning(f"[AgentOS] Invalid skill type for {skill_name}: {type(skill_cap)}")
        except Exception as e:
            logger.error(f"[AgentOS] Failed to load skills: {e}")

    async def _setup_completion_handlers(self):
        """设置完成事件处理器"""
        if self._completion_event:
            return

        async def on_task_completed(event: Event):
            if event.event_type != EventType.TASK_COMPLETED:
                return
            logger.info("[AgentOS] TASK_COMPLETED event received")
            if self._completion_event and not self._completion_event.is_set():
                self._completion_event.set()

        async def on_task_failed(event: Event):
            if event.event_type != EventType.TASK_FAILED:
                return
            logger.info("[AgentOS] TASK_FAILED event received")
            if self._completion_event and not self._completion_event.is_set():
                self._completion_event.set()

        if self.bus:
            self.bus.subscribe(on_task_completed)
            self.bus.subscribe(on_task_failed)
        logger.info("[ProductionAgentOS] Completion handlers registered")

    async def run(self, task: str, timeout: float = 300.0) -> Dict[str, Any]:
        logger.info("=" * 70)
        logger.info("任务启动 (DynamicPlan 框架)")
        logger.info("=" * 70)

        await self.initialize()

        # 设置完成处理器
        await self._setup_completion_handlers()
        self._completion_event = asyncio.Event()

        # 创建 Workers
        self.workers = [Worker(
            f"W-{i}",
            self.capability_registry,
            self.tool_registry,
            self.runtime,
            self.critic
        ) for i in range(self.worker_count)]
        for w in self.workers:
            w.set_event_bus(self.bus)
            w.set_engine(self.engine)

        # 订阅 Workers
        for w in self.workers:
            if self.bus:
                self.bus.subscribe(w.on_step_ready)

        # 加载计划
        tools = self.tool_registry._tools
        skills = self.capability_registry.get_instructable_schemas()
        plan = await self.planner.plan(task, list(tools.values()), skills, caller="ProductionAgentOS.run")
        logger.info(f"[AgentOS] Plan generated: plan_id={plan.plan_id}, steps={list(plan.steps.keys())}, plan_len={len(plan.steps)}")

        if not plan or not plan.steps:
            logger.error("[AgentOS] Plan is empty, cannot proceed")
            return {"status": "error", "error": "Plan generation failed - no steps produced"}

        # 设置 DynamicPlan（包含 Step 对象）用于执行
        self.engine.dynamic_plan = DynamicPlan()
        self.engine.dynamic_plan.steps = {
            step_id: Step(
                step_id=step_id,
                step_type="llm" if step_data.target_agent == "llm" else "skill",
                task=step_data.task,
                depends_on=step_data.depends_on,
                input_data=step_data.input_data,
                skill_name=step_data.target_agent
            )
            for step_id, step_data in plan.steps.items()
        }
        self.engine.dynamic_plan.dag = plan.dag

        self.engine.set_plan(plan)
        for w in self.workers:
            w.plan = self.engine.plan

        await self.engine.start()

        try:
            if self._completion_event:
                await asyncio.wait_for(self._completion_event.wait(), timeout=timeout)
            else:
                await asyncio.wait_for(self._wait_for_completion(timeout=timeout), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("[AgentOS] Timeout waiting for completion")

        # 返回结果
        completed_count = sum(1 for s in self.engine.plan.steps.values() if s.status == StepState.COMPLETED)
        failed_count = sum(1 for s in self.engine.plan.steps.values() if s.status == StepState.FAILED)
        total = len(self.engine.plan.steps)
        completed_from_artifacts = len(self.engine.state.artifacts)
        completed_count = max(completed_count, completed_from_artifacts)

        # 收集所有步骤的实际输出结果
        step_results = {}
        for step_id, step in self.engine.plan.steps.items():
            # 检查步骤是否有输出（通过 artifacts 获取）
            step_output = None
            if self.engine.state and self.engine.state.artifacts:
                # 查找与步骤相关的 artifact
                for artifact_id, artifact in self.engine.state.artifacts.items():
                    if artifact_id == step_id or artifact_id.startswith(step_id):
                        step_output = extract_output(artifact)
                        break
            
            if step.status == StepState.COMPLETED and step_output:
                # 清理输出格式，提取实际内容
                cleaned_output = self._clean_output(step_output)
                step_results[step_id] = {
                    "step_id": step_id,
                    "status": "completed",
                    "output": cleaned_output,
                    "skill_name": step.target_agent if hasattr(step, 'target_agent') else "unknown"
                }
            elif step.status == StepState.FAILED:
                step_results[step_id] = {
                    "step_id": step_id,
                    "status": "failed",
                    "error": step_output if step_output else "Unknown error",
                    "skill_name": step.target_agent if hasattr(step, 'target_agent') else "unknown"
                }

        # 收集最终结果（最后一个完成的步骤的输出）
        final_result = None
        if self.engine.state.artifacts:
            # 使用最后一个artifact作为最终结果
            final_artifact = list(self.engine.state.artifacts.values())[-1]
            final_result = self._clean_output(extract_output(final_artifact))
        elif step_results:
            # 使用最后一个完成的步骤的输出作为最终结果
            last_completed_step = None
            for step_id, result in step_results.items():
                if result["status"] == "completed":
                    last_completed_step = result
            if last_completed_step:
                final_result = last_completed_step["output"]

        await self._cleanup()

        return {
            "status": "completed" if failed_count == 0 and completed_count > 0 else "partial" if completed_count > 0 else "failed",
            "total_steps": total,
            "completed_steps": completed_count,
            "failed_steps": failed_count,
            "plan_id": plan.plan_id if plan else "unknown",
            "final_result": final_result,
            "step_results": step_results,
            "artifacts_count": len(self.engine.state.artifacts) if self.engine.state else 0
        }

    def _clean_output(self, output: str) -> str:
        """清理输出格式，提取实际内容"""
        if not output:
            return ""
        
        # 如果是 Artifact 对象，提取 value 字段
        if "Artifact(value=" in output:
            # 提取 Artifact 的 value 内容
            import re
            match = re.search(r"Artifact\(value='([^']*)'", output)
            if match:
                return match.group(1)
        
        # 如果是 JSON 字符串，尝试解析
        try:
            import json
            data = json.loads(output)
            if isinstance(data, dict):
                if "output" in data:
                    return str(data["output"])
                elif "result" in data:
                    return str(data["result"])
                elif "content" in data:
                    return str(data["content"])
            return str(data)
        except:
            pass
        
        # 返回原始输出
        return output

    async def _wait_for_completion(self, timeout: float):
        """等待完成（备用方案）"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.engine and self.engine.plan:
                steps = self.engine.plan.steps
                completed_count = sum(1 for s in steps.values() if s.status == StepState.COMPLETED)
                total = len(steps)
                if total > 0:
                    all_done = all(
                        s.status in [StepState.COMPLETED, StepState.FAILED, StepState.BLOCKED]
                        for s in steps.values()
                    )
                    if all_done:
                        logger.info(f"[AgentOS] _wait_for_completion: All steps completed ({completed_count}/{total})")
                        return
            await asyncio.sleep(0.5)
        logger.warning(f"[AgentOS] _wait_for_completion timeout after {timeout}s")

    async def _cleanup(self):
        """清理资源"""
        logger.info("[AgentOS] Cleaning up resources...")
        if self._completion_event:
            self._completion_event.clear()
            self._completion_event = None
        if self.engine:
            await self.engine.shutdown()
        if self._tool_executor_pool:
            await self._tool_executor_pool.shutdown()
        if self.llm:
            try:
                await self.llm.close()
            except Exception:
                logger.exception("[AgentOS] Error closing LLM")
        if self.bus:
            await self.bus.shutdown()
        logger.info("[AgentOS] Cleanup completed")

    async def shutdown(self):
        """关闭系统"""
        logger.info("[AgentOS] Shutdown initiated")
        if self.llm:
            try:
                await self.llm.close()
            except Exception:
                pass
        logger.info("[AgentOS] Shutdown completed")


# =========================
# 运行入口
# =========================
async def main():
    import traceback
    agent = ProductionAgentOS(worker_count=2, skills_dir="skills", tools_dir="tools")
    task = "分析一下今天的市场"
    try:
        result = await agent.run(task, timeout=300)
        
        # 返回完整的 JSON 结果
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
    except Exception as e:
        error_result = {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        print(json.dumps(error_result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
