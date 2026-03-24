import asyncio
import logging
import sys
import threading
from typing import Dict, Any, Optional
from datetime import datetime
import uuid

from agent_v7 import ProductionAgentOS

logger = logging.getLogger(__name__)


class LogCapture:
    def __init__(self):
        self.logs = []
        self.lock = threading.Lock()

    def add_log(self, level: str, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = {
            "timestamp": timestamp,
            "level": level,
            "message": message
        }
        with self.lock:
            self.logs.append(log_entry)

    def get_logs(self) -> list:
        with self.lock:
            return self.logs.copy()

    def clear_logs(self):
        with self.lock:
            self.logs.clear()


class AgentTask:
    def __init__(self, task_id: str, task_description: str):
        self.task_id = task_id
        self.task_description = task_description
        self.status = "pending"
        self.result = None
        self.error = None
        self.log_capture = LogCapture()
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None


class AgentWrapper:
    def __init__(self):
        self.tasks: Dict[str, AgentTask] = {}
        self.loop = None
        self.loop_thread = None
        self._initialized = False

    def _ensure_initialized(self):
        if not self._initialized:
            self._init_event_loop()
            self._initialized = True

    def _init_event_loop(self):
        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()

        self.loop_thread = threading.Thread(target=run_loop, daemon=True)
        self.loop_thread.start()
        
        while self.loop is None:
            threading.Event().wait(0.01)

    def create_task(self, task_description: str) -> str:
        self._ensure_initialized()
        task_id = str(uuid.uuid4())
        task = AgentTask(task_id, task_description)
        self.tasks[task_id] = task
        return task_id

    def execute_task(self, task_id: str):
        self._ensure_initialized()
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")

        task.status = "running"
        task.started_at = datetime.now()
        task.log_capture.add_log("INFO", f"Starting task: {task.task_description}")

        try:
            future = asyncio.run_coroutine_threadsafe(
                self._run_agent(task),
                self.loop
            )
            future.result(timeout=300)
        except asyncio.TimeoutError:
            task.status = "failed"
            task.error = "Task execution timeout (300s)"
            task.log_capture.add_log("ERROR", "Task execution timeout")
        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            task.log_capture.add_log("ERROR", f"Task failed: {str(e)}")

        task.completed_at = datetime.now()

    async def _run_agent(self, task: AgentTask):
        class LogHandler(logging.Handler):
            def __init__(self, capture):
                super().__init__()
                self.capture = capture

            def emit(self, record):
                message = self.format(record)
                self.capture.add_log(record.levelname, message)

        log_handler = LogHandler(task.log_capture)
        log_handler.setFormatter(logging.Formatter('%(message)s'))
        
        root_logger = logging.getLogger()
        root_logger.addHandler(log_handler)

        try:
            agent = ProductionAgentOS(worker_count=2, skills_dir="skills", tools_dir="tools")
            result = await agent.run(task.task_description, timeout=300)
            task.status = "completed"
            task.result = result
            task.log_capture.add_log("INFO", "Task completed successfully")
        finally:
            root_logger.removeHandler(log_handler)

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        task = self.tasks.get(task_id)
        if not task:
            return None

        return {
            "task_id": task.task_id,
            "status": task.status,
            "result": task.result,
            "error": task.error,
            "created_at": task.created_at.isoformat() if task.created_at else None,
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None
        }

    def get_task_logs(self, task_id: str) -> list:
        task = self.tasks.get(task_id)
        if not task:
            return []
        return task.log_capture.get_logs()

    def cleanup_old_tasks(self, max_age_hours: int = 1):
        """清理超过指定时间的旧任务"""
        cutoff = datetime.now().timestamp() - max_age_hours * 3600
        to_remove = [
            task_id for task_id, task in self.tasks.items()
            if task.created_at.timestamp() < cutoff
        ]
        for task_id in to_remove:
            del self.tasks[task_id]
        return len(to_remove)
