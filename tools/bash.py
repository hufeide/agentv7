"""Bash 命令执行工具"""
import asyncio
import logging
import shlex
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

NETWORK_TIMEOUT_COMMANDS = ["curl", "wget", "ssh", "nc", "telnet"]
DEFAULT_TIMEOUT = 5
MAX_TIMEOUT = 10


def _inject_timeout_flags(command: str, timeout: float) -> str:
    """
    为网络命令自动注入超时参数，防止命令永久挂起

    Args:
        command: 原始命令
        timeout: 超时秒数

    Returns:
        修改后的命令（如果需要），否则原样返回
    """
    parts = command.strip().split()
    if not parts:
        return command

    cmd_name = parts[0]

    if cmd_name == "curl" and "--max-time" not in command and "-m" not in command:
        safe_timeout = min(timeout, MAX_TIMEOUT)
        return f"curl --max-time {safe_timeout} {command[6:].strip()}"
    elif cmd_name == "wget" and "--timeout" not in command and "-T" not in command:
        safe_timeout = min(timeout, MAX_TIMEOUT)
        return f"wget --timeout {safe_timeout} {command[6:].strip()}"
    return command


class BashTool:
    """Bash 命令执行工具"""

    def __init__(self, timeout: float = DEFAULT_TIMEOUT, max_output_size: int = 10000):
        self.timeout = timeout
        self.max_output_size = max_output_size

    async def execute(self, command: str) -> Dict[str, Any]:
        """
        执行 Bash 命令

        Args:
            command: 要执行的命令

        Returns:
            {
                "success": bool,
                "stdout": 标准输出,
                "stderr": 标准错误,
                "exit_code": 退出码,
                "error": 错误信息 (如果失败)
            }
        """
        safe_command = _inject_timeout_flags(command, self.timeout)
        if safe_command != command:
            logger.info(f"[Bash] Original command too long, using: {safe_command[:100]}...")

        logger.info(f"[Bash] Executing: {safe_command[:100]}...")

        try:
            process = await asyncio.create_subprocess_shell(
                safe_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout + 5.0
                )

                stdout_text = stdout.decode('utf-8', errors='ignore')
                stderr_text = stderr.decode('utf-8', errors='ignore')

                if len(stdout_text) > self.max_output_size:
                    stdout_text = stdout_text[:self.max_output_size] + "\n... (output truncated)"
                if len(stderr_text) > self.max_output_size:
                    stderr_text = stderr_text[:self.max_output_size] + "\n... (error truncated)"

                exit_code = process.returncode
                success = exit_code == 0

                logger.info(f"[Bash] Exit code: {exit_code}, Success: {success}")

                return {
                    "success": success,
                    "stdout": stdout_text,
                    "stderr": stderr_text,
                    "exit_code": exit_code
                }

            except asyncio.TimeoutError:
                try:
                    process.kill()
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning(f"[Bash] Process kill timeout, force terminating")
                logger.error(f"[Bash] Command timed out after {self.timeout}s: {safe_command[:80]}")
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": f"Command timed out after {self.timeout}s",
                    "exit_code": -1,
                    "error": f"Timeout: command took more than {self.timeout}s"
                }

        except Exception as e:
            logger.error(f"[Bash] Execution failed: {e}")
            return {
                "success": False,
                "stdout": "",
                "stderr": "",
                "exit_code": -1,
                "error": str(e)
            }

    async def execute_with_timeout(self, command: str, timeout: float = None) -> Dict[str, Any]:
        """执行带自定义超时的命令"""
        original_timeout = self.timeout
        if timeout:
            self.timeout = timeout
        try:
            return await self.execute(command)
        finally:
            self.timeout = original_timeout

    async def run_python(self, code: str, filename: str = "script.py") -> Dict[str, Any]:
        """运行 Python 代码"""
        # 写入临时文件
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name

        try:
            result = await self.execute(f"python3 {temp_file}")
            return result
        finally:
            try:
                os.unlink(temp_file)
            except:
                pass

    async def run_script(self, script: str, language: str = "bash") -> Dict[str, Any]:
        """运行脚本"""
        if language == "python":
            return await self.run_python(script)
        else:
            return await self.execute(script)
