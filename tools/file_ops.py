"""文件读写工具"""
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class FileReadTool:
    """文件读取工具"""

    def __init__(self, base_dir: str = "."):
        self.base_dir = base_dir

    async def execute(self, filepath: str) -> Dict[str, Any]:
        """执行文件读取（符合工具接口）"""
        return await self.read_file(filepath)

    def read_file_sync(self, filepath: str) -> Dict[str, Any]:
        """
        读取文件内容

        Args:
            filepath: 文件路径（相对于 base_dir）

        Returns:
            {
                "content": 文件内容,
                "success": bool,
                "error": str (如果失败)
            }
        """
        try:
            # 安全检查：防止目录穿越
            full_path = os.path.abspath(os.path.join(self.base_dir, filepath))
            if not full_path.startswith(os.path.abspath(self.base_dir)):
                return {
                    "content": "",
                    "success": False,
                    "error": "Invalid path: path traversal detected"
                }

            if not os.path.exists(full_path):
                return {
                    "content": "",
                    "success": False,
                    "error": f"File not found: {filepath}"
                }

            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()

            return {
                "content": content,
                "success": True,
                "metadata": {
                    "path": filepath,
                    "size": len(content),
                    "exists": True
                }
            }

        except UnicodeDecodeError:
            return {
                "content": "",
                "success": False,
                "error": "Failed to decode file as UTF-8"
            }
        except Exception as e:
            return {
                "content": "",
                "success": False,
                "error": str(e)
            }

    async def list_dir(self, dirpath: str = ".") -> Dict[str, Any]:
        """列出目录内容"""
        try:
            full_path = os.path.abspath(os.path.join(self.base_dir, dirpath))
            if not full_path.startswith(os.path.abspath(self.base_dir)):
                return {
                    "files": [],
                    "success": False,
                    "error": "Invalid path"
                }

            items = []
            for item in os.listdir(full_path):
                item_path = os.path.join(full_path, item)
                items.append({
                    "name": item,
                    "type": "dir" if os.path.isdir(item_path) else "file"
                })

            return {
                "files": items,
                "success": True,
                "metadata": {
                    "path": dirpath,
                    "count": len(items)
                }
            }
        except Exception as e:
            return {
                "files": [],
                "success": False,
                "error": str(e)
            }


class FileWriteTool:
    """文件写入工具"""

    def __init__(self, base_dir: str = "."):
        self.base_dir = base_dir

    async def write_file(
        self,
        filepath: str,
        content: str,
        mode: str = "w",
        overwrite: bool = False
    ) -> Dict[str, Any]:
        """
        写入文件内容

        Args:
            filepath: 文件路径
            content: 文件内容
            mode: 写入模式 ('w' 覆盖, 'a' 追加)
            overwrite: 是否覆盖已存在文件

        Returns:
            {
                "success": bool,
                "error": str (如果失败)
            }
        """
        try:
            # 安全检查
            full_path = os.path.abspath(os.path.join(self.base_dir, filepath))
            if not full_path.startswith(os.path.abspath(self.base_dir)):
                return {
                    "success": False,
                    "error": "Invalid path: path traversal detected"
                }

            # 检查文件是否存在
            if os.path.exists(full_path) and not overwrite and mode == "w":
                return {
                    "success": False,
                    "error": f"File already exists: {filepath}. Use overwrite=True to replace."
                }

            # 确保目录存在
            dir_path = os.path.dirname(full_path)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)

            with open(full_path, mode, encoding='utf-8') as f:
                f.write(content)

            return {
                "success": True,
                "metadata": {
                    "path": filepath,
                    "size": len(content),
                    "mode": mode
                }
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def append_file(self, filepath: str, content: str) -> Dict[str, Any]:
        """追加到文件"""
        return await self.write_file(filepath, content, mode="a", overwrite=False)

    async def execute(self, filepath: str, content: str, mode: str = "w", overwrite: bool = False) -> Dict[str, Any]:
        """执行文件写入（符合工具接口）"""
        return await self.write_file(filepath, content, mode=mode, overwrite=overwrite)
