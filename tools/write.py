"""文件读写工具"""
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)



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
            overwrite: 是否覆盖已存在文件 (支持布尔值或字符串 'True'/'False')

        Returns:
            {
                "success": bool,
                "error": str (如果失败)
            }
        """
        try:
            # 统一路径分隔符，处理 Windows 风格路径
            filepath = filepath.replace("\\", "/")

            # 安全检查：防止目录穿越
            base_dir = self.base_dir
            full_path = os.path.abspath(os.path.join(base_dir, filepath))
            real_base = os.path.abspath(base_dir)

            # 允许写入当前目录或子目录，或者 /tmp 目录
            allowed = (
                full_path.startswith(real_base) or
                full_path.startswith("/tmp/")
            )
            if not allowed:
                return {
                    "success": False,
                    "error": f"Invalid path: path traversal detected. Base: {real_base}, Attempted: {full_path}"
                }

            # 检查文件是否存在
            # 统一 overwrite 参数类型
            overwrite_bool = overwrite if isinstance(overwrite, bool) else str(overwrite).lower() == 'true'
            if os.path.exists(full_path) and not overwrite_bool and mode == "w":
                return {
                    "success": False,
                    "error": f"File already exists: {filepath}. Use overwrite=True to replace."
                }

            # 确保目录存在（对于 /tmp 路径，使用 /tmp 作为 base_dir）
            dir_path = os.path.dirname(full_path)
            if dir_path:
                # 如果是 /tmp 路径，确保 /tmp 存在
                if full_path.startswith("/tmp/"):
                    tmp_dir = "/tmp"
                    if not os.path.exists(tmp_dir):
                        os.makedirs(tmp_dir, exist_ok=True)
                elif not os.path.exists(dir_path):
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
        # 确保 overwrite 是布尔值
        overwrite_bool = overwrite if isinstance(overwrite, bool) else str(overwrite).lower() == 'true'
        return await self.write_file(filepath, content, mode=mode, overwrite=overwrite_bool)
