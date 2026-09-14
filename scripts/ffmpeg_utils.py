"""
ffmpeg / ffprobe 定位与调用工具。

跨平台解决「找到可用的 ffmpeg 二进制」这一前置问题，查找优先级：
1. 环境变量 FFMPEG_BINARY / FFPROBE_BINARY（显式指定，最高优先级）；
2. 系统 PATH 中的 ffmpeg / ffprobe；
3. imageio-ffmpeg 内置的 ffmpeg（pip 装上就有，无需系统安装）；
4. 若干常见安装路径（Windows / macOS / Linux）。

注意：imageio-ffmpeg 只带 ffmpeg、不带 ffprobe。
因此 probe_video 在找不到 ffprobe 时会退化为「解析 ffmpeg -i 的 stderr」。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import List, Optional, Sequence

# 需要在 Windows 下隐藏子进程弹出的黑框
_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


class FFmpegNotFoundError(RuntimeError):
    """找不到可用的 ffmpeg / ffprobe 时抛出，附带修复建议。"""


# --------------------------------------------------------------------------- #
# 二进制定位
# --------------------------------------------------------------------------- #
def _common_ffmpeg_paths() -> List[str]:
    """常见安装路径候选（按平台给出，找不到也无所谓）。"""
    home = os.path.expanduser("~")
    return [
        # Windows
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
        os.path.join(home, "scoop", "shims", "ffmpeg.exe"),
        # macOS（Homebrew）
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        # Linux
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
    ]


def _common_ffprobe_paths() -> List[str]:
    home = os.path.expanduser("~")
    return [
        r"C:\ffmpeg\bin\ffprobe.exe",
        r"C:\Program Files\ffmpeg\bin\ffprobe.exe",
        r"C:\ProgramData\chocolatey\bin\ffprobe.exe",
        os.path.join(home, "scoop", "shims", "ffprobe.exe"),
        "/opt/homebrew/bin/ffprobe",
        "/usr/local/bin/ffprobe",
        "/usr/bin/ffprobe",
        "/usr/local/bin/ffprobe",
    ]


def _try_imageio_ffmpeg() -> Optional[str]:
    """尝试用 imageio-ffmpeg 内置的 ffmpeg 二进制。"""
    try:
        import imageio_ffmpeg  # type: ignore

        path = imageio_ffmpeg.get_ffmpeg_exe()
        if path and os.path.exists(path):
            return path
    except Exception:
        pass
    return None


def find_ffmpeg() -> str:
    """返回可用的 ffmpeg 可执行文件路径。"""
    env = os.environ.get("FFMPEG_BINARY")
    if env and os.path.exists(env):
        return env

    which = shutil.which("ffmpeg")
    if which:
        return which

    for p in _common_ffmpeg_paths():
        if os.path.exists(p):
            return p

    builtin = _try_imageio_ffmpeg()
    if builtin:
        return builtin

    raise FFmpegNotFoundError(
        "未找到 ffmpeg。请任选一种方式安装：\n"
        "  1) pip install imageio-ffmpeg   （最省事，自带二进制，推荐）\n"
        "  2) 设置环境变量 FFMPEG_BINARY 指向 ffmpeg 可执行文件\n"
        "  3) 系统安装：winget install Gyan.FFmpeg / brew install ffmpeg / apt install ffmpeg"
    )


def find_ffprobe() -> Optional[str]:
    """返回可用的 ffprobe 路径；找不到时返回 None（调用方可退化为解析 ffmpeg 输出）。"""
    env = os.environ.get("FFPROBE_BINARY")
    if env and os.path.exists(env):
        return env

    which = shutil.which("ffprobe")
    if which:
        return which

    for p in _common_ffprobe_paths():
        if os.path.exists(p):
            return p

    # imageio-ffmpeg 目录里偶尔会附带 ffprobe，顺手找一下
    ff = _try_imageio_ffmpeg()
    if ff:
        cand = os.path.join(os.path.dirname(ff), "ffprobe.exe" if sys.platform == "win32" else "ffprobe")
        if os.path.exists(cand):
            return cand

    return None


# 进程内缓存，避免重复探测
_CACHE: dict = {}


def ffmpeg_bin() -> str:
    if "ffmpeg" not in _CACHE:
        _CACHE["ffmpeg"] = find_ffmpeg()
    return _CACHE["ffmpeg"]


def ffprobe_bin() -> Optional[str]:
    if "ffprobe" not in _CACHE:
        _CACHE["ffprobe"] = find_ffprobe()
    return _CACHE["ffprobe"]


# --------------------------------------------------------------------------- #
# 命令执行
# --------------------------------------------------------------------------- #
def run(cmd: Sequence[str], capture: bool = True, timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    """执行子进程命令（不经过 shell，避免路径注入与转义问题）。

    Args:
        cmd: 参数列表，例如 [ffmpeg, "-i", "a.mp4", ...]
        capture: True 时捕获 stdout/stderr 为文本
        timeout: 秒；超时抛 subprocess.TimeoutExpired
    """
    return subprocess.run(
        list(cmd),
        capture_output=capture,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        creationflags=_CREATE_NO_WINDOW,
    )


def ffmpeg(*args: str, timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    """便捷调用：ffmpeg_bin() + 参数。"""
    return run([ffmpeg_bin(), *args], timeout=timeout)
