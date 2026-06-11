"""文本预处理器 — 对输入文本进行标准化处理，消除常见绕过手段"""

import re
import unicodedata
from typing import Dict, Set

from app.logger import get_logger

logger = get_logger("preprocessor")

# 常见变体字符映射表
_VARIANT_MAP: Dict[str, str] = {
    # 数字替代字母
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "8": "b",
    # 全角字符 → 半角
    "０": "0",
    "１": "1",
    "２": "2",
    "３": "3",
    "４": "4",
    "５": "5",
    "６": "6",
    "７": "7",
    "８": "8",
    "９": "9",
    "Ａ": "a",
    "Ｂ": "b",
    "Ｃ": "c",
    "Ｄ": "d",
    "Ｅ": "e",
    "Ｆ": "f",
    "Ｇ": "g",
    "Ｈ": "h",
    "Ｉ": "i",
    "Ｊ": "j",
    "Ｋ": "k",
    "Ｌ": "l",
    "Ｍ": "m",
    "Ｎ": "n",
    "Ｏ": "o",
    "Ｐ": "p",
    "Ｑ": "q",
    "Ｒ": "r",
    "Ｓ": "s",
    "Ｔ": "t",
    "Ｕ": "u",
    "Ｖ": "v",
    "Ｗ": "w",
    "Ｘ": "x",
    "Ｙ": "y",
    "Ｚ": "z",
    "ａ": "a",
    "ｂ": "b",
    "ｃ": "c",
    "ｄ": "d",
    "ｅ": "e",
    "ｆ": "f",
    "ｇ": "g",
    "ｈ": "h",
    "ｉ": "i",
    "ｊ": "j",
    "ｋ": "k",
    "ｌ": "l",
    "ｍ": "m",
    "ｎ": "n",
    "ｏ": "o",
    "ｐ": "p",
    "ｑ": "q",
    "ｒ": "r",
    "ｓ": "s",
    "ｔ": "t",
    "ｕ": "u",
    "ｖ": "v",
    "ｗ": "w",
    "ｘ": "x",
    "ｙ": "y",
    "ｚ": "z",
    # 同音/形近替代
    "艹": "草",
    "操": "操",
    "鈤": "日",
    "曰": "日",
}


class Preprocessor:
    """
    文本预处理器

    处理流水线：
    原始文本 → Unicode归一化 → 去除干扰符 → 繁简转换 → 大小写统一 → 变体还原 → 净化文本
    """

    def __init__(
        self,
        skip_chars: str = "",
        traditional_to_simplified: bool = True,
        normalize_variants: bool = True,
        remove_special_chars: bool = True,
    ):
        self._skip_chars: Set[str] = set(skip_chars)
        self._do_t2s = traditional_to_simplified
        self._do_variants = normalize_variants
        self._do_remove_special = remove_special_chars
        self._t2s_converter = None

        # 延迟加载繁简转换器
        if self._do_t2s:
            self._init_t2s()

    def _init_t2s(self) -> None:
        """初始化繁体转简体转换器"""
        try:
            from opencc import OpenCC
            self._t2s_converter = OpenCC("t2s")
        except ImportError:
            logger.warning("opencc 未安装，繁简转换功能不可用")
            self._do_t2s = False
        except Exception as e:
            logger.warning(f"繁简转换器初始化失败: {e}")
            self._do_t2s = False

    def normalize(self, text: str) -> str:
        """
        主入口：对文本执行完整的标准化流水线

        Args:
            text: 原始文本

        Returns:
            标准化后的文本
        """
        if not text:
            return ""

        # Step 1: Unicode 归一化 (NFKC)
        text = self._unicode_normalize(text)

        # Step 2: 去除干扰字符
        if self._do_remove_special:
            text = self._remove_skip_chars(text)

        # Step 3: 繁体转简体
        if self._do_t2s and self._t2s_converter:
            text = self._traditional_to_simplified(text)

        # Step 4: 统一小写
        text = text.lower()

        # Step 5: 变体字还原
        if self._do_variants:
            text = self._normalize_variants(text)

        return text

    def _unicode_normalize(self, text: str) -> str:
        """Unicode NFKC 归一化"""
        return unicodedata.normalize("NFKC", text)

    def _remove_skip_chars(self, text: str) -> str:
        """去除可忽略的干扰字符"""
        if not self._skip_chars:
            return text
        return "".join(ch for ch in text if ch not in self._skip_chars)

    def _traditional_to_simplified(self, text: str) -> str:
        """繁体转简体"""
        if self._t2s_converter:
            try:
                return self._t2s_converter.convert(text)
            except Exception:
                return text
        return text

    def _normalize_variants(self, text: str) -> str:
        """变体字还原"""
        result = []
        for ch in text:
            result.append(_VARIANT_MAP.get(ch, ch))
        return "".join(result)
