"""为未覆盖的 E-AC-3 JOC/EMDF 变体生成结构化错误和维修报告。"""
import hashlib
import json
from pathlib import Path


def bytes_descriptor(data):
    """返回足以识别载荷、但不会复制整段载荷的摘要。"""
    raw = bytes(data)
    return {
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "prefix_hex": raw[:32].hex(),
        "suffix_hex": raw[-16:].hex() if raw else "",
    }


class UnsupportedVariantError(ValueError):
    """表示输入结构有效或疑似有效，但当前解析器没有覆盖该变体。"""

    def __init__(self, stage, variant, message, *, frame=None, details=None):
        self.stage = str(stage)
        self.variant = str(variant)
        self.message = str(message)
        self.frame = frame
        self.details = dict(details or {})
        super().__init__(self.__str__())

    def add_context(self, *, frame=None, details=None):
        if self.frame is None and frame is not None:
            self.frame = int(frame)
        if details:
            for key, value in details.items():
                self.details.setdefault(key, value)
        self.args = (self.__str__(),)
        return self

    def to_dict(self):
        return {
            "report_schema": 1,
            "error": "unsupported_variant",
            "stage": self.stage,
            "variant": self.variant,
            "frame": self.frame,
            "message": self.message,
            "details": self.details,
        }

    def __str__(self):
        where = f" frame={self.frame}" if self.frame is not None else ""
        detail = json.dumps(self.details, ensure_ascii=False, separators=(",", ":"))
        return f"[{self.stage}/{self.variant}]{where} {self.message}; details={detail}"


def write_variant_report(path, error, *, input_path=None, output_path=None):
    """将变体错误写成可交给后续维修工具的 JSON。"""
    report = error.to_dict()
    if input_path is not None:
        report["input"] = str(Path(input_path))
    if output_path is not None:
        report["requested_output"] = str(Path(output_path))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return target
