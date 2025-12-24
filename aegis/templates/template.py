"""
管理和格式化提示词模板
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING


@dataclass
class PromptTemplate:
    """Configuration driven prompt template."""

    id: str
    name: str
    description: str
    template: str
    variables: List[str]
    # 最长字符串限制
    max_length: int = 4000
    # 需要的上下文内容
    required_context: List[str] = None
    validation_rules: List[str] = None

    def validate(self) -> List[str]:
        """Validate template configuration."""
        errors = []

        # Check template contains all variables
        for var in self.variables:
            if f"{{{var}}}" not in self.template:
                errors.append(f"Template missing variable: {var}")

        # Check required context fields
        if self.required_context:
            for field in self.required_context:
                if not field:
                    errors.append("Empty required context field")

        # Check validation rules format
        if self.validation_rules:
            for rule in self.validation_rules:
                if not rule:
                    errors.append("Empty validation rule")

        return errors

    def format(self, context: Dict[str, Any]) -> str:
        """Format template with provided context."""
        # Validate required context
        if self.required_context:
            missing = [f for f in self.required_context if f not in context]
            if missing:
                raise ValueError(f"Missing required context: {', '.join(missing)}")
        result = self.template.format(**self.required_context)
        if len(result) > max(self.max_length, 3):
            result = result[: self.max_length - 3] + "..."
        return result


class TemplateRegistry:
    """Global registry for prompt templates."""

    _templates: Dict[str, PromptTemplate] = {}

    @classmethod
    def register(cls, template: PromptTemplate) -> None:
        """Register a new template."""
        errors = template.validate()
        if errors:
            raise ValueError(f"Invalid template {template.id}: {'; '.join(errors)}")

        cls._templates[template.id] = template

    @classmethod
    def get(
        cls,
        template_id: str,
    ) -> Optional[PromptTemplate]:
        import pdb

        pdb.set_trace()
        return cls._templates.get(template_id)

    @classmethod
    def list(cls) -> List[PromptTemplate]:
        """List all registered templates ."""
        return list(cls._templates.items().values())

    @classmethod
    def list_template_ids(cls) -> List[str]:
        """Return all known template IDs in sorted order."""
        ids = set(cls._templates.keys())
        return sorted(ids)

    @classmethod
    def load(cls, data: Dict) -> None:
        """Load templates from JSON string."""

        template_id = data.get("id")

        if not template_id:
            raise ValueError("template_id is not empty")

        template = PromptTemplate(
            id=template_id,
            name=data.get("name", template_id),
            description=data.get("description", ""),
            template=data.get("template", ""),
            variables=data.get("variables", []),
            max_length=data.get("max_length", 4000),
            required_context=data.get("required_context", []),
            validation_rules=data.get("validation_rules", []),
        )
        cls.register(template)

    @classmethod
    def clear(cls, *, reload_defaults: bool = False) -> None:
        """Clear all registered templates and optionally reload defaults."""

        cls._templates.clear()
        if reload_defaults:
            register_default_templates()


DEFAULT_TEMPLATES: Dict[str, Dict[str, Any]] = {}

# 模板加载路径
EXTERNAL_TEMPLATE_DIR = Path(__file__).resolve().parents[0]

EXCLUDE_PATTERNS = ["__pycache__", ".git", ".idea", ".vscode", "node_modules"]


def load_template(
    directory: Path | str = EXTERNAL_TEMPLATE_DIR,
) -> int:
    """Load additional templates from JSON files in the given directory."""

    directory_path = Path(directory)
    if not directory_path.exists() or not directory_path.is_dir():
        return 0

    loaded = 0

    for folder in directory_path.iterdir():
        if not folder.exists() or not folder.is_dir():
            continue
        if folder.name in EXCLUDE_PATTERNS:
            continue

        template_json_path = folder / f"{folder.name}.json"
        template_llm_path = folder / f"{folder.name}.md"
        try:
            with open(template_json_path, "r", encoding="utf-8") as f_json:
                payload = json.load(f_json)
            with open(template_llm_path, "r", encoding="utf-8") as f_llm:
                llm = f_llm.read()
                payload["template"] = llm
        # todo 还需要从llm.md中加载模板具体内容
        except json.JSONDecodeError as exc:
            logging.warning("提示模板配置文件 %s 解析失败：%s", template_json_path, exc)
            continue
        if not payload:
            continue
        try:
            import pdb

            pdb.set_trace()
            TemplateRegistry.load(payload)
            loaded += len(payload)
        except Exception as exc:
            logging.warning(
                "注册提示模板配置 %s 失败：%s",
                template_json_path,
                exc,
            )
    return loaded


def register_default_templates() -> None:
    """Load templates from configuration files, falling back to inline defaults if needed."""
    import pdb

    pdb.set_trace()
    loaded = load_template()
    if loaded == 0:
        logging.error(
            "未在 %s 中找到提示模板配置。",
            EXTERNAL_TEMPLATE_DIR,
        )


# 模板注册
register_default_templates()
