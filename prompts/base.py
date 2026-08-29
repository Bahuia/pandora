"""
Pandora Prompt Template System

Jinja2-based template engine for all prompt generation.
"""

import re
from pathlib import Path
from typing import Any, Optional, Dict
import sys
sys.path.append("..")
from jinja2 import Template, FileSystemLoader, Environment
from utils.logger import setup_logger


class PromptTemplate:
    """
    Single prompt template with variable substitution.
    """

    def __init__(self, content: str, name: str = "unnamed"):
        """
        Initialize prompt template.

        Args:
            content: Template string with {{ variables }}
            name: Template name for logging
        """
        self.content = content
        self.name = name
        self.logger = setup_logger("pandora.prompts")
        self.template = Template(content)

    def render(self, **kwargs) -> str:
        """
        Render template with provided variables.

        Args:
            **kwargs: Variables to substitute

        Returns:
            Rendered prompt string
        """
        try:
            return self.template.render(**kwargs)
        except Exception as e:
            self.logger.error(f"Template rendering failed for {self.name}: {e}")
            raise

    @classmethod
    def from_file(cls, file_path: str, name: Optional[str] = None) -> "PromptTemplate":
        """
        Load template from file.

        Args:
            file_path: Path to template file
            name: Optional template name

        Returns:
            PromptTemplate instance
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Template file not found: {file_path}")

        content = path.read_text(encoding="utf-8")
        template_name = name or path.stem

        return cls(content, name=template_name)


class TemplateEngine:
    """
    Template engine for managing and rendering prompts.

    Loads task-specific Jinja templates. Demonstrations are supplied by the
    verified-memory retriever in ``core.agent`` rather than static data files.
    """

    def __init__(
        self,
        template_dir: str = "./prompts",
    ):
        """
        Initialize template engine.

        Args:
            template_dir: Directory containing template files
        """
        self.template_dir = Path(template_dir)
        self.logger = setup_logger("pandora.prompts.engine")

        # Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(self.template_dir),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Cache for loaded templates
        self._templates: Dict[str, PromptTemplate] = {}


    def get_template(self, name: str, task: Optional[str] = None) -> PromptTemplate:
        """
        Get a template by name.

        Args:
            name: Template name (e.g., "schema_linking", "code_reasoning")
            task: Optional task name for task-specific templates

        Returns:
            PromptTemplate instance
        """
        # Check cache
        cache_key = f"{task}/{name}" if task else name
        if cache_key in self._templates:
            return self._templates[cache_key]

        # Try task-specific template first
        if task:
            paths = [
                self.template_dir / task / f"{name}.txt",
                self.template_dir / "tasks" / task / f"{name}.txt",
            ]
        else:
            paths = [
                self.template_dir / f"{name}.txt",
            ]

        # Find first existing path
        template_path = None
        for path in paths:
            if path.exists():
                template_path = path
                break

        if not template_path:
            raise FileNotFoundError(
                f"Template '{name}' not found (task={task}). "
                f"Searched: {[str(p) for p in paths]}"
            )

        # Load template
        template = PromptTemplate.from_file(str(template_path), name=cache_key)
        self._templates[cache_key] = template

        self.logger.debug(f"Loaded template: {cache_key} from {template_path}")

        return template

    def create_system_message(self, task: str) -> str:
        """
        Create system message for a task.

        Args:
            task: Task name

        Returns:
            System message string
        """
        system_file = self.template_dir / "shared" / "system_instructions.txt"

        if system_file.exists():
            template = PromptTemplate.from_file(str(system_file), "system")
            return template.render(task=task)

        # Default system message
        return f"""You are an expert AI assistant for {task.upper()} tasks.
Your role is to generate accurate Python Pandas code to answer questions based on structured data.

Follow these guidelines:
1. Think step-by-step before writing code
2. Use the provided schema to understand data structure
3. Write clean, efficient, and correct code
4. Handle edge cases appropriately
5. Return your answer in the specified JSON format"""
