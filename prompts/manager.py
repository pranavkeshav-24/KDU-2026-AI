# prompts/manager.py
import yaml
from pathlib import Path
from typing import Optional, List
from jinja2 import Template
from dataclasses import dataclass
import threading


@dataclass
class PromptDefinition:
    name: str
    category: str
    version: str
    system_prompt: str
    user_template: str
    metadata: dict
    few_shot_examples: list


class PromptManager:
    """
    Loads, versions, and renders prompt templates from YAML files.

    Versioning strategy:
    - Each category has versioned YAML files: v1.yaml, v2.yaml, etc.
    - Router requests specific version or 'latest'
    - 'latest' always resolves to the highest version with status=active
    - Deprecated versions still work but log a warning

    AWS Future Replacement: Amazon S3 + Amazon DynamoDB
    ────────────────────────────────────────────────────
    Replace local YAML files with:
    - Amazon S3 for prompt artifact storage (S3 versioning enabled,
      every prompt change creates a new object version — full rollback)
    - Amazon DynamoDB for prompt registry metadata, version resolution,
      and status queries (active/deprecated/experimental)
    - AWS Lambda@Edge for globally cached prompt delivery (< 10ms anywhere)

    This gives global availability, versioned rollback, audit trail,
    and enables A/B testing different prompt versions via DynamoDB flags.

    Migration path:
      1. Upload prompt YAMLs to S3 bucket with versioning enabled
      2. Store metadata + active version mapping in DynamoDB table
      3. PromptManager fetches from S3 via presigned URL or SDK
      4. Cache in Lambda /tmp for the invocation lifetime
    """

    def __init__(self, registry_path: str = None):
        if registry_path is None:
            registry_path = str(Path(__file__).parent)
        self._registry_path = Path(registry_path)
        self._cache: dict = {}
        self._lock = threading.Lock()

    def render(
        self, category: str, version: str = "v1", variables: dict = None
    ) -> str:
        """Returns fully rendered prompt string ready for LLM invocation."""
        prompt_def = self._load(category, version)
        vars_ = variables or {}

        system = prompt_def.system_prompt.strip()
        user_tmpl = Template(prompt_def.user_template)
        user = user_tmpl.render(**vars_).strip()

        # Inject few-shot examples if present
        if prompt_def.few_shot_examples:
            examples_block = self._format_examples(prompt_def.few_shot_examples)
            system = system + "\n\n" + examples_block

        return f"[SYSTEM]\n{system}\n\n[USER]\n{user}"

    def get_metadata(self, category: str, version: str = "v1") -> dict:
        """Returns prompt metadata without rendering — used by eval framework."""
        prompt_def = self._load(category, version)
        return prompt_def.metadata

    def list_versions(self, category: str) -> List[str]:
        """Lists all available versions for a category."""
        category_path = self._registry_path / category
        if not category_path.exists():
            return []
        return sorted([f.stem for f in category_path.glob("v*.yaml")])

    def list_categories(self) -> List[str]:
        """Lists all categories that have prompt files."""
        if not self._registry_path.exists():
            return []
        return [
            d.name for d in self._registry_path.iterdir()
            if d.is_dir() and not d.name.startswith("_") and not d.name.startswith(".")
            and any(d.glob("v*.yaml"))
        ]

    def _load(self, category: str, version: str) -> PromptDefinition:
        cache_key = f"{category}:{version}"
        with self._lock:
            if cache_key in self._cache:
                return self._cache[cache_key]

            prompt_path = self._registry_path / category / f"{version}.yaml"
            if not prompt_path.exists():
                # Fallback to v1 if requested version doesn't exist
                prompt_path = self._registry_path / category / "v1.yaml"
            if not prompt_path.exists():
                # Ultimate fallback: use the generic fallback prompt
                prompt_path = self._registry_path / "fallback" / "v1.yaml"

            with open(prompt_path) as f:
                data = yaml.safe_load(f)

            prompt_def = PromptDefinition(
                name=data["metadata"]["name"],
                category=data["metadata"]["category"],
                version=data["metadata"]["version"],
                system_prompt=data["system_prompt"],
                user_template=data["user_template"],
                metadata=data["metadata"],
                few_shot_examples=data.get("few_shot_examples", []),
            )
            self._cache[cache_key] = prompt_def
            return prompt_def

    def _format_examples(self, examples: list) -> str:
        formatted = ["## Examples\n"]
        for i, ex in enumerate(examples, 1):
            formatted.append(f"Example {i}:")
            formatted.append(f"Input: {ex['input']}")
            formatted.append(f"Output: {ex['output']}\n")
        return "\n".join(formatted)

    def invalidate_cache(self, category: Optional[str] = None):
        """Invalidate cache for hot-reload on config change."""
        with self._lock:
            if category:
                keys = [k for k in self._cache if k.startswith(f"{category}:")]
                for k in keys:
                    del self._cache[k]
            else:
                self._cache.clear()
