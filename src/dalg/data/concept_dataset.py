import csv
import json
from typing import List, Tuple, Union, Optional
import pandas as pd
from pathlib import Path


class ConceptDataset:
    def __init__(
        self,
        path: Union[str, Path],
        *,
        prompt_field: str = "prompt",
        json_key: Optional[str] = None,
        dedup: bool = False,
    ):
        """
        Load a list of prompt strings from a CSV, JSON, or JSONL file.

        Args:
            path: Path to a .csv, .json, or .jsonl file.
            prompt_field: Column/key name that holds the prompt string (default: 'prompt').
            json_key: If the top-level JSON is a dict and you only want a specific key's
                      list, provide its name. If None, all list-like values are used.
            dedup: If True, remove duplicate prompts while preserving order.
        """
        self.path = Path(path)
        self.prompt_field = prompt_field
        self.json_key = json_key
        self.data: List[str] = []

        suffix = self.path.suffix.lower()
        if suffix == ".csv":
            self._load_csv()
        elif suffix == ".json":
            self._load_json()
        elif suffix == ".jsonl":
            self._load_jsonl()
        else:
            raise ValueError(f"Unsupported file type: {self.path.suffix} (use .csv, .json, or .jsonl)")

        if dedup:
            self._deduplicate_in_place()

    def _extract_prompt_from_dict(self, d: dict) -> Optional[str]:
        if not isinstance(d, dict):
            return None
        val = d.get(self.prompt_field)
        if isinstance(val, str) and val.strip():
            return val.strip()
        for k in ("sentence", "prompt", "text"):
            val = d.get(k)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return None

    def _load_csv(self):
        with self.path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if self.prompt_field in row and isinstance(row[self.prompt_field], str):
                    p = row[self.prompt_field].strip()
                    if p:
                        self.data.append(p)

    def _load_json(self):
        with self.path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            self._extend_from_sequence(obj)
        elif isinstance(obj, dict):
            if self.json_key is not None:
                self._extend_from_sequence(obj.get(self.json_key, []))
            else:
                for seq in obj.values():
                    self._extend_from_sequence(seq)
        else:
            raise ValueError("Unsupported JSON structure: expected list or dict at the top level.")

    def _load_jsonl(self):
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("{") or line.startswith("["):
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, dict):
                        p = self._extract_prompt_from_dict(obj)
                        if p:
                            self.data.append(p)
                    elif isinstance(obj, list):
                        self._extend_from_sequence(obj)
                else:
                    self.data.append(line)

    def _extend_from_sequence(self, seq):
        if not isinstance(seq, list):
            return
        for item in seq:
            if isinstance(item, str):
                p = item.strip()
                if p:
                    self.data.append(p)
            elif isinstance(item, dict):
                p = self._extract_prompt_from_dict(item)
                if p:
                    self.data.append(p)

    def _deduplicate_in_place(self):
        seen = set()
        deduped = []
        for p in self.data:
            if p not in seen:
                seen.add(p)
                deduped.append(p)
        self.data = deduped

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_batches(self, batch_size: int) -> List[dict]:
        """
        Yield batches of prompts.

        Args:
            batch_size: Number of samples per batch.

        Returns:
            List of dicts, each with key "prompt" mapping to a list of strings.
        """
        batches = []
        for i in range(0, len(self.data), batch_size):
            batches.append({"prompt": list(self.data[i:i + batch_size])})
        return batches


class SupervisedConceptDataset:
    def __init__(self, path: str):
        """
        Load (prompt, label) pairs from a CSV or JSON file.

        Supported CSV columns (tried in order):
          "prompt"/"text"/"sentence"  x  "label"/"concept"

        Supported JSON structures:
          - List[dict]: each dict should contain a prompt field
            ("prompt", "text", or "sentence") and a label field ("label" or "concept").
          - dict: keys are label strings, values are lists of prompt strings.

        Args:
            path: Path to a .csv or .json file.
        """
        self.path = path
        self.data: List[Tuple[str, str]] = []

        if path.endswith(".csv"):
            self._load_csv()
        elif path.endswith(".json"):
            self._load_json()
        else:
            raise ValueError(f"Unsupported file type: {path} (use .csv or .json)")

    # ---- helpers ----

    @staticmethod
    def _find_columns(columns) -> Tuple[Optional[str], Optional[str]]:
        """Return (prompt_col, label_col) for the first recognised column pair."""
        for pc, lc in [("prompt", "label"), ("text", "label"),
                       ("sentence", "concept"), ("sentence", "label")]:
            if pc in columns and lc in columns:
                return pc, lc
        return None, None

    def _add_pair(self, prompt, label):
        p, y = str(prompt).strip(), str(label).strip()
        if p and y:
            self.data.append((p, y))

    # ---- loaders ----

    def _load_csv(self):
        df = pd.read_csv(self.path, encoding="utf-8")
        pc, lc = self._find_columns(df.columns)
        if pc is None:
            raise ValueError(
                f"No recognised prompt/label columns in {self.path}. "
                f"Found: {list(df.columns)}"
            )
        for p, y in zip(df[pc].dropna(), df[lc].dropna()):
            self._add_pair(p, y)

    def _load_json(self):
        with open(self.path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        if isinstance(obj, list):
            for item in obj:
                if not isinstance(item, dict):
                    continue
                prompt = item.get("prompt") or item.get("text") or item.get("sentence")
                label = item.get("label") or item.get("concept")
                if prompt is not None and label is not None:
                    self._add_pair(prompt, label)
        elif isinstance(obj, dict):
            for label, prompts in obj.items():
                for prompt in (prompts if isinstance(prompts, list) else []):
                    if prompt is not None:
                        self._add_pair(prompt, label)
        else:
            raise ValueError(
                f"Unsupported JSON structure in {self.path}: expected list or dict."
            )

    # ---- public interface ----

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[str, str]:
        return self.data[idx]

    def select(self, s: Union[slice, range, List[int]]) -> "SupervisedConceptDataset":
        """
        Return a new SupervisedConceptDataset containing only the entries at self.data[s].

        Args:
            s: A slice, range, or list of integer indices selecting the desired subset.

        Returns:
            A new SupervisedConceptDataset with the selected (prompt, label) pairs.
        """
        subset = SupervisedConceptDataset.__new__(SupervisedConceptDataset)
        subset.path = self.path
        if isinstance(s, slice):
            subset.data = self.data[s]
        else:
            subset.data = [self.data[i] for i in s]
        return subset

    def get_batches(self, batch_size: int) -> List[dict]:
        """
        Yield batches of (prompt, label) pairs.

        Args:
            batch_size: Number of samples per batch.

        Returns:
            List of dicts with keys "prompt" and "label", each mapping to a list of strings.
        """
        batches = []
        for i in range(0, len(self.data), batch_size):
            batch = self.data[i:i + batch_size]
            prompts, labels = zip(*batch) if batch else ([], [])
            batches.append({"prompt": list(prompts), "label": list(labels)})
        return batches
