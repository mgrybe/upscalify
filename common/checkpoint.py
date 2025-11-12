import torch
from pathlib import Path
from typing import Any, Mapping

class Checkpoint:

    def __init__(self):
        self._items: dict[str, Any] = {}
        self._params: dict[str, Any] = {}
        self.checkpoint = None

    def _get_key(self, checkpointable: Any) -> str:
        return str(type(checkpointable))

    def add_checkpointable(self, checkpointable: Any):
        if not callable(getattr(checkpointable, "state_dict", None)):
            raise ValueError(f"{checkpointable!r} must implement state_dict().")

        self._items[f"{self._get_key(checkpointable)}.state_dict"] = checkpointable

    def add_params(self, **kwargs: Mapping[str, Any]):
        self._params.update(kwargs)

    def get_params(self, param: str):
        return self.checkpoint['params'][param]

    def get_state_dict(self, obj: Any):
        return self.checkpoint[f"{self._get_key(obj)}.state_dict"]

    def create(self, **kwargs: Mapping[str, Any]):
        checkpoint = {name: obj.state_dict() for name, obj in self._items.items()}

        self.add_params(**kwargs)
        checkpoint['params'] = self._params

        return checkpoint

    def save(self, checkpoint_path: str, **kwargs: Mapping[str, Any]):
        checkpoint_path = Path(checkpoint_path)
        tmp = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")

        self.checkpoint = self.create(**kwargs)

        torch.save(self.checkpoint, tmp)
        tmp.replace(checkpoint_path)

        return self.checkpoint

    def load(self, checkpoint_path: str):
        self.checkpoint = torch.load(Path(checkpoint_path))

        return self.checkpoint

    def apply(self):
        for name, obj in self._items.items():
            obj.load_state_dict(self.checkpoint[name])

        return self.checkpoint