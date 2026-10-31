from abc import ABC, abstractmethod


class Task(ABC):
    """Implement model/data/metrics for the chosen research task."""

    @abstractmethod
    def train(self, config, project_root, run_dir):
        """Use train/validation only; save selected checkpoint and actual logs."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, config, project_root, run_dir):
        """Load a specified checkpoint and report actual metrics on the selected split."""
        raise NotImplementedError
