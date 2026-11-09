"""Registered tasks. Heavy imports (torch, timm) happen inside the task methods, not here."""
from .skin_lesion import SkinLesionTask

TASKS = {"skin_lesion": SkinLesionTask}
