# Copyright (c) 2026 Jean Disset
# MIT License - see LICENSE file for details.

from typing import Any, Literal
from pydantic import BaseModel, Field
import numpy as np


class Metric(BaseModel):
    value: Any
    description: str
    scope: Literal['individual', 'global'] = 'individual'

    class Config:
        arbitrary_types_allowed = True

    def model_dump(self, **kwargs):
        d = super().model_dump(**kwargs)
        if isinstance(d['value'], np.ndarray):
            d['value'] = d['value'].tolist()
        elif isinstance(d['value'], (np.floating, np.integer)):
            d['value'] = float(d['value'])
        return d


class TaskMetrics(BaseModel):
    task_name: str = ''
    individual: dict[str, Metric] = Field(default_factory=dict)
    global_metrics: dict[str, Metric] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    def add(self, name: str, value: Any, description: str, scope: Literal['individual', 'global'] = 'individual'):
        metric = Metric(value=value, description=description, scope=scope)
        if scope == 'global':
            self.global_metrics[name] = metric
        else:
            self.individual[name] = metric

    def model_dump(self, **kwargs):
        return {
            'task_name': self.task_name,
            'individual': {k: v.model_dump(**kwargs) for k, v in self.individual.items()},
            'global': {k: v.model_dump(**kwargs) for k, v in self.global_metrics.items()},
        }
