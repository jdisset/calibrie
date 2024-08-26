import logging
from rich.logging import RichHandler
from calibry.utils import ArbitraryModel, Context

from typing import List, Dict, Any, Optional, Union
from pydantic import Field, ConfigDict
from matplotlib.figure import Figure


MISSING = object()


class DiagnosticFigure(ArbitraryModel):
    fig: Figure
    name: str
    source_task: str = ""

    def __init__(self, fig, name, source_task="", **kwargs):
        super(DiagnosticFigure, self).__init__(
            fig=fig, name=name, source_task=source_task, **kwargs
        )


class Task(ArbitraryModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True)

    diagnostics_settings: dict[str, Any] = Field(default_factory=dict)

    def initialize(self, ctx: Context) -> Context:
        return Context()

    def process(self, ctx: Context) -> Context:
        return Context()

    def diagnostics(self, ctx: Context, **kw) -> Optional[List[DiagnosticFigure]]:
        pass

    __hash__ = object.__hash__


class Pipeline(ArbitraryModel):
    tasks: List[Task]
    loglevel: str = "NOTSET"

    def model_post_init(self, *args):
        super().model_post_init(*args)
        # set up logging
        self._log = logging.getLogger('calibry')
        if self._log.hasHandlers():
            self._log.handlers.clear()
        logging_handler = RichHandler()
        logging_handler.setFormatter(logging.Formatter(datefmt="%Y-%m-%dT%H:%M:%S%z "))
        logging_handler._log_render.show_path = False
        self._log.addHandler(logging_handler)
        self._log.setLevel(self.loglevel)
        for task in self.tasks:
            task._log = self._log.getChild(task.__class__.__name__)

    def initialize(self):
        self._context = Context()
        for task in self.tasks:
            self._log.debug(f"Initializing task {task.__class__.__name__}")
            self._context.update(task.initialize(self._context))

    def apply_all(self, input):
        self._context.pipeline_input = input
        last_output = None
        for task in self.tasks:
            self._log.debug(f"Applying task {task.__class__.__name__}")
            last_output = task.process(self._context)
            self._context.update(last_output)
        return last_output

    def diagnostics(self, task_name, idx=None, **kw) -> Optional[List[DiagnosticFigure]]:
        names = [t.__class__.__name__ for t in self.tasks]
        if task_name not in names:
            raise ValueError(f"Task {task_name} not found in pipeline")
        # how many tasks with this name?
        n = names.count(task_name)
        if n == 1:
            idx = 0
        elif idx is None:
            raise ValueError(
                f"Task {task_name} found {n} times in pipeline. Please specify which one to use with the `idx` argument"
            )
        figs = self.tasks[names.index(task_name) + idx].diagnostics(self._context, **kw)
        return figs

    def get_unique_task_names(self):
        names = [t.__class__.__name__ for t in self.tasks]
        unique_names = []
        for i, n in enumerate(names):
            total_count = names.count(n)
            if total_count == 1:
                unique_names.append(n)
            else:
                current_count = names[:i].count(n)
                unique_names.append(f"{n}_{current_count}")
        return unique_names

    def all_diagnostics(self, **kw):
        unique_names = self.get_unique_task_names()
        all_figs = []
        for task, name in zip(self.tasks, unique_names):
            task_kwargs = {}
            for k, v in task.diagnostics_settings.items():
                print(f"Setting {k} to {v} of type {type(v)}")
                task_kwargs[k] = task.diagnostics_settings[k]
            task_kwargs.update(kw)
            print(f"Generating diagnostics for {name} with kwargs {task_kwargs}")
            self._log.debug(f"Generating diagnostics for {name}")
            figs = task.diagnostics(self._context, **task_kwargs)
            if figs:
                if isinstance(figs, list):
                    for fig in figs:
                        fig.source_task = name
                    all_figs.extend(figs)
                else:
                    figs.source_task = name
                    all_figs.append(figs)
        return all_figs
