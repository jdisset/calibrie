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
    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_default=True, protected_namespaces=()
    )

    diagnostics_settings: dict[str, Any] = Field(default_factory=dict)
    priority: float = Field(default=0, description="Execution order of the task. Lower is earlier.")
    notes: str = ""

    def model_post_init(self, *args):
        super().model_post_init(*args)
        self._name: str = ""

    def initialize(self, ctx: Context) -> Context:
        return Context()

    def process(self, ctx: Context) -> Context:
        return Context()

    def diagnostics(self, ctx: Context, **kw) -> Optional[List[DiagnosticFigure]]:
        pass

    __hash__ = object.__hash__


class Pipeline(ArbitraryModel):
    tasks: Dict[str, Task]
    loglevel: str = "NOTSET"
    notes: str = ""
    name: str = ""

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
        self._ordered_task_list: List[Task] = []
        for task_name, task in self.tasks.items():
            task._log = self._log.getChild(task_name)

    def order_tasks_and_assign_names(self):
        # make sure each task has its name
        for task_name, task in self.tasks.items():
            task._name = task_name
        self._ordered_task_list = sorted(self.tasks.values(), key=lambda t: t.priority)

    def initialize(self):
        self.order_tasks_and_assign_names()

        self._context = Context()
        for task in self._ordered_task_list:
            self._log.debug(f"Initializing task {task.__class__.__name__}")
            self._context.update(task.initialize(self._context))

    def apply_all(self, input):
        self.order_tasks_and_assign_names()

        self._context.pipeline_input = input
        last_output = None
        for task in self._ordered_task_list:
            print(f"Applying task {task.__class__.__name__}")
            self._log.debug(f"Applying task {task.__class__.__name__}")
            last_output = task.process(self._context)
            self._context.update(last_output)
        return last_output

    def diagnostics(self, task_name, **kw) -> Optional[List[DiagnosticFigure]]:
        self.order_tasks_and_assign_names()

        if task_name not in self.tasks:
            raise ValueError(f"Task {task_name} not found in pipeline")
        task = self.tasks[task_name]
        figs = task.diagnostics(self._context, **kw)
        return figs

    def all_diagnostics(self, **kw):
        self.order_tasks_and_assign_names()

        all_figs = []
        for task in self._ordered_task_list:
            task_kwargs = task.diagnostics_settings.copy()
            task_kwargs.update(kw)
            print(f"Generating diagnostics for {task._name} with kwargs {task_kwargs}")
            self._log.debug(f"Generating diagnostics for {task._name}")
            figs = task.diagnostics(self._context, **task_kwargs)
            if figs:
                if isinstance(figs, list):
                    for fig in figs:
                        fig.source_task = task._name
                    all_figs.extend(figs)
                else:
                    figs.source_task = task._name
                    all_figs.append(figs)
        return all_figs
