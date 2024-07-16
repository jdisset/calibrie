import logging
from rich.logging import RichHandler
from calibry.utils import ArbitraryModel, Context

from typing import List, Dict, Any, Optional, Union
from pydantic import Field


MISSING = object()


class Task(ArbitraryModel):


    def initialize(self, ctx: Context) -> Context:
        return Context()

    def process(self, ctx: Context) -> Context:
        return Context()

    def diagnostics(self, ctx: Context, **kw) -> Any:
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
        for task in self.tasks:
            self._log.debug(f"Applying task {task.__class__.__name__}")
            last_output = task.process(self._context)
            self._context.update(last_output)

    def diagnostics(self, task_name, idx=None, **kw):
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
        updt = self.tasks[names.index(task_name) + idx].diagnostics(self._context, **kw)
        if updt is not None:
            self._context.update(updt)
