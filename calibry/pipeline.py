import logging
from rich.logging import RichHandler

from typing import List, Dict, Any, Optional, Union

class Task:
    def __init__(self, **kwargs):
        self.params = kwargs
        self.log = logging.getLogger("calibry")

    def initialize(self, **_):
        pass

    def apply(self, **_):
        pass

    def diagnostics(self):
        pass


class Pipeline:
    def __init__(self, tasks: List[Task], loglevel="NOTSET"):
        logger_name = f"calibry_{tasks[0].__class__.__name__}"
        self.log = logging.getLogger(logger_name)
        if self.log.hasHandlers():
            self.log.handlers.clear()
        logging_handler = RichHandler()
        logging_handler.setFormatter(logging.Formatter(datefmt="%Y-%m-%dT%H:%M:%S%z "))
        logging_handler._log_render.show_path = False
        self.log.addHandler(logging_handler)
        self.log.setLevel(loglevel)
        self.tasks = tasks
        for task in self.tasks:
            task.log = self.log.getChild(task.__class__.__name__)
        self.context = {}

    def initialize(self, **kw):
        for task in self.tasks:
            self.log.debug(f"Initializing task {task.__class__.__name__}")
            self.context.update(task.initialize(**self.context, **kw))

    def apply_all(self, input):
        self.context.update({'pipeline_input': input})
        for task in self.tasks:
            self.log.debug(f"Applying task {task.__class__.__name__}")
            last_output = task.process(**self.context)
            self.context.update(last_output)
        # return first key of last output
        return list(last_output.values())[0]


    def all_diagnostics(self):
        for task in self.tasks:
            task.diagnostics()

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
        updt = self.tasks[names.index(task_name) + idx].diagnostics(**self.context, **kw)
        if updt is not None:
            self.context.update(updt)

