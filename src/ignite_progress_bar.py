import torch.utils.data as data
import ignite
from blackhc.progress_bar import create_progress_bar
from ignite.engine import Engine
from typing import Callable, Any, Optional


class IgniteProgressBar(object):
    def __init__(self, desc: Optional[Callable[[Any], str]], log_interval: int) -> None:
        self.log_interval = log_interval
        self.desc = desc
        self.progress_bar: Optional[Any] = None

    def attach(self, engine: Engine) -> None:
        engine.add_event_handler(ignite.engine.Events.EPOCH_STARTED, self.on_start)
        engine.add_event_handler(ignite.engine.Events.EPOCH_COMPLETED, self.on_complete)
        engine.add_event_handler(ignite.engine.Events.ITERATION_COMPLETED, self.on_iteration_complete)

    def on_start(self, engine: Engine) -> None:
        dataloader = engine.state.dataloader
        self.progress_bar = create_progress_bar(len(dataloader) * dataloader.batch_size)

        if self.desc is not None:
            print(self.desc(engine))
        self.progress_bar.start()

    def on_complete(self, engine: Engine) -> None:
        assert self.progress_bar is not None
        self.progress_bar.finish()

    def on_iteration_complete(self, engine: Engine) -> None:
        dataloader = engine.state.dataloader
        iter = (engine.state.iteration - 1) % len(dataloader) + 1

        if iter % self.log_interval == 0:
            assert self.progress_bar is not None
            self.progress_bar.update(self.log_interval * dataloader.batch_size)


def ignite_progress_bar(
    engine: Engine, desc: Optional[Callable[[Any], str]] = None, log_interval: int = 0
) -> IgniteProgressBar:
    wrapper = IgniteProgressBar(desc, log_interval)
    wrapper.attach(engine)

    return wrapper
