import collections
from typing import Optional, Any, Dict, List

import ignite
from blackhc.laaos import StoreRoot
from ignite.engine import Engine
from torch.utils.data import DataLoader


def epoch_chain(engine: Engine, then_engine: Engine, dataloader: DataLoader) -> None:
    @engine.on(ignite.engine.Events.EPOCH_COMPLETED)
    def on_complete(_: Any) -> None:
        then_engine.run(dataloader)


def chain(engine: Engine, then_engine: Engine, dataloader: DataLoader) -> None:
    @engine.on(ignite.engine.Events.COMPLETED)
    def on_complete(_: Any) -> None:
        then_engine.run(dataloader)


def log_epoch_results(engine: Engine, name: str, trainer: Engine) -> None:
    @engine.on(ignite.engine.Events.COMPLETED)
    def log(_: Any) -> None:
        metrics = engine.state.metrics
        avg_accuracy = metrics["accuracy"]
        avg_nll = metrics["nll"]
        print(
            f"{name} Results - Epoch: {trainer.state.epoch if trainer.state else 0}  "
            f"Avg accuracy: {avg_accuracy*100:.2f}% Avg loss: {avg_nll:.2f}"
        )


def log_results(engine: Engine, name: str) -> None:
    @engine.on(ignite.engine.Events.COMPLETED)
    def log(_: Any) -> None:
        metrics = engine.state.metrics
        avg_accuracy = metrics["accuracy"]
        avg_nll = metrics["nll"]
        print(f"{name} Results  " f"Avg accuracy: {avg_accuracy*100:.2f}% Avg loss: {avg_nll:.2f}")


def store_iteration_results(engine: Engine, store_object: List[Any], log_interval: int = 1000) -> None:
    @engine.on(ignite.engine.Events.ITERATION_COMPLETED)
    def log(_: Any) -> None:
        if engine.state.iteration % log_interval == 0:
            store_object.append(engine.state.output)


def store_epoch_results(engine: Engine, store_object: StoreRoot, name: Optional[str] = None) -> None:
    @engine.on(ignite.engine.Events.EPOCH_COMPLETED)
    def log(_: Any) -> None:
        metrics = engine.state.metrics
        if isinstance(store_object, collections.MutableSequence):
            store_object.append(metrics)
        else:
            store_object[name] = metrics
