from io import IOBase
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Type, Union

from fastNLP import DataSet, Vocabulary
from omegaconf import DictConfig, ListConfig
from pytorch_lightning import Trainer
from src.torch_struct import StructDistribution
from src.utils.config import Config
from torch import Tensor
from torchmetrics import Metric

InputDict = Dict[str, Tensor]
TensorDict = Dict[str, Tensor]
AnyDict = Dict[str, Any]
GenDict = (dict, DictConfig)
GenList = (list, ListConfig)

if TYPE_CHECKING:
    from .data.datamodule import DataModule
    from .encoders.base import EncoderBase
    from .models.basic import BasicModel
    from .modules.embeddings import Embedding
    from .runners import BasicRunner
    from .task_specific.base import TaskBase
    from .utils.var_pool import VarPool
