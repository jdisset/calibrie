import jax
from jax import jit, vmap
from jax.tree_util import Partial as partial
import jax.numpy as jnp
import numpy as np

from . import plots, utils
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from tqdm import tqdm


