from ...utils import Registry
from .base import ComposedEvalutor

EVALUATORS = Registry('evaluator')


def build_evaluator(cfg):
    """Build evalutor."""

    if isinstance(cfg, list):
        evaluators = [build_evaluator(_cfg) for _cfg in cfg]
        return ComposedEvalutor(evaluators)
    else:
        return EVALUATORS.build(cfg)
