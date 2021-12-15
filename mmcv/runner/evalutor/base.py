import os.path as osp
import pickle
import shutil
import tempfile
import warnings
from abc import ABCMeta, abstractmethod
from typing import Dict, List

import torch
import torch.distributed as dist


class BaseEvaluator(metaclass=ABCMeta):
    """Evaluator base interface.

    Args:
        dataset_meta (dict): Dataset meta information
    """

    def __init__(self):
        self._dataset_meta = None
        self.results = []

        if dist.is_available() and dist.is_initialized():
            self.rand = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rand = 0
            self.world_size = 1

    @property
    def dataset_meta(self):
        return self._dataset_meta

    @dataset_meta.setter
    def dataset_meta(self, dataset_meta: Dict):
        self._dataset_meta = dataset_meta

    def evaluate(self):
        # check the results and raise a warning if it is empty
        if len(self.results) == 0:
            warnings.warn(
                f'{self.__class__.__name__} got empty `self._results`. Please '
                'ensure that the processed results are properly added into '
                '`self._results` in `process` method.')

        if self.world_size == 0:
            # non-distributed evaluation
            return self.compute_metrics(self.results)

        else:
            # distributed evaluation
            results = self._collect_results()

            if self.rank == 0:
                return self.compute_metrics(results)
            else:
                return None

    def _collect_results(self) -> List:
        """Collected results in distributed environments."""
        # create temp dir
        t_tempdir = torch.full((512, ),
                               ord(' '),
                               dtype=torch.uint8,
                               device='cuda')
        if self.rank == 0:
            tempdir = tempfile.mkdtemp()
            tempdir = t_tempdir.new(bytearray(tempdir.encode()))
            t_tempdir[:len(tempdir)] = tempdir

        dist.broadcast(t_tempdir, 0)
        tempdir = t_tempdir.cpu().numpy().tobytes().decode().rstrip()
        # synchronizes all processes to make sure tmpdir exist
        dist.barrier()
        # dump the part result
        with open(osp.join(tempdir, f'part_{self.rank}.pkl'), 'wb') as f:
            pickle.dump(self.results, f, protocol=2)
        # synchronizes all processes for loading pickle file
        dist.barrier()
        # collect all parts
        if self.rand != 0:
            return None
        part_list = []
        for i in range(self.world_size):
            with open(osp.join(tempdir, f'part_{i}.pkl'), 'rb') as f:
                part_list.append(pickle.load(f))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # remove temp dir
        shutil.rmtree(tempdir)

        return ordered_results

    @abstractmethod
    def process(self, data_samples: Dict, predictions: Dict):
        """Process one batch of model samples and model predictions. The
        processed results should be stored in a list `self.results`.

        Args:
            data_samples (dict): The sample information from dataset
            predictions (dict): The model prediction
        """

    @abstractmethod
    def compute_metrics(self, results: List) -> Dict:
        """Compute the evaluation metrics over the dataset.

        Args:
            results (list): list of processed batch results.

        Returns (dict): evaluation results where each item is the name and
            value of one evaluation metric.
        """


class ComposedEvaluator(BaseEvaluator):

    def __init__(self, evaluators: List[BaseEvaluator]):
        self.evaluators = evaluators.copy()
        self._dataset_meta = None

    @property
    def dataset_meta(self):
        return self._dataset_meta

    @dataset_meta.setter
    def dataset_meta(self, dataset_meta: Dict):
        self._dataset_meta = dataset_meta
        for evaluator in self.evaluators:
            evaluator.dataset_meta = dataset_meta

    def process(self, data_samples: Dict, predictions: Dict):
        """Invoke process method of each wrapped evaluator."""

        for evalutor in self.evaluators:
            evalutor.process(input, data_samples, predictions)

    def evaluate(self) -> Dict:

        eval_results = {}
        for evaluator in self.evaluators:
            _eval_results = evaluator.evaluate()

            # Check metric name conflicts
            for name in _eval_results.keys():
                # TODO: Automatically handle the name conflict instead of
                # raising an exception.
                if name in eval_results:
                    raise ValueError(
                        'There are multiple evaluators with the same metric '
                        f'name {name}')

            eval_results.update(_eval_results)

        return eval_results
