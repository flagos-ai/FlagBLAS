from typing import Callable, Dict, Optional, Tuple

from flag_blas.runtime.shape_classifier import ShapeClassifier


class KernelDispatcher:
    def __init__(self):
        self._kernels: Dict[str, Dict[str, Callable]] = {}
        self._config_names: Dict[str, Dict[str, str]] = {}
        self._classifiers: Dict[str, ShapeClassifier] = {}

    def register_op(
        self, op_name: str, classifier: ShapeClassifier
    ):
        self._classifiers[op_name] = classifier
        self._kernels[op_name] = {}
        self._config_names[op_name] = {}

    def register_kernel(
        self,
        op_name: str,
        transa: int,
        transb: int,
        shape_class: str,
        kernel_fn: Callable,
        config_name: Optional[str] = None,
    ):
        key = f"{transa}_{transb}_{shape_class}"
        self._kernels[op_name][key] = kernel_fn
        if config_name is not None:
            self._config_names[op_name][key] = config_name

    def classify(self, op_name: str, **shape_args) -> str:
        classifier = self._classifiers.get(op_name)
        if classifier is None:
            return "default"
        return classifier.classify(**shape_args)

    def dispatch(
        self,
        op_name: str,
        transa: int,
        transb: int,
        **shape_args,
    ) -> Tuple[Callable, str]:
        shape_class = self.classify(op_name, **shape_args)

        key = f"{transa}_{transb}_{shape_class}"
        if key in self._kernels.get(op_name, {}):
            kernel = self._kernels[op_name][key]
            config_name = self._config_names.get(op_name, {}).get(key, op_name)
            return kernel, config_name

        default_key = f"{transa}_{transb}_default"
        if default_key in self._kernels.get(op_name, {}):
            kernel = self._kernels[op_name][default_key]
            config_name = self._config_names.get(op_name, {}).get(default_key, op_name)
            return kernel, config_name

        raise KeyError(
            f"No kernel registered for op={op_name}, shape_class={shape_class}"
        )

    def get_config_name(self, op_name: str, shape_class: str) -> Optional[str]:
        return self._config_names.get(op_name, {}).get(shape_class)


_global_dispatcher = KernelDispatcher()


def get_dispatcher() -> KernelDispatcher:
    return _global_dispatcher
