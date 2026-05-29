from typing import Callable, Dict, Optional, Tuple


class ShapeClassifier:
    def __init__(
        self,
        rules: Dict[str, Callable[..., bool]],
        default: str = "default",
        key_fn: Optional[Callable[..., Tuple]] = None,
    ):
        self.rules = rules
        self.default = default
        self.key_fn = key_fn

    def classify(self, **shape_args) -> str:
        for class_name, rule_fn in self.rules.items():
            if rule_fn(**shape_args):
                return class_name
        return self.default

    def get_cache_key(self, **shape_args) -> Tuple:
        shape_class = self.classify(**shape_args)
        if self.key_fn is not None:
            return (shape_class,) + self.key_fn(**shape_args)
        return (shape_class,) + tuple(v for v in shape_args.values())
