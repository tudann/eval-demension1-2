"""Video quality evaluation for dimensions D1-D6.

Heavy runtime dependencies are imported lazily so registry and configuration tools also work in
lightweight environments.
"""

__all__ = ["EvaluationRequest", "evaluate"]


def __getattr__(name):
    if name in __all__:
        from .pipeline import EvaluationRequest, evaluate
        return {"EvaluationRequest": EvaluationRequest, "evaluate": evaluate}[name]
    raise AttributeError(name)
