# src/inference/__init__.py
from .preprocessor import preprocess_row
from .predictor    import predict

__all__ = ['preprocess_row', 'predict']
