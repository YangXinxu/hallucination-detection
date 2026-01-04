"""Detection methods for hallucination detection. 

Available methods:
- lapeigvals: Laplacian eigenvalue-based (EMNLP 2025)
- lookback_lens: Attention ratio analysis
- entropy: Token/attention entropy-based
- ensemble: Combined methods (voting, stacking, concat)
- hypergraph: Hypergraph neural network (HyperCHARM)

Example:
    from src.methods import create_method, LapEigvalsMethod, HypergraphMethod
    
    # By name
    method = create_method("hypergraph")
    
    # Direct instantiation
    method = HypergraphMethod()
    
    # Train
    method.fit(features_list)
    
    # Predict
    prediction = method.predict(features)
"""

from .base import BaseMethod, create_method
from . lapeigvals import LapEigvalsMethod, LapEigvalsFullMethod
from .lookback_lens import LookbackLensMethod, AttentionStatsMethod
from . entropy import EntropyMethod, PerplexityMethod
from .ensemble import EnsembleMethod, AutoEnsembleMethod
from .hypergraph import HypergraphMethod, HypergraphTokenMethod

__all__ = [
    # Base
    "BaseMethod",
    "create_method",
    
    # LapEigvals
    "LapEigvalsMethod",
    "LapEigvalsFullMethod",
    
    # Lookback Lens
    "LookbackLensMethod",
    "AttentionStatsMethod",
    
    # Entropy
    "EntropyMethod",
    "PerplexityMethod",
    
    # Ensemble
    "EnsembleMethod",
    "AutoEnsembleMethod",
    
    # Hypergraph
    "HypergraphMethod",
    "HypergraphTokenMethod",
]