"""
farmfederate_rag — RAG extension for the FarmFederate federated learning framework.

Quick-start:
    from farmfederate_rag import setup_federated_rag, FedRAGConfig, example_inference

    server, clients = setup_federated_rag(
        existing_classification_models=your_models,
        existing_llm_encoders=None,       # extracted automatically
        farm_configs=[
            {"farm_id": 0, "region": "south_asia",      "crops": ["rice", "wheat"]},
            {"farm_id": 1, "region": "disease_endemic",  "crops": ["potato", "tomato"]},
            {"farm_id": 2, "region": "global",           "crops": ["cotton", "maize"]},
        ],
        config=FedRAGConfig(num_rounds=8, local_epochs=3),
    )
"""

from .rag_core import (
    Document,
    AgriculturalChunker,
    RetrieverEncoder,
    RAGQueryBuilder,
    FarmVectorStore,
    ContextAssembler,
)
from .federated_rag_training import (
    RetrieverContrastiveLoss,
    AdvisoryGenerator,
    FedRAGConfig,
    FedRAGClient,
    FedRAGServer,
)
from .integration import (
    KnowledgeBaseBuilder,
    FarmFederateRAG,
    setup_federated_rag,
    example_inference,
)
from .advisory_and_eval import (
    LLMAdvisoryGenerator,
    RAGEvaluator,
    evaluation_protocol,
)

__all__ = [
    # rag_core
    "Document",
    "AgriculturalChunker",
    "RetrieverEncoder",
    "RAGQueryBuilder",
    "FarmVectorStore",
    "ContextAssembler",
    # federated_rag_training
    "RetrieverContrastiveLoss",
    "AdvisoryGenerator",
    "FedRAGConfig",
    "FedRAGClient",
    "FedRAGServer",
    # integration
    "KnowledgeBaseBuilder",
    "FarmFederateRAG",
    "setup_federated_rag",
    "example_inference",
    # advisory_and_eval
    "LLMAdvisoryGenerator",
    "RAGEvaluator",
    "evaluation_protocol",
]
