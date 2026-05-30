from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class RAGConfig:
    knowledge_base_path: str = "./knowledge_base"
    index_save_path: str = "./knowledge_base_store"

    embedding_model: str = "D:/modelscope/hub/models/microsoft/harrier-oss-v1-0___6b"
    llm_model: str = "kimi-k2.6"

    top_k: int = 3

    temperature: float = 0.1
    max_tokens: int = 2048

    def __post_init__(self):
        pass

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RAGConfig":
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_path": self.knowledge_base_path,
            "index_save_path": self.index_save_path,
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model,
            "top_k": self.top_k,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


DEFAULT_CONFIG = RAGConfig()
