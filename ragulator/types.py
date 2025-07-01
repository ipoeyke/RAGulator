from pydantic import BaseModel, model_validator
from typing import List, Optional, Union
from typing_extensions import Self
from ragulator.utils import load_config

config = load_config('ragulator/config.yaml')['infer']

class ConfigResponse(BaseModel):
    config: dict

class PipelineRequest(BaseModel):
    llm_response: str
    context: str
    minimum_sentence_length: int = config['minimum_sentence_length']
    context_window_length: int = config['context_window_length']
    threshold: float = config['threshold']
    return_probas: bool = config['return_probas']

class PipelineRequestBatch(BaseModel):
    llm_responses: List[str]
    contexts: List[str]
    minimum_sentence_length: int = config['minimum_sentence_length']
    context_window_length: int = config['context_window_length']
    threshold: float = config['threshold']
    return_probas: bool = config['return_probas']

    @model_validator(mode='after')
    def check_texts(self) -> Self:
        n_responses = len(self.llm_responses)
        n_contexts = len(self.contexts)
        if n_responses == 0 or n_contexts == 0:
            raise ValueError("Both llm_responses and contexts must not be empty.")
        if n_responses != n_contexts:
            raise ValueError(
                f"Number of llm_responses ({n_responses}) and contexts ({n_contexts}) must be equal."
            )
        return self

class PipelineResponse(BaseModel):
    prediction: Optional[Union[int, float]]
    sentences_predictions: List[Union[int, float]]
    sentences_evaluated: List[str]

class PipelineResponseBatch(BaseModel):
    predictions: Optional[List[Union[int, float]]]
    sentences_predictions: List[List[Union[int, float]]]
    sentences_evaluated: List[List[str]]
