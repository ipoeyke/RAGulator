from pydantic import BaseModel, root_validator
from typing import List, Optional
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

    @root_validator
    def check_lengths_match(cls, values):
        llm_responses = values.get('llm_responses')
        contexts = values.get('contexts')
        if not llm_responses or not contexts:
            raise ValueError("Both llm_responses and contexts must not be empty.")
        if len(llm_responses) != len(contexts):
            raise ValueError(
                f"Number of llm_responses ({len(llm_responses)}) and contexts ({len(contexts)}) must be the same."
            )
        return values

class PipelineResponse(BaseModel):
    prediction: Optional[float]
    sentences_evaluated: List[str]

class PipelineResponseBatch(BaseModel):
    predictions: Optional[List[float]]
    sentences_evaluated: List[List[str]]