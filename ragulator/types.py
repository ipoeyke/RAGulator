from pydantic import BaseModel
from typing import List
from ragulator.utils import load_config

config = load_config('ragulator/config.yaml')

class ConfigResponse(BaseModel):
    config: dict

class PipelineRequest(BaseModel):
    llm_response: str
    context: str
    min_sentence_length: int = config['infer']['min_sentence_length']
    threshold: float = config['infer']['threshold']
    return_probas: bool = config['infer']['return_probas']

class PipelineRequestBatch(BaseModel):
    llm_responses: List[str]
    contexts: List[str]
    min_sentence_length: int = config['infer']['min_sentence_length']
    threshold: float = config['infer']['threshold']
    return_probas: bool = config['infer']['return_probas']

class PipelineResponse(BaseModel):
    prediction: float
    sentences_evaluated: List[str]

class PipelineResponseBatch(BaseModel):
    predictions: List[float]
    sentences_evaluated: List[List[str]]