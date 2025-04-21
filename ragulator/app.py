import ray
from ray import serve
from fastapi import FastAPI
from ragulator.infer import RAGulator
from ragulator.types import ConfigResponse, PipelineRequest, PipelineRequestBatch, PipelineResponse, PipelineResponseBatch
from ragulator.utils import load_config

app = FastAPI()

@serve.deployment
@serve.ingress(app)
class RAGulatorDeployment:
    def __init__(self) -> None:
        """Initialise deployment."""
        self.config = load_config('ragulator/config.yaml')['infer']
        self.model = RAGulator(device=self.config['device'], batch_size=self.config['batch_size'])

    @app.get("/get_config")
    async def get_config(self) -> ConfigResponse:
        """View the default configuration of the model."""
        return ConfigResponse(config=self.config)

    @app.post("/predict")
    async def predict(self, request: PipelineRequest) -> PipelineResponse:
        """Predict for a single LLM response-context pair."""
        sentences = self.model.sentencize(request.llm_response)
        contexts = [request.context] * len(sentences)
        preds = self.model.infer_batch(
            sentences=sentences,
            contexts=contexts,
            context_window_len=request.context_window_length,
            ooc_threshold=request.threshold,
            return_probas=request.return_probas
        )
        
        return PipelineResponse(
            prediction=preds[0],
            sentences_evaluated=sentences
        )

    @app.post("/batch_predict")
    async def batch_predict(self, request: PipelineRequestBatch) -> PipelineResponseBatch:
        """Batch predict for multiple LLM response-context pairs."""
        sentences, contexts = [], []
        for r, c in zip(request.llm_responses, request.contexts):
            s = self.model.sentencize(r)
            sentences.append(s)
            contexts.append(c * len(s))
        preds = self.model.infer_batch(
            sentences=sentences,
            contexts=contexts,
            context_window_len=request.context_window_length,
            ooc_threshold=request.threshold,
            return_probas=request.return_probas
        )

        return PipelineResponseBatch(
            predictions=preds,
            sentences_evaluated=sentences
        )

# entry point to start Ray Serve
def main():
    ray.init(ignore_reinit_error=True)
    serve.run(RAGulatorDeployment.bind(), blocking=True)

if __name__ == "__main__":
    main()
