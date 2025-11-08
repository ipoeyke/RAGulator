import numpy as np
import ray
from ray import serve
from fastapi import FastAPI
from ragulator.infer import RAGulator
from ragulator.types import ConfigResponse, PipelineRequest, PipelineRequestBatch, PipelineResponse, PipelineResponseBatch
from ragulator.utils import load_config, filter_sentences

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
        """Endpoint to view default config.
        
        Args:
            None
            
        Returns:
            ConfigResponse: deployed model configuration
        """
        return ConfigResponse(config=self.config)

    @app.post("/predict")
    async def predict(self, request: PipelineRequest) -> PipelineResponse:
        """Endpoint to predict OOC on a single LLM response.
        
        Args:
            payload (PipelineRequest): payload
            
        Returns:
            PipelineResponse: response
        """
        sentences = self.model.sentencize(request.llm_response)
        sentences = filter_sentences(sentences, request.minimum_sentence_length)
        contexts = [request.context] * len(sentences)

        response_pred, preds = None, []
        if len(sentences) > 0:
            # get predictions at sentence-level
            preds = self.model.infer_batch(
                sentences=sentences,
                contexts=contexts,
                context_window_len=request.context_window_length,
                ooc_threshold=request.threshold,
                return_probas=request.return_probas
            )
            # get predictions at response-level
            if request.return_probas:
                response_pred = 1 if all(p >= request.threshold for p in preds) else 0
            else:
                response_pred = 1 if all(preds) else 0
        
        return PipelineResponse(
            prediction=response_pred,
            sentences_predictions=preds,
            sentences_evaluated=sentences
        )

    @app.post("/batch_predict")
    async def batch_predict(self, request: PipelineRequestBatch) -> PipelineResponseBatch:
        """Endpoint to predict OOC on multiple LLM responses.
        
        Args:
            payload (PipelineRequestBatch): payload
            
        Returns:
            PipelineResponseBatch: response
        """
        nested_sentences, nested_contexts, pair_ids = [], [], []
        valid_pair_counter = 0
        for r, c in zip(request.llm_responses, request.contexts):
            s = self.model.sentencize(r)
            s = filter_sentences(s, request.minimum_sentence_length)
            if len(s) > 0:
                nested_sentences.append(s)
                nested_contexts.append([c] * len(s))
                pair_ids.extend([valid_pair_counter] * len(s))
                valid_pair_counter += 1
        
        response_preds, nested_preds = [], []
        if valid_pair_counter > 0:
            # get predicdtions at sentence-level
            sentences = [s for ele in nested_sentences for s in ele]
            contexts = [c for ele in nested_contexts for c in ele]
            preds = self.model.infer_batch(
                sentences=sentences,
                contexts=contexts,
                context_window_len=request.context_window_length,
                ooc_threshold=request.threshold,
                return_probas=request.return_probas
            )
            # get predictions at response-level
            preds_arr = np.array(preds)
            pair_ids = np.array(pair_ids)
            for i in range(valid_pair_counter):
                indices = np.where(pair_ids == i)
                values = preds_arr[indices]
                nested_preds.append(values.tolist())
                if request.return_probas:
                    # take min. probability as the response overall probability
                    response_preds.append(1 if np.min(values).item() >= request.threshold else 0)
                else:
                    # if ALL OOC then pair is OOC else IC
                    response_preds.append(int(np.all(values)))

        return PipelineResponseBatch(
            predictions=response_preds,
            sentences_predictions=nested_preds,
            sentences_evaluated=nested_sentences
        )

# entry point to start Ray Serve
def main():
    ray.init(ignore_reinit_error=True)
    serve.run(RAGulatorDeployment.bind(), blocking=True)

if __name__ == "__main__":
    main()
