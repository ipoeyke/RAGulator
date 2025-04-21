import numpy as np
from spacy.lang.en import English
import torch
import transformers
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
from typing import Any, Iterator, Union
import warnings
warnings.filterwarnings("ignore")

class RAGulator:
    def __init__(self, model_name='deberta-v3-large', batch_size=32, device='cpu'):
        if model_name == 'deberta-v3-large':
            # use ragulator-deberta-v3-large HF model
            model_path = 'ipoeyke/ragulator-deberta-v3-large'
            self.tokenizer = DebertaV2Tokenizer.from_pretrained(model_path)
            self.model = DebertaV2ForSequenceClassification.from_pretrained(
                model_path,
                num_labels=2,
                output_attentions=False,
                output_hidden_states=False
            )
        else:
            # not supported for now
            raise ValueError("Other models currently not supported")

        self.model.eval()
        self.device = torch.device(device)
        self.model.to(self.device)
        self.batch_size = batch_size
        self.softmax = torch.nn.Softmax(dim=-1)
        transformers.logging.set_verbosity_error()
        
        # SpaCy
        self.spacy = English()
        self.spacy.add_pipe('sentencizer')
    
    def infer_batch(
        self,
        sentences: list[str],
        contexts: list[str],
        context_window_len: int = 490, # slightly lower than 512-token overall limit
        ooc_threshold: float = 0.5,
        return_probas: bool = True
        ) -> list[Union[int, float]]:
        """Perform inference for a batch of sentence-context pairs.

        For long-context inference, each pair will be split into multiple sub-sequence
        inputs and passed to BERT model. The outputs are then aggregated back into a
        single predicted label for each given pair.
        (Logic: only if ALL splits are OOC, then sentence-context pair is overall OOC,
        else it is in-context.)
        
        Args:
            sentences: list of LLM-generated sentences
            contexts: list of corresponding reference contexts
            context_window_len: size of window to use for splitting long contexts
            ooc_threshold: if returning labels, set the prediction threshold 
            return_probas: if True, returns OOC probabilities instead of label
            
        Returns:
            list of predictions for each pair (1 for OOC, 0 for in-context)
            OR
            list of probabilities for each pair
        """
        assert isinstance(sentences, list) and isinstance(contexts, list)
        assert len(sentences) == len(contexts)

        # break long-context pairs into multiple chunks,
        # include a pair ID to track original pair position
        split_sentences, split_contexts, pair_ids = [], [], []
        for i, (s, c) in enumerate(zip(sentences, contexts)):
            windows = self.__window_text(c, seq_len=context_window_len)
            split_contexts.extend(windows)
            split_sentences.extend([s] * len(windows))
            pair_ids.extend([i] * len(windows))

        # tokenize and inference
        outputs = []
        counter = 0
        for batch in self.__batch_tokenize(split_sentences, split_contexts):
            out = self.__infer(batch)
            if not return_probas:
                # output binary label
                out = np.where(out >= ooc_threshold, 1, 0)
            outputs.append(out)
            counter += 1

        outputs = np.concatenate(outputs, axis=0)

        # aggregate based on original pair ID
        assert len(outputs) == len(split_contexts) == len(pair_ids)
        pair_ids = np.array(pair_ids)
        preds = []
        for i in range(len(contexts)):
            indices = np.where(pair_ids == i)
            values = outputs[indices]
            if return_probas:
                # take min. probability as the pair's overall probability
                preds.append(np.min(values).item())
            else:
                # if ALL OOC then pair is OOC else IC
                preds.append(int(np.all(values)))
            
        assert len(preds) == len(sentences) == len(contexts)
        
        return preds

    def sentencize(self, text: str) -> list[str]:
        """Sentencize a text using SpaCy.
        
        Args:
            text (str): text to be sentencized
        
        Returns:
            list of sentences
        """
        return [sent.text.strip() for sent in self.spacy(text).sents]

    def __infer(self, batch: dict) -> np.array:
        """Perform BERT forward pass for a single batch of tokenized 
        sentence-context-pair sub-sequences. Maximum sequence length is 512 tokens.
        NOTE: forward pass takes place on GPU, but inputs/outputs shall reside on CPU.
        
        Args:
            batch: dict containing encoded sentence-context pairs
            
        Returns:
            numpy array of predicted OOC probabilities for each sub-sequence
        """
        inputs = {k: v.to(self.device) for k,v in batch.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        probas = self.softmax(outputs.logits).cpu().numpy()
        
        return probas[:,1] # probability of OOC only

    def __batch_tokenize(self, sentences: list[str], contexts: list[str]) -> Iterator[dict]:
        """Generator to tokenize sentence-context pairs and yield one batch at a time.
        
        Args:
            sentences: list of sentences to tokenize
            context: list of corresponding reference contexts to tokenize
        
        Returns:
            generator of batches of encoded sentence-context pairs
        """
        assert isinstance(sentences, list) and isinstance(contexts, list)
        assert len(sentences) == len(contexts)

        for s_batch, c_batch in zip(
            self.__to_chunks(sentences, self.batch_size),
            self.__to_chunks(contexts, self.batch_size)
        ):
            encoded_batch = self.tokenizer(
                s_batch,
                c_batch,
                add_special_tokens=True,
                return_token_type_ids=True,
                return_attention_mask=True,
                padding='max_length',
                max_length=512,
                truncation='longest_first',
                return_tensors='pt'
            )
            
            yield encoded_batch

    def __to_chunks(self, lst: list[Any], n: int) -> Iterator[list]:
        """Generator to yield successive n-sized chunks from a list.
        
        Args:
            lst: list to generate chunks from
            n: size of each chunk
            
        Returns:
            generator of n-sized chunks
        """
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def __window_text(self, texts: Union[str, list[str]], seq_len: int = 512) -> list[str]:
        """Split text into windows of >=1 FULL sentences with total token len <= seq_len.
        
        Args:
            texts: text(s) to be split into windows containing full sentences
            seq_len: maximum number of tokens in each window
        
        Returns:
            list of windows of multiple sentences, each having token length <= seq_len
        """
        window, windows = [], []
        token_len = 0

        # convert to list of sentences if not already
        if isinstance(texts, str):
            texts = self.sentencize(texts)

        # windowing
        for i, s in enumerate(texts):
            cur_len = self.__count_tokens(s)
            token_len += cur_len
            if token_len > seq_len and len(window) > 0:
                # append window
                windows.append(" ".join(window))
                # restart window, add current sentence
                window = [s]
                token_len = cur_len
            else:
                window.append(s)
                    
        if window:
            windows.append(" ".join(window))
            
        return windows

    def __count_tokens(self, text: str) -> int:
        """Count tokens in a text using SpaCy."""
        return len(self.spacy(text))
