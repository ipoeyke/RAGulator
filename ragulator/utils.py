import yaml

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def filter_sentences(sentences: list[str], min_sentence_len: int) -> list[str]:
    """Filter sentences by minimum character length.
    
    Args:
        sentences (list[str]): list of sentences to filter
        min_sentence_len (int): minimum length of sentences to include
    
    Returns:
        list of filtered sentences
    """
    return [sent for sent in sentences if len(sent) >= min_sentence_len]
