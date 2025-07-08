from ..base import GenericTokeniser, Encoding, EncodingType, SPECIAL_TOKENS, Vocab
from ...datasets.base import TokeniserTrainingDataset

from ...tktkt.tokenizers_adapter import TokenizersPreprocessor

from tktkt.models.bpe.knockout import BPEKnockout
class BPEKnockout(GenericTokeniser):
    """
    A class for Byte Pair Encoding (BPE) tokenization with knockout functionality.
    
    Inherits from GenericTokeniser and provides additional methods for knockout operations.
    
    This class supports training a BPE tokeniser on a dataset, loading a pre-trained tokeniser, and saving the tokeniser to a file.
    
    Arguments:
        model (BPE | None): The BPE model to use. If None, a new BPE model will be created.
        normaliser (Normalizer | None): The normalizer to use for text normalization.
        pre_tokeniser (PreTokenizer | None): The pre-tokenizer to use for initial tokenization.
        post_processor (PostProcessor | None): The post-processor to apply after tokenization.
        decoder (Decoder | None): The decoder to use for converting tokens back to text.
        special_tokens_set (str): The set of special tokens to use. Defaults to "default". Picks from the SPECIAL_TOKENS dictionary.
    """
    pass