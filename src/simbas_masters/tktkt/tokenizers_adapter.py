import json
import os
from typing import Dict, List

from langcodes import Language
from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers
from tktkt.interfaces.preparation import Preprocessor, Pretokeniser, TextMapper
from tktkt.models.bpe.knockout import BPEKnockout
from tktkt.preparation.boundaries import BoundaryMarker, BoundaryMarkerLocation

from . import sadilar_config
# NOTE You must ALWAYS add your custom configurations to the CONFIGS dictionary.
# mine are added in sadilar_config.py when I import it.

## CLASSES
# these classes (if adapted) would be added to the `tktkt.interfaces.preparation`
# maybe under their own module, e.g. `tktkt.interfaces.preparation.tokenizers_adapter`
class TokenizersNormaliser(TextMapper):
    """
    A :obj:`CustomNormaliser` wraps the normalizer from a :class:`~tokenizers.Tokenizer` to work with according to :class:`tktkt.interfaces`.

    It provides a method to convert text using the normalizer's normalization process.

    Args:
        normaliser (tokenizers.normalizers.Normalizer): An instance of :class:`~tokenizers.normalizers.Normalizer` to be used for text normalization.sing the normalizer's normalization process.
    """
    def __init__(self, normaliser: normalizers.Normalizer):
        self.normaliser = normaliser

    @classmethod
    def from_tokeniser(cls, tokeniser: Tokenizer|str) -> 'TokenizersNormaliser':
        """
        Class method to create a CustomNormaliser from a :class:`~tokenizers.Tokenizer` instance.

        Args:
            tokeniser (tokenizers.Tokenizer): An instance of :class:`~tokenizers.Tokenizer` from which to extract the normalizer. If a string is provided, it is treated as a file path to a JSON file containing a previously serialized :class:`~tokenizers.Tokenizer`.
        
        Returns:
            :class:`CustomNormaliser`: A new instance of CustomNormaliser initialized with the normalizer from the Tokenizer.
        """
        if isinstance(tokeniser, str):
            if not os.path.exists(tokeniser):
                raise FileNotFoundError(f"The specified path does not exist: {tokeniser}")
            if not tokeniser.endswith('.json'):
                raise ValueError(f"The specified path must point to a JSON file: {tokeniser}")
            tokeniser = Tokenizer.from_file(tokeniser)

        normaliser = tokeniser.normalizer
        if normaliser is None:
            return cls(normalizers.Sequence([]))
        return cls(tokeniser.normalizer)
    
    def convert(self, text: str) -> str:
        """
        Convert the input text using the normalizer's normalization process.
        Args:
            text (str): The input text to be normalized.
        Returns:
            str: The normalized text.
        """
        return self.normaliser.normalize_str(text)

class TokenizersPretokeniser(Pretokeniser):
    """
    A :obj:`CustomPretokeniser` wraps the pre-tokenizer and decoder from a :class:`~tokenizers.Tokenizer` to work with according to :class:`tktkt.interfaces`.

    It provides a method to convert text using the pre-tokenizer's pre-tokenization process.

    Attributes:
        encoder (tokenizers.pre_tokenizers.PreTokenizer): An instance of :class:`~tokenizers.pre_tokenizers.PreTokenizer` to be used for text pre-tokenization.
        decoder (tokenizers.decoders.Decoder): An instance of :class:`~tokenizers.decoders.Decoder` to be used for decoding pre-tokenized strings back to their original form.
    """
    def __init__(self, encoder: pre_tokenizers.PreTokenizer, decoder: decoders.Decoder):
        self.encoder: pre_tokenizers.PreTokenizer = encoder
        self.decoder: decoders.Decoder = decoder
    
    @classmethod
    def from_tokeniser(cls, tokeniser: Tokenizer|str) -> 'TokenizersPretokeniser':
        """
        Class method to create a CustomPretokeniser from a :class:`~tokenizers.Tokenizer` instance.

        Args:
            tokeniser (tokenizers.Tokenizer): An instance of :class:`~tokenizers.Tokenizer` from which to extract the pre-tokenizer. If a string is provided, it is treated as a file path to a JSON file containing a previously serialized :class:`~tokenizers.Tokenizer`.
        Returns:
            :class:`CustomPretokeniser`: A new instance of CustomPretokeniser initialized with the pre-tokenizer from the Tokenizer.
        """
        if isinstance(tokeniser, str):
            if not os.path.exists(tokeniser):
                raise FileNotFoundError(f"The specified path does not exist: {tokeniser}")
            if not tokeniser.endswith('.json'):
                raise ValueError(f"The specified path must point to a JSON file: {tokeniser}")
            tokeniser = Tokenizer.from_file(tokeniser)

        encoder = tokeniser.pre_tokenizer
        decoder = tokeniser.decoder

        if encoder is None:
            encoder = pre_tokenizers.Sequence([])
        if decoder is None:
            decoder = decoders.Sequence([])
        return cls(encoder, decoder)
    
    @classmethod
    def from_path(cls, path: str) -> 'TokenizersPretokeniser':
        """
        Class method to create a CustomPretokeniser from a file path.

        Args:
            path (str): The file path to a local JSON file containing a previously serialized :class:`~tokenizers.Tokenizer`.
        Returns:
            :class:`CustomPretokeniser`: A new instance of CustomPretokeniser initialized with the pre-tokenizer from the :class:`~tokenizers.Tokenizer` loaded from the file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"The specified path does not exist: {path}")
        if not path.endswith('.json'):
            raise ValueError(f"The specified path must point to a JSON file: {path}")
        tokeniser = Tokenizer.from_file(path)
        return cls.from_tokeniser(tokeniser)
    
    def split(self, text: str) -> List[str]:
        """
        Split an entire text (e.g. a sentence) into smaller pre-token strings.

        It is these pre-token strings that will SEPARATELY be passed to the tokeniser.
        
        Args:
            text (str): The input text to be pre-tokenized.
        
        Returns:
            List[str]: A list of pre-tokenized strings.
        """
        return [w for w, _ in self.encoder.pre_tokenize_str(text)]
    
    def invertTokens(self, pretokens: List[str]) -> List[str]:
        """
        Invert any string transformations applied in the process of splitting a string into pretokens.

        May also apply a merging operation into a smaller list if that is appropriate.

        Args:
            pretokens (List[str]): A list of pre-tokenized strings to be inverted.

        Returns:
            List[str]: A list of inverted strings, which are the original tokens before pre-tokenization.
        """
        return [self.decoder.decode(pretokens)]

class TokenizersPreprocessor(Preprocessor):
    """
    A :obj:`CustomPreprocessor` combines a :class:`CustomNormaliser` and a :class:`CustomPretokeniser` to provide a complete pre-processing pipeline.
    It inherits from :class:`tktkt.interfaces.Preprocessor` and implements the necessary methods for text normalization and pre-tokenization.
    """
    def __init__(self, normaliser: TokenizersNormaliser, pre_tokeniser: TokenizersPretokeniser):
        super().__init__(
            uninvertible_mapping=normaliser,
            splitter= pre_tokeniser,
        )

    @classmethod
    def from_tokeniser(cls, tokeniser: Tokenizer|str) -> 'TokenizersPreprocessor':
        """
        Class method to create a CustomPreprocessor from a :class:`~tokenizers.Tokenizer` instance.
        Args:
            tokeniser (tokenizers.Tokenizer|str): An instance of :class:`~tokenizers.Tokenizer` from which to extract the normalizer and pre-tokenizer. If a string is provided, it is treated as a file path to a JSON file containing a previously serialized :class:`~tokenizers.Tokenizer`.
        Returns:
            :class:`CustomPreprocessor`: A new instance of CustomPreprocessor initialized with the normalizer and pre-tokenizer from the Tokenizer.
        """
        if isinstance(tokeniser, str):
            if not os.path.exists(tokeniser):
                raise FileNotFoundError(f"The specified path does not exist: {tokeniser}")
            if not tokeniser.endswith('.json'):
                raise ValueError(f"The specified path must point to a JSON file: {tokeniser}")
            tokeniser = Tokenizer.from_file(tokeniser)

        normaliser = TokenizersNormaliser.from_tokeniser(tokeniser)
        pre_tokeniser = TokenizersPretokeniser.from_tokeniser(tokeniser)
        return cls(normaliser, pre_tokeniser)
    
    @classmethod
    def from_path(cls, path: str):
        tokeniser = Tokenizer.from_file(path)
        return cls.from_tokeniser(tokeniser)

## FUNCTIONS
def detectBoundaryMarkerFromTokeniser(tokeniser: Tokenizer) -> BoundaryMarker:
    """
    Copied from `tktkt.preparation.boundaries.detectBoundaryMarkerFromTokeniser`.
    Just changed the tokeniser from `hf_tokeniser` to `tokeniser`.
    """
    CHAR = "a"
    N = 50

    token_with_potential_prefix = tokeniser.encode(" " + CHAR*N).tokens[0]
    if CHAR in token_with_potential_prefix and token_with_potential_prefix.rstrip(CHAR) and token_with_potential_prefix != token_with_potential_prefix.rstrip(CHAR):
        prefix = token_with_potential_prefix.rstrip(CHAR)
        return BoundaryMarker(prefix, detached=True, location=BoundaryMarkerLocation.START)

    token_with_potential_suffix = tokeniser.encode(CHAR*N + " ").tokens[-1]
    if CHAR in token_with_potential_suffix and token_with_potential_suffix.lstrip(CHAR) and token_with_potential_suffix != token_with_potential_suffix.lstrip(CHAR):
        suffix = token_with_potential_suffix.lstrip(CHAR)
        return BoundaryMarker(suffix, detached=True, location=BoundaryMarkerLocation.END)

    # continuation = hf_tokeniser.tokenize("a"*100)[1].rstrip("a")  # TODO: Does TkTkT even support BERT continuation?
    # print("P:", prefix)
    # print("S:", suffix)
    return BoundaryMarker("", detached=True, location=BoundaryMarkerLocation.START)

# these following functions would perhaps be in the `tktkt.models.bpe.knockout` module
# just like how there is a fromHuggingFace method in the BPEKnockout class.
def get_info_from_tokeniser(tokeniser: Tokenizer) -> Dict[str, any]:
    """
    Extracts information from a Tokenizer instance.

    It assumes that the tokeniser is one of Unigram or BPE tokenisers, which have a vocabulary.
    Args:
        tokeniser (Tokenizer): An instance of :class:`~tokenizers.Tokenizer` from which to extract information.
    Returns:
        dict: A dictionary containing the vocabulary size, boundary marker, type of tokeniser, and merges (if applicable).
    Raises:
        ValueError: If the tokeniser type is not supported (i.e., not "BPE" or "Unigram").
    """
    info = json.loads(tokeniser.to_str())
    model = info.get("model", {})
    type_ = model.get("type", "")
    if type_ not in ["BPE", "Unigram"]:
        raise ValueError(f"Unsupported tokeniser type: {type_}. Only 'BPE' and 'Unigram' are supported.")
    vocab = model.get("vocab", {}) if type_ == "BPE" else model.get("vocab", []) # Unigram vocab is a list, BPE vocab is a dict
    merges = model.get("merges", [])
    return {
        "vocab": vocab,
        "boundary_marker": detectBoundaryMarkerFromTokeniser(tokeniser),
        "type": type_,
        "merges": merges,
    }

def getBPEKnockoutFromTokeniser(tokeniser: Tokenizer|str, language: Language|str) -> BPEKnockout:
    """
    Get a BPEKnockout instance from a Tokenizer or a file path to a Tokenizer JSON file.

    Args:
        tokeniser (Tokenizer|str): A Tokenizer instance or a string representing the path to a Tokenizer JSON file.
        language (Union[str, Language]): The language for which the BPEKnockout is being created. This is used to set the `language` attribute of the BPEKnockout instance to get the correct language-specific configurations.
    
    Returns:
        BPEKnockout: An instance of BPEKnockout initialized with the Tokenizer's vocabulary.
    """
    if isinstance(tokeniser, str):
        if not os.path.exists(tokeniser):
            raise FileNotFoundError(f"The specified path does not exist: {tokeniser}")
        if not tokeniser.endswith('.json'):
            raise ValueError(f"The specified path must point to a JSON file: {tokeniser}")
        tokeniser = Tokenizer.from_file(tokeniser)
    
    info = get_info_from_tokeniser(tokeniser)
    if info["type"] != "BPE":
        raise ValueError(f"Expected a BPE tokeniser, but got {info['type']}.")
    if not info["merges"]:
        raise ValueError("The BPE tokeniser does not have any merges defined. Please check the tokeniser file.")
    if not info["boundary_marker"]:
        raise ValueError("The BPE tokeniser does not have a boundary marker defined. Please check the tokeniser file.")
    if not info["vocab"]:
        raise ValueError("The BPE tokeniser does not have a vocabulary defined. Please check the tokeniser file.")
    
    return BPEKnockout(
        preprocessor=TokenizersPreprocessor.from_tokeniser(tokeniser),
        boundary_marker=info["boundary_marker"],
        unk_type=tokeniser.model.unk_token,

        vocab= info["vocab"],
        merges=info["merges"],

        language=language
    )