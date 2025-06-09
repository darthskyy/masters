from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

from ..datasets.base import TokeniserTrainingDataset

# the supported tokeniser implementations
# for each tokenisation algorithm
SUPPORTED_ALGORITHMS = {
    "bpe": # [ ] first to be implemented
        (
            "huggingface",
            "sentencepiece",
            "tktkt",
            "tokenizers",
        ),
    "unigram": # [ ] not so important for now
        (
            "huggingface",
            "sentencepiece",
            "tokenizers",
        ),
    "bpe_knockout": # [ ] first to be implemented
        (
            "tktkt"
        ),
}

SPECIAL_TOKENS = {
    "minimal": ["<unk>", "<pad>"],
    "default": ["<unk>", "<pad>", "<s>", "</s>", "<mask>"],
    "extended": ["<unk>", "<pad>", "<s>", "</s>", "<mask>", "<cls>", "<sep>"],
}
class Vocab:
    """
    A simple vocabulary wrapper that maps tokens to IDs and vice versa.
    This class provides methods to add tokens, retrieve their IDs, and check for token existence.
    It is designed to be used with tokenisers to manage the vocabulary of tokens.

    Attributes:
        _token_to_id (dict): A dictionary mapping tokens to their unique IDs.
        _id_to_token (dict): A dictionary mapping IDs back to their corresponding tokens.
        _special_tokens (dict): A dictionary for special tokens (not used in this implementation).
        _next_id (int): The next available ID for a new token.
    """

    def __init__(self, token_to_id: dict[str, int] | None = None):
        if token_to_id is None:
            token_to_id = {}
        self._token_to_id = token_to_id
        self._id_to_token = {v: k for k, v in token_to_id.items()}
        self._next_id = max(self._token_to_id.values(), default=-1) + 1
        self._special_tokens = {}  # Not used in this implementation, but can be extended later
        self._next_id = max(self._token_to_id.values(), default=-1) + 1        
    
    def add_token(self, token: str) -> int:
        """Add a token and return its ID"""
        if token not in self._token_to_id:
            self._token_to_id[token] = self._next_id
            self._id_to_token[self._next_id] = token
            self._next_id += 1
        return self._token_to_id[token]
    
    def get_id(self, token: str, default=None) -> int | None:
        """Get ID for a token. Returns None if token is not found."""
        return self._token_to_id.get(token, default)
    
    def get_token(self, id: int, default=None) -> str:
        """Get token for an ID. Returns None if ID is not found."""
        return self._id_to_token.get(id, default)
    
    def __len__(self) -> int:
        return len(self._token_to_id)
    
    def __contains__(self, token: str) -> bool:
        return token in self._token_to_id
    
    # Just for convience, to access the token to ID mapping directly
    @property
    def token_to_id(self) -> dict:
        return self._token_to_id.copy()

class Encoding:
    """
    A simple class to represent a tokenisation operation.
    This class is used to encapsulate the tokenised text and its corresponding IDs.
    It provides methods to retrieve the tokenised text and the IDs of the tokens.

    Attributes:
        tokens (list[str]): The list of tokenised strings.
        ids (list[int]): The list of IDs corresponding to the tokens.
    """

    def __init__(self, tokens: list[str], ids: list[int], offsets: list[tuple[int, int]] | None = None):
        self.tokens = tokens
        self.ids = ids
        self.offsets = offsets
    
    def __len__(self) -> int:
        """Return the number of tokens in the tokenisation."""
        return len(self.tokens)
    
    def __str__(self):
        """Return a string representation of the tokenisation."""
        return f"{self.tokens}"
    


# tokenisation type which supports:
# - Tokenisation object
# - list of strings (tokens)
# - list of integers (IDs)
type EncodingType = Union[Encoding, list[str], list[int]]

class GenericTokeniser(ABC):
    """
    Base class for my tokenisers.
    This is for helping in building a consistent interface and pipeline for tokenisation.
    It defines the basic structure and methods that all tokenisers should implement.

    Arguments:
        implementation (str): The implementation of the tokeniser (e.g., "huggingface", "sentencepiece", "tktkt").
        algorithm (str): The algorithm used by the tokeniser (e.g., "bpe", "unigram", "bpe_knockout").
    Raises:
        ValueError: If the implementation or algorithm is not supported.
    Supported implementations and algorithms are defined in the SUPPORTED_IMPLEMENTATIONS dictionary.
    Supported implementations:
        - "huggingface": Uses the Hugging Face Transformers library.
        - "sentencepiece": Uses the SentencePiece library.
        - "tktkt": The tokenisation library by Thomas Bauwens.
        - "tokenizers": Uses the Tokenizers library from Hugging Face.

    Supported algorithms:
        - "bpe": Byte Pair Encoding, a common tokenisation algorithm.
        - "unigram": A unigram language model for tokenisation.
        - "bpe_knockout": A variant of BPE which is more morphologically aware.
    """

    def __init__(self, implementation: str, algorithm: str):
        """
        Initialize the tokeniser with the specified implementation and algorithm.

        Arguments:
            implementation (str): The implementation of the tokeniser (e.g., "huggingface", "sentencepiece", "tktkt").
            algorithm (str): The algorithm used by the tokeniser (e.g., "bpe", "unigram", "bpe_knockout").
        Raises:
            ValueError: If the implementation or algorithm is not supported.
        """
        if algorithm not in SUPPORTED_ALGORITHMS:
            raise ValueError(f"Unsupported algorithm '{algorithm}' for implementation '{implementation}'")
        
        if implementation not in SUPPORTED_ALGORITHMS[algorithm]:
            raise ValueError(f"Unsupported implementation '{implementation}' for algorithm '{algorithm}'")
        

        self.implementation = implementation
        self.algorithm = algorithm
        self._is_trained = False
        self.vocab = Vocab()

    # Abstract methods to be implemented by subclasses
    @classmethod
    @abstractmethod
    def load(cls, path: Path | str) -> "GenericTokeniser":
        """
        Load a pre-trained tokeniser from the specified path.

        Args:
            path (Union[str, Path]): The path to the pre-trained tokeniser.
            algorithm (str): The algorithm used by the tokeniser.
            vocab_size (int, optional): The vocabulary size for the tokeniser.

        Returns:
            GenericTokeniser: An instance of the tokeniser.
        """
        pass

    @abstractmethod
    def save(self, path: Path | str) -> None:
        """
        Save the tokeniser to the specified path.

        Args:
            path (Union[str, Path]): The path where the tokeniser will be saved.
        """
        pass
    
    @abstractmethod
    def train(self, dataset: TokeniserTrainingDataset, vocab_size: int | None = None, **kwargs) -> None:
        """
        Train the tokeniser on the provided dataset.

        Args:
            dataset (TokeniserTrainingDataset): The dataset to train the tokeniser on.
            **kwargs: Additional keyword arguments for training configuration.
        """
        pass
    
    # NOTE: Huge MAYBE here (is it necessary to have this method?)
    # @abstractmethod
    # def get_parameters(self) -> dict:
    #     """
    #     Get the parameters of the tokeniser.

    #     Returns:
    #         dict: A dictionary containing the tokeniser's parameters.
    #     """
    #     pass
    
    @abstractmethod
    def encode(self, text: str) -> EncodingType:
        """
        Encode the input text into tokens.

        Args:
            text (str): The input text to be encoded.

        Returns:
            TokenisationType: A tokenisation object, a list of tokens, or a list of IDs.
        """
    
    @abstractmethod
    def decode(self, tokens: EncodingType) -> str:
        """
        Decode the list of tokens back into text.

        Args:
            tokens (Union[Tokenisation, list[str], list[int]]): The tokenised input to be decoded.

        Returns:
            str: The decoded text.
        """
    
    @abstractmethod
    def batch_encode(self, texts: list[str]) -> list[list[str]]:
        """
        Encode a batch of input texts into tokens.

        Args:
            texts (list[str]): A list of input texts to be encoded.

        Returns:
            list[list[str]]: A list of lists, where each inner list contains tokens for the corresponding text.
        """
        pass

    @abstractmethod
    def batch_decode(self, token_batches: list[list[str]]) -> list[str]:
        """
        Decode a batch of token lists back into texts.

        Args:
            token_batches (list[list[str]]): A list of lists of tokens to be decoded.

        Returns:
            list[str]: A list of decoded texts corresponding to the input token batches.
        """
        pass
    
    # Concrete methods
    def is_trained(self) -> bool:
        """
        Check if the tokeniser has been trained.

        Returns:
            bool: True if the tokeniser is trained, False otherwise.
        """
        return self._is_trained
    
    def __call__(self, text: str) -> list[str]:
        """
        Tokenise the input text.

        Args:
            text (str): The input text to be tokenised.

        Returns:
            list[str]: A list of tokens.
        """
        if not self.is_trained():
            raise ValueError("The tokeniser has not been trained yet.")
        
        return self.encode(text)
    
    def train_from_file(self, path: Path | str, vocab_size: int | None = None, **kwargs) -> None:
        """
        Train the tokeniser from a file.

        Args:
            path (Union[str, Path]): The path to the training data file.
            vocab_size (int, optional): The vocabulary size for the tokeniser.
            **kwargs: Additional keyword arguments for training configuration.
        """
        dataset = TokeniserTrainingDataset(path=path, language_code="N/A")
        self.train(dataset, vocab_size=vocab_size, **kwargs)
        
    def get_vocab(self) -> Vocab:
        """
        Get the vocabulary of the tokeniser.

        Returns:
            Vocab: The vocabulary object containing token to ID mappings.
        """
        return self.vocab
    
    def get_ids(self, tokens: list[str]) -> list[int]:
        """
        Get the IDs for a list of tokens.

        Args:
            tokens (list[str]): A list of tokens to get IDs for.

        Returns:
            list[int]: A list of IDs corresponding to the input tokens.
        """
        return [self.vocab.get_id(token) for token in tokens]
    
    def get_tokens(self, ids: list[int]) -> list[str]:
        """
        Get the tokens for a list of IDs.

        Args:
            ids (list[int]): A list of IDs to get tokens for.

        Returns:
            list[str]: A list of tokens corresponding to the input IDs.
        """
        return [self.vocab.get_token(id) for id in ids]
    
    @property
    def vocab_size(self) -> int:
        """
        Get the size of the vocabulary.

        Returns:
            int: The number of unique tokens in the vocabulary.
        """
        return len(self.vocab)

    @property
    def config(self) -> dict:
        """
        Get the configuration of the tokeniser.

        Returns:
            dict: A dictionary containing the tokeniser's configuration.
        """
        return {
            "implementation": self.implementation,
            "algorithm": self.algorithm,
            "vocab_size": self.vocab_size,
            "is_trained": self._is_trained,
            "vocab": self.vocab.token_to_id,
        }