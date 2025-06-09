from abc import ABC, abstractmethod
from pathlib import Path

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

    def __init__(self):
        self._token_to_id = {}
        self._id_to_token = {}
        self._special_tokens = {}
        self._next_id = 0
    
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
    def __call__(self, text: str) -> list[str]:
        """
        Tokenise the input text.

        Args:
            text (str): The input text to be tokenised.

        Returns:
            list[str]: A list of tokens.
        """
    
    @abstractmethod
    def train(self, dataset: TokeniserTrainingDataset, vocab_size: int | None = None, **kwargs) -> None:
        """
        Train the tokeniser on the provided dataset.

        Args:
            dataset (TokeniserTrainingDataset): The dataset to train the tokeniser on.
            **kwargs: Additional keyword arguments for training configuration.
        """
        pass
    
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
    def encode(self, text: str) -> list[str]:
        """
        Encode the input text into tokens.

        Args:
            text (str): The input text to be encoded.

        Returns:
            list[str]: A list of encoded tokens.
        """
    
    @abstractmethod
    def decode(self, tokens: list[str]) -> str:
        """
        Decode the list of tokens back into text.

        Args:
            tokens (list[str]): The list of tokens to be decoded.

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
    
    def get_vocab(self) -> Vocab:
        """
        Get the vocabulary of the tokeniser.

        Returns:
            Vocab: The vocabulary object containing token to ID mappings.
        """
        return self.vocab
    
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