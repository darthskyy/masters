from pathlib import Path

from tokenizers import Tokenizer, pre_tokenizers, decoders, normalizers, models
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers.processors import PostProcessor
from tokenizers.decoders import Decoder
from tokenizers.normalizers import Normalizer
from tokenizers.trainers import BpeTrainer
from ..base import GenericTokeniser, Encoding, EncodingType, SPECIAL_TOKENS, Vocab
from ...datasets.base import TokeniserTrainingDataset

class BPEModel(GenericTokeniser):
    """
    A class for Byte Pair Encoding (BPE) tokenization.
    
    Inherits from GenericTokeniser and uses the Tokenizer class from the tokenizers library.

    This class supports training a BPE tokeniser on a dataset, loading a pre-trained tokeniser, and saving the tokeniser to a file.

    Arguments:
        model (BPE | None): The BPE model to use. If None, a new BPE model will be created.
        normaliser (Normalizer | None): The normalizer to use for text normalization.
        pre_tokeniser (PreTokenizer | None): The pre-tokenizer to use for initial tokenization.
        post_processor (PostProcessor | None): The post-processor to apply after tokenization.
        decoder (Decoder | None): The decoder to use for converting tokens back to text.
        special_tokens_set (str): The set of special tokens to use. Defaults to "default". Picks from the SPECIAL_TOKENS dictionary.

    """
    
    def __init__(
        self,
        model: BPE | None = None,
        normaliser: Normalizer | None = None,
        pre_tokeniser: PreTokenizer | None = None,
        post_processor: PostProcessor | None = None,
        decoder: Decoder | None = None,
        special_tokens: list[str] = SPECIAL_TOKENS["default"],
    ):
        """
        Initializes the BPEModel with optional vocabulary and merges.
        

        """
        super().__init__(implementation="tokenizers", algorithm="bpe")
        self.tokeniser = Tokenizer(model=model or BPE())
        self.tokeniser.normalizer = normaliser
        self.tokeniser.pre_tokenizer = pre_tokeniser
        self.tokeniser.post_processor = post_processor
        self.tokeniser.decoder = decoder
        self.special_tokens = special_tokens
        self.vocab = None
        self._is_trained = self.tokeniser.get_vocab_size() > 0
        if self._is_trained:
            self.tokeniser.add_special_tokens(special_tokens)
            self.vocab = Vocab(self.tokeniser.get_vocab())

    @classmethod
    def load(cls, path: str) -> "BPEModel":
        """
        Load a pre-trained BPE tokeniser from the specified path.
        
        Args:
            path (str): The path to the pre-trained tokeniser.
        
        Returns:
            BPEModel: An instance of the BPEModel.
        """
        tokenizer = Tokenizer.from_file(path)
        if not isinstance(tokenizer.model, BPE):
            raise ValueError(f"Expected a BPE model, but got {type(tokenizer.model)}.")
        special_tokens = [item.content for _, item in tokenizer.get_added_tokens_decoder().items()]
        return cls(
            model=tokenizer.model,
            normaliser=tokenizer.normalizer,
            pre_tokeniser=tokenizer.pre_tokenizer,
            post_processor=tokenizer.post_processor,
            decoder=tokenizer.decoder,
            special_tokens=special_tokens,
        )
    
    def save(self, path: Path | str) -> None:
        """
        Save the BPE tokeniser to the specified path.
        
        Args:
            path (str): The path where the tokeniser will be saved.
        """
        if not self._is_trained:
            raise ValueError("Tokeniser is not trained. Please train the tokeniser before saving.")
        if self.vocab is None:
            self.vocab = Vocab(self.tokeniser.get_vocab())

        path = path if path.endswith(".json") else f"{path}.json"
        path = Path(path)
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        
        # TODO: implement a saving method that saves in a consistent format over all tokenisers
        self.tokeniser.save(path.as_posix())

    def train(self, dataset: TokeniserTrainingDataset, vocab_size: int = 30000, **kwargs) -> None:
        """
        Train the BPE tokeniser on the provided dataset.
        
        Args:
            dataset (TokeniserTrainingDataset): The dataset to train the tokeniser on.
            vocab_size (int): The size of the vocabulary to be created.
        """
        trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=self.special_tokens, **kwargs)
        # FIXME: add special tokens to the trainer
        self.tokeniser.add_special_tokens(self.special_tokens)
        self.tokeniser.train_from_iterator(dataset, trainer=trainer)
        self.vocab = Vocab(self.tokeniser.get_vocab())
        self._is_trained = True
    
    # TODO: implement the other abstract methods from GenericTokeniser
    # [x] encode
    # [x] decode
    # [x] batch_encode
    # [x] batch_decode

    def encode(self, text: str) -> Encoding:
        """
        Encode a string into token IDs.
        
        Args:
            text (str): The input text to encode.
        
        Returns:
            EncodingType: The encoded representation of the text.
        """
        if not self._is_trained:
            raise ValueError("Tokeniser is not trained. Please train the tokeniser before encoding text.")
        encoding = self.tokeniser.encode(text)
        return Encoding(
            ids=encoding.ids,
            tokens=encoding.tokens,
            offsets=encoding.offsets,
        )

    def decode(self, tokens: EncodingType) -> str:
        """
        Decode token IDs back into a string.
        
        Args:
            tokens (Encoding | list[int] | list[str]): The tokens to decode. Can be an Encoding object, a list of integers, or a list of strings.
        
        Returns:
            str: The decoded string.
        """
        if not self._is_trained:
            raise ValueError("Tokeniser is not trained. Please train the tokeniser before decoding text.")
        
        # Determine the type of tokens and extract IDs accordingly
        # maybe in the future we can support more types of tokens
        if isinstance(tokens, Encoding):
            ids = tokens.ids
        elif isinstance(tokens, list):
            if len(tokens) == 0:
                return ""
            elif isinstance(tokens[0], int):
                ids = tokens
            elif isinstance(tokens[0], str):
                ids = self.get_ids(tokens)
        else:
            raise TypeError("tokens must be an instance of Encoding or a list of integers or strings.")
        
        return self.tokeniser.decode(ids)

    def batch_encode(self, texts: list[str]) -> list[Encoding]:
        """
        Encode a list of strings into token IDs.
        
        Args:
            texts (list[str]): The input texts to encode.
        
        Returns:
            list[Encoding]: A list of Encodings for each input text.
        """
        if not self._is_trained:
            raise ValueError("Tokeniser is not trained. Please train the tokeniser before encoding text.")
        
        # using encode_batch_fast for better performance
        # encodings = self.tokeniser.encode_batch_fast(texts)
        # result = [
        #     Encoding(
        #         ids=encoding.ids,
        #         tokens=self.get_tokens(encoding.ids),
        #     ) for encoding in encodings
        # ]

        # for now we are doing batch decoding slow
        encodings = self.tokeniser.encode_batch(texts)
        result = [
            Encoding(
                ids=encoding.ids,
                tokens=encoding.tokens,
                offsets=encoding.offsets,
            ) for encoding in encodings
        ]
        
        return result

    def batch_decode(self, token_batches: list[EncodingType]) -> list[str]:
        """
        Decode a list of token IDs back into strings.
        
        Args:
            token_batches (list[Encoding | list[int] | list[str]]): A list of token batches to decode. Each batch can be an Encoding object, a list of integers, or a list of strings.
            All elements in the list should be of the same type.
        
        Returns:
            list[str]: A list of decoded strings.
        """
        if not self._is_trained:
            raise ValueError("Tokeniser is not trained. Please train the tokeniser before decoding text.")
        if not token_batches:
            return []
        # Determine the type of tokens and extract IDs accordingly
        if isinstance(token_batches[0], Encoding):
            ids_batches = [encoding.ids for encoding in token_batches]
        elif isinstance(token_batches[0], list):
            if len(token_batches[0]) == 0:
                return []
            elif isinstance(token_batches[0][0], int):
                ids_batches = token_batches
            elif isinstance(token_batches[0][0], str):
                ids_batches = [self.get_ids(t) for t in token_batches]
            else:
                raise TypeError("tokens must be a list of Encoding, a list of lists of integers, or a list of lists of strings.")
        else:
            raise TypeError("tokens must be a list of Encoding, a list of lists of integers, or a list of lists of strings.")
        
        # still thinking about what to do with special tokens
        return self.tokeniser.decode_batch(ids_batches)