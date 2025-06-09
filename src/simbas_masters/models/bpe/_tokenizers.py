# from ..base import GenericTokeniser
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers.processors import PostProcessor
from tokenizers.decoders import Decoder
from tokenizers.normalizers import Normalizer
from tokenizers.trainers import BpeTrainer 
from ..base import GenericTokeniser, SPECIAL_TOKENS
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
        special_tokens_set: str = "default",  # Default to the 'default' set of special tokens
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
        self.special_tokens = SPECIAL_TOKENS.get(special_tokens_set, SPECIAL_TOKENS["default"])
        self._is_trained = self.tokeniser.get_vocab_size() > 0
        if self._is_trained:
            self.tokeniser.add_special_tokens(SPECIAL_TOKENS[special_tokens_set])

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
        return cls(
            model=tokenizer.model,
            normaliser=tokenizer.normalizer,
            pre_tokeniser=tokenizer.pre_tokenizer,
            post_processor=tokenizer.post_processor,
            decoder=tokenizer.decoder
        )
    
    def save(self, path: str) -> None:
        """
        Save the BPE tokeniser to the specified path.
        
        Args:
            path (str): The path where the tokeniser will be saved.
        """
        # TODO: implement a saving method that saves in a consistent format over all tokenisers
        
        self.tokeniser.save(path)

    def train(self, dataset: TokeniserTrainingDataset, vocab_size: int = 30000) -> None:
        """
        Train the BPE tokeniser on the provided dataset.
        
        Args:
            dataset (TokeniserTrainingDataset): The dataset to train the tokeniser on.
            vocab_size (int): The size of the vocabulary to be created.
        """
        trainer = BpeTrainer(vocab_size=vocab_size)
        self.tokeniser.train_from_iterator(dataset, trainer=trainer, special_tokens=self.special_tokens)
        self._is_trained = True
    
    # TODO: implement the other abstract methods from GenericTokeniser