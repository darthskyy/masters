from pathlib import Path
from abc import ABC, abstractmethod
from typing import Iterator, List, Optional, Union, Tuple, Iterable
from ..paths import DATA_DIR
from pathlib import Path

class TokeniserTrainingDataset:
    """Base class for tokeniser training datasets."""
    
    def __init__(self, 
                 language_code: str,
                 path: Optional[Union[str, Path]] = None,
                 glob: Optional[str] = "*",
                 max_files: Optional[int] = -1,
                 max_lines_per_file: Optional[int] = -1,
                 ):
        """
        Initialize a tokeniser training dataset.
        
        Arguments:
            language_code: Language code (zu, xh, nr, ss)
            max_files: Maximum number of files to load (None for all)
            max_lines_per_file: Maximum number of lines to load per file (None for all)
            custom_path: Custom path to dataset files (overrides default path construction)
        """
        self.language_code = language_code
        self.max_files = max_files
        self.max_lines_per_file = max_lines_per_file
        self.glob = glob
        
        if path is not None:
            self.data_path = Path(path)
        else:
            raise ValueError("A path must be provided for the dataset.")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.data_path}")
    
    def iter_files(self) -> Iterator[Path]:
        """Iterate over dataset file(s)."""
        # If the path is a file, yield it directly
        if self.data_path.is_file():
            yield self.data_path
            return

        files = list(self.data_path.glob(self.glob))
        
        if self.max_files != -1 and self.max_files is not None:
            files = files[:self.max_files]
            
        for file_path in files:
            yield file_path
    
    def iter_lines(self) -> Iterator[str]:
        """Iterate over all lines in all files."""
        for file_path in self.iter_files():
            # TODO: handle posibilities of different file formats (e.g. csv, json, etc.)
            with open(file_path, 'r', encoding='utf-8') as f:
                if self.max_lines_per_file != -1 and self.max_lines_per_file is not None:
                    for i, line in enumerate(f):
                        if i >= self.max_lines_per_file:
                            break
                        # NOTE: should we skip empty lines
                        # and should we yield the line as is or strip it?
                        # right now we are stripping it and skipping empty lines
                        if line.strip():  # Skip empty lines
                            yield line.strip()
                        # yield line
                else:
                    for line in f:
                        if line.strip():  # Skip empty lines
                            yield line.strip()
                        # yield line
    
    def get_corpus(self) -> List[str]:
        """Return all lines as a list."""
        return list(self.iter_lines())
    
    def save(self, output_path: Optional[Union[str, Path]] = None) -> None:
        """
        Save the processed data to a file.
        
        Arguments:
            output_path: Path to save the processed data (default is the dataset path with '.txt' extension)
        """
        if output_path is None:
            output_path = self.data_path.with_suffix('.txt')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for line in self.iter_lines():
                f.write(line + '\n')
        
        print(f"Processed data saved to {output_path}")

    def __repr__(self) -> str:
        return f"TokeniserTrainingDataset(language_code={self.language_code}, path={self.data_path}, max_files={self.max_files}, max_lines_per_file={self.max_lines_per_file})"
    def __str__(self) -> str:
        return f"TokeniserTrainingDataset for {self.language_code} at {self.data_path} with max_files={self.max_files} and max_lines_per_file={self.max_lines_per_file}"
    def __len__(self) -> int:
        """Return the number of lines in the dataset."""
        return sum(1 for _ in self.iter_lines())
    def __iter__(self) -> Iterator[str]:
        """Iterate over all lines in the dataset."""
        return self.iter_lines()

class TokenRef:
    """Base class for token references."""
    trunc_chars = 50
    def __init__(self, text: str, tokens: List[str]):
        self.text = text
        self.tokens = tokens
    
    def produce(self) -> Tuple[str, List[str]]:
        """Return the text and tokens as a tuple."""
        return self.text, self.tokens
    
    def __repr__(self) -> str:
        text = self.text[:self.trunc_chars] + "..." if len(self.text) > self.trunc_chars else self.text
        tokens_str = str(self.tokens)
        if len(tokens_str) > self.trunc_chars:
            tokens_str = tokens_str[:self.trunc_chars] + "..."
        return f"TokenRef(text={text!r}, tokens={tokens_str})"
    
    def __str__(self) -> str:
        text = self.text[:self.trunc_chars] + "..." if len(self.text) > self.trunc_chars else self.text
        tokens_repr = ", ".join(self.tokens)
        if len(tokens_repr) > self.trunc_chars:
            tokens_repr = tokens_repr[:self.trunc_chars] + "..."
        return f"TokenRef: {text} -> {tokens_repr}"
    
    def __len__(self) -> int:
        return len(self.tokens)
    
    def __iter__(self):
        return iter(self.tokens)

class TokeniserEvaluationDatasetBase(ABC, TokenRef):
    """Base class for tokeniser evaluation datasets."""
    
    def __init__(self, language_code: str):
        self.language_code = language_code
    
    @abstractmethod
    def _validate(self):
        """Validate the dataset files exist and are accessible."""
        pass

    @abstractmethod
    def generate(self, **kwargs) -> Iterator[TokenRef]:
        """Generate tokenised objects from the dataset."""
        pass

    def __iter__(self) -> Iterator[TokenRef]:
        """Iterate over tokenised objects."""
        return self.generate()
    
    def evaluate_tokeniser(self, tokenise_func, eval_func):
        """
        Evaluate a tokeniser against this dataset.
        Arguments:
            tokenise_func: Function to apply for tokenisation
                - takes a single argument: text to be tokenised and returns a list of tokens
            eval_func: Function to evaluate the tokenised output
                - takes two arguments: expected tokens and tokenised output
        Returns:
            [dict[str, dict]]: A list of dictionaries containing evaluation results for each tokenisation.
            e.g.
        """
        results = []
        for token_ref in self.generate():
            text, tokens = token_ref.produce()
            tokenised_output = tokenise_func(text)
            score = eval_func(tokens, tokenised_output)
            results.append(
                {
                    "text": text,
                    "expected_tokens": tokens,
                    "tokenised_output": tokenised_output,
                    "score": score
                }
            )
        return results

class TokeniserEvalutionDataset(TokeniserEvaluationDatasetBase):
    """
    Concrete class for tokeniser evaluation datasets (from my expectations)
    
    Arguments:
        language_code (str): Language code (zu, xh, nr, ss)
        ref_path (Path | str): Path to the reference file
        token_path (Path | str): Path to the tokenised file

    """
    
    def __init__(self, language_code: str, text_path: Path | str, token_ref_path: Path | str, token_separator: str = "_"):
        super().__init__(language_code)
        self.text_path = Path(text_path)
        self.token_ref_path = Path(token_ref_path)
        self.token_separator = token_separator
    
    @classmethod
    def from_contained(cls, language_code: str, path: Path | str, token_separator: str = "_") -> 'TokeniserEvalutionDataset':
        """
        Create a TokeniserEvalutionDataset from a contained path.
        
        Arguments:
            language_code (str): Language code (zu, xh, nr, ss)
            path (Path | str): Path to the dataset directory
        """
        return cls(language_code, 
                   ref_path=Path(path) / f"{language_code}_ref.txt", 
                   token_path=Path(path) / f"{language_code}_token.txt",
                   token_separator=token_separator)
    
    def _validate(self):
        """Load the dataset from the specified path."""
        if not self.text_path.exists():
            raise FileNotFoundError(f"Reference file not found: {self.text_path}")
        if not self.token_ref_path.exists():
            raise FileNotFoundError(f"Tokenised file not found: {self.token_ref_path}")
    
    def generate(self, **kwargs) -> Iterator[TokenRef]:
        """Generate tokenised objects from the dataset."""
        self._validate()
        
        with open(self.text_path, 'r', encoding='utf-8') as text_file, \
             open(self.token_ref_path, 'r', encoding='utf-8') as token_file:
            for line, token_line in zip(text_file, token_file):
                ref_text = line.strip()
                tokens = token_line.strip().replace(self.token_separator, " ").split()
                if ref_text and tokens:
                    yield TokenRef(text=ref_text, tokens=tokens)