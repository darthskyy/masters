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

class EvaluationDataset:
    """
    Simple class for simple binary evaluation datasets.
    Arguments:
        language_code (str): Language code (zu, xh, nr, ss)
        token_ref_path (Path | str): Path to the tokenised reference file
        token_pred_path (Path | str): Path to the tokenised prediction file
        token_separator (str): Separator used in the tokenised file (default is "_")
        metric: Evaluation metric function to use (default is None)
            - should take two arguments: expected tokens and predicted tokens, in that order
            - should return a dictionary with evaluation results

    """
    def __init__(self,
                 language_code: str,
                 token_ref_path: Union[str, Path],
                 token_pred_path: Union[str, Path],
                 token_separator: str = "_",
                 metric = None,
                 ):
        self.language_code = language_code
        self.token_ref_path = Path(token_ref_path)
        self.token_pred_path = Path(token_pred_path)
        self.token_separator = token_separator
        self.metric = metric
        
        self._validate()

    def _validate(self):
        """Validate the dataset files exist and are accessible."""
        if not self.token_ref_path.exists():
            raise FileNotFoundError(f"Reference file not found: {self.token_ref_path}")
        if not self.token_pred_path.exists():
            raise FileNotFoundError(f"Prediction file not found: {self.token_pred_path}")
    
    @classmethod
    def from_contained(cls, language_code: str, path: Union[str, Path], token_separator: str = "_", metric = None) -> 'EvaluationDataset':
        """
        Create an EvaluationDataset from a contained path.
        
        Arguments:
            language_code (str): Language code (zu, xh, nr, ss)
            path (Path | str): Path to the dataset directory
                - should contain files named "{language_code}_ref.txt" and "{language_code}_pred.txt"
            token_separator (str): Separator used in the tokenised file (default is "_")
            metric: Evaluation metric function to use (default is None)
                - should take two arguments: expected tokens and predicted tokens, in that order
                - should return a dictionary with evaluation results
        """
        return cls(
            language_code, 
            token_ref_path=Path(path) / f"{language_code}_ref.txt", 
            token_pred_path=Path(path) / f"{language_code}_pred.txt",
            token_separator=token_separator,
            metric=metric
            )

    def evaluate(self):
        """
        Evaluate the tokenised predictions against the reference tokens.
        
        Returns:
            List[dict]: A list of dictionaries containing evaluation results for each tokenisation.
            Each dictionary contains:
                - "expected_tokens": List of expected tokens from the reference file
                - "predicted_tokens": List of predicted tokens from the prediction file
                - "score": Evaluation score calculated by the metric function
        Raises:
            ValueError: If no evaluation metric is provided.
        """
        if self.metric is None:
            raise ValueError("No evaluation metric provided. Please set a metric function.")
        results = []
        with open(self.token_ref_path, 'r', encoding='utf-8') as ref_file, \
             open(self.token_pred_path, 'r', encoding='utf-8') as pred_file:
            for ref_line, pred_line in zip(ref_file, pred_file):
                ref_tokens = ref_line.strip().split(self.token_separator)
                pred_tokens = pred_line.strip().split(self.token_separator)
                
                if ref_tokens and pred_tokens:
                    score = self.metric(ref_tokens, pred_tokens)
                    results.append({
                        "expected_tokens": ref_tokens,
                        "predicted_tokens": pred_tokens,
                        "score": score
                    })
        return results
    