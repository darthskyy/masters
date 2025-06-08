from pathlib import Path
from typing import Iterator, List, Optional, Union
from ..paths import DATA_DIR

class TokenizerTrainingDataset:
    """Base class for tokenizer training datasets."""
    
    def __init__(self, 
                 language_code: str,
                 path: Optional[Union[str, Path]] = None,
                 glob: Optional[str] = "*",
                 max_files: Optional[int] = -1,
                 max_lines_per_file: Optional[int] = -1,
                 ):
        """
        Initialize a tokenizer training dataset.
        
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
            with open(file_path, 'r', encoding='utf-8') as f:
                if self.max_lines_per_file != -1 and self.max_lines_per_file is not None:
                    for line in f:
                        if line.strip():  # Skip empty lines
                            yield line.strip()
                else:
                    for i, line in enumerate(f):
                        if i >= self.max_lines_per_file:
                            break
                        if line.strip():  # Skip empty lines
                            yield line.strip()
    
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
        return f"TokenizerTrainingDataset(language_code={self.language_code}, path={self.data_path}, max_files={self.max_files}, max_lines_per_file={self.max_lines_per_file})"
    def __str__(self) -> str:
        return f"TokenizerTrainingDataset for {self.language_code} at {self.data_path} with max_files={self.max_files} and max_lines_per_file={self.max_lines_per_file}"
    def __len__(self) -> int:
        """Return the number of lines in the dataset."""
        return sum(1 for _ in self.iter_lines())
    def __iter__(self) -> Iterator[str]:
        """Iterate over all lines in the dataset."""
        return self.iter_lines()