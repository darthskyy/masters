"""
This module provides a dataset class for the Sadilar Morphological Dataset.
It is designed to handle morphological data, particularly for languages with rich morphology.

The data that is read is in TSV format, and the dataset is structured to handle inflectional morphology.

Sadilar Base Files are in the following format:
word{TAB}morph_analysis(morphs+tags){TAB}decomposition/canonical_form{TAB}lemma

Sadilar MorphParse Edition files are in the following format:
word{TAB}morph_analysis(morphs+tags){TAB}segmentation/surface_form{TAB}morphological_tag
"""

from typing import Iterator, Tuple, Union
import langcodes
from pathlib import Path
from modest.formats.tsv import iterateTsv
from modest.interfaces.datasets import ModestDataset, M
from modest.interfaces.morphologies import WordDecomposition, WordSegmentation

# redefining Languageish here for clarity of what it is.
Languageish = Union[langcodes.Language, str]

SOUTH_AFRICAN_LANGUAGES = {
    langcodes.find("English"): "en",
    langcodes.find("South Ndebele"): "nr",
    langcodes.find("siSwati"): "ss",
    langcodes.find("isiXhosa"): "xh",
    langcodes.find("isiZulu"): "zu",
}

BASE_DIR = "sadilar_morph_data" # This should be a default

## CLASSES
# NOTE: I am copying the format from the MODEST morphynet dataset in the MODEST repository.
# the files copied are src/modest/formats/morphynet.py
class SadilarMorphologySurface(WordSegmentation):
    """
    Represents a surface morphological analysis from the Sadilar Morphological Dataset.
    The surface form is acquired from scripts by Moeng et. al. (2021). [MorphSegment](https://github.com/TumiMoeng/MORPH_SEGMENT)

    This WordSegmentation subclass uses the surface form (morphs) of the word meaning there are no algorithms which need to be applied to the word to segment it.

    Arguments:
        word (str): The word unaltered, as it appears in the dataset.
        analysis (str): The morphological analysis/surface segmentation of the word, which is a string of morphemes separated by a separator.
        morpheme_separator (str): The separator used in the surface form to separate morphemes. Defaults to "_".

    Example:
        >>> from sadilar_morph import SadilarMorphologySurface
        >>> analysis = SadilarMorphologySurface(word="ezelulekayo", analysis="eze_lulek_a_yo")
        >>> analysis.segment()
        ('eze', 'lulek', 'a', 'yo')
    """
    
    def __init__(self, word: str, analysis: str, morpheme_separator: str = "_"):
        super().__init__(word=word)
        self.word = word
        self.analysis = analysis
        self.morpheme_separator = morpheme_separator  # Separator used in the surface form

    def segment(self) -> Tuple[str, ...]:
        """
        Segments the word into its morphemes based on the surface form.

        Returns:
            Tuple[str, ...]: A tuple of morphemes derived from the surface form.
        """
        return tuple(self.analysis.split(self.morpheme_separator)) if self.analysis else (self.analysis,)

class SadilarMorphologyCanonical(WordDecomposition):
    """
    Represents a canonical morphological analysis from the Sadilar Morphological Dataset.
    This class is used to encapsulate the word and its morphological analysis.

    This WordDecomposition subclass uses the canonical form of the word, which is a string of morphemes separated by a separator.

    Arguments:
        word (str): The word, as it appears in the dataset.
        analysis (str): The morphological analysis/canonical decomposition of the word, which is a string of morphemes separated by a separator.
        morpheme_separator (str): The separator used in the canonical form to separate morphemes. Defaults to "_".

    Example:
        >>> from sadilar_morph import SadilarMorphologyCanonical
        >>> analysis = SadilarMorphologyCanonical(word="ezelulekayo", analysis="ezi_lulek_a_yo")
        >>> analysis.decompose()
        ('ezi', 'lulek', 'a', 'yo')
    """
    
    def __init__(self, word: str, analysis: str, morpheme_separator: str = "_"):
        super().__init__(word=word)
        self.word = word  # The surface form of the word
        self.analysis = analysis  # The morphological analysis of the word
        self.morpheme_separator = morpheme_separator  # Separator used in the analysis

    def decompose(self) -> Tuple[str, ...]:
        """
        Decomposes the word into its morphemes based on the canonical form.
        Returns:
            tuple[str, ...]: A tuple of morphemes derived from the canonical form.
        """
        return self.analysis.split(self.morpheme_separator) if self.analysis else (self.word,)


# the files copied are src/modest/datasets/morphynet
class SadilarMorphDataset(ModestDataset[M]):
    """
    A dataset class for the Sadilar Morphological Dataset.
    This class is designed to handle morphological data.

    Requires data to be in directory structure:
    BASEDIR/
        ├── surface/
        │   └── {lang_code}.tsv
        └── canonical/
            └── {lang_code}.tsv

    where {lang_code} is the language code as per Sadilar's conventions.

    Arguments:
        language (Languageish): The language for which the dataset is to be loaded.
    Attributes:
        _subset (str): The subset of the dataset, which is "inflectional" for this dataset.
    """
    
    def __init__(self, language: Languageish, set_name: str = None):
        super().__init__(name="SadilarMorph", language=language)
        self._subset = "inflectional"  # This dataset is focused on inflectional morphology
        assert set_name in ["surface", "canonical"], "set_name must be either 'surface' or 'canonical'."
        self._set_name = set_name
        self._sadilar_code = SOUTH_AFRICAN_LANGUAGES.get(self._language)
        if self._sadilar_code is None:
            raise ValueError(f"Language not in Sadilar Morphological Dataset: {self._language}")
    
    def _get(self) -> Path:
        if self._set_name is None:
            raise ValueError("Set name must be provided to access the dataset.")
        # Construct the path to the dataset based on the language code
        return Path(f"{BASE_DIR}/{self._set_name}/{self._sadilar_code.upper()}.tsv")
    
class SadilarMorphDataset_Surface(SadilarMorphDataset[SadilarMorphologySurface]):
    """
    A dataset class for the surface morphological analysis from the Sadilar Morphological Dataset.
    This class reads the surface forms of words and segments them into morphemes.
    The surface form is acquired from scripts by Moeng et. al. (2021): [MorphSegment](https://github.com/TumiMoeng/MORPH_SEGMENT).

    The surface form is a string of morphemes separated by a separator, which is typically an underscore (_).

    NOTE: for now the dataset sets the word to lowercase, as the surface form (morphs) is typically in lowercase. Needs to be checked if this is always the case because it may hurt some words with important capitalization.
    
    Arguments:
        language (Languageish): The language for which the dataset is to be loaded.
    Attributes:
        _subset (str): The subset of the dataset, which is "inflectional" for this dataset.
    This dataset is focused on inflectional morphology, specifically surface segmentation.

    Example:
        >>> from sadilar_morph import SadilarMorphDataset_Surface
        >>> # Iterate through the dataset and print the word and its segmented morphemes
        >>> # Note: Assuming that ezelulekayo is the first word in the dataset
        >>> dataset = SadilarMorphDataset_Surface(language="zulu")
        >>> for item in dataset.generate():
        ...     print(item.word, item.segment())
        ezelulekayo ('eze', 'lulek', 'a', 'yo')

    """

    def __init__(self, language: Languageish):
        # Added set_name to the constructor to specify the type of dataset
        super().__init__(language=language, set_name="surface")
    
    def _generate(self, path: Path, **kwargs) -> Iterator[SadilarMorphologySurface]:
        """
        Generates instances of SadilarMorphologySurface from the TSV file at the given path.
        This method reads the TSV file, extracts the word and its surface analysis, and yields instances of SadilarMorphologySurface.

        NOTE: The word is converted to lowercase, as the surface form (morphs) is typically in lowercase.
        NOTE: The method ensures that each word is only yielded once, even if it appears multiple times in the dataset. Maybe removing this would be a form of weighting the dataset, but for now it is not needed.

        Arguments:
            path (Path): The path to the TSV file containing the morphological data.
        Returns:
            Iterator[SadilarMorphologySurface]: An iterator yielding instances of SadilarMorphologySurface.
        """

        prev: SadilarMorphologySurface = None

        seen = set()
        for parts in iterateTsv(path):
            word, analysis = parts[0].lower(), parts[2]  # Assuming the third column is the surface form of the word
            if word in seen:
                continue
            seen.add(word)

            curr = SadilarMorphologySurface(word=word, analysis=analysis)
            if prev is None:
                prev = curr
                continue

            yield prev
            prev = curr

class SadilarMorphDataset_Canonical(SadilarMorphDataset[SadilarMorphologyCanonical]):
    """
    A dataset class for the canonical morphological analysis from the Sadilar Morphological Dataset.
    This class reads the canonical forms of words and decomposes them into morphemes.
    The canonical form is a string of morphemes separated by a separator, which is typically an underscore (_).

    Arguments:
        language (Languageish): The language for which the dataset is to be loaded.
    Attributes:
        _subset (str): The subset of the dataset, which is "inflectional" for this dataset.
    This dataset is focused on inflectional morphology, specifically canonical decomposition.
    Example:
        >>> from sadilar_morph import SadilarMorphDataset_Canonical
        >>> # Iterate through the dataset and print the word and its decomposed morphemes
        >>> # Note: Assuming that ezelulekayo is the first word in the dataset
        >>> dataset = SadilarMorphDataset_Canonical(language="zulu")
        >>> for item in dataset.generate():
        ...     print(item.word, item.decompose())
        ezelulekayo ('ezi', 'lulek', 'a', 'yo')
    """

    def __init__(self, language: Languageish):
        # Added set_name to the constructor to specify the type of dataset
        super().__init__(language=language, set_name="canonical")
    
    def _generate(self, path: Path, **kwargs) -> Iterator[SadilarMorphologyCanonical]:
        """
        Generates instances of SadilarMorphologyCanonical from the TSV file at the given path.
        This method reads the TSV file, extracts the word and its canonical analysis, and yields instances of SadilarMorphologyCanonical.

        NOTE: The word is converted to lowercase, as the canonical form is typically in lowercase.
        NOTE: The method ensures that each word is only yielded once, even if it appears multiple times in the dataset. Maybe removing this would be a form of weighting the dataset, but for now it is not needed.

        Arguments:
            path (Path): The path to the TSV file containing the morphological data.
        Returns:
            Iterator[SadilarMorphologyCanonical]: An iterator yielding instances of SadilarMorphologyCanonical.
        """

        prev: SadilarMorphologyCanonical = None

        seen = set()
        for parts in iterateTsv(path):
            word, analysis = parts[0].lower(), parts[2]
            if word in seen:
                continue
            seen.add(word)

            curr = SadilarMorphologyCanonical(word=word, analysis=analysis)
            if prev is None:
                prev = curr
                continue
            yield prev
            prev = curr


def main():
    dataset = SadilarMorphDataset_Surface(language="zulu")

    i = 0
    for item in dataset.generate():
        i += 1
        if i >= 10:
            break
        print(item.word, item.segment())
    print("done")
    pass

if __name__ == '__main__':
    main()