"""
This module provides a dataset class for the Sadilar Morphological Dataset.
It is designed to handle morphological data, particularly for languages with rich morphology.

The data that is read is in TSV format, and the dataset is structured to handle inflectional morphology.

Sadilar Base Files are in the following format:
word{TAB}morph_analysis(morphs+tags){TAB}decomposition/canonical_form{TAB}lemma

Sadilar MorphParse Edition files are in the following format:
word{TAB}morph_analysis(morphs+tags){TAB}segmentation/surface_form{TAB}morphological_tag
"""

from typing import Iterator
import langcodes
from pathlib import Path
from modest.formats.tsv import iterateTsv
from modest.interfaces.datasets import ModestDataset, M, Languageish
from modest.interfaces.morphologies import WordDecomposition, WordSegmentation

SOUTH_AFRICAN_LANGUAGES = {
    langcodes.find("English"): "en",
    langcodes.find("South Ndebele"): "nr",
    langcodes.find("siSwati"): "ss",
    langcodes.find("isiXhosa"): "xh",
    langcodes.find("isiZulu"): "zu",
}

BASE_DIR = "sadilar-morph"

# NOTE: I am copying the format from the MODEST morphynet dataset in the MODEST repository.
# the files copied are src/modest/formats/morphynet.py
class SadilarMorphologySurface(WordSegmentation):
    """
    Represents a surface morphological analysis from the Sadilar Morphological Dataset.
    The surface form is acquired from scripts by Moeng et. al. (2021). [MorphSegment](https://github.com/TumiMoeng/MORPH_SEGMENT)
    """
    
    def __init__(self, word: str, analysis: str, morpheme_separator: str = "_"):
        super().__init__(word=word)
        self.word = word
        self.analysis = analysis
        self.morpheme_separator = morpheme_separator  # Separator used in the surface form

    def segment(self) -> tuple[str, ...]:
        """
        Segments the word into its morphemes.
        Returns a tuple of morphemes.
        """
        return tuple(self.analysis.split(self.morpheme_separator)) if self.analysis else (self.analysis,)

class SadilarMorphologyCanonical(WordDecomposition):
    """
    Represents a canonical morphological analysis from the Sadilar Morphological Dataset.
    This class is used to encapsulate the word and its morphological analysis.
    """
    
    def __init__(self, word: str, analysis: str, morpheme_separator: str = "_"):
        super().__init__(word=word)
        self.analysis = analysis  # The morphological analysis of the word
        self.word = word  # The surface form of the word
        self.morpheme_separator = morpheme_separator  # Separator used in the analysis

    def decompose(self) -> tuple[str, ...]:
        """
        Decomposes the word into its morphemes.
        Returns a tuple of morphemes.
        """
        return self.analysis.split(self.morpheme_separator) if self.analysis else (self.word,)


# the files copied are src/modest/datasets/morphynet
class SadilarMorphDataset(ModestDataset[M]):
    """
    A dataset class for the Sadilar Morphological Dataset.
    This class is designed to handle morphological data.

    Requires data to be in directory structure:
    BASEDIR/
        └── {lang_code}.tsv


    where {language} is the language code as per Sadilar's conventions.
    """
    
    def __init__(self, language: Languageish):
        super().__init__(name="SadilarMorph", language=language)
        self._subset = "inflectional"  # This dataset is focused on inflectional morphology
    
    def _get(self) -> Path:
        sadilar_code = SOUTH_AFRICAN_LANGUAGES.get(self._language)
        if sadilar_code is None:
            raise ValueError(f"Language not in Sadilar Morphological Dataset: {self._language}")
        # Construct the path to the dataset based on the language code
        return Path(f"{BASE_DIR}/{sadilar_code.upper()}.tsv")
    
class SadilarMorphDataset_Surface(SadilarMorphDataset[SadilarMorphologySurface]):
    """
    A dataset class for the surface morphological analysis from the Sadilar Morphological Dataset.
    This class reads the surface forms of words and segments them into morphemes.
    """
    
    def _generate(self, path: Path, **kwargs) -> Iterator[SadilarMorphologySurface]:
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
    """
    
    def _generate(self, path: Path, **kwargs) -> Iterator[SadilarMorphologyCanonical]:
        for parts in iterateTsv(path):
            if len(parts) < 3:
                continue
            word = parts[0]
            analysis = parts[2]
            yield SadilarMorphologyCanonical(word=word, analysis=analysis)


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