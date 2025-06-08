"""
Authors:
    - Cael Marquard
    - Simbarashe Mawere
Date:
    - 2024/07/07

Description:
    - This script downloads, reads the SADII files and formats them into a TSV format.
    - The TSV format is as follows:
        - word{tab}canonical_segmentation{tab}morphological_parse
    - The script assumes that the directory structure is as follows:
        - data/
            - TEST/
                - SADII.{lang}.*
            - TRAIN/
                - SADII.{lang}.*
        - script/
            - data_prep.py
    - The script outputs the files in the following format:
        - word{tab}canonical_segmentation{tab}morphological_parse
    - The script skips the English files.
"""
from collections import defaultdict

import re
import io
import os
import shutil
import requests
import tqdm
import zipfile
from pathlib import Path
from ..paths import DATA_DIR

# for extracting the data
DIRS = ["TEST", "TRAIN"]

# for downloading the data
URL = "https://repo.sadilar.org/bitstream/handle/20.500.12185/546/SADII_CTexT_2022-03-14.zip?sequence=4&isAllowed=y"
BASE_DIR = DATA_DIR / "sadilar_raw"

LANGUAGES = ["EN", "NR", "SS", "XH", "ZU"]

class SadilarDataPreparation:
    @staticmethod
    def download_and_extract_sadilar(source_url: str = URL, out_dir: Path | str = BASE_DIR, clean_dir: bool = True, overwrite: bool = False, verbose: bool = False, combine: bool = False):
        """
        Downloads a zip file and extracts it into a specified directory

        Arguments:
            file_url (str): The URL of the zip file to download
            out_dir (str): The directory to extract the contents of the zip file into

        """
        out_dir = Path(out_dir)
        if os.path.exists(out_dir):
            if overwrite:
                print(f"Directory {out_dir} already exists. Overwriting...")
                shutil.rmtree(out_dir)
            else:
                print(f"Directory {out_dir} already exists. Use --overwrite to overwrite it.")
                return
        else:
            print(f"Creating directory {out_dir}...")
            os.makedirs(out_dir, exist_ok=True)

        r = requests.get(source_url)
        content = io.BytesIO(r.content)

        with zipfile.ZipFile(content, "r") as zip_ref:
            zip_ref.extractall(out_dir)
            print(f"Extracted files to {out_dir}")
        if clean_dir:
            SadilarDataPreparation.clean_out_dir(out_dir, verbose=verbose)
            if combine:
                SadilarDataPreparation.combine_subsets(out_dir, verbose=verbose)
        else:
            print("Skipping removal of extra files. Use --remove_extras to remove them.")

    @staticmethod
    def clean_out_dir(out_dir: Path | str, verbose: bool = True):
        """
        Removes unnecessary files and directories from the extracted zip file. Made for the SADII_CTexT dataset

        Arguments:
            out_dir (str): The directory containing the extracted files
        """
        # removing unnecessary file
        # Check the operating system
        out_dir = Path(out_dir)
        shutil.rmtree(out_dir / "Protocols", ignore_errors=True)
        if verbose:
            print(f"Removed 'Protocols' directory from {out_dir.relative_to(DATA_DIR.parent)}")

        children = os.listdir(out_dir)

        for child in children:
            if "README" in child:
                if verbose:
                    print(f"Removing {child} from {out_dir.relative_to(DATA_DIR.parent)}")
                os.remove(out_dir / child)
                continue
            for file_ in os.listdir(out_dir / child):
                # renaming the files to keep only the language code
                lang, ext = file_.split(".")[1], file_.split(".")[-1]
                # keeping all files
                os.rename(out_dir / child / file_, out_dir / child / f"{lang}.{ext}")
                if verbose:
                    print(f"Renamed {file_} to {lang}.{ext} in {(out_dir / child).relative_to(DATA_DIR.parent)}")
        
    @staticmethod
    def combine_subsets(out_dir: Path | str, verbose: bool = True, languages: list[str] = LANGUAGES):
        """
        Combines the subsets of the SADII_CTexT dataset into a single file

        Arguments:
            out_dir (str): The directory containing the extracted files
        """
        out_dir = Path(out_dir)
        subsets = os.listdir(out_dir)

        if not os.path.exists(out_dir / "COMBINED"):
            os.makedirs(out_dir / "COMBINED")

        for lang in languages:
            lang_data = ""
            for subset in subsets:
                with open(out_dir / subset / f"{lang}.txt", "r", encoding="utf-8") as f:
                    lang_data += f.read()
            
            with open(out_dir / "COMBINED" / f"{lang}.txt", "w", encoding="utf-8") as f:
                f.write(lang_data)
            if verbose:
                print(f"Combined files for {lang}.")

    def format_line(line: str) -> str:
        """
        Removes the POS columns from the line then adds the separated canonical segmentation of the word and morphological tags.

        Arguments:
            line (str): The line to format:
                word{tab}analysis/parsed{tab}lemma{tab}pos

        Returns:
            str: The formatted line:
                word{tab}lemma{tab}canonical_segmentation{tab}morpheme_tags

        Example:
            >>> format_line("abanengi\taba[AdjPref2a]-nengi[AdjStem]\tnengi\tADJ02a")
            "abanengi\tnengi\taba_nengi\tAdjPref2a_AdjStem\tADJ02a"
        """
        if "<LINE#" in line:
            return line

        line = line.rstrip()
        raw, parsed, lemma, pos = line.split("\t")
        segmentation, tags = SadilarDataPreparation.split_tags(parsed)
        line = [raw, lemma, "_".join(segmentation), "_".join(tags), pos]
        return "\t".join(line) + "\n"

    def split_tags(text: str) -> tuple[list[str], list[str]]:
        """
        Split a word into its canonical segmentation and morpheme tags.

        Arguments:
            text (str): The word to split

        Returns:
            (str, list[str]): The canonical segmentation and morpheme tags

        Example:
            split_tags("a[DET]b[N]") -> (["a", "b"], ["DET", "N"])
        """
        split = [
            morpheme
            for morpheme in re.split(r"\[[a-zA-Z-_0-9]*?]-?", text)
            if morpheme != ""
        ]
        return (split, re.findall(r"\[([a-zA-Z-_0-9]*?)]", text))

    def reformat_file(file_path: Path | str, out_path: Path | str):
        """
        Reformats a file to the TSV format.
        
        Assumptions:
            - input file is in the format:
            word{tab}analysis/parsed{tab}lemma{tab}pos
        Arguments:
            file_path (str): The path to the file to reformat
            out_path (str): The path to the output file
        """
        file_path = Path(file_path)
        out_path = Path(out_path)
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        lines = [SadilarDataPreparation.format_line(line) for line in lines if line.strip() != ""]

        with open(out_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"Reformatted\n\t{file_path} ->\n\t{out_path}")
    
    @staticmethod
    def group_lines_from_file(file_path: Path | str) -> dict[str, list[tuple[str, str]]]:
        """
        Processes a file and groups the text by line number.

        Assumptions:
            - The file contains lines that start with "<LINE" followed by a line number.
            - The input file has been format so that the lines are in the manner of:
                "<LINE#1>
                word1\tlemma1\tcanonical_segmentation1\tmorpheme1\tpos1\n
                word2\tlemma2\tcanonical_segmentation2\tmorpheme2\tpos2\n
                ...
                <LINE#2>
                word3\tlemma3\tcanonical_segmentation3\tmorpheme3\tpos3\n
                word4\tlemma4\tcanonical_segmentation4\tmorpheme4\tpos4\n
                ...
                "
        
        This function expects the file to contain lines that start with "<LINE" followed by a line number.
        It groups the text lines that follow each line number into a dictionary where the keys are the line numbers and the values are lists of text lines.
        The function reads the file line by line, identifies the line numbers, and collects the corresponding text lines until the next line number is encountered.

        Arguments:
            file_path (str): The path to the file to process
        Returns:
            dict[str, list[tuple[str, str]]]: A dictionary where the keys are line numbers and the values are lists of tuples containing words and their corresponding morphemes (index 0 and 1 of the tuple respectively).
        """
        # Dictionary to group text by line number
        line_groups = defaultdict(list)
        file_path = Path(file_path)
        
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        curr_line = None
        for line in lines:
            line = line.rstrip()
            if "<LINE" in line:
                # Extract the line number
                curr_line = line
                line_groups[curr_line] = ([], [])  # Initialize a new group for this line
            elif curr_line is not None:
                # Process the line and add it to the current line group
                word = line.split("\t")[0]
                morpheme = line.split("\t")[2]
                line_groups[curr_line][0].append(word)
                line_groups[curr_line][1].append(morpheme)
        
        return line_groups

    def sentencify(sentence_fragments: list[str], ignore_punctuation: bool = False) -> str:
        """
        Joins a list of words and punctuation into a single string based on punctuation rules.
        
        Args:
            sentence_fragments (list): A list of strings representing words and punctuation marks.
            ignore_punctuation (bool): If True, punctuation will not be processed and will be included as is.
        
        Returns:
            str: A single string representing a sentence, with appropriate spacing around punctuation.

        Example:
            >>> sentencify(["Hello", ",", "World", "!", "This", "is", "a", "test", "sentence", "."])
            "Hello, World! This is a test sentence."
        """
        line = " ".join(sentence_fragments)
        line = re.sub(r"\s+", " ", line)  # Replace multiple spaces with a single space
        if ignore_punctuation:
            return line.strip()
        
        space_before_punct = ["!", "?", ".", ",", ";", ":", ")", "]", "}", "’", "”"]
        space_after_punct = ["(", "[", "{", "‘", "“"]
        space_before_after_punct = ["-", "_", "=", "+", "*", "/"]
        for punc in space_before_punct:
            line = line.replace(f" {punc}", punc)
        for punc in space_after_punct:
            line = line.replace(f"{punc} ", punc)
        for punc in space_before_after_punct:
            line = line.replace(f" {punc} ", punc)
        return line.strip()

    @staticmethod
    def sentencify_file(file_path: Path | str, progress: bool = True):
        """
        Reads a file, processes it to join sentence fragments into complete sentences, and writes the result back a
        
        Args:
            file_path (Path | str): The path to the file to process.
        
        Assumptions:
            - The file contains lines that start with "<LINE" followed by a line number.
            - The input file has been formatted so that the lines are in the manner of:
                "<LINE#1>
                word1\tlemma1\tcanonical_segmentation1\tmorpheme1\tpos1\n
                word2\tlemma2\tcanonical_segmentation2\tmorpheme2\tpos2\n
                ...
                "
        Returns:
            None: The function creates two new files in the same directory as the input file:
            - One file containing sentences formed from the words.
            - Another file containing sentences formed from the morphemes.
        """

        file_path = Path(file_path)
        suffix = file_path.suffix
        stem = file_path.stem

        line_groups = SadilarDataPreparation.group_lines_from_file(file_path)
        # Create sentences from the words in each line group
        word_sentences = []
        morpheme_sentences = []
        
        if progress:
            iter_ = tqdm.tqdm(line_groups.items(), desc="Processing lines", unit="line")
        else:
            iter_ = line_groups.items()
        for line_id, line in iter_:
            if len(line_groups) > 0:
                # Process words into proper sentences with punctuation handling
                words = line[0]
                words = [decapitalise(word, separator="-") for word in words]  # Decapitalise words
                # HACK: unhyphenate words that are hyphenated
                words = [word.replace("-", "") for word in words]
                word_sentence = SadilarDataPreparation.sentencify(words)
                word_sentences.append(word_sentence + "\n")
                
                # Process morphemes (treating them as tokens without punctuation rules)
                morphemes = line[1]
                morphemes = [decapitalise(morpheme, separator="_") for morpheme in morphemes]
                morpheme_sentence = SadilarDataPreparation.sentencify(morphemes, ignore_punctuation=True)
                morpheme_sentences.append(decapitalise(morpheme_sentence, separator="_") + "\n")

        # Write the sentences to new files
        sentence_file_path = file_path.parent / f"{stem}_sentences{suffix}"
        with open(sentence_file_path, "w", encoding="utf-8") as f:
            f.writelines(word_sentences)

        morpheme_file_path = file_path.parent / f"{stem}_morphemes{suffix}"
        with open(morpheme_file_path, "w", encoding="utf-8") as f:
            f.writelines(morpheme_sentences)
        
        print(f"Processed sentences and morphemes from {file_path} into:\n\t{sentence_file_path}\n\t{morpheme_file_path}")


# auxiliary function to decapitalise words
def decapitalise(word: str, separator: str = "-") -> str:
    """
    Decapitalises a word, handling hyphenated words by decapitalising each part
    except for those that are fully uppercase.
    Args:
        word (str): The word to decapitalise.
        separator (str): The separator used in hyphenated words. Default is '-'.
    Returns:
        str: The decapitalised word.
    Example:
        >>> decapitalise("Burger-King")
        'burger-king'
        >>> decapitalise("the-WHO-Organization")
        'the-who-organization'
        >>> decapitalise("A-B-C")
        'a-b-c'
        >>> decapitalise("A")
        'a'
    """

    # short-circuit for non-hyphenated words
    if separator not in word:
        if word.isupper() and len(word) > 1:
            return word
        else:
            return word.lower()

    parts = word.split(separator)
    for i, part in enumerate(parts):
        if part.isupper():
            continue
        else:
            parts[i] = part.lower()
    return separator.join(parts)