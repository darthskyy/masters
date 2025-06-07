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
            line (str): The line to format. word{tab}analysis/parsed{tab}{lemma}{tab}pos

        Returns:
            str: The formatted line. {word}{tab}{lemma}{tab}{canonical_segmentation}{tab}{morpheme_tags}

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
    def group_lines_from_file(file_path: Path | str) -> dict[str, list[str]]:
        """
        Processes a file and groups the text by line number.

        This function expects the file to contain lines that start with "<LINE" followed by a line number.
        It groups the text lines that follow each line number into a dictionary where the keys are the line numbers and the values are lists of text lines.
        The function reads the file line by line, identifies the line numbers, and collects the corresponding text lines until the next line number is encountered.

        Arguments:
            file_path (str): The path to the file to process
        Returns:
            dict[str, list[str]]: A dictionary where the keys are line numbers and the values are lists of text lines
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
            elif curr_line is not None:
                # Process the line and add it to the current line group
                out_ = line.split("\t")[0]
                line_groups[curr_line].append(out_)
        
        return line_groups

    def sentencify(sentence_fragments: list[str]) -> str:
        """
        Joins the lines in a group into a single string based on punctuation rules.
        
        Args:
            line_group (list): The group of lines to join
        
        Returns:
            str: The joined string

        Example:
            >>> sentencify(["Hello", ",", "World", "!", "This", "is", "a", "test", "sentence", "."])
            "Hello, World! This is a test sentence."
        """
        space_before_punct = ["!", "?", ".", ",", ";", ":", ")", "]", "}", "’", "”"]
        space_after_punct = ["(", "[", "{", "‘", "“"]
        space_before_after_punct = ["-", "_", "=", "+", "*", "/"]
        line = " ".join(sentence_fragments)
        line = re.sub(r"\s+", " ", line)  # Replace multiple spaces with a single space
        
        for punc in space_before_punct:
            line = line.replace(f" {punc}", punc)
        for punc in space_after_punct:
            line = line.replace(f"{punc} ", punc)
        for punc in space_before_after_punct:
            line = line.replace(f" {punc} ", punc)
        return line