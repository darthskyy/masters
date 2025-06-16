from langcodes import Language
from pathlib import Path
from tokenizers import Tokenizer, pre_tokenizers
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from typing import Union
import warnings

from bpe_knockout.auxiliary.tokenizer_interface import HuggingFaceTokeniserPath
from bpe_knockout.project.config import ProjectConfig

from tktkt.models.bpe.knockout import CONFIGS
from tktkt.models.bpe.knockout import BPEKnockout
from tktkt.preparation.boundaries import BoundaryMarker, BoundaryMarkerLocation

from .sadilar_morph import SadilarMorphDataset_Surface

LINEAR_WEIGHTER  = lambda f: f # redefining the reweighter function here for clarity, as it is used in the project config.

# FIXME: For now, I am still using the HuggingFaceTokeniserPath class but ideally I should
# create a new class that uses Tokenizer from the `tokenizers` library directly.
def setupNdebele() -> ProjectConfig:
    return ProjectConfig(
        language_name="South Ndebele",
        lemma_counts=None,
        morphologies=SadilarMorphDataset_Surface("South Ndebele"),
        base_tokeniser=HuggingFaceTokeniserPath(Path("tokenisers/nr_tokeniser.json")),
        reweighter=LINEAR_WEIGHTER,
    )

def setupSwati() -> ProjectConfig:
    return ProjectConfig(
        language_name="Swati",
        lemma_counts=None,
        morphologies=SadilarMorphDataset_Surface("Swati"),
        base_tokeniser=HuggingFaceTokeniserPath(Path("tokenisers/ss_tokeniser.json")),
        reweighter=LINEAR_WEIGHTER,
    )

def setupXhosa() -> ProjectConfig:
    return ProjectConfig(
        language_name="Xhosa",
        lemma_counts=None,
        morphologies=SadilarMorphDataset_Surface("Xhosa"),
        base_tokeniser=HuggingFaceTokeniserPath(Path("tokenisers/xh_tokeniser.json")),
        reweighter=LINEAR_WEIGHTER,
    )

def setupZulu() -> ProjectConfig:
    return ProjectConfig(
        language_name="Zulu",
        lemma_counts=None,
        morphologies=SadilarMorphDataset_Surface("Zulu"),
        base_tokeniser=HuggingFaceTokeniserPath(Path("tokenisers/zu_tokeniser.json")),
        reweighter=LINEAR_WEIGHTER,
    )

CUSTOM_CONFIGS = {
    "nr": setupNdebele(),
    "ss": setupSwati(),
    "xh": setupXhosa(),
    "zu": setupZulu(),
}

# Registering the custom configurations with the main CONFIGS dictionary.
for lang, config in CUSTOM_CONFIGS.items():
    CONFIGS[lang] = config
