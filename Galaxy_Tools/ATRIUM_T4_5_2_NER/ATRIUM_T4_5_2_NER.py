import json
from pathlib import Path
import argparse
import logging
import os
import urllib.request
import zipfile
import spacy
import shutil

logger = logging.getLogger(__name__)

def process(input: Path, model: Path, output: Path):
    logger.info("Reading input file '%s'", input)

    # read the text out of the input JSON file
    with open(input, "r") as f:    
        file_content = json.load(f)

    text = file_content["text"]

    ner = spacy.load(model)

    annotations = ner(text).ents

    spans = []

    for annotation in annotations:
        spans.append({
            "label": "PLACE",
            "start": annotation.start_char,
            "end": annotation.end_char,
            "span_text": text[annotation.start_char:annotation.end_char]
        })

    if "spans" in file_content:
        file_content["spans"].extend(spans)
    else:
        file_content["spans"] = spans

    with open(output, 'w') as f:
        json.dump(file_content, f)

def download_model(url: str, dest_parent: Path):
    """Download and extract a spaCy model wheel, returning the loadable model dir.

    This is used only for self-contained tests (via --download); in production
    the model directory comes from the data manager via --model-dir.
    """

    os.makedirs(dest_parent, exist_ok=True)
    wheel_path = dest_parent / "model.whl"
    extract_dir = dest_parent / "_model"
    urllib.request.urlretrieve(url, wheel_path)
    with zipfile.ZipFile(wheel_path) as zf:
        zf.extractall(extract_dir)

    # find all the meta.json files in the wheel and check
    # that there is only 1 --
    meta_file = list(extract_dir.glob("*/meta.json"))
    if len(meta_file) == 0:
        raise RuntimeError("Couldn't find meta.json. Is this really a spaCy model?")
    elif len(meta_file) != 1:
        raise RuntimeError("Found multiple meta.json mfiles. Is this really a spaCy model?")

    # read the metadata out of the file
    with open(meta_file[0]) as f:
        metadata = json.load(f)

    # get the lang, name, and version from the metadat and
    # build the appropriate variables we want later
    base_name = metadata["name"]
    lang = metadata["lang"]
    model_name = f"{lang}_{base_name}"
    version = metadata["version"]

    print(f"Extracting {model_name} v{version}")

    # this is the data directory inside the extracted wheel, if it
    # doesn't exist then we have a big problem
    data_dir = extract_dir / model_name / f"{model_name}-{version}"
    if not data_dir.is_dir():
        raise RuntimeError(f"Model data directory not found in wheel: {data_dir}")

    # move the extracted data directory to where Galaxy can find it
    dest_dir = dest_parent / f"{model_name}_{version}"
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    shutil.move(str(data_dir), str(dest_dir))

    # Clean up the wheel and leftover extraction directory.
    wheel_path.unlink()
    shutil.rmtree(extract_dir, ignore_errors=True)

    return dest_dir


if __name__ == "__main__":
    
    # initiate the input arguments parser
    parser = argparse.ArgumentParser(
        prog=__file__, description="ATRIUM T4.5.2 NER Component")

    # add long and short argument descriptions for input file path (directory containing files to be processed)
    parser.add_argument(
        "--input", "-i", 
        required=True,
        help="Input JSON File to process")

    parser.add_argument(
        "--model", "-m",
        required=True,
        help="The spaCy model to use (should be en_deberta_v3_base_ner_historical_place)"
    )

    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the default model used by the tool -- normally used for testing only"
    )

    # add long and short argument descriptions for output file path (directory to write processed files to)
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output JSON file")
    
    # parse and clean command line arguments
    args = parser.parse_args()
    input_path: Path = Path(args.input.strip())
    output_path: Path = Path(args.output.strip())
    
    if args.download:
        model_path: Path = Path(
            download_model(
                "https://huggingface.co/NikosKprl/en_deberta_v3_base_ner_historical_place/resolve/main/en_deberta_v3_base_ner_historical_place-1.0-py3-none-any.whl?download=true",
                Path(os.path.join(os.getcwd(), "spacy_models"))
            )
        )
    else:
        model_path: Path = Path(args.model.strip())

    process(input_path, model_path, output_path)

    if args.download:
         shutil.rmtree(model_path, ignore_errors=True)
    
   
