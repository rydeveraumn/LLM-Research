import urllib.request

# File name for the example text in the book
FILE_PATH = "/home/jusun/dever120/datasets/llm-research/the-verdict.txt"


def download_test_data() -> None:
    """
    Function to download the example text in building a
    LLM from scratch.
    """
    # Downloading the data for execution
    url = (
        "https://raw.githubusercontent.com/rasbt/"
        "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
        "the-verdict.txt"
    )

    urllib.request.urlretrieve(url, FILE_PATH)


def load_data() -> str:
    """
    Function that loads in data for building an LLM from
    scratch.
    """
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        raw_text = f.read()

    print(f"Total number of characters: {len(raw_text)}")
    return raw_text
