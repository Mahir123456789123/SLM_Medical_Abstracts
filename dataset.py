import pandas as pd
import ast
import re

# =========================================================
# LOAD DATASET
# =========================================================

INPUT_FILE = "./pubmed_dataset/pubmed_abstracts.csv"
OUTPUT_FILE = "pubmed_corpus.txt"

df = pd.read_csv(INPUT_FILE)

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def remove_urls(text):
    """
    Remove URLs from text
    """
    return re.sub(r"http\S+|www\S+|https\S+", "", text)


def clean_text(text):
    """
    Basic text cleanup
    """
    text = remove_urls(text)

    # remove extra whitespace/newlines/tabs
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_abstract(cell):
    """
    Extract abstract text from dataset cell

    Expected formats:
    (['abstract text'], 'title')
    or similar tuple/list structures
    """

    if pd.isna(cell):
        return None

    try:
        # convert string representation -> python object
        parsed = ast.literal_eval(cell)

        # Example:
        # (
        #   ['abstract text here'],
        #   'paper title'
        # )

        # get first element
        abstract_part = parsed[0]

        # if list of paragraphs/sentences
        if isinstance(abstract_part, list):
            abstract = " ".join(
                str(x) for x in abstract_part if x
            )

        else:
            abstract = str(abstract_part)

        abstract = clean_text(abstract)

        # skip empty or tiny abstracts
        if len(abstract.split()) < 20:
            return None

        return abstract

    except Exception:
        return None


# =========================================================
# FIND TEXT COLUMNS
# =========================================================

text_columns = []

for col in df.columns:

    col_lower = col.lower()

    # skip index columns
    if "unnamed" in col_lower:
        continue

    # skip link/url columns
    if "link" in col_lower or "url" in col_lower:
        continue

    text_columns.append(col)

print("Using columns:")
print(text_columns)

# =========================================================
# EXTRACT ALL ABSTRACTS
# =========================================================

all_abstracts = []

for idx, row in df.iterrows():

    for col in text_columns:

        abstract = extract_abstract(row[col])

        if abstract:
            all_abstracts.append(abstract)

    if idx % 100 == 0:
        print(f"Processed {idx} rows...")

# =========================================================
# REMOVE DUPLICATES
# =========================================================

all_abstracts = list(dict.fromkeys(all_abstracts))

print(f"\nTotal abstracts collected: {len(all_abstracts)}")

# =========================================================
# SAVE TO TXT
# =========================================================

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    for abstract in all_abstracts:

        f.write(abstract)
        f.write("\n\n")

print(f"\nDONE")
print(f"Saved corpus to: {OUTPUT_FILE}")