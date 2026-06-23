from pathlib import Path

import pandas as pd

try:
    import kagglehub
except ImportError:
    kagglehub = None




DATASET_NAME = "venkataanuhyatummala/flores200data"
BASE_DIR = Path(__file__).resolve().parents[1]
SOURCE_LANG = "eng_Latn"
TARGET_LANG = "vec_Latn"
SOURCE_FILENAME = "eng_Latn.dev"
TARGET_FILENAME = "vec_Latn.dev"
DEFAULT_DOMAIN = "unknown"
DIALECT_LABEL = "normalized_venetian"
SOURCE_TYPE = "manual_translation"
OUTPUT_JSONL = BASE_DIR / f"{SOURCE_LANG}_{TARGET_LANG}_dataset.jsonl"


def find_source_file(source_dir: Path, filename: str) -> Path:
    direct_path = source_dir / filename
    if direct_path.exists():
        return direct_path

    matches = list(source_dir.rglob(filename))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Could not find {filename} inside {source_dir}")


def resolve_source_dir() -> Path:
    local_files_exist = all(
        (BASE_DIR / filename).exists() for filename in (SOURCE_FILENAME, TARGET_FILENAME)
    )
    if local_files_exist:
        return BASE_DIR

    if kagglehub is None:
        raise ImportError(
            "kagglehub is not installed and the local dataset files were not found. "
            "Install it with: pip install kagglehub[pandas-datasets]"
        )

    return Path(kagglehub.dataset_download(DATASET_NAME))


def read_lines(source_dir: Path, filename: str) -> list[str]:
    file_path = find_source_file(source_dir, filename)
    return file_path.read_text(encoding="utf-8").splitlines()


def build_translation_dataset(source_dir: Path) -> pd.DataFrame:
    source_lines = read_lines(source_dir, SOURCE_FILENAME)
    target_lines = read_lines(source_dir, TARGET_FILENAME)

    if len(source_lines) != len(target_lines):
        raise ValueError(
            f"Line count mismatch: {SOURCE_FILENAME} has {len(source_lines)} rows, "
            f"{TARGET_FILENAME} has {len(target_lines)} rows."
        )

    records = []
    for index, (source_text, target_text) in enumerate(zip(source_lines, target_lines), start=1):
        records.append(
            {
                "id": f"{index:04d}",
                "source_lang": SOURCE_LANG,
                "target_lang": TARGET_LANG,
                "source_text": source_text,
                "target_text": target_text,
                "domain": DEFAULT_DOMAIN,
                "dialect_label": DIALECT_LABEL,
                "source_type": SOURCE_TYPE,
                "translation_prompt": f"translate English to Venetian: {source_text}",
            }
        )

    return pd.DataFrame(records)


def save_jsonl(df: pd.DataFrame, output_path: Path) -> Path:
    df.to_json(output_path, orient="records", lines=True, force_ascii=False)
    return output_path


def main() -> None:
    source_dir = resolve_source_dir()
    df = build_translation_dataset(source_dir)
    output_path = save_jsonl(df, OUTPUT_JSONL)

    print(f"Using source directory: {source_dir}")
    print(f"Saved {len(df)} rows -> {output_path}")
    print(df.head())


if __name__ == "__main__":
    main()
