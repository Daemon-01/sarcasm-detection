"""Unified feature extraction dispatcher for SarcasTone.

Dispatches text and sentiment extraction for the text branch.
Audio and multimodal modalities remain unimplemented stubs per AGENTS.md Section 8.
"""

import argparse
import sys
from pathlib import Path

# Add src to sys.path if not present
repo_root = Path(__file__).resolve().parents[1]
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from scripts.extract_text import main as extract_text_main


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract SarcasTone Modality Features")
    parser.add_argument(
        "--modality",
        choices=["text", "audio", "multimodal"],
        default="text",
        help="Feature modality to extract (default: text)",
    )
    args, unknown = parser.parse_known_args()

    if args.modality == "text":
        extract_text_main()
    else:
        raise NotImplementedError(
            f"Modality '{args.modality}' is not implemented in the current text-branch-only scope. "
            "See docs/sarcastone_text_branch_okf/interfaces/text-branch-output.md and AGENTS.md Section 8."
        )


if __name__ == "__main__":
    main()

