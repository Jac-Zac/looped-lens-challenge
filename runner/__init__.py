"""Runs a submission: one fresh process per dataset, environment variables in,
files out.

This package is everything that executes inside the untrusted runner task, and
``submit.py --local`` calls the same ``run_submission`` on a participant's
machine. It imports nothing else from this repository.

The contract with a method
--------------------------
The submission directory holds exactly one entry file, ``main.py`` or
``main.ipynb``. For each dataset it is started in a new process, with the
submission directory as the working directory and these variables set:

    DATASET_NAME   the dataset's name
    DATASET_PATH   a directory holding the dataset's input files
    OUTPUT_DIR     an empty directory; everything written here is the result
    NDIF_API_KEY   the submitter's NDIF key        (nnsight reads it)
    NDIF_HOST      the competition's NDIF server   (nnsight reads it)
    HF_TOKEN       the submitter's Hugging Face token, when they sent one

plus whatever the competition declares for that dataset (``MODEL_ID``, say).
A run succeeds when the process exits 0 and ``OUTPUT_DIR`` is not empty. What
the files in it mean is for the evaluator to decide.
"""

from runner.run import ENTRY_FILES, Dataset, DatasetRun, RunReport, find_entry, load_manifest, run_submission

__all__ = ["ENTRY_FILES", "Dataset", "DatasetRun", "RunReport", "find_entry", "load_manifest", "run_submission"]
