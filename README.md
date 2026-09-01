# FISAD Analysis Code

This repository contains the analysis code used for the technical validation reported in the FISAD Data Descriptor ("FISAD: A Large-Scale Full-Body Apparel Dataset for Studying Apparent Personality Trait Perception") and the companion reliability/methodology paper. All scripts operate on `FISAD_data.csv`, available at https://doi.org/10.5281/zenodo.22238732.

## Structure

- **`attribute_extraction/`** — vision-based extraction of apparel colour attributes (Tops Colour, Bottom Colour), using Claude Haiku 4.5 to complete fields that were incomplete in the original ChatGPT-4.0-derived annotation. Requires an Anthropic API key (`ANTHROPIC_API_KEY` environment variable).
- **`reliability_analysis/`** — the inter-rater reliability decomposition and Spearman-Brown annotation-density projection reported in both papers' Technical Validation / Results sections.
- **`predictive_validation/`** — cross-validated predictive baselines: `attribute_based_baseline.py` (discretized apparel attributes, 5 regression models) and `clip_baseline.py` (raw CLIP image embeddings, bypassing attribute labelling entirely).
- **`causal_screening/`** — the discovery/estimation split with false-discovery-rate correction (`discovery_estimation_fdr_screening.py`) and the five-seed stability check (`five_seed_stability_check.py`) reported in the reliability/methodology paper's Section 4.3.

## Requirements

See `requirements.txt`. Note that `attribute_extraction/fill_colors.py` requires network access (to fetch images and call the Anthropic API) and an API key; `predictive_validation/clip_baseline.py` requires network access (to fetch images) and downloads a CLIP model on first run.

## Usage

Each script can be run independently from a directory containing `FISAD_data.csv` (or with the path specified via command-line argument / environment variable — see each script's docstring). For example:

```bash
pip install -r requirements.txt
python3 reliability_analysis/reliability_decomposition.py --input FISAD_data.csv
```

## Notes on reproducibility

- `reliability_analysis/reliability_decomposition.py` and `causal_screening/*.py` are deterministic given the same input CSV and depend only on standard scientific Python libraries.
- `attribute_extraction/fill_colors.py` and `predictive_validation/clip_baseline.py` call external AI model APIs (Anthropic Claude, and a locally-downloaded CLIP model respectively); results may vary slightly between runs due to model updates, though the overall pattern reported in the papers (near-zero predictive R2 across models, robust reliability estimates) has been independently reproduced across multiple runs during this project's development.
- The apparel attribute data released in `FISAD_data.csv` reflects a combination of the original ChatGPT-4.0-derived annotation (for garment type, sleeve, hair) and Claude Haiku 4.5-derived annotation (for the previously-incomplete colour fields). See the Data Descriptor's Methods section for full disclosure of this AI-vs-AI provenance.

## Citation

If you use this code, please cite the FISAD Data Descriptor and dataset (https://doi.org/10.5281/zenodo.22238732).

## License

Code in this repository is released under the MIT License (see LICENSE). The dataset itself (`FISAD_data.csv`) is released separately under CC BY 4.0 — see the dataset's own Zenodo record for details.
