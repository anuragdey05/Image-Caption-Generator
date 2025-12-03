# Image-Caption-Generator

This repo prepares subsets of the ArtEmis annotations and the WikiArt images for caption generation experiments.

## Data Layout
- `data/official_data/`
	- Place the ArtEmis CSVs here:
		- `artemis_dataset_release_v0.csv`
		- `ola_dataset_release_v0.csv` (optional)
	- Example path used by code: `data/official_data/artemis_dataset_release_v0.csv`.

- `data/wikiart.zip`
	- Upload the WikiArt archive as `data/wikiart.zip`.
	- When unzipped, it should produce `data/wikiart/<art_style>/<image>.jpg` structure, e.g.
		- `data/wikiart/Realism/rembrandt_woman-standing-with-raised-hands.jpg`

- Generated outputs
	- `data/artemis_sample.csv`: stratified sample of ArtEmis rows.
	- `data/wikiart_sample/<art_style>/*.jpg`: copied images for the sample.
	- `data/wikiart_sample_128/<art_style>/*.jpg`: resized images (128×128).
	- Logs: `data/wikiart_logs.txt`, `data/wikiart_copy_missing.txt`, `data/wikiart_resize_errors.txt`.

## Setup
Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Usage
Run the preparation script end-to-end (sample ArtEmis and copy WikiArt images):

```bash
python src/pre_processing.py
```

That script will:
- Optionally unzip `data/wikiart.zip` into `data/wikiart/` (you can uncomment those lines in `src/pre_processing.py`).
- Create a stratified sample from `data/official_data/artemis_dataset_release_v0.csv` into `data/artemis_sample.csv`.
- Copy corresponding images from `data/wikiart` into `data/wikiart_sample/<art_style>/`.
- Resize copied images to 128×128 into `data/wikiart_sample_128/<art_style>/`.

## Notes
- The ArtEmis dataset may contain multiple annotations per painting. Copy counts reflect operations; unique files on disk can be fewer due to filename collisions. The resize step processes actual files present in `data/wikiart_sample`.
- Missing images are logged to `data/wikiart_copy_missing.txt` with the style and filename base.
- Ensure style folder names in `data/wikiart` match the `art_style` values in the CSV.

## Troubleshooting
- Bad zip entries: corrupt files during unzip are logged in `data/wikiart_logs.txt`. Re-download the archive or skip corrupt entries.
- If images use extensions other than `.jpg`, adjust the copy logic or normalize file names as needed.
# Image Caption Generator - Data Processing

- Purpose: Process WikiArt + Artemis datasets to create a 5k stratified subset and 128x128 resized images.

## Prerequisites
- Activate the virtualenv: `source venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt` (or handled automatically by the workspace tools)

## Dataset Locations
- Artemis CSV: `data/official_data/artemis_dataset_release_v0.csv`
- WikiArt Zip: `data/wikiart.zip`

## Steps (Automated)
- Unzip WikiArt
- Stratified sample 5,000 rows by style + emotion
- Copy corresponding images to a new folder
- Resize images to 128x128

## Run
```bash
source venv/bin/activate

/Users/anurag/Desktop/IML_A3/venv/bin/python scripts/process_data.py \
	--zip data/wikiart.zip \
	--extract-dir data/wikiart \
	--artemis-csv data/official_data/artemis_dataset_release_v0.csv \
	--samples 5000 \
	--out-csv data/official_data/artemis_5k.csv \
	--copy-dir data/wikiart_5k \
	--resize-dir data/wikiart_5k_128
```

## Outputs
- Sampled CSV: `data/official_data/artemis_5k.csv` (same columns as source, image paths updated relative to `data/wikiart_5k`)
- Copied images: `data/wikiart_5k/`
- Resized images: `data/wikiart_5k_128/`

