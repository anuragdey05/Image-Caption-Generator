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

## Notes
- The script expects columns `style` and `emotion`. If your CSV uses `art_style` or `label`, it will auto-normalize.
- It also requires an image path column (e.g., `image_path`). If your CSV differs, adjust `load_artemis_csv` in `scripts/process_data.py`.
○ Setup and execution instructions
○ Dataset preprocessing steps