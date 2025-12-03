import argparse
import os
import zipfile
from pathlib import Path
import shutil
import sys
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from PIL import Image

"""
Preparing the ArtEmis Dataset and WikiArt images for training
and preprocessing.
"""

class DataPreparator:

    def __init__(self, data_dir: Path = Path('data')):
        self.data_dir = Path(data_dir)
        self.zip_path = self.data_dir / 'wikiart.zip'
        self.extract_dir = self.data_dir
        self.artemis_sample = None

    def unzip_wikiart(self) -> Path:
        """
        Unzip WikiArt from `data/wikiart.zip` into `data/wikiart/`.
        Returns the extraction directory path.
        """
        self.extract_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.data_dir / 'wikiart_logs.txt'
        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            for member in zf.namelist():
                try:
                    zf.extract(member, path=self.extract_dir)
                except Exception as e:
                    # Log the corrupt or problematic entry and continue
                    with open(log_path, 'a', encoding='utf-8') as lf:
                        lf.write(f"Failed to extract: {member} | Error: {e}\n")
                    continue
                print(f'Extracted: {member}')
        return self.extract_dir

    def stratified_artemis(
        self,
        artemis_csv: Path = Path('data/official_data/artemis_dataset_release_v0.csv'),
        out_csv: Path = Path('data/artemis_sample.csv'),
        n_samples: int = 2000,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Load Artemis CSV and perform stratified sampling based on art style and emotion.

        Arguments:
        - artemis_csv: path to full Artemis CSV with columns 'art_style', 'emotion'.
        - out_csv: where to write the reduced CSV.
        - n_samples: number of rows to sample (default 2000).
        - random_state: RNG seed for reproducibility.

        Returns the sampled DataFrame.
        """

        # Convert to Path if needed
        artemis_csv = Path(artemis_csv)
        out_csv = Path(out_csv)

        # Check if CSV exists
        if not artemis_csv.exists():
            raise FileNotFoundError(f"Artemis CSV not found: {artemis_csv}")

        df = pd.read_csv(artemis_csv)

        # Create stratification label combining art_style and emotion
        strat = df['art_style'].astype(str) + '||' + df['emotion'].astype(str)

        # Use StratifiedShuffleSplit with integer and fallback to fraction if it doesnt work
        try:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=n_samples, random_state=random_state)
            for _, idx in splitter.split(df, strat):
                sample = df.iloc[idx].copy()
                break
        except ValueError:
            frac = n_samples / len(df)
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=frac, random_state=random_state)
            for _, idx in splitter.split(df, strat):
                sample = df.iloc[idx].copy()
                break
        
        print(f"Stratified sample created ({len(sample)})")

        # Save sampled DataFrame
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        sample.to_csv(out_csv, index=False)
        self.artemis_sample = sample
        
        return
    
    # Based on the artemis sampling, it samples the WikiArt images
    def copy_images_wikiart(self, dest_dir: Path = None):
        """
        Copy sampled WikiArt images based on the sampled Artemis DataFrame.
        
        Arguments:
        - dest_dir: destination directory to copy sampled images into.
        
        """
        # Load the df
        try:
            df = pd.read_csv(self.artemis_sample)
        except Exception:
            df = pd.read_csv(self.data_dir / 'artemis_sample.csv')
       
        #Check if stratified_artemis has been run
        if df is None or df.empty:
            raise ValueError("No sampled DataFrame found. Run stratified_artemis() first.")

        # Source and destination roots
        source_root = self.data_dir / 'wikiart'
        dest_root = self.data_dir / 'wikiart_sample' if dest_dir is None else dest_dir
        dest_root.mkdir(parents=True, exist_ok=True)

        # Check for 'painting' and 'art_style' columns
        if {'painting', 'art_style'} - set(df.columns):
            raise ValueError("Sampled DataFrame must contain 'painting' and 'art_style' columns.")

        copied, missing = 0, 0
        for _, row in df.iterrows():
            style = str(row['art_style']).strip()
            p = Path(str(row['painting']).strip())
            fname = p.name if p.suffix else f"{p.name}.jpg"
            src_path = source_root / style / fname
            
            if not src_path.exists():
                # Log missing source image path and continue
                copy_log = self.data_dir / 'wikiart_copy_missing.txt'
                with open(copy_log, 'a', encoding='utf-8') as lf:
                    lf.write(f"Missing: style='{style}', file='{fname}'\n")
                missing += 1
                continue

            # Destination path under style subfolder
            dst_dir = dest_root / style
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst_path = dst_dir / Path(src_path).name
            shutil.copy2(src_path, dst_path)
            copied += 1

        print(f"Copied: {copied} images to {dest_root}. Missing: {missing}")

        return self
    
    def resize_images(
        self,
        src_dir: Path = Path('data/wikiart_sample'),
        out_dir: Path = Path('data/wikiart_sample_128'),
        size=(128, 128)
    ):
        out_dir.mkdir(parents=True, exist_ok=True)
        count, errors = 0, 0
        error_log = self.data_dir / 'wikiart_resize_errors.txt'
        for root, _, files in os.walk(src_dir):
            for fname in files:
                src_path = Path(root) / fname
                rel = src_path.relative_to(src_dir)
                dst_path = out_dir / rel.with_suffix('.jpg')
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with Image.open(src_path) as img:
                        img = img.convert('RGB')
                        img = img.resize(size, Image.LANCZOS)
                        img.save(dst_path, format='JPEG', quality=90)
                    count += 1
                except Exception as e:
                    errors += 1
                    with open(error_log, 'a', encoding='utf-8') as lf:
                        lf.write(f"Failed to resize: {src_path} | Error: {e}\n")

        return count, errors
    
if __name__ == '__main__':
    
    #Unzipping the WikiArt Dataset
    preparator = DataPreparator()
    out_dir = preparator.unzip_wikiart()
    print(f"WikiArt extracted to: {out_dir}")

    #Creating a stratified sample of the ArtEmis Dataset
    preparator.stratified_artemis()

    #Creating sample WikiArt Dataset based on the ArtEmis Sample
    preparator.copy_images_wikiart()

    #Resizing the sampled WikiArt images to 128x128
    resized, errors = preparator.resize_images()
    print(f"Resized: {resized}, Errors: {errors}")