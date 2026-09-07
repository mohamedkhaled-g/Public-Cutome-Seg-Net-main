import os
import sys
import time
import rarfile
from tqdm import tqdm
import webbrowser
from pathlib import Path

# For Mega.nz downloads
try:
    from mega import Mega

    MEGA_AVAILABLE = True
except ImportError:
    MEGA_AVAILABLE = False


class DatasetManager:
    def __init__(self):
        self.mega_url = "https://mega.nz/file/ak92xKSC#ZAKefRsPUPfq1L-Mch4L9fftjUZ9wVo9bUfuYq0Wltk"
        self.local_filename = "Dataset_600.rar"
        self.dataset_dir = os.path.abspath('./dataset')
        self.file_path = os.path.abspath(self.local_filename)
        self.min_expected_size = 100 * 1024 * 1024  # 100MB minimum

        # Alternative approach: check for dataset in these locations too
        self.alt_dataset_locations = [
            "./dataset/Dataset_600",
            "K:/Dataset_600",
            "./Dataset_600"
        ]

    def prepare_dataset(self):
        """Main method to ensure dataset is ready"""
        os.makedirs(self.dataset_dir, exist_ok=True)

        # Check if already extracted
        if os.path.exists(os.path.join(self.dataset_dir, 'train_data')):
            print("✓ Dataset already extracted")
            return self.dataset_dir

        # Try to find dataset in alternative locations
        for location in self.alt_dataset_locations:
            if os.path.exists(os.path.join(location, 'train_data')):
                print(f"✓ Found dataset at {location}")
                return location

        # Check if RAR file exists locally and is valid
        if self._check_file_exists():
            print("✓ Found existing RAR file")
            if self._extract_dataset():
                return self.dataset_dir

        # Try to download from Mega.nz
        if self._download_from_mega():
            if self._extract_dataset():
                return self.dataset_dir

        # If all fails, show manual instructions
        self._show_manual_instructions()
        return None

    def _check_file_exists(self):
        """Check if file exists and meets minimum size"""
        if not os.path.exists(self.file_path):
            return False

        file_size = os.path.getsize(self.file_path)
        if file_size < self.min_expected_size:
            print(f"⚠ File too small ({file_size / 1024 / 1024:.2f} MB), not valid")
            return False

        return True

    def _download_from_mega(self):
        """Download the dataset from Mega.nz using the mega.py library"""
        if not MEGA_AVAILABLE:
            print("⚠ Mega.py library not available. Install with: pip install mega.py")
            return False

        try:
            print("⬇ Downloading from Mega.nz (this may take a while)...")

            # Extract file ID and key from URL
            file_id = self.mega_url.split('/file/')[1].split('#')[0]
            file_key = self.mega_url.split('#')[1]

            # Login to Mega (anonymous)
            mega = Mega()
            m = mega.login()

            # Download the file with progress tracking
            print(f"Downloading file ID: {file_id}")
            print("This is a large file (approx. 4.4GB). Please be patient...")

            # Define a progress callback
            def callback(current, total):
                if not hasattr(callback, 'pbar'):
                    callback.pbar = tqdm(total=total, unit='B', unit_scale=True)
                callback.pbar.update(current - callback.pbar.n)

            # Download the file
            m.download_url(self.mega_url, dest_path=os.path.dirname(self.file_path),
                           dest_filename=self.local_filename, progress_callback=callback)

            print("✓ Download completed")
            return True
        except Exception as e:
            print(f"✗ Mega download failed: {str(e)}")
            return False

    def _extract_dataset(self):
        """Extract with comprehensive error handling"""
        print("\nExtracting dataset...")
        try:
            # Verify before extraction
            if not self._verify_download():
                print("✗ File verification failed before extraction")
                return False

            with rarfile.RarFile(self.file_path) as rf:
                print(f"Extracting {len(rf.namelist())} files to {self.dataset_dir}...")
                rf.extractall(self.dataset_dir)

            # Verify extraction
            required = ['train_data/data', 'train_data/ground-truth']
            for path in required:
                if not os.path.exists(os.path.join(self.dataset_dir, path)):
                    raise RuntimeError(f"Missing directory: {path}")

            print("✓ Extraction successful")
            return True

        except rarfile.NotRarFile:
            print("✗ Error: Not a valid RAR file")
            print("This usually means the download failed")
            print(f"Please delete {self.file_path} and try again")
            return False
        except rarfile.NeedFirstVolume:
            print("✗ Error: This is a multi-part RAR archive and the first volume is missing")
            return False
        except rarfile.BadRarFile:
            print("✗ Error: Invalid RAR file format")
            return False
        except rarfile.PasswordRequired:
            print("✗ Error: RAR file is password protected")
            return False
        except rarfile.RarCannotExec:
            print("✗ Error: Cannot find unrar executable")
            print("Please install WinRAR or 7-Zip and make sure it's in your system PATH")
            print("On Windows, you need to download unrar.exe and add it to your PATH")
            print("On Linux/Mac: 'apt-get install unrar' or 'brew install unrar'")
            return False
        except rarfile.RarExecError:
            print("✗ Error running unrar executable")
            print("Make sure you have rarfile module installed: pip install rarfile")
            print("And WinRAR/7-Zip/unrar installed on your system")
            return False
        except Exception as e:
            print(f"✗ Extraction failed: {str(e)}")
            print("Please make sure:")
            print("1. The download completed successfully")
            print("2. You have WinRAR/7-Zip installed")
            print("3. You have rarfile module installed: pip install rarfile")
            return False

    def _verify_download(self):
        """Verify the downloaded file"""
        if not os.path.exists(self.file_path):
            return False

        file_size = os.path.getsize(self.file_path)
        if file_size < self.min_expected_size:
            print(f"✗ File too small ({file_size / 1024 / 1024:.2f} MB)")
            return False

        # Quick check if it's a valid RAR file
        try:
            with rarfile.RarFile(self.file_path) as rf:
                # Just check a few files instead of full test
                for name in list(rf.namelist())[:5]:
                    if rf.getinfo(name).file_size == 0:
                        continue
                    try:
                        rf.open(name).read(100)
                    except:
                        return False
            return True
        except:
            return False

    def _show_manual_instructions(self):
        """Comprehensive manual instructions"""
        print("\n" + "=" * 60)
        print("MANUAL DOWNLOAD REQUIRED")
        print("=" * 60)

        if not MEGA_AVAILABLE:
            print("The mega.py library is not installed. You can either:")
            print("1. Install it with: pip install mega.py")
            print("   Then run this script again")
            print("\nOR")

        print("\n2. Download the dataset manually:")
        print("   - Go to: https://mega.nz/file/ak92xKSC#ZAKefRsPUPfq1L-Mch4L9fftjUZ9wVo9bUfuYq0Wltk")
        print(f"   - Save the file as: {self.file_path}")
        print("\n3. Alternatively, place the extracted dataset in any of these directories:")
        for loc in self.alt_dataset_locations:
            print(f"   - {loc}")
        print("\n4. Run the script again after downloading")

        print("\nImportant notes:")
        print("- The dataset file should be about 4.4GB when fully downloaded")
        print("- Make sure you have rarfile module installed (pip install rarfile)")
        print("- On Windows, you need WinRAR or 7-Zip installed")
        print("\nOpening download page in browser...")
        print("=" * 60)

        try:
            webbrowser.open("https://mega.nz/file/ak92xKSC#ZAKefRsPUPfq1L-Mch4L9fftjUZ9wVo9bUfuYq0Wltk")
        except:
            pass

        sys.exit(1)