# Stop Sign Detection and Attack

Simple tool to detect stop signs in images and generate adversarial examples.

## Project Structure
```
project/
├── input_images/     # Put original images here
├── output_images/    # Detection results appear here
├── noisy_images/     # Adversarial images stored here
├── main.py          # Stop sign detector
├── noiser.py        # Adversarial attack generator
└── requirements.txt
```

## Setup Virtual Environment
```bash
# Create virtual environment
python -m venv env

# Activate virtual environment
source env/bin/activate

# Install requirements
pip install -r requirements.txt
```

## Usage
1. Place images in `input_images/` directory
2. Run detector:
   ```bash
   python main.py
   ```
3. Run adversarial attack:
   ```bash
   python noiser.py
   ```

Detection results and attack reports will be saved in their respective output directories (`output_images/` and `noisy_images/`).