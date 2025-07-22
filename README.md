# VisionNet: Simple Image Classifier

A beginner-friendly image classification project using deep learning. This repository provides a clean, modular codebase for training and evaluating image classifiers on custom or standard datasets.

## Features

- Modular code structure for easy customization
- Support for common image formats
- Training, validation, and testing scripts
- Model checkpointing and logging
- Easily extensible for new datasets or architectures

## Project Structure

- `main.py` — Entry point for training and evaluation
- `model.py` — Neural network model definitions
- `train.py` — Training loop and utilities
- `data_loader.py` — Data loading and augmentation
- `utils.py` — Helper functions
- `outputs/` — Saved models and results (ignored by git)
- `requirements.txt` — Python dependencies

## Setup

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/visionnet-image-classifier.git
   cd visionnet-image-classifier
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

Train the model:
```sh
python main.py --train
```

Evaluate the model:
```sh
python main.py --eval --checkpoint outputs/best_model.pth
```

## Customization

- Modify `model.py` to try different architectures.
- Update `data_loader.py` for your own dataset.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.

---

*Happy coding!*
