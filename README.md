# PhotoPost

**AI-assisted photo discovery, aesthetic scoring, caption generation, and music recommendations for social media content.**

PhotoPost is a collection of Python-based AI/ML components designed to support different stages of preparing a social media post. The repository contains separate modules for finding pose and photo references, ranking images by aesthetic quality, generating and ranking captions, and recommending music based on a desired vibe.

> **Repository status:** The components are implemented as separate scripts/modules. This repository snapshot does not contain a single integrated frontend or application orchestrator connecting all four workflows.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Technology Stack](#technology-stack)
- [Getting Started](#getting-started)
- [Running the Components](#running-the-components)
  - [1. Pose and Image Recommendation](#1-pose-and-image-recommendation)
  - [2. NIMA Aesthetic Photo Scoring](#2-nima-aesthetic-photo-scoring)
  - [3. BLIP Caption Generation](#3-blip-caption-generation)
  - [4. Vibe-Based Song Recommendation](#4-vibe-based-song-recommendation)
- [Environment Variables](#environment-variables)
- [Model Files](#model-files)
- [Implementation Notes and Limitations](#implementation-notes-and-limitations)
- [Contributing](#contributing)
- [License](#license)

## Overview

Preparing a social media post involves more than taking a photograph. Choosing a pose, selecting the strongest image from a set, writing a caption that matches the image, and finding suitable music are separate tasks that can take time.

PhotoPost explores how computer vision and language models can assist with these tasks through four components:

1. **Discover:** Find pose and photo references using a user-described outfit, location, and visual vibe.
2. **Select:** Estimate the aesthetic quality of photographs and rank them.
3. **Caption:** Describe an image, generate caption candidates, and rank them for image-text relevance.
4. **Recommend:** Extract vibe keywords and recommend music matching the intended mood.

## Features

### CLIP-based pose and image recommendation

- Builds a search prompt from outfit, location, vibe, and gender inputs.
- Retrieves image results through SerpAPI.
- Uses CLIP (`openai/clip-vit-base-patch32`) to calculate image-text similarity.
- Applies a critical-keyword filter to reject results that do not sufficiently match key prompt details.
- Combines CLIP similarity with image sharpness, brightness, and contrast measurements.
- Saves the five highest-ranked images for each session.

### NIMA aesthetic photo scoring

- Implements a Neural Image Assessment (NIMA) model.
- Supports MobileNetV2 and EfficientNetB0 backbones in the model builder.
- Predicts a probability distribution over aesthetic ratings from 1 to 10.
- Computes a mean aesthetic score from the predicted distribution.
- Includes an AVA-style data pipeline, training, evaluation, inference, and TensorFlow Lite utilities.
- Includes a FastAPI endpoint intended to accept multiple images and return them ranked by predicted mean aesthetic score.

### BLIP caption generation and CLIP ranking

- Generates an image description using Salesforce BLIP image captioning.
- Uses an OpenRouter-hosted language model to generate caption candidates from the image description and a user-provided vibe prompt.
- Supports an optional tone instruction.
- Uses OpenAI CLIP image-text similarity to rank generated captions.
- Prints the top-ranked captions for each image.

### Vibe-based song recommendation

- Extracts mood and vibe keywords from a natural-language prompt.
- Supports KeyBERT/Sentence Transformers for keyword extraction when installed.
- Uses an OpenRouter language model for keyword extraction and song recommendations.
- Requests recommendations across Bollywood and Hollywood.
- Filters duplicate title-and-artist pairs across repeated recommendation batches within the same process.

## Architecture

The repository currently contains four independently executable components:

```text
                         User-provided context
                                  |
             +--------------------+--------------------+
             |                    |                    |
       Outfit / location       Image + vibe          Vibe prompt
        / photo vibe             / tone                  |
             |                    |                     |
             v                    v                     v
      Pose recommender       BLIP captioning       Song recommender
             |                    |                     |
       SerpAPI search          BLIP model         KeyBERT / OpenRouter
             |                    |                     |
       CLIP similarity       Caption candidates    Vibe keywords
             |                    |                     |
       Image quality         CLIP text-image             v
       (OpenCV)              similarity ranking     Song suggestions
             |                    |
             v                    v
       Top 5 reference       Ranked captions
          images

                  Separate NIMA scoring component
                              |
                       Uploaded images
                              |
                       NIMA model inference
                              |
                    Predicted rating distribution
                              |
                    Mean aesthetic score / ranking
```

The diagram represents the implemented component-level workflows. It does not imply that a single application currently coordinates them automatically.

## Repository Structure

```text
Photopost/
├── .github/
│   ├── templates/
│   │   └── pull_request_template.md
│   └── workflows/
│       └── lint.yml
│
├── caption-generation-blip-script/
│   ├── Script.py
│   └── requirements.txt
│
├── models/
│   ├── nima_best_finetune.h5
│   ├── nima_best_finetune.weights.h5
│   ├── nima_best_head.h5
│   ├── nima_best_head.weights.h5
│   ├── nima_final.h5
│   ├── nima_final.keras
│   ├── nima_savedmodel/
│   └── train_log.csv
│
├── nima-aesthetic-photo-scorer/
│   ├── data.py
│   ├── eval.py
│   ├── infer.py
│   ├── loss.py
│   ├── model.py
│   ├── router.py
│   ├── tflite_convert.py
│   ├── tflite_infer.py
│   ├── train.py
│   └── utils.py
│
├── pose-recommender-clip-script/
│   ├── script.py
│   └── requirements.txt
│
├── song_recommendation/
│   └── song-recommender-script/
│       ├── song_recommendation.py
│       └── requirements.txt
│
├── CONTRIBUTING.md
├── LICENSE
└── readme.md
```

## Technology Stack

| Area | Technologies |
|---|---|
| Programming language | Python |
| Computer vision and image processing | OpenCV, Pillow |
| Vision-language models | OpenAI CLIP |
| Image captioning | Salesforce BLIP, Hugging Face Transformers |
| Aesthetic assessment | NIMA, TensorFlow, Keras, MobileNetV2, EfficientNetB0 |
| Language-model integration | OpenRouter API, OpenAI-compatible SDK |
| Keyword extraction | KeyBERT, Sentence Transformers |
| API layer | FastAPI |
| Image search | SerpAPI |
| Data and numerical processing | NumPy, Pandas |
| Model execution | PyTorch, TensorFlow, TensorFlow Lite |

## Getting Started

### Prerequisites

- Python installed on your system.
- Git.
- API credentials for the external services used by the component you intend to run.
- A compatible PyTorch/TensorFlow installation for the relevant model.
- Sufficient memory and storage for downloading and loading pretrained models.

Clone the repository:

```bash
git clone https://github.com/ZephyriumX/Photopost.git
cd Photopost
```

Each component has its own dependency file. Install dependencies from the relevant component directory rather than installing all requirements into one environment. The component requirement files pin different versions of shared libraries, including PyTorch and NumPy.

A virtual environment is recommended for each component:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Then install that component's requirements as described below.

## Running the Components

### 1. Pose and Image Recommendation

**Location:** `pose-recommender-clip-script/`

This script asks for an outfit, location, desired vibe, and gender. It builds a search query, retrieves image results, filters them using CLIP similarity, calculates basic image-quality measures, and saves the top five results.

#### Setup

Create a SerpAPI account and obtain an API key. Set it as an environment variable:

```env
SERPAPI_KEY=your_serpapi_key
```

Install the component dependencies:

```bash
cd pose-recommender-clip-script
pip install -r requirements.txt
```

Run:

```bash
python script.py
```

The script prompts for:

- Outfit
- Location
- Desired photo vibe
- Gender

Results are saved in session-specific folders under `fetched/`:

```text
fetched/
└── session_1/
    ├── top_1.jpg
    ├── top_2.jpg
    ├── top_3.jpg
    ├── top_4.jpg
    └── top_5.jpg
```

The script creates a new session number on subsequent runs.

### 2. NIMA Aesthetic Photo Scoring

**Location:** `nima-aesthetic-photo-scorer/`

The NIMA module predicts a distribution over aesthetic ratings from 1 to 10. Its inference script calculates the mean score using the predicted distribution.

#### Inference with a saved H5 model

The inference script accepts a model path and one or more image paths:

```bash
python infer.py --model ../models/nima_final.h5 --images path/to/image1.jpg path/to/image2.jpg
```

The script prints the mean aesthetic score and predicted rating distribution for each image.

#### Training

The training pipeline expects:

- An AVA-style CSV file.
- An image directory containing the corresponding JPEG files.
- The first CSV column to contain image identifiers or filenames.
- The next ten columns to contain vote counts for ratings 1 through 10.

Example command:

```bash
python train.py \
  --ava_csv path/to/ava_dataset.csv \
  --images_dir path/to/ava_images \
  --out_dir ../models \
  --epochs 10 \
  --batch_size 32
```

On Windows PowerShell, enter the command on one line or use PowerShell's backtick for line continuation.

Optional training arguments include:

- `--backbone`: `MobileNetV2` or an EfficientNetB0 fallback.
- `--val_split`: validation fraction.
- `--limit`: limit the number of samples for quick experiments.
- `--loss`: `emd` or `kl`.
- `--freeze_backbone_epochs`: head-only training epochs.
- `--finetune_layers`: number of final backbone layers to unfreeze.
- `--mixed_precision`: enable mixed-precision training where supported.

#### FastAPI router

The module includes a FastAPI router with a `POST /nima/rate` endpoint. It accepts multiple uploaded image files and returns a list ranked by predicted mean aesthetic score.

The router currently looks for a TensorFlow Lite model at:

```text
nima-aesthetic-photo-scorer/models/nima.tflite
```

That file is not included in this repository snapshot. Generate or place the expected TFLite model at that path before using the endpoint. The router and package layout also need the import/package setup described in [Implementation Notes and Limitations](#implementation-notes-and-limitations).

### 3. BLIP Caption Generation

**Location:** `caption-generation-blip-script/`

The caption pipeline:

1. Loads an image and generates a description using BLIP.
2. Generates five caption candidates through an OpenRouter language model.
3. Calculates CLIP image-text similarity for each candidate.
4. Sorts candidates by similarity and prints the top results.

#### Setup

Create a `.env` file in the working directory with:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
```

Install dependencies:

```bash
cd caption-generation-blip-script
pip install -r requirements.txt
```

Place the images to process in a directory named `session_1` relative to the script's working directory. The current script uses this fixed directory.

Run:

```bash
python Script.py --prompt "cinematic, confident photoshoot vibe" --tone "witty" --top_k 3
```

Arguments:

| Argument | Description |
|---|---|
| `--prompt` | Optional vibe prompt for caption generation |
| `--tone` | Optional caption tone |
| `--top_k` | Number of highest-ranked captions to print; defaults to 3 |

If no prompt is provided, the script attempts to load `prompt.txt`; if that file is absent, it uses a built-in default prompt.

### 4. Vibe-Based Song Recommendation

**Location:** `song_recommendation/song-recommender-script/`

The song recommender extracts keywords from a prompt and asks an OpenRouter language model to return Bollywood and Hollywood song recommendations in JSON format.

#### Setup

Create a `.env` file in the working directory:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
```

Install dependencies:

```bash
cd song_recommendation/song-recommender-script
pip install -r requirements.txt
```

Run:

```bash
python song_recommendation.py --prompt "cinematic, confident, high-energy photoshoot vibe"
```

Useful arguments:

| Argument | Description |
|---|---|
| `--prompt` | Required description of the desired vibe |
| `--no-keybert` | Disable KeyBERT keyword extraction |
| `--num-each` | Requested number of songs per category |
| `--model` | OpenRouter model identifier; defaults to `gpt-4o-mini` |
| `--tone` | Optional tone hint |

The script runs interactively and asks whether another recommendation batch is required.

## Environment Variables

| Variable | Used by | Purpose |
|---|---|---|
| `SERPAPI_KEY` | Pose recommender | Access image-search results through SerpAPI |
| `OPENROUTER_API_KEY` | Caption generator | Access the OpenRouter chat-completions API |
| `OPENROUTER_API_KEY` | Song recommender | Access the OpenRouter chat-completions API |

Do not commit `.env` files, API keys, or other credentials to version control.

## Model Files

The repository includes NIMA model artifacts under `models/`, including:

- `nima_final.h5`
- `nima_final.keras`
- `nima_best_head.h5`
- `nima_best_head.weights.h5`
- `nima_best_finetune.h5`
- `nima_best_finetune.weights.h5`
- `nima_savedmodel/`
- `train_log.csv`

The training script saves Keras H5/Keras model files and attempts to export a TensorFlow SavedModel. The inference script's current loading logic explicitly handles `.h5` files or TensorFlow checkpoint directories; it does not directly load the `.keras` file through its current branch.

The CLIP and BLIP pretrained checkpoints are loaded through their respective libraries and may be downloaded the first time the scripts run.

## Implementation Notes and Limitations

The following details are important when setting up this repository:

1. **Independent components:** The pose recommender, NIMA scorer, caption generator, and song recommender are separate workflows. The repository snapshot does not include a unified frontend or a central service that chains all four together.
2. **NIMA TFLite artifact:** `router.py` expects `nima-aesthetic-photo-scorer/models/nima.tflite`. This file is not present in the supplied repository snapshot, although other NIMA model artifacts are included in the root `models/` directory.
3. **NIMA Python package layout:** The directory `nima-aesthetic-photo-scorer` contains modules using relative imports. Because the directory name contains hyphens, it is not directly importable as a normal Python package. Rename it to a valid Python package name (for example, `nima_aesthetic_photo_scorer`) and update import paths before using it as an installed/module-based package or running the FastAPI router cleanly.
4. **Pose recommender credentials:** The pose recommender requires `SERPAPI_KEY`. The script does not provide a complete user-facing credential setup flow.
5. **Caption image directory:** The caption script currently reads from `./session_1`. This path is fixed in the code rather than exposed as a command-line argument.
6. **Separate dependency environments:** The requirements files pin different versions of libraries such as PyTorch and NumPy. Use separate virtual environments for the individual components to avoid dependency conflicts.
7. **External model/API availability:** First-time model downloads require internet access. API-backed features require valid credentials and may incur usage charges according to the provider's terms.
8. **Image-search rights:** Retrieved images may be subject to copyright, licensing, and platform terms. Check the source and usage rights before redistributing or publishing fetched images.

This README documents the code present in the repository snapshot. It does not claim that the components have been integrated, deployed, or validated as one end-to-end application.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting changes.

The repository contribution guide describes a branch-based workflow and conventional commit prefixes such as:

- `feat:` for new features
- `fix:` for bug fixes
- `chore:` for maintenance
- `docs:` for documentation

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
