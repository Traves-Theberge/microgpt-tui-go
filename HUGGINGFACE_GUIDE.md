# Hugging Face Hosting Guide

Follow these steps to host your model checkpoint on Hugging Face.

## Prerequisites

Before starting, you need:

1. A **Hugging Face Account**: Create one at [huggingface.co](https://huggingface.co/).
2. A **User Access Token**: Generate a token with "Write" access at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
3. **Curl** for the CLI installation.

## 1. Create a Hugging Face Repository

1. Go to [huggingface.co/new](https://huggingface.co/new).
2. Set the owner (your username).
3. Repository Name: `[YOUR_MODEL_NAME]` (e.g., `my-transformer-v1`).
4. Select **Model** as the type.
5. Set Visibility (Public or Private).
6. Click **Create repository**.

## 2. Prepare the Model Card (README.md)

Create a file named `README.md` with the following content. This will be the landing page for your model.

````markdown
---
language: en
license: mit
tags:
- transformer
- [YOUR_TAGS]
---

# [MODEL_NAME]

Provide a brief description of your model here. Mention the architecture, purpose, and any high-level details.

## Model Details
- **Architecture**: [e.g., Transformer, CNN, etc.]
- **Parameters**: [Number of params]
- **Training Data**: [Dataset name/description]
- **Tokenization**: [Type of tokenization used]

## Usage

Describe how to run the model.

### Loading the Model
1. Download the required weights/checkpoint files from this repository.
2. Follow the implementation-specific instructions for your engine.

```bash
# Example command
[YOUR_COMMAND] [PATH_TO_MODEL] "Your input here"
```

## Attribution

Provide any necessary attribution or credits here.
````

## 3. Upload the Model Files

You can upload files directly through the Hugging Face web interface or use the CLI for better reliability.

### CLI Instructions

1. Install the standalone Hugging Face CLI (`hf`) using Curl:

   ```bash
   curl -LsSf https://hf.co/cli/install.sh | bash
   ```

   *Note: Alternatively, you can use the Python-based library:* `pip install --upgrade huggingface_hub`

2. Log in to your account:

   ```bash
   huggingface-cli login
   ```

3. Upload the model files:

   ```bash
   huggingface-cli upload <your-username>/[MODEL_REPO_NAME] ./[LOCAL_PATH_TO_FILES] [REMOTE_PATH]
   ```

4. Upload the model card:

   ```bash
   huggingface-cli upload <your-username>/[MODEL_REPO_NAME] README.md README.md
   ```
