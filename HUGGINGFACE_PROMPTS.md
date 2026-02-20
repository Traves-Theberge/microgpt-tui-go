# Hugging Face Hosting: Prompt Guide

This document provides a sequence of prompts that you can give to an AI agent to automate the hosting of your model and the deployment of an OpenAI-compatible API on Hugging Face.

---

## Prerequisites

Before starting, ensure you have:

1. A **Hugging Face Account**: [Sign up here](https://huggingface.co/join).
2. A **User Access Token**: [Create one here](https://huggingface.co/settings/tokens) (ensure it has "Write" permissions).
3. **Curl installed** on your system.

---

## Phase 1: Preparation and CLI Setup

**Prompt:**
> "I want to host my model on Hugging Face. Please start by installing the Hugging Face CLI (`hf`) in my local environment using the standalone installation script: `curl -LsSf https://hf.co/cli/install.sh | bash`. Once installed, help me login using my Hugging Face write token."

---

## Phase 2: Model Repository Deployment

**Prompt:**
> "Now that the CLI is ready, I need to create a new model repository on Hugging Face called `[YOUR_USERNAME]/[MODEL_NAME]`.
>
> Please perform the following:
>
> 1. Create the repository.
> 2. Generate a professional `README.md` (Model Card) that describes the model architecture, parameters, and provides usage instructions.
> 3. Upload the `README.md` and the model weights or checkpoint files to the repository."

---

## Phase 3: Code Refactoring for API Compatibility

**Prompt:**
> "I want to deploy an OpenAI-compatible API for this model on a Hugging Face Space. Before we create the Space, we need to refactor the codebase to make the model logic reusable.
>
> Please:
>
> 1. Restructure the project to extract the core inference logic into a dedicated package.
> 2. Isolate the model math, architecture, and tokenizer/preprocessing logic from the main application.
> 3. Ensure any existing tools or UIs are updated to use this new reusable package."

---

## Phase 4: Developing the API Server

**Prompt:**
> "Now, implement an API server that:
>
> 1. Uses the refactored inference package.
> 2. Implements an OpenAI-compatible `/v1/chat/completions` endpoint.
> 3. Implements a `/v1/models` endpoint.
> 4. Includes a root `/` handler that returns a simple 'API is running' message.
> 5. Configures the server to listen on port `7860` (the standard port for Hugging Face Spaces)."

---

## Phase 5: Containerization and Space Deployment

**Prompt:**
> "Finally, let's deploy the API to a Hugging Face Space.
>
> Please:
>
> 1. Create a multi-stage `Dockerfile` that builds the API server and bundles the required model weights.
> 2. Create a `README.md` for the Space with the required YAML metadata (e.g., `sdk: docker`, `app_port: 7860`).
> 3. Create a new Space on Hugging Face called `[YOUR_USERNAME]/[SPACE_NAME]` and upload the Dockerfile, metadata, and necessary source code to it."

---

## Phase 6: Verification

**Prompt:**
> "The Space is deployed! Please run a `curl` command to verify that the root endpoint returns a success message, and then send a test `POST` request to `/v1/chat/completions` to ensure the model generates a response correctly."
