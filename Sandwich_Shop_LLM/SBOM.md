# Software Bill of material (SBOM)

## Host
This project was tested with the following, but it is expected to work with Windows 10/11 and other nvidia RTX (with sufficient memory) running on AMD64 arch
- Windows 10.0.19045.5608
- Nvidia RTX 3090 Driver 32.0.15.6094
- Docker Desktop 4.39.0 (184744)
- docker-compose-plugin 2.33.1-1\~ubuntu.24.04\~noble

### WSL
These utilities are installed inside the Windows services for Linux.

#### nvidia-container-toolkit
NVIDIA Container Runtime Hook version 1.17.5
- commit: f785e908a7f72149f8912617058644fd84e38cde
- Installed by apt-get package manager

#### nvidia-cuda-toolkit
- nvidia-cuda-toolkit: 12.0.140~12.0.1-4build4
- nvidia-cuda-toolkit-doc: 12.0.1-4build4                          

## Models
The following models were evaluated during our demo, however we only ended up using a few of them.

### lmstudio-community/Codestral-22B-v0.1-GGUF
- link ```https://huggingface.co/lmstudio-community/Codestral-22B-v0.1-GGUF/blob/main/Codestral-22B-v0.1-Q3_K_M.gguf```
- sha256: 758c965d21d07c5b03b4fcabd42a2197ba507ab2b5e8374c19c288eb99293f4d

### bartowski/Dolphin3.0-Llama3.1-8B-GGUF
- link: ```https://huggingface.co/bartowski/Dolphin3.0-Llama3.1-8B-GGUF/blob/main/Dolphin3.0-Llama3.1-8B-Q4_K_S.gguf```
- sha256: 3d8f3404e0d0f2f914fd3e3134f4b3e2af09d26a8d4aaa06f1fc369ff19bbe0a

### TheBloke/Mistral-7B-Instruct-v0.1-GGUF
- link: ```https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/blob/main/mistral-7b-instruct-v0.1.Q4_K_S.gguf```
- sha256: f1b7f1885029080be49aff49c83f87333449ef727089546e0d887e2f17f0d02e

### bartowski/Mistral-Nemo-Instruct-2407-GGUF
- link: ```https://huggingface.co/bartowski/Mistral-Nemo-Instruct-2407-GGUF/blob/main/Mistral-Nemo-Instruct-2407-Q4_K_M.gguf```
- sha256: 7c1a10d202d8788dbe5628dc962254d10654c853cae6aaeca0618f05490d4a46

### Llama-3.2-11B-Vision-Instruct.Q4_K_M.gguf
- link: ```https://huggingface.co/leafspark/Llama-3.2-11B-Vision-Instruct-GGUF/blob/main/Llama-3.2-11B-Vision-Instruct.Q4_K_M.gguf```
- sha256: 652e85aa1e14c9087a4ccc3ab516fb794cbcf152f8b4b8d3c0b828da4ada62d9

## Inference Service

### llama.cpp:full-cuda 
- Registry link: ```https://github.com/ggml-org/llama.cpp/pkgs/container/llama.cpp/373238619?tag=full-cuda-b4879```
- sha256: af38c79b9e47304faf54047e2f8571d4ac885b48a9075dae657aa79e7b7285a1

## Front End

### mintplexlabs/anythingllm:latest
- Registry Link: ```https://hub.docker.com/layers/mintplexlabs/anythingllm/latest/images/sha256-0ea31171797d8ad285b124adf7136a6ac603249c240cf7bdd04ac69e74e7d301```
- sha256: c8479cb1e85561a1dca92dfdad6c3d6055b317a0b94f99e1f70c438ecb6606f4
