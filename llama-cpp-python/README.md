# Installation

```sh
python -m venv .venv && source .venv/bin/activate
pip install llama-cpp-python[server]
```

Pour Mistral :

```sh
git clone --recurse-submodules https://github.com/abetlen/llama-cpp-python.git
cd llama-cpp-python/
make build
```

# Lancement API

`python3 -m llama_cpp.server --model /media/2To-nvme/dev/llm/Mistral/mistral-7b-v0.1.Q5_K_M.gguf`

# Avec GPU

```sh
docker run -it -p 7860:7860 --platform=linux/amd64 --gpus all \
    -v "/media/2To-nvme/dev/llm/huggingface:/data" \
	-e HF_HOME="/data" \
	-e REPO_ID="TheBloke/Mistral-7B-Instruct-v0.1-GGUF" \
	-e MODEL_FILE="mistral-7b-instruct-v0.1.Q4_K_M.gguf" \
	registry.hf.space/spacesexamples-llama-cpp-python-cuda-gradio:latest
```

Le code source est ici : https://huggingface.co/spaces/SpacesExamples/llama-cpp-python-cuda-gradio/tree/main

Ensuite ouvrir http://0.0.0.0:7860/ et c'est ultra-rapide tout en n'utilisant que 5Go de VRAM.

Ou alors en local :
```sh
source .venv/bin/activate
cd /media/2To-nvme/dev/llm/llama-cpp-python
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export PATH=$PATH:$CUDA_HOME/bin
#git clone --recurse-submodules https://github.com/abetlen/llama-cpp-python.git
#cd llama-cpp-python/
pip uninstall -y llama-cpp-python
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install -e .[server] --no-cache-dir

python3 -m llama_cpp.server --model /media/2To-nvme/dev/llm/Mistral/mistral-7b-v0.1.Q5_K_M.gguf

cd -
python test_llama-cpp-python.py
```
