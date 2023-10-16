

```sh
python -m venv .venv && source .venv/bin/activate
pip install vllm
python -m vllm.entrypoints.openai.api_server --dtype float16 --quantization awq --model TheBloke/Mistral-7B-v0.1-AWQ
#  mistralai/Mistral-7B-v0.1 => Pas assez de VRAM
# , mistralai/Mistral-7B-Instruct-v0.1

```