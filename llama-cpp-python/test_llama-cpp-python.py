"""
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python

# Conversion du modèle GGML en GGUF
/media/data-nvme/dev/src/llm/llama.cpp$ python3 convert-llama-ggmlv3-to-gguf.py -i ../llama-cpp-python/llama-2-7b-32k-instruct.ggmlv3.q2_K.bin -o ../llama-cpp-python/llama-2-7b-32k-instruct.gguf.q2_K.bin

"""
from llama_cpp import Llama
from time import time
model_path="/media/ben/2To/dev/llm/Mistral/Mistral-7B-v0.1-GGUF/mistral-7b-v0.1.Q5_K_M.gguf"

start = time()

# GPU
lcpp_llm = None
lcpp_llm = Llama(
    model_path=model_path,
    n_threads=32, # CPU cores
    n_batch=512, # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    n_ctx=2048, # Context window
    n_gpu_layers=50, # change n_gpu_layers if you have more or less VRAM 
)

# See the number of layers in GPU
print(f"{lcpp_llm.params.n_gpu_layers=}")

prompt = "Write a linear regression in python"
prompt_template=f'''[INST] Write code to solve the following coding problem that obeys the constraints and passes the example test cases. Please wrap your code answer using ```:
{prompt}
[/INST]
'''

response = lcpp_llm(
    prompt=prompt_template,
    max_tokens=256,
    temperature=0.5,
    top_p=0.95,
    repeat_penalty=1.1,
    top_k=50,
    stop = ['USER:'], # Dynamic stopping when such token is detected.
    echo=True # return the prompt
)
print(response)
print(response["choices"][0]["text"])

print("Time taken: ", time() - start, "s")
