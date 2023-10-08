# Running llm on Mac
## 1.Install Git Lfs
mac install:
```
brew install git-lfs

#Verify that the installation is successful
git lfs install
## If `Git LFS initialized` is displayed, the installation is successful
```

## 2. Install llm all-MiniLM-L6-v2
### ①. Install all-MiniLM-L6-v2
all-MiniLM-L6-v2 (used to convert text vectors)
```
git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
```

### ②. Set the vector model path in langchain
Set the absolute path of the model in HuggingFaceEmbeddings in `src/langchain_example/llm.py`

model_name='/xx/xxx'


## 3. Install llm ChatGlm2-6B
### ①. Install chatglm2-6b
chatglm2-6b  (AI Chat)
```
git clone https://huggingface.co/THUDM/chatglm2-6b
```

### ②. Quantize chatglm.cpp
- chatglm.cpp: Similar to `llama.cpp` CPU quantization accelerated reasoning scheme to achieve real-time conversation on Mac notebook

#### a. Clone chatglm.cpp
```
git clone --recursive https://github.com/li-plus/chatglm.cpp.git && cd chatglm.cpp
```
If you forgot the -- recursive flag when cloning the repository, run the following command in the `chatglm.cpp` folder:
```
git submodule update --init --recursive
```

#### b. Install dependency
```
python3 -m pip install -U pip
python3 -m pip install torch tabulate tqdm transformers sentencepiece
```

#### c. Convert model
convert ChatGLM2-6B to quantized GGML format using `convert.py`
```
python3 chatglm_cpp/convert.py -i THUDM/chatglm2-6b -t q4_0 -o chatglm2-ggml.bin
```

#### d. Compile
cmake needs to be prepared in advance
```
cmake -B build
cmake --build build -j --config Release
```

#### e. Run model
```
./build/bin/main -m chatglm2-ggml.bin -p 你好 --top_p 0.8 --temp 0.8
# 你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。
```

### ③. Python Building
#### a. Install chatglm-cpp
```
CMAKE_ARGS="-DGGML_METAL=ON" pip install -U chatglm-cpp
```

#### b. Install chatglm-cpp[api]
```
pip install 'chatglm-cpp[api]'
```

#### c. Start the api server for LangChain
```
MODEL=./chatglm2-ggml.bin uvicorn chatglm_cpp.langchain_api:app --host 127.0.0.1 --port 8001
```

### ④. Set API Server Endpoint in langchain
Set `endpoint_url` in `src/langchain_example/llm.py`

endpoint_url = "http://127.0.0.1:8001"