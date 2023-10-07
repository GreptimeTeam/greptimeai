# Mac 笔记本上运行 llm
## 1.安装 Git Lfs
mac 安装方式:
```
brew install git-lfs

#校验是否安装成功
git lfs install
## 若显示 Git LFS initialized. 则安装成功
```

## 2. 安装大模型 all-MiniLM-L6-v2
### ①. 安装 all-MiniLM-L6-v2
all-MiniLM-L6-v2 (用于转换文本向量)
```
git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
```

### ②. 在 langchain 中设置向量模型路径
在 src/langchain_example/llm.py 中的 HuggingFaceEmbeddings 设置 model 的绝对路径

model_name='/xx/xxx'


## 3. 安装大模型 ChatGlm2-6B
### ①. 安装 chatglm2-6b
chatglm2-6b  (用于 AI Chat)
```
git clone https://huggingface.co/THUDM/chatglm2-6b
```

### ②. 使用 chatglm.cpp 量化
- chatglm.cpp: 类似 llama.cpp 的 CPU 量化加速推理方案，实现 Mac 笔记本上实时对话

#### a. 克隆 chatglm.cpp 仓库
```
git clone --recursive https://github.com/li-plus/chatglm.cpp.git && cd chatglm.cpp
```
如果在克隆存储库时忘记了——recursive 标志，请在 chatglm.cpp 文件夹中运行以下命令:
```
git submodule update --init --recursive
```

#### b. 安装相关依赖
```
python3 -m pip install -U pip
python3 -m pip install torch tabulate tqdm transformers sentencepiece
```

#### c. 转换模型
使用 convert.py 将 ChatGLM2-6B 转换为量化的GGML格式
```
python3 chatglm_cpp/convert.py -i THUDM/chatglm2-6b -t q4_0 -o chatglm2-ggml.bin
```

#### d. 编译
需要提前准备 cmake 
```
cmake -B build
cmake --build build -j --config Release
```

#### e. 模型运行
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

### ④. 在 langchain 中设置 API Server Endpoint
在 src/langchain_example/llm.py 中设置 endpoint_url

endpoint_url = "http://127.0.0.1:8001"