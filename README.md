# greptimeai

Observability and analytics tool for LLM framework, service, etc. You can find more
examples and guides on [greptimeai-cookbook][greptimeai-cookbook]

## Installation

To start, ensure you have Python 3.8 or newer. If you just
want to use the package, run:

```sh
pip install --upgrade greptimeai
```

## Usage

To get started, create a service by registering [greptimeai][greptimeai], and get:

- host
- database
- username
- password

Set it as the `GREPTIMEAI_xxx` environment variable before using the library:

```bash
export GREPTIMEAI_HOST=''
export GREPTIMEAI_DATABASE=''
export GREPTIMEAI_USERNAME=''
export GREPTIMEAI_PASSWORD=''
```

#### LangChain

LangChain provides a callback system that allows you to hook into the various stages of your LLM
application. Import GreptimeCallbackHandler, which helps to collect metrics and traces to
GreptimeCloud.

```python
from greptimeai.langchain.callback import GreptimeCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

callbacks = [GreptimeCallbackHandler()]
llm = OpenAI()
prompt = PromptTemplate.from_template("1 + {number} = ")

# Constructor callback: First, let's explicitly set the GreptimeCallbackHandler
# when initializing our chain
chain = LLMChain(llm=llm, prompt=prompt, callbacks=callbacks)
chain.run(number=2)

# Request callbacks: Finally, let's use the request `callbacks` to achieve the same result
chain = LLMChain(llm=llm, prompt=prompt)
chain.run(number=2, callbacks=callbacks)
```

This example needs to be configured with your OpenAI account's private API key which is available on
our [developer platform](openai). Set it as the `OPENAI_API_KEY` environment variable:

```bash
export OPENAI_API_KEY='sk-...'
```

#### OpenAI

TODO

[greptimeai]: https://console.greptime.cloud/ai
[greptimeai-cookbook]: https://github.com/GreptimeTeam/greptimeai-cookbook
[openai]: https://platform.openai.com/account/api-keys
