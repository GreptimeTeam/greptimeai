# greptimeai

Observability and analytics tool for LLM framework, service, etc. You can find more
examples and guides on [greptimeai-cookbook][greptimeai-cookbook]

## Installation

To start, ensure you have Python 3.8 or newer. If you just want to use the package, run:

```sh
pip install --upgrade greptimeai
```

## Setup

To get started, create a service by registering [greptimeai][greptimeai], and get:

- host
- database
- token

Set it as the `GREPTIMEAI_xxx` environment variable before using the library:

```bash
export GREPTIMEAI_HOST=''
export GREPTIMEAI_DATABASE=''
export GREPTIMEAI_TOKEN=''
```

or you can pass them via parameters.

## Examples

- [langchain](./examples/langchain.ipynb)
- [openai](./examples/openai.ipynb)

## Contributing

Contributions are highly encouraged!

Pull requests that add support for or fix a bug in a feature will likely be accepted after review.

## Licensing

All code in this repository is licensed under the [Apache License 2.0](LICENSE).

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be licensed as above, without any additional terms or conditions.

[greptimeai]: https://console.greptime.cloud/ai
[greptimeai-cookbook]: https://github.com/GreptimeTeam/greptimeai-cookbook
