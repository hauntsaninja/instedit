# instedit

Get an editable install, instantly!

PEP 660 is slow. `instedit` is fast. If your build backend is setuptools, then by default your
editable installs will not work well with static type checkers and will have additional runtime
overhead.

So instead `instedit` plays fast and loose with the rules and only pays attention to standards when
it suits it. Your mileage may vary.

## Installation
```bash
pipx install 'git+https://github.com/hauntsaninja/instedit'
```
`instedit` is happy to work from outside your project environment. It will respect the `VIRTUAL_ENV`
environment variable, or you can pass in a target python with the `--python` flag.

## Usage
```bash
instedit
```
Installs the project in the current directory editably!

```
λ instedit --help
usage: instedit [-h] [--python PYTHON] [path]

positional arguments:
  path             Path to the project to install editably

options:
  -h, --help       show this help message and exit
  --python PYTHON  Python to install to (defaults to Python from VIRTUAL_ENV if set, otherwise sys.executable)
```
