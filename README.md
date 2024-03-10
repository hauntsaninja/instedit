# instedit

Get an editable install, instantly!

PEP 660 is slow. `instedit` is fast. If your build backend is setuptools, then by default your
editable installs will not work well with static type checkers and will have additional runtime
overhead.

So instead `instedit` plays fast and loose with the rules and only pays attention to standards when
it suits it. Your mileage may vary.

`instedit` is happy to work from outside your project environment.
