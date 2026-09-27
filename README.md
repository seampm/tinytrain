# tinytrain — neural network training in C++, from scratch

Companion to [tinyinfer](https://github.com/seampm/tinyinfer). Where tinyinfer
*runs* models, tinytrain *teaches* them: a from-scratch autograd engine,
optimizers, and everything needed to train a small transformer with zero ML
frameworks. A trained checkpoint loads directly into tinyinfer — one pipeline,
built entirely by hand.

Status: scaffolding. The engine, training runs, and benchmarks land here as
they're built and measured.
