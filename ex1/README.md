Prerequisites
=============

The only prerequisites should be Rust and its package manager Cargo, which
usually come together. Cargo will pull in other requirements. The simplest way
to get started is with the [rustup script](https://rustup.rs/).

Running
=======

You can run like so:

 $ cargo run -- -s 41cc89fe9eb4edfc4c947510b3dd20ff -vv

The switch -s supplies a seed to the PRNG and the number of -v flags passed
control verbosity.

The final results for this seed are as follows::

    Final results
    #0 <Gene 01fa x1: fa = 4.90625 x2: 01 = 1.015625> fitness: 38.269287
    #1 <Gene 01fa x1: fa = 4.90625 x2: 01 = 1.015625> fitness: 38.269287
    #2 <Gene 01fa x1: fa = 4.90625 x2: 01 = 1.015625> fitness: 38.269287
    #3 <Gene 01fa x1: fa = 4.90625 x2: 01 = 1.015625> fitness: 38.269287
    #4 <Gene 01fa x1: fa = 4.90625 x2: 01 = 1.015625> fitness: 38.269287
    #5 <Gene 01fa x1: fa = 4.90625 x2: 01 = 1.015625> fitness: 38.269287
    #6 <Gene 01fa x1: fa = 4.90625 x2: 01 = 1.015625> fitness: 38.269287
    #7 <Gene 01fa x1: fa = 4.90625 x2: 01 = 1.015625> fitness: 38.269287
    #8 <Gene 01fa x1: fa = 4.90625 x2: 01 = 1.015625> fitness: 38.269287
    #9 <Gene 01fa x1: fa = 4.90625 x2: 01 = 1.015625> fitness: 38.269287
