Preprocessing
-------------

[FinnPos](https://github.com/mpsilfve/FinnPos) is used for lemmatisation. It is
a prerequisites iff you want to run the preprocessing stage. If you only want
to run the optimisation, you can skip this step and use the dump included as
1991.dump.

You will also need the OpenSubtitles2018 corpus in Finnish.
[It can be obtained from the OPUS project.](http://opus.nlpl.eu/OpenSubtitles2018.php)

You can generate a corpus dump of all films from 1991 like so:

    $ cargo run --release --bin proj-parse /path/to/OpenSubtitles2018/xml/fi/1991 1991.dump

Optimisation
------------

You can run it like so:

    $ cargo run --release --bin proj -- --neighborhood-size 8 --too-soon 2.5 --pop-size 64 --generations 100 1991.dump -s 43b0d543b602348229637c7fc3f4f41d --initial-gen-method mix --initial-ind-length 20 --z-star 0,0,0,-50 --scale 100,1000000000,100,100 --save-solutions solutions.8.64.100.dump

You can then view the solution like so:

    $ cargo run --release --bin print-solutions ../OpenSubtitles2018/xml/fi/1991 solutions.8.64.100.dump 1991.dump

For example, for the above solution #2 is

    1: Siinä .
    2: Hienoo
    3: Ei .
    4: Kuule
    5: Tulevaisuuden
    6: Ei !
    7: Selvä .
    8: Hei .
    9: Herätys !
    10: Aamen .
    11: Liikettä !
    12: Salaman .
    13: Takanasi !
    14: Hienoa .
    15: Barnabas !
    16: Älä komentele .
    17: Hitto !
    18: Ei hätää .
    19: ― En .
    20: Kuulen sinut !
