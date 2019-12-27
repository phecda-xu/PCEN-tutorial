# PCEN

    This repository contains static pcen and trainable pcen based on tensorflow.
    Note: trainable pcen never test in any architecture.

## About pcen

Per-channel energy normalization (PCEN), an acoustic frontend of AGC. Proposed in [Trainable Frontend For Robust and Far-Field Keyword Spotting](https://arxiv.org/pdf/1607.05666v1.pdf)
Learn from [Per-Channel Energy Normalization: Why and How]() for more information.

## Enviroment

```
ubuntu 16.04 LTS
python 3.5
package version see in requirements.txt
```

## usage

```
from NetWork import RPCEN
pcen = RPCEN(trainable=True)
out = pcen.gen_pcen(input)

default trainable means getting the trainable layer.
```

```
from NetWork import FPCEN
pcen = FPCEN()
out = pcen.gen_pcen(input)

use two IIRs with s= 0.015 and s=0.08
```

- Note:

    we initial the four parameters with the static value,
    so in Main.py you will get the same results, with dtype is 'float64'.
    when dtype is 'float32' there is sligthly difference between the two results.

## Reference

- [pytorch-pcen](https://github.com/daemon/pytorch-pcen)
- [static pcen](https://github.com/librosa/librosa/issues/615)

## TODO

Test trainable pcen layer in kws.