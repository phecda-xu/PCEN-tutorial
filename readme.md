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

## Static pcen

```
Main.py

function staticPcen(sig, sr)
```

## Trainable pcen

```
Main.py

function trainablePcen(sig, sr)
```


## Reference

- [pytorch-pcen](https://github.com/daemon/pytorch-pcen)
- [static pcen](https://github.com/librosa/librosa/issues/615)

## TODO

Test trainable pcen layer in kws.