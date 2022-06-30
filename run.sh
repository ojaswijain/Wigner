#!/bin/sh
del ana.pkl
python feeder.py
python wigner.py > out2.txt