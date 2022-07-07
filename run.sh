#!/bin/sh
del ana.pkl
python feeder.py
python simple_wigner.py > out.txt