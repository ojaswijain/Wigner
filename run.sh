#!/bin/sh
del ana.pkl
python gen.py
python simple_wigner.py > out.txt