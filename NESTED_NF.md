# Nested NF — split one Gaussian bucket, don’t throw the 16 away
NF4’s 16 codes are the **first description** (equal-mass bins of a normal). 
The plug is **which sub-bin inside that cell**, not IEEE 12 LSBs, not a new uniform lattice.
```
one NF4 cell --16-way Gaussian split--> 16 sub-cells
16 × 16 = 256 = NF8
another split → toward BF16 residual
```
Hole (VRAM, H-TILE qpack) = NF4 nibble. 
Plug (RAM, 4 bits/w) = sub-index 0..15. 
Inflate = `NF8_CELLS[nf4][plug] * absmax`.
Lossless **vs NF8** if the plug is stored. Not bit-exact BF16; that leftover is the next plane.
Toy Gaussian 4096: NF4 RMSE 1.85e-3; nested NF8 1.29e-4; uniform INT8 1.14e-4.
`python3 nested_nf.py`
