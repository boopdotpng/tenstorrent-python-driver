# blackhole-py
A full replacement of **tt-metal** and **tt-umd**, written in Python. 

## requirements 
- **tt-kmd**

## instructions 
Run `setup-deps.sh` to install the tarball containing the risc-v compiler, include headers, and libraries. Previously, you had to build a whole copy of `tt-metal` to get these headers, but the tarball contains the minimal set of files required to run all tt-metal kernels. 


Currently, I have add1 and a very naive, slow matmul working. 
