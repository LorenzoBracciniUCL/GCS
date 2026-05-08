# GCS 
## Superposition of Quantum Gaussian Processes: Gaussian Cat States
### Main Reference: [arXiv:2510.01156](https://arxiv.org/abs/2510.01156)

This methodology generalized the covariance matrix formalism of Quantum Optics for n modes to inculde interactions with N qubits, resulting in superpositions of quantum gaussian processes, i.e. dynamics and measurements of Gaussian-branched Cat States (qubits-modes entangled state). 

The libabry in Python implements all the Tables given in our [paper](https://arxiv.org/abs/2510.01156), thus, solving the non-linear quantum dynamics, both closed and open, via numerical implementations (Table 1 and 3) and, when possible, semi-analythical implementations (Table 2). Measurements of the qubits and modes are possible. The library has built-in functions for visualization and plotting, for instance, generating animation of time-dependent Wigner functions. 

Furthermore, a Mathematica notebooks is provided to implement the anlaytical solutions of linear qubit-mode dynamcis (table 2 and 5).

We provide the following Notebooks:
- Exapmple_Unitary: examples of all possible Python implementations of Unitary Dynamcis
- Exapmple_Open: examples of all possible Python implementations of Open Dynamcis
- Example_Animation_Wigner: examples of plotting functions of the first moments, qubit resduced density matrix and Animation of Wigner function, both for the statistical mixture of Gaussian process and post-measurement states (with wigner negativities).
- Exapmple_Paper: The two examples presented in the [paper](https://arxiv.org/abs/2510.01156): (a) Measurement-Based Entanglement between two qubits with a mediating resonator, and (b) a Stern-Gerlach Interferometer of a levitated mass. 
