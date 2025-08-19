

A comprehensive implementation of **_Matrix Decomposition Techniques_**, forming the foundations of various scientific and engineering applications. 

To gain a deeper understanding of how Orthogonalization & matrices Decomposition works in real-life applications, like,
- **Signal Processing**
- **Control Systems and Robotics**
- **Solving Linear Systems i.e. *AX = B***

With mathematical intuition & visual introspection, the project makes most of the abstract concepts easier to grasp and connect with practical applications.

> This is the first draft of the project. More implementations—such as **Householder Reflection**, **Bidiagonalization**, **LU Decomposition**, and others—will be added soon.

## Features / What's Inside

For a vector space, building *Orthonormal Basis* is the most essential part of Linear Algebra. 

You can find top discussions about certain topics like Orthogonality, and why do we need it, ....here.

#### Gram-Schmidt Orthogonalization

One of the technique for creating Orthonormal Basis from a vector space (*assuming linear independence*) by using a Vector Projection Elimination method. 

Here's a snippet,


## 🧪 Testing

To test the export process, run `.github/scripts/build.py` from the root directory.

```bash
uv run .github/scripts/build.py
```

This will export all notebooks in a folder called `_site/` in the root directory. Then to serve the site, run:

```bash
python -m http.server -d _site
```

This will serve the site at `http://localhost:8000`.
