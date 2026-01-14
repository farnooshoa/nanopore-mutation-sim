from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from Bio.PDB import PDBParser

VDW = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47,
    "P": 1.80, "S": 1.80, "CL": 1.75, "BR": 1.85, "I": 1.98
}

def infer_element(atom) -> str:
    el = (atom.element or "").strip().upper()
    if el:
        return el
    name = atom.get_name().strip().upper()
    if len(name) >= 2 and name[:2] in {"CL", "BR"}:
        return name[:2]
    return name[0] if name else "C"

def load_coords(pdb_path: Path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pore", str(pdb_path))

    coords, elements = [], []
    for atom in structure.get_atoms():
        el = infer_element(atom)
        if el == "H":
            continue
        coords.append(atom.coord.astype(float))
        elements.append(el)

    coords = np.asarray(coords, dtype=float)
    elements = np.asarray(elements, dtype=object)
    if coords.shape[0] < 1000:
        raise ValueError("Too few atoms loaded—did the PDB parse correctly?")
    return coords, elements

def pca_axis(coords: np.ndarray) -> np.ndarray:
    X = coords - coords.mean(axis=0, keepdims=True)
    cov = np.cov(X.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)]
    return axis / np.linalg.norm(axis)

def radius_profile(coords: np.ndarray, elements: np.ndarray, n_points: int = 250):
    axis = pca_axis(coords)
    center = coords.mean(axis=0)

    proj = (coords - center) @ axis
    z_min, z_max = float(proj.min()), float(proj.max())

    zs = np.linspace(z_min, z_max, n_points)
    sample_points = center + np.outer(zs, axis)

    tree = cKDTree(coords)
    dists, idx = tree.query(sample_points, k=1)

    nearest_elements = elements[idx]
    vdw = np.array([VDW.get(str(e), 1.70) for e in nearest_elements], dtype=float)
    radii = np.maximum(dists - vdw, 0.0)

    return zs, radii

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdb_path", type=str)
    ap.add_argument("--out_prefix", type=str, default="outputs/out")
    ap.add_argument("--n_points", type=int, default=250)
    args = ap.parse_args()

    pdb_path = Path(args.pdb_path)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    coords, elements = load_coords(pdb_path)
    zs, radii = radius_profile(coords, elements, n_points=args.n_points)

    csv_path = Path(out_prefix.as_posix() + "_radius_profile.csv")
    csv_path.write_text("z_along_axis_A,radius_A\n" + "\n".join(
        f"{z:.4f},{r:.4f}" for z, r in zip(zs, radii)
    ))
    print(f"Saved: {csv_path}")

    plt.figure()
    plt.plot(zs, radii)
    plt.xlabel("z along inferred pore axis (Å)")
    plt.ylabel("approx free radius (Å)")
    plt.title(f"Radius profile (v0): {pdb_path.name}")
    png_path = Path(out_prefix.as_posix() + "_radius_profile.png")
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {png_path}")

if __name__ == "__main__":
    main()
