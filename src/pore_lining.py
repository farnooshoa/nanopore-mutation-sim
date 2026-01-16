from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser

def infer_element(atom) -> str:
    el = (atom.element or "").strip().upper()
    if el:
        return el
    name = atom.get_name().strip().upper()
    if len(name) >= 2 and name[:2] in {"CL", "BR"}:
        return name[:2]
    return name[0] if name else "C"

def pca_axis(coords: np.ndarray) -> np.ndarray:
    X = coords - coords.mean(axis=0, keepdims=True)
    cov = np.cov(X.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)]
    return axis / np.linalg.norm(axis)

def point_line_dist_and_z(p: np.ndarray, center: np.ndarray, axis: np.ndarray):
    v = p - center
    z = float(v @ axis)
    perp = v - z * axis
    d = float(np.linalg.norm(perp))
    return d, z

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdb_path", type=str)
    ap.add_argument("--cutoff_A", type=float, default=6.0, help="pore-lining cutoff (Angstrom)")
    ap.add_argument("--out_prefix", type=str, default="outputs/7AHL")
    args = ap.parse_args()

    pdb_path = Path(args.pdb_path)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pore", str(pdb_path))

    # gather all heavy-atom coords for axis inference
    all_coords = []
    for atom in structure.get_atoms():
        if infer_element(atom) == "H":
            continue
        all_coords.append(atom.coord.astype(float))
    all_coords = np.asarray(all_coords, dtype=float)

    center = all_coords.mean(axis=0)
    axis = pca_axis(all_coords)

    rows = []
    for model in structure:
        for chain in model:
            for res in chain:
                if res.id[0] != " ":  # skip hetero/water
                    continue
                res_atoms = [a for a in res.get_atoms() if infer_element(a) != "H"]
                if not res_atoms:
                    continue

                best_d = 1e9
                best_z = None
                best_atom = None
                for a in res_atoms:
                    d, z = point_line_dist_and_z(a.coord.astype(float), center, axis)
                    if d < best_d:
                        best_d = d
                        best_z = z
                        best_atom = a.get_name().strip()

                if best_d <= args.cutoff_A:
                    resid = res.id[1]
                    icode = res.id[2].strip()
                    rows.append([chain.id, res.get_resname(), resid, icode, best_atom, best_d, best_z])

    if not rows:
        raise SystemExit("No pore-lining residues found. Try increasing --cutoff_A")

    # sort by z along pore axis
    rows.sort(key=lambda r: r[-1])

    csv_path = Path(out_prefix.as_posix() + "_pore_lining_residues.csv")
    header = "chain,resname,resid,icode,closest_atom,min_dist_to_axis_A,z_along_axis_A\n"
    csv_path.write_text(header + "\n".join(
        f"{c},{rn},{rid},{ic},{atm},{d:.4f},{z:.4f}"
        for c, rn, rid, ic, atm, d, z in rows
    ))
    print(f"Saved: {csv_path}")

    # plot: z vs min distance to axis
    zs = np.array([r[-1] for r in rows], dtype=float)
    ds = np.array([r[-2] for r in rows], dtype=float)

    plt.figure()
    plt.scatter(zs, ds, s=10)
    plt.xlabel("z along inferred pore axis (Å)")
    plt.ylabel("min atom distance to axis (Å)")
    plt.title(f"Pore-lining residues (cutoff {args.cutoff_A} Å): {pdb_path.name}")
    png_path = Path(out_prefix.as_posix() + "_pore_lining_residues.png")
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {png_path}")

if __name__ == "__main__":
    main()
