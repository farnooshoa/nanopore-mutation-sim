import sys
from pathlib import Path
import requests

def download_pdb(pdb_id: str, out_path: Path) -> Path:
    pdb_id = pdb_id.upper()
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    r = requests.get(url, timeout=30)
    r.raise_for_status()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(r.content)
    return out_path

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/download_pdb.py <PDB_ID>  (e.g., 7AHL)")
        raise SystemExit(2)

    pdb_id = sys.argv[1]
    out_file = Path("data/pdb") / f"{pdb_id.upper()}.pdb"
    path = download_pdb(pdb_id, out_file)
    print(f"Saved: {path}")
