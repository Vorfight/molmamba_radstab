# data.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors, Crippen, Lipinski, rdMolOps, rdmolops
from rdkit.Chem.Scaffolds import MurckoScaffold

from torch_geometric.data import Data, Batch

# -----------------------------
# Config
# -----------------------------
ATOM_LIST = list(range(1, 119))  # H..Og
HYBRIDIZATIONS = [
    Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3,
    Chem.HybridizationType.SP3D, Chem.HybridizationType.SP3D2
]
BOND_TYPES = [
    Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC
]

RDKit_DESC_FUNCS = [
    Descriptors.MolWt,
    Crippen.MolLogP,
    Descriptors.TPSA,
    Lipinski.NumHDonors,
    Lipinski.NumHAcceptors,
    Descriptors.NumRotatableBonds,
    Descriptors.FractionCSP3,
    rdMolDescriptors.CalcNumRings,
    rdMolDescriptors.CalcNumAromaticRings,
    Descriptors.BertzCT,
    Descriptors.MolMR,
]

# -----------------------------
# Utilities
# -----------------------------
def one_hot_encode(x, choices):
    vec = [0] * len(choices)
    try:
        idx = choices.index(x)
        vec[idx] = 1
    except ValueError:
        pass
    return vec

def safe_float(x: float) -> float:
    if x is None or np.isnan(x) or np.isinf(x):
        return 0.0
    return float(x)

def smiles_to_mol(smi: str) -> Optional[Chem.Mol]:
    if not isinstance(smi, str) or not smi.strip():
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    Chem.Kekulize(mol, clearAromaticFlags=False)
    return mol

def get_scaffold_smiles(smi: str, include_chirality: bool = False) -> str:
    mol = smiles_to_mol(smi)
    if mol is None:
        return ""
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaf, isomericSmiles=include_chirality) if scaf is not None else ""

# -----------------------------
# Atom/Bond featurization
# -----------------------------
def atom_features(atom: Chem.Atom) -> List[float]:
    Z = atom.GetAtomicNum()
    degree = atom.GetDegree()
    formal_charge = atom.GetFormalCharge()
    chiral_tag = int(atom.GetChiralTag())
    radical_e = atom.GetNumRadicalElectrons()
    hybrid = atom.GetHybridization()
    aromatic = int(atom.GetIsAromatic())
    total_h = atom.GetTotalNumHs()
    in_ring = int(atom.IsInRing())
    mass = atom.GetMass() * 1e-2  # scale
    v = []
    v += one_hot_encode(Z, ATOM_LIST)
    v += [degree, formal_charge, chiral_tag, radical_e]
    v += one_hot_encode(hybrid, HYBRIDIZATIONS)
    v += [aromatic, total_h, in_ring, mass]
    return [safe_float(x) for x in v]

def bond_features(bond: Chem.Bond) -> List[float]:
    if bond is None:
        # Self-loop placeholder
        return [1, 0, 0, 0, 0, 0]
    btype = bond.GetBondType()
    conj = int(bond.GetIsConjugated())
    in_ring = int(bond.IsInRing())
    v = one_hot_encode(btype, BOND_TYPES)
    v += [conj, in_ring]
    return [safe_float(x) for x in v]

def mol_to_atom_graph(mol: Chem.Mol) -> Data:
    assert mol is not None
    N = mol.GetNumAtoms()
    atom_feat = [atom_features(mol.GetAtomWithIdx(i)) for i in range(N)]
    x = torch.tensor(atom_feat, dtype=torch.float)

    edge_index = []
    edge_attr = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edge_index.append([i, j])
        edge_attr.append(bf)
        edge_index.append([j, i])
        edge_attr.append(bf)

    # Добавим self-loops с простыми признаками
    for i in range(N):
        edge_index.append([i, i])
        edge_attr.append(bond_features(None))

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

# -----------------------------
# Fragment graph (ring systems)
# -----------------------------
def _ring_atom_sets(mol: Chem.Mol) -> List[List[int]]:
    """Возвращает список колец (в виде списков атомных индексов)."""
    ri = mol.GetRingInfo()
    atom_rings = list(ri.AtomRings())
    # Объединим кольца в «кольцевые системы» (если кольца пересекаются по атомам)
    systems: List[set] = []
    for ring in atom_rings:
        rset = set(ring)
        merged = False
        for s in systems:
            if len(s.intersection(rset)) > 0:
                s.update(rset)
                merged = True
                break
        if not merged:
            systems.append(set(rset))
    return [sorted(list(s)) for s in systems] if systems else []

def submol_from_atom_indices(mol: Chem.Mol, atom_ids: List[int]) -> Optional[Chem.Mol]:
    if not atom_ids:
        return None
    try:
        emol = Chem.PathToSubmol(mol, atom_ids, useQuery=False)
        Chem.SanitizeMol(emol)
        return emol
    except Exception:
        return None

def rdkit_descriptor_vector(mol: Chem.Mol) -> List[float]:
    vals = []
    for fn in RDKit_DESC_FUNCS:
        try:
            vals.append(safe_float(fn(mol)))
        except Exception:
            vals.append(0.0)
    return vals

def mol_to_fragment_graph(mol: Chem.Mol) -> Data:
    """Фрагменты = кольцевые системы. Узловые признаки — RDKit-дескрипторы фрагмента.
       Если колец нет, один узел = дескрипторы всей молекулы.
       Рёбра: полный неориентированный граф между фрагментами (простая, но стабильная связность).
    """
    rings = _ring_atom_sets(mol)
    frag_mols: List[Chem.Mol] = []
    if len(rings) == 0:
        frag_mols = [mol]
    else:
        for atom_ids in rings:
            sm = submol_from_atom_indices(mol, atom_ids)
            if sm is not None and sm.GetNumAtoms() > 0:
                frag_mols.append(sm)
        if len(frag_mols) == 0:
            frag_mols = [mol]

    x = torch.tensor([rdkit_descriptor_vector(fm) for fm in frag_mols], dtype=torch.float)

    F = len(frag_mols)
    edge_index = []
    edge_attr = []
    if F == 1:
        # self-loop
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        edge_attr = torch.ones(1, 1, dtype=torch.float)
    else:
        for i in range(F):
            for j in range(i + 1, F):
                edge_index.append([i, j])
                edge_index.append([j, i])
                edge_attr.append([1.0])
                edge_attr.append([1.0])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# -----------------------------
# Numeric features (environment + RDKit of main mol)
# -----------------------------
def numeric_features(mol: Chem.Mol, diel_const: float, conc_stand: float) -> np.ndarray:
    desc = rdkit_descriptor_vector(mol)
    env = [safe_float(diel_const), safe_float(conc_stand)]
    return np.asarray(env + desc, dtype=np.float32)

# -----------------------------
# Dataset
# -----------------------------
@dataclass
class Item:
    atom_data: Data
    frag_data: Data
    num_feats: Tensor  # shape [D]
    y: Optional[Tensor]
    meta: Dict

class RadiationDataset(Dataset):
    """
    Ожидаемый CSV с колонками:
      - smiles (str)
      - solv_smiles (str) [может быть пустым, пока не используется в признаках]
      - diel_const (float)
      - conc_stand (float)
      - dc_stand (float)  # таргет
    """
    def __init__(self, csv_path: str):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        required = ["smiles", "diel_const", "conc_stand", "dc_stand"]
        for col in required:
            if col not in self.df.columns:
                raise ValueError(f"CSV must contain column '{col}'")
        # Опционально: solv_smiles
        if "solv_smiles" not in self.df.columns:
            self.df["solv_smiles"] = ""

        self._cache: List[Optional[Item]] = [None] * len(self.df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Item:
        cached = self._cache[idx]
        if cached is not None:
            return cached

        row = self.df.iloc[idx]
        smi = str(row["smiles"])
        diel = float(row["diel_const"])
        conc = float(row["conc_stand"])
        yval = float(row["dc_stand"])
        solv = str(row.get("solv_smiles", ""))

        mol = smiles_to_mol(smi)
        if mol is None:
            # создадим пустой dummy-граф, чтобы не падать
            atom_data = Data(x=torch.zeros((1, len(ATOM_LIST) + 4 + len(HYBRIDIZATIONS) + 4), dtype=torch.float),
                             edge_index=torch.tensor([[0], [0]], dtype=torch.long),
                             edge_attr=torch.ones(1, 6, dtype=torch.float))
            frag_data = Data(x=torch.zeros((1, len(RDKit_DESC_FUNCS)), dtype=torch.float),
                             edge_index=torch.tensor([[0], [0]], dtype=torch.long),
                             edge_attr=torch.ones(1, 1, dtype=torch.float))
            numf = np.zeros((2 + len(RDKit_DESC_FUNCS),), dtype=np.float32)
        else:
            atom_data = mol_to_atom_graph(mol)
            frag_data = mol_to_fragment_graph(mol)
            numf = numeric_features(mol, diel_const=diel, conc_stand=conc)

        item = Item(
            atom_data=atom_data,
            frag_data=frag_data,
            num_feats=torch.from_numpy(numf),
            y=torch.tensor([yval], dtype=torch.float),
            meta={"smiles": smi, "solv_smiles": solv, "diel_const": diel, "conc_stand": conc}
        )
        self._cache[idx] = item
        return item

# -----------------------------
# Collate for DataLoader
# -----------------------------
@dataclass
class BatchPack:
    atom_batch: Batch
    frag_batch: Batch
    num_feats: Tensor  # [B, D]
    y: Optional[Tensor]  # [B, 1]
    meta: List[Dict]

def collate_batch(items: List[Item]) -> BatchPack:
    atom_list = [it.atom_data for it in items]
    frag_list = [it.frag_data for it in items]
    atom_batch = Batch.from_data_list(atom_list)
    frag_batch = Batch.from_data_list(frag_list)
    num_feats = torch.stack([it.num_feats for it in items], dim=0)
    y = torch.stack([it.y for it in items], dim=0) if items[0].y is not None else None
    meta = [it.meta for it in items]
    return BatchPack(atom_batch=atom_batch, frag_batch=frag_batch, num_feats=num_feats, y=y, meta=meta)

# -----------------------------
# Optional: feature standardizer (для числовых признаков)
# -----------------------------
class FeatureStandardizer:
    """Простая стандартизация (x - mean) / std для num_feats. Хранит параметры."""
    def __init__(self):
        self.mean_: Optional[Tensor] = None
        self.std_: Optional[Tensor] = None

    def fit(self, dataset: RadiationDataset, indices: Optional[List[int]] = None):
        xs = []
        if indices is None:
            indices = list(range(len(dataset)))
        for idx in indices:
            xs.append(dataset[idx].num_feats)
        X = torch.stack(xs, 0)
        self.mean_ = X.mean(dim=0)
        self.std_ = X.std(dim=0).clamp_min(1e-6)
        return self

    def transform(self, num_feats: Tensor) -> Tensor:
        assert self.mean_ is not None and self.std_ is not None, "Call fit() first"
        return (num_feats - self.mean_) / self.std_

    def fit_transform(self, dataset: RadiationDataset, indices: Optional[List[int]] = None) -> List[Tensor]:
        self.fit(dataset, indices)
        xs = []
        if indices is None:
            indices = list(range(len(dataset)))
        for idx in indices:
            xs.append(self.transform(dataset[idx].num_feats))
        return xs

# -----------------------------
# Expose helper for scaffold-based CV (используем в utils.py)
# -----------------------------
def scaffold_for_row(smiles: str) -> str:
    """Bemis–Murcko scaffold SMILES (без хиральности)."""
    return get_scaffold_smiles(smiles, include_chirality=False)