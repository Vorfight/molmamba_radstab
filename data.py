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
from rdkit.Chem import rdMolDescriptors, Descriptors, Crippen, Lipinski, rdmolops
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

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
    try:
        # стандартная санитизация; RDKit сам корректно выставит ароматичность
        mol = Chem.MolFromSmiles(smi, sanitize=True)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)  # на всякий случай, если SMILES кривоват
        return mol
    except Exception:
        return None

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
# Numeric features (separate for molecule and solvent)
# -----------------------------
def numeric_features_mol(mol: Optional[Chem.Mol]) -> np.ndarray:
    """RDKit descriptors of the main molecule ONLY (no solvent, no diel_const)."""
    if mol is None:
        return np.zeros((len(RDKit_DESC_FUNCS),), dtype=np.float32)
    desc = rdkit_descriptor_vector(mol)
    return np.asarray(desc, dtype=np.float32)


def numeric_features_solv(solv_mol: Optional[Chem.Mol], diel_const: float) -> np.ndarray:
    """RDKit descriptors of solvent PLUS diel_const as the last element."""
    if solv_mol is None:
        desc = [0.0] * len(RDKit_DESC_FUNCS)
    else:
        desc = rdkit_descriptor_vector(solv_mol)
    env = [safe_float(diel_const)]  # appended at the end
    return np.asarray(desc + env, dtype=np.float32)

# -----------------------------
# Dataset
# -----------------------------
@dataclass
class Item:
    atom_data: Data
    frag_data: Data
    # Combined numeric features: [num_mol | num_solv] where
    #   num_mol  = RDKit descriptors of main molecule
    #   num_solv = RDKit descriptors of solvent + [diel_const]
    num_feats_mol: Tensor      # shape [D_mol]
    num_feats_solv: Tensor     # shape [D_solv]
    conc: Tensor               # shape [1] (standardize outside)
    y: Optional[Tensor]
    meta: Dict

class RadiationDataset(Dataset):
    """
    Ожидаемый CSV с колонками:
      - smiles (str)
      - solvent_smiles (str) [может быть пустым, пока не используется в признаках]
      - diel_const (float)
      - concentration (float)
      - dose_constant (float)  # таргет
    """
    def __init__(self, csv_path: str):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        # Требуем только SMILES молекулы; остальные колонки опциональны и заполняются безопасными значениями
        if "smiles" not in self.df.columns:
            raise ValueError("CSV must contain column 'smiles'")
        if "solvent_smiles" not in self.df.columns:
            self.df["solvent_smiles"] = ""
        if "diel_const" not in self.df.columns:
            self.df["diel_const"] = 0.0
        if "concentration" not in self.df.columns:
            self.df["concentration"] = 0.0
        if "dose_constant" not in self.df.columns:
            # таргет может отсутствовать на инференсе/SSL-предобучении
            self.df["dose_constant"] = np.nan

        # Expose sizes of NUM splits
        self.num_mol_dim = len(RDKit_DESC_FUNCS)
        self.num_solv_dim = len(RDKit_DESC_FUNCS) + 1  # solvent RDKit + diel_const

        self._cache: List[Optional[Item]] = [None] * len(self.df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Item:
        cached = self._cache[idx]
        if cached is not None:
            return cached

        row = self.df.iloc[idx]
        smi = str(row["smiles"])
        diel = safe_float(row.get("diel_const", 0.0))
        conc = safe_float(row.get("concentration", 0.0))
        y_raw = row.get("dose_constant", np.nan)
        solv = str(row.get("solvent_smiles", ""))
        y_tensor: Optional[Tensor]
        if pd.isna(y_raw):
            y_tensor = None
        else:
            try:
                y_tensor = torch.tensor([float(y_raw)], dtype=torch.float)
            except Exception:
                y_tensor = None

        mol = smiles_to_mol(smi)
        solv_mol = smiles_to_mol(solv) if isinstance(solv, str) and solv.strip() else None
        if mol is None:
            # создадим пустой dummy-граф, чтобы не падать
            atom_data = Data(x=torch.zeros((1, len(ATOM_LIST) + 4 + len(HYBRIDIZATIONS) + 4), dtype=torch.float),
                             edge_index=torch.tensor([[0], [0]], dtype=torch.long),
                             edge_attr=torch.ones(1, 6, dtype=torch.float))
            frag_data = Data(x=torch.zeros((1, len(RDKit_DESC_FUNCS)), dtype=torch.float),
                             edge_index=torch.tensor([[0], [0]], dtype=torch.long),
                             edge_attr=torch.ones(1, 1, dtype=torch.float))
        else:
            atom_data = mol_to_atom_graph(mol)
            frag_data = mol_to_fragment_graph(mol)

        numf_mol = numeric_features_mol(mol)
        numf_solv = numeric_features_solv(solv_mol, diel_const=diel)

        item = Item(
            atom_data=atom_data,
            frag_data=frag_data,
            num_feats_mol=torch.from_numpy(numf_mol),
            num_feats_solv=torch.from_numpy(numf_solv),
            conc=torch.tensor([conc], dtype=torch.float),
            y=y_tensor,
            meta={"smiles": smi, "solvent_smiles": solv, "diel_const": diel, "concentration": conc}
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
    num_feats_mol: Tensor   # [B, D_mol]
    num_feats_solv: Tensor  # [B, D_solv]
    conc: Tensor            # [B, 1]
    y: Optional[Tensor]     # [B, 1]
    meta: List[Dict]

def collate_batch(items: List[Item]) -> BatchPack:
    atom_list = [it.atom_data for it in items]
    frag_list = [it.frag_data for it in items]
    atom_batch = Batch.from_data_list(atom_list)
    frag_batch = Batch.from_data_list(frag_list)
    num_feats_mol = torch.stack([it.num_feats_mol for it in items], dim=0)
    num_feats_solv = torch.stack([it.num_feats_solv for it in items], dim=0)
    conc = torch.stack([it.conc for it in items], dim=0).view(-1, 1)  # [B,1]
    has_y = all(it.y is not None for it in items)
    y = torch.stack([it.y for it in items], dim=0) if has_y else None
    meta = [it.meta for it in items]
    return BatchPack(atom_batch=atom_batch, frag_batch=frag_batch,
                     num_feats_mol=num_feats_mol, num_feats_solv=num_feats_solv,
                     conc=conc, y=y, meta=meta)

# -----------------------------
# Optional: feature standardizer (для числовых признаков)
# -----------------------------
class FeatureStandardizer:
    """Простая стандартизация (x - mean) / std для произвольного набора тензоров.
    Можно вызывать fit на датасете по имени поля (num_feats_mol/num_feats_solv). Хранит параметры.
    """
    def __init__(self):
        self.mean_: Optional[Tensor] = None
        self.std_: Optional[Tensor] = None

    def fit(self, dataset: Optional[Dataset] = None, indices: Optional[List[int]] = None, field: str = None, X: Optional[Tensor] = None):
        """Либо передайте X (Tensor [N, F]), либо dataset+field (строка: 'num_feats_mol' или 'num_feats_solv')."""
        if X is None:
            if dataset is None or field is None:
                raise ValueError("Provide either X tensor or (dataset and field) to fit FeatureStandardizer")
            if indices is None:
                indices = list(range(len(dataset)))
            xs = []
            for idx in indices:
                it = dataset[idx]
                val = getattr(it, field)
                xs.append(val)
            X = torch.stack(xs, 0).float()
        else:
            if not torch.is_tensor(X):
                X = torch.tensor(X, dtype=torch.float)
        self.mean_ = X.mean(dim=0)
        self.std_ = X.std(dim=0).clamp_min(1e-8)
        return self

    def transform(self, x: Tensor) -> Tensor:
        assert self.mean_ is not None and self.std_ is not None, "Call fit() first or set mean_/std_ manually"
        return (x - self.mean_) / self.std_

    def fit_transform(self, dataset: Optional[Dataset] = None, indices: Optional[List[int]] = None, field: str = None, X: Optional[Tensor] = None) -> Tensor:
        self.fit(dataset=dataset, indices=indices, field=field, X=X)
        if X is None:
            if dataset is None or field is None:
                raise ValueError("Provide either X tensor or (dataset and field) to fit_transform FeatureStandardizer")
            if indices is None:
                indices = list(range(len(dataset)))
            xs = []
            for idx in indices:
                it = dataset[idx]
                val = getattr(it, field)
                xs.append(val)
            X = torch.stack(xs, 0).float()
        return self.transform(X)

# -----------------------------
# Expose helper for scaffold-based CV (используем в utils.py)
# -----------------------------
def scaffold_for_row(smiles: str) -> str:
    """Bemis–Murcko scaffold SMILES (без хиральности)."""
    return get_scaffold_smiles(smiles, include_chirality=False)