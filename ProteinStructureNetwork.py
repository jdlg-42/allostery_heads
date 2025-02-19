from Bio import PDB
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple

class ProteinStructureNetwork:
    """
    Creates and analyzes protein structure networks from PDB files
    """
    
    def __init__(self, distance_threshold: float = 5.0):
        """
        Initialize PSN builder
        
        Args:
            distance_threshold: Distance cutoff for considering residue contacts (Angstroms)
        """
        self.distance_threshold = distance_threshold
        self.parser = PDB.PDBParser(QUIET=True)
        self.structure = None
        self.network = None
        
    def build_network(self, pdb_file: str, chain_id: str = 'A') -> nx.Graph:
        """
        Build protein structure network from PDB file
        
        Args:
            pdb_file: Path to PDB file
            chain_id: Chain identifier to process
            
        Returns:
            NetworkX graph representing the protein structure network
        """
        # Load structure
        structure = self.parser.get_structure('protein', pdb_file)
        self.structure = structure
        
        # Create graph
        G = nx.Graph()
        
        # Get chain
        chain = structure[0][chain_id]
        
        # Add nodes (residues)
        for residue in chain:
            if PDB.is_aa(residue):
                res_id = residue.get_id()[1]  # Residue number
                G.add_node(res_id, 
                          amino_acid=PDB.Polypeptide.three_to_one(residue.get_resname()),
                          position=residue['CA'].get_coord())  # Use CA coordinates
        
        # Add edges based on distance threshold
        residue_list = [r for r in chain if PDB.is_aa(r)]
        for i, res1 in enumerate(residue_list):
            for res2 in residue_list[i+1:]:
                distance = self._calculate_residue_distance(res1, res2)
                if distance <= self.distance_threshold:
                    G.add_edge(res1.get_id()[1], 
                             res2.get_id()[1], 
                             distance=distance)
        
        self.network = G
        return G
    
    def _calculate_residue_distance(self, res1: PDB.Residue, res2: PDB.Residue) -> float:
        """Calculate distance between CA atoms of two residues"""
        return np.linalg.norm(res1['CA'].get_coord() - res2['CA'].get_coord())
    
    def get_attention_annotated_network(self, head_idx: int, 
                                      attention_maps: torch.Tensor) -> nx.Graph:
        """
        Add attention scores as edge attributes to the network
        
        Args:
            head_idx: Index of attention head to use
            attention_maps: Attention maps from ESM2
            
        Returns:
            Network with attention scores as edge weights
        """
        if self.network is None:
            raise ValueError("Must build network first using build_network()")
            
        G = self.network.copy()
        
        # Get attention map for specific head
        attention = attention_maps[0, 0, head_idx].cpu().numpy()
        
        # Add attention scores to existing edges
        for u, v in G.edges():
            G[u][v]['attention'] = attention[u-1, v-1]  # -1 for 0-based indexing
            
        return G