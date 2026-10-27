import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
import pandas as pd

def plot_genotype_network_landscape(G: nx.Graph, df: pd.DataFrame, fitness_col: str, dose_val: float):
    """
    Plots the genotype transition network overlaying fitness values under a specific drug environment.
    """
    plt.figure(figsize=(10, 7))
    pos = nx.spring_layout(G, seed=42)
    
    fitness_map = dict(zip(df['genotype'], df[fitness_col]))
    node_colors = [fitness_map[node] for node in G.nodes()]
    
    nodes = nx.draw_networkx_nodes(
        G, pos, 
        node_color=node_colors, 
        cmap=plt.cm.viridis, 
        node_size=700,
        alpha=0.9
    )
    
    nx.draw_networkx_edges(G, pos, alpha=0.4, edge_color='gray', width=1.5)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', font_color='white')
    
    cbar = plt.colorbar(nodes)
    cbar.set_label('Cellular Fitness / Survival Rate', rotation=270, labelpad=15)
    
    plt.title(f"Pharmacogenomic Fitness Landscape at Drug Dose: {dose_val:.2f} µM", fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('fitness_network_landscape.png', dpi=300)
    plt.close()

def plot_seascape_heatmap(seascape_df: pd.DataFrame, drug_concentrations: np.ndarray):
    """
    Generates a Seascape Heatmap illustrating how fitness rankings change dynamically across drug doses.
    """
    plt.figure(figsize=(12, 8))
    
    fitness_cols = [f"fitness_dose_{conc:.2f}" for conc in drug_concentrations]
    heatmap_data = seascape_df.set_index('genotype')[fitness_cols]
    
    sns.heatmap(
        heatmap_data, 
        cmap="YlGnBu", 
        annot=True, 
        fmt=".2f", 
        cbar_kws={'label': 'Fitness (0 = Lethal, 1 = Max Growth)'},
        xticklabels=[f"{c:.1f}" for c in drug_concentrations]
    )
    
    plt.title("Pharmacogenomic Fitness Seascape Across Drug Concentrations", fontsize=14, fontweight='bold')
    plt.xlabel("Drug Concentration (µM)", fontsize=12)
    plt.ylabel("Genotype Sequence", fontsize=12)
    plt.tight_layout()
    plt.savefig('fitness_seascape_heatmap.png', dpi=300)
    plt.close()