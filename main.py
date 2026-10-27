import numpy as np
from src.data_loader import generate_synthetic_pharmacogenomic_dataset, calculate_fitness_seascape
from src.network_builder import build_genotype_network
from src.fitness_landscape import plot_genotype_network_landscape, plot_seascape_heatmap

def main():
    print("Executing Computational Pharmacogenomic Seascape Analysis Pipeline...")
    
    # 1. Define drug dosage gradient (Environment)
    drug_doses = np.linspace(0.1, 10.0, 10)
    
    # 2. Load dataset & calculate fitness seascape
    df = generate_synthetic_pharmacogenomic_dataset(num_variants=16)
    seascape_df = calculate_fitness_seascape(df, drug_doses)
    
    # 3. Construct Network Graph
    G = build_genotype_network(seascape_df)
    print(f"Graph built with {G.number_of_nodes()} genotype nodes and {G.number_of_edges()} mutation transitions.")
    
    # 4. Generate Visualizations
    selected_dose_col = f"fitness_dose_{drug_doses[4]:.2f}"
    plot_genotype_network_landscape(G, seascape_df, fitness_col=selected_dose_col, dose_val=drug_doses[4])
    plot_seascape_heatmap(seascape_df, drug_doses)
    
    print("Analysis complete. Figures saved to root repository.")

if __name__ == "__main__":
    main()