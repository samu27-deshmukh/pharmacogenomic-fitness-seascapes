import pandas as pd
import numpy as np

def generate_synthetic_pharmacogenomic_dataset(num_variants: int = 16) -> pd.DataFrame:
    """
    Generates a synthetic pharmacogenomic genotype dataset representing single/double amino acid
    mutations (e.g., wildtype vs variants) and their baseline drug binding IC50 affinities.
    """
    genotypes = [f"{i:04b}" for i in range(num_variants)]
    np.random.seed(42)
    base_ic50 = []
    
    for g in genotypes:
        num_mutations = g.count('1')
        # Synergistic/epistatic resistance increases IC50 non-linearly with mutation count
        ic50_val = 0.5 * (3.5 ** num_mutations) + np.random.normal(0, 0.2)
        base_ic50.append(max(0.1, round(ic50_val, 3)))
        
    df = pd.DataFrame({
        'genotype': genotypes,
        'num_mutations': [g.count('1') for g in genotypes],
        'ic50_uM': base_ic50
    })
    
    return df

def calculate_fitness_seascape(df: pd.DataFrame, drug_concentrations: np.ndarray) -> pd.DataFrame:
    """
    Calculates dynamic dose-dependent cellular fitness across drug environments (concentration gradient)
    using the Hill dose-response equation.
    """
    hill_coeff = 2.0
    seascape_df = df.copy()
    
    for conc in drug_concentrations:
        col_name = f"fitness_dose_{conc:.2f}"
        seascape_df[col_name] = 1.0 / (1.0 + (conc / seascape_df['ic50_uM']) ** hill_coeff)
        
    return seascape_df