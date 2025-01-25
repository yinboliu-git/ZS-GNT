import pandas as pd
import os


# Function to load data with file paths as parameters
def load_data(gene_disease_path, disease_attributes_path, gene_attributes_path, drug_interactions_path):
    print('p1: 开始加载数据...')
    gene_disease_df = pd.read_excel(gene_disease_path)
    disease_df = pd.read_excel(disease_attributes_path)
    gene_df = pd.read_excel(gene_attributes_path)
    drug_df = pd.read_csv(drug_interactions_path, sep='\t')
    print(gene_disease_df.shape)
    print(drug_df.shape)
    return gene_disease_df, disease_df, gene_df, drug_df

if __name__ == "__main__":
    gene_disease_path = '../gene_disease/geneDiseaseNetwork.xlsx'
    disease_attributes_path = '../diseaseAttributes.xlsx'
    gene_attributes_path = '../geneAttributes.xlsx'
    drug_interactions_path = '../gene_drug/interactions.tsv'

    gene_disease_df, disease_df, gene_df, drug_df = load_data(gene_disease_path,
                                                              disease_attributes_path,
                                                              gene_attributes_path,
                                                              drug_interactions_path)
