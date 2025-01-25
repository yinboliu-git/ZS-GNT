import pandas as pd
import os


# Function to load data with file paths as parameters
def load_data(gene_disease_path, disease_attributes_path, gene_attributes_path, drug_interactions_path):
    print('p1: 开始加载数据...')
    gene_disease_df = pd.read_excel(gene_disease_path)
    disease_df = pd.read_excel(disease_attributes_path)
    gene_df = pd.read_excel(gene_attributes_path)
    drug_df = pd.read_csv(drug_interactions_path, sep='\t')

    return gene_disease_df, disease_df, gene_df, drug_df


# Function to find genes by diseaseNID
def find_genes_by_disease(df, diseaseNID):
    return df[df['diseaseNID'] == diseaseNID]['geneNID'].unique()


# Function to find diseases by geneNID
def find_diseases_by_gene(df, geneNID):
    return df[df['geneNID'] == geneNID]['diseaseNID'].unique()


# Function to get diseaseNID from diseaseCID
def get_dNid(disease_df, diseaseCID):
    return disease_df[disease_df['diseaseId'] == diseaseCID]['diseaseNID'].values.item()


# Function to generate the disease-gene association result
def generate_disease_gene_associations(input_diseaseCID, gene_disease_df, disease_df, gene_df, save_dir=None):
    input_diseaseNID = get_dNid(disease_df, input_diseaseCID)
    initial_genes = find_genes_by_disease(gene_disease_df, input_diseaseNID)
    print(f'过滤出基因:{initial_genes}')
    all_related_diseases = {disease for gene in initial_genes for disease in
                            find_diseases_by_gene(gene_disease_df, gene)}
    unique_related_diseases = list(all_related_diseases)

    related_diseases_df = pd.DataFrame(unique_related_diseases, columns=['diseaseNID'])
    merged_df = pd.merge(related_diseases_df, gene_disease_df, on='diseaseNID', how='left')
    result_df = merged_df[['diseaseNID', 'geneNID']].drop_duplicates()

    result_df = pd.merge(result_df, disease_df[['diseaseNID', 'diseaseId', 'diseaseName']], on='diseaseNID', how='left')
    result_df = pd.merge(result_df, gene_df[['geneNID', 'geneId', 'geneName']], on='geneNID', how='left')
    result_df = result_df[['diseaseNID', 'diseaseId', 'diseaseName', 'geneNID', 'geneId', 'geneName']].drop_duplicates()
    result_df.to_csv(save_dir + 'disease_gene_ass_result.csv', index=False)
    print(result_df.head())
    print(f'disease_gene_ass_result shape:{result_df.shape}')
    print(f'{input_diseaseCID}的第一级基因个数{len(initial_genes)}')
    print(f'{input_diseaseCID}的第一级疾病的个数{len(set(unique_related_diseases))}')
    print(f'{input_diseaseCID}的第二级基因的个数{len(set(merged_df["geneNID"]))}')

    return result_df


# Function to merge with drug data
def merge_with_drug_data(result_df, drug_df, save_dir=None):
    drug_df = drug_df[['entrez_id', 'drug_concept_id']].drop_duplicates()
    merged_df = pd.merge(result_df, drug_df, left_on='geneId', right_on='entrez_id', how='left')
    merged_df = merged_df.drop(columns=['entrez_id']).drop_duplicates()
    return merged_df


# Function to filter and sample merged_df
def filter_and_sample(merged_df, input_diseaseCID, sample_size=500000):
    # Step 1: Filter rows where 'diseaseId' equals input_diseaseCID
    filtered_df = merged_df[merged_df['diseaseId'] == input_diseaseCID]

    # Step 2: Drop the rows already in `filtered_df` from `merged_df`
    remaining_df = merged_df[merged_df['diseaseId'] != input_diseaseCID]
    if sample_size is not None:
        # Step 3: Randomly sample 500,000 rows from remaining_df
        sample_size = min(sample_size, len(remaining_df))
        sampled_df = remaining_df.sample(n=sample_size, random_state=42)

    # Step 4: Concatenate filtered_df and sampled_df
    result_df = pd.concat([filtered_df, sampled_df], ignore_index=True)

    return result_df


def p1_main(input_diseaseCID,
            gene_disease_df,
            disease_df,
            gene_df,
            drug_df):
    # Import the input disease CID
    print(f'p1: {input_diseaseCID} 开始运行...')
    save_dir = f'./savedata/{input_diseaseCID}/'
    os.makedirs(save_dir, exist_ok=True)


    # Generate the disease-gene association result
    result_df = generate_disease_gene_associations(input_diseaseCID, gene_disease_df, disease_df, gene_df,save_dir)
    print("Disease-gene association result:")

    # Merge with drug data and save the final result
    merged_df = merge_with_drug_data(result_df, drug_df, save_dir)
    print(f'{input_diseaseCID}的药物的个数{len(set(merged_df["drug_concept_id"]))}')
    print(f'{input_diseaseCID}的总边的个数{merged_df.shape[0]}')

    # merged_df = filter_and_sample(merged_df, input_diseaseCID, sample_size=50000)
    merged_df.to_csv(save_dir + 'result_with_drugs.csv', index=False)
    print(f'disease_gene_drug_ass_result shape:{merged_df.shape}')
    print("Final result with drugs:")
    print('p1: 运行完成...')

if __name__ == "__main__":
    gene_disease_path = '../gene_disease/geneDiseaseNetwork.xlsx'
    disease_attributes_path = '../diseaseAttributes.xlsx'
    gene_attributes_path = '../geneAttributes.xlsx'
    drug_interactions_path = '../gene_drug/interactions.tsv'

    gene_disease_df, disease_df, gene_df, drug_df = load_data(gene_disease_path,
                                                              disease_attributes_path,
                                                              gene_attributes_path,
                                                              drug_interactions_path)
    for id in ['C1333977', 'C0033860', 'C0279672']: # 'C0033860', 'C0279672'
        p1_main(id,gene_disease_df, disease_df, gene_df, drug_df)