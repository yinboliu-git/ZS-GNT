import pandas as pd
from configs import input_diseaseCID
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
#
# def load_approved_drugs(filepath, delimiter=','):
#     extracted_values = []
#     with open(filepath, 'r') as f:
#         for line in f:  # 遍历文件中的所有行
#             line = line.strip()  # 去除行两端的空白字符
#             if delimiter in line:  # 如果行中有逗号
#                 # 获取逗号前的部分并添加到列表
#                 extracted_values.append(line.split(delimiter)[0])
#     extracted_values = extracted_values[1:]
#     approved_drugs=np.array(extracted_values)
#     approved_drugs = pd.DataFrame(approved_drugs, columns=['Parent Molecule'])
#     approved_drug_set = set(extracted_values)
#     return approved_drugs, approved_drug_set


def load_approved_drugs(filepath,delimiter):
    approved_drugs = pd.read_csv(filepath, delimiter=delimiter)
    approved_drug_set = set(approved_drugs['Parent Molecule'].values)
    return approved_drugs, approved_drug_set

def find_and_save_top_genes_and_drugs(disease_id, top_g_n, top_dg_n, disease_gene_file, gene_drug_file, output_file,
                                      approved_drugs, approved_drug_set, disease_attributes, gene_attributes,
                                      save_dir=None):
    # Load data
    disease_gene_scores = pd.read_csv(disease_gene_file)
    gene_drug_scores = pd.read_csv(gene_drug_file)


    # Adjust the 'Target' column in gene_drug_scores to remove the 'chembl:' prefix
    gene_drug_scores['Target'] = gene_drug_scores['Target'].apply(
        lambda x: x.split(':')[-1] if isinstance(x, str) else np.nan)

    gene_drug_scores = gene_drug_scores[gene_drug_scores['Target'].isin(approved_drug_set)]
    # Filter top N genes for the given disease
    top_genes = disease_gene_scores[disease_gene_scores['Source'] == disease_id]
    top_genes = top_genes.sort_values(by='Prediction', ascending=False).head(top_g_n)
    print(f'Top Gene shape is {top_genes.shape}')
    columns = ['diseaseId',
               'geneId',
               'disease-gene score',
               'drugId',
               'gene-drug score',
               'diseaseName',
               'geneName',
               'drugName',]
    # Prepare the final DataFrame
    final_results = pd.DataFrame(
        columns=columns
                 )

    print(final_results.keys())

    # For each top gene, find top M drugs
    for _, gene_row in top_genes.iterrows():
        gene_id = gene_row['Target']
        gene_score = gene_row['Prediction']
        top_drugs = gene_drug_scores[gene_drug_scores['Source'] == gene_id]
        top_drugs = top_drugs.sort_values(by='Prediction', ascending=False).head(top_dg_n)

        # Filter by approved drugs
        top_drugs = top_drugs[top_drugs['Target'].isin(approved_drug_set)]

        for _, drug_row in top_drugs.iterrows():
            drug_id = 'chembl:' + drug_row['Target']  # Re-add 'chembl:' prefix for consistency
            drug_score = drug_row['Prediction']
            drug_name = approved_drugs.loc[approved_drugs['Parent Molecule'] == drug_row['Target'], 'Name'].values[
                0]  # Get drug name
            gene_name = gene_attributes.loc[gene_attributes['geneId'] == gene_id, 'geneName'].values[0]  # Get gene name
            disease_name = disease_attributes.loc[disease_attributes['diseaseId'] == disease_id, 'diseaseName'].values[
                0]  # Get disease name

            # Append to the DataFrame
            new_row = pd.DataFrame({
                'diseaseId': [disease_id],
                'geneId': [gene_id],
                'disease-gene score': [gene_score],
                'drugId': [drug_id],
                'gene-drug score': [drug_score],
                'diseaseName': [disease_name],
                'geneName': [gene_name],
                'drugName': [drug_name]
            })

            # Use pd.concat() to add the new row to final_results
            final_results = pd.concat([final_results, new_row], ignore_index=True)

    # Save to CSV
    final_results = final_results.drop_duplicates(subset=['diseaseId', 'geneId', 'drugId'])
    final_results.to_csv(output_file, index=False)
    return final_results


def load_data(disease_attributes_path='../diseaseAttributes.xlsx',
             gene_attributes_path='../geneAttributes.xlsx',
             Drug_approved='../DOWNLOAD-Drug_approved_max.csv',
             delimiter=','
                          ):
    approved_drugs, approved_drug_set = load_approved_drugs(Drug_approved, delimiter)
    disease_attributes = pd.read_excel(disease_attributes_path)
    gene_attributes = pd.read_excel(gene_attributes_path)
    return disease_attributes, gene_attributes, approved_drugs, approved_drug_set


# Usage
def p3_main(input_diseaseCID,
            disease_attributes,
            gene_attributes,
            approved_drugs,
            approved_drug_set,
            top_genes=20,
            top_drugs=20,
           ):
    save_dir = f'./savedata/{input_diseaseCID}/'
    print('p3: 开始过滤最终表格....')


    disease_id = input_diseaseCID # replace with your actual disease ID

    disease_gene_file = save_dir + 'disease_gene_scores.csv'
    gene_drug_file = save_dir + 'gene_drug_scores.csv'
    output_file = save_dir + 'final_gene_drug_relationships.csv'
    results_df = find_and_save_top_genes_and_drugs(disease_id, top_genes,
                                                   top_drugs, disease_gene_file,
                                                   gene_drug_file, output_file,
                                                   approved_drugs, approved_drug_set,
                                                   disease_attributes, gene_attributes,
                                                   save_dir=save_dir)
    print(results_df)
    print(results_df.shape)
    print(f'p3: 最终表格已保存....\n')
    print(f'p3: {input_diseaseCID} 全部完成....\n')

if __name__ == '__main__':
    load_data()
    p3_main('C1333977')