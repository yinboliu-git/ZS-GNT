from pipeline1_get_disease_gene_drug import p1_main, load_data
from pipeline2_GCN_train_predict import p2_main
from pipeline3_pre_pdata_and_drug_name import p3_main
from pipeline4_get_top_drug import get_top_drug_main

if __name__ == '__main__':
    input_diseaseCID_list = ['C1333977'] # 'C1333977','C0033860','C0279672'
    gene_disease_path = '../gene_disease/geneDiseaseNetwork.xlsx'
    disease_attributes_path = '../diseaseAttributes.xlsx'
    gene_attributes_path = '../geneAttributes.xlsx'
    drug_interactions_path = '../gene_drug/interactions.tsv'

    gene_disease_df, disease_df, gene_df, drug_df = load_data(gene_disease_path, disease_attributes_path, gene_attributes_path,drug_interactions_path)

    sample_size = min(100, len(disease_df))
    # if random 100 disease:
    sampled_disease_cid_list = disease_df['diseaseId'].sample(n=sample_size, random_state=42).tolist()

    # other:
    sampled_disease_cid_list = ['C1333977', 'C0033860', 'C0279672']  #

    for input_diseaseCID in sampled_disease_cid_list:
        p1_main(input_diseaseCID,
                gene_disease_df, disease_df, gene_df, drug_df)

    for input_diseaseCID in sampled_disease_cid_list:
        p2_main(input_diseaseCID)

    for input_diseaseCID in sampled_disease_cid_list:
        p3_main(input_diseaseCID)

    get_top_drug_main()
