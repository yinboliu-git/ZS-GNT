import copy
import os
import pandas as pd
from collections import defaultdict


# 创建 Name-EFO 和 EFO-Name 字典
def create_name_efo_dictionaries(filepath, delimiter):
    data = pd.read_csv(filepath, delimiter=delimiter)
    name_to_efo = defaultdict(list)
    efo_to_name = defaultdict(list)

    for _, row in data.iterrows():
        name = row['Parent Molecule Name']
        # efo_terms = row['EFO Terms'].split('|') if isinstance(row['EFO Terms'], str) else [row['EFO Terms']]
        efo_terms = row['MESH ID']
        name_to_efo[name].append(efo_terms)
        efo_to_name[efo_terms].append(name)

    return name_to_efo, efo_to_name


# 获取一级子文件夹名称
def get_subfolder_names(folder_path):
    return [f.name for f in os.scandir(folder_path) if f.is_dir()]


# 处理 disease-drug 数据
def process_disease_drug_data(filepath):
    data = pd.read_csv(filepath)
    result = defaultdict(list)
    grouped = data.groupby(['diseaseName', 'drugName'])

    for (disease, drug), group in grouped:
        gene_count = group['geneName'].nunique()
        result[disease].append([drug, gene_count,list(group['geneName'].values)])

    return dict(result)


# 合并字典
def merge_dicts(dict1, dict2):
    for key, value in dict2.items():
        if key in dict1:
            dict1[key].extend(value)
        else:
            dict1[key] = value


# 获取 top N 的药物
def get_top_n_drugs(drug_tuple, n):
    sorted_drugs = sorted(drug_tuple, key=lambda x: x[1], reverse=True)
    return [drug[0] for drug in sorted_drugs[:n]]

def get_top_n_drugs_list(drug_tuple, n):
    sorted_drugs = sorted(drug_tuple, key=lambda x: x[1], reverse=True)
    return [drug for drug in sorted_drugs[:n]]


def flatten_and_format(drugs):
    """将列表展平，并且格式化为前部分用逗号分隔，后部分用分号分隔"""
    formatted_drugs = []

    # 处理非嵌套元素（疾病名和药物名）
    non_nested = [str(item) for item in drugs if not isinstance(item, list)]
    # 处理嵌套的子列表
    nested = [item for item in drugs if isinstance(item, list)]

    # 拼接非嵌套部分
    formatted_drugs.append(",".join(non_nested))

    # 拼接嵌套部分，后续的子列表用分号分隔
    for sublist in nested:
        formatted_drugs.append(";".join(map(str, sublist)))

    return ",".join(formatted_drugs)
# 主逻辑
def get_top_drug_main():
    disease_attributes_path = '../diseaseAttributes.xlsx'

    disease_df = pd.read_excel(disease_attributes_path)

    sample_size = min(100, len(disease_df))
    sampled_disease_cid_list = disease_df['diseaseId'].sample(n=sample_size, random_state=42).tolist()

    save_dir = '../code_search_param/savedata/'
    Drug_approved = '../DOWNLOAD-Drug_approved_max.csv'
    Drug_mesh_id = save_dir + 'mesh_id.csv'
    delimiter = ';'
    meshid = pd.read_csv(Drug_mesh_id, index_col=None, header=0)
    # 创建 Name-EFO 和 EFO-Name 字典
    name_to_efo, efo_to_name = create_name_efo_dictionaries(Drug_approved, delimiter)

    # 获取所有子文件夹
    subfolders = get_subfolder_names(save_dir)

    # 初始化 disease-drug 字典
    disease_drug_dict = {}
    NotGene_disease = []
    disease_id_name = {}
    disease_meshid_name = {}
    disease_oldid_name = {}
    disease_numbers = []
    # 打开文件进行写入
    with open("sampled_disease.txt", "w") as file:
        # 每10个元素一行
        for i in range(0, len(sampled_disease_cid_list), 10):
            # 获取当前行的10个元素
            line = sampled_disease_cid_list[i:i + 10]

            # 将当前行的元素转换为字符串，并用空格或逗号连接
            file.write(" ".join(map(str, line)) + "\n")
    for sub in sampled_disease_cid_list:
        subf = f'{save_dir}/{sub}/final_gene_drug_relationships.csv'
        # 检查文件是否存在
        if not os.path.exists(subf):
            # 文件不存在，记录疾病名称并跳过
            disease_name = sub  # 以文件夹名称为疾病名称
            NotGene_disease.append(disease_name)
            continue

        dd_dict = process_disease_drug_data(subf)
        merge_dicts(disease_drug_dict, dd_dict)
        disease_id_name[sub] = list(dd_dict.keys())[0]
        try:
            disease_meshid_name[meshid[meshid['CID']==sub].iloc[0,-1]] = list(dd_dict.values())[0]
            disease_oldid_name[sub] = list(dd_dict.values())[0]
        except:
            continue
    print(disease_id_name)

    # 计算 Top10 和 Top50 准确率
    top10_acc = []
    top5_acc = []
    best_durg = []
    number_dict = defaultdict(list)
    name_dict = {}
    with open("disease_drug_gene.txt", "w") as file:
        for id, (disease, drug_tuple) in zip(disease_id_name.keys(), disease_drug_dict.items()):
            # 获取排名前10的药物
            top_drugs = get_top_n_drugs_list(drug_tuple, 10)

            for drugs in top_drugs:
                # 在药物列表前插入疾病名称
                drugs = copy.deepcopy(drugs)
                drugs.insert(0, id)
                # 将疾病和药物写入文件，以逗号分隔
                formatted_drugs = flatten_and_format(drugs)
                file.write(formatted_drugs + "\n")
    disease_mm_list = []
    for (did, _),(disease, drug_tuple) in zip(disease_oldid_name.items(),disease_meshid_name.items()):
        true_drugs = efo_to_name[disease] if disease in efo_to_name else []
        top_drugs = get_top_n_drugs(drug_tuple, 10)
        if true_drugs == []:
            continue
        disease_mm_list.append(did)
        print(disease)
        print(true_drugs)
        predict_drugs = drug_tuple
        name_dict[disease] = drug_tuple
        for i in range(1,11):
            top_drugs = get_top_n_drugs(predict_drugs, i)
            # top10_drugs = get_top_n_drugs(predict_drugs, 10)

            a = len(set(top_drugs) & set(true_drugs)) if true_drugs else 0

            number_dict[i].append(a)

        # 记录最佳药物
        # top_drug = max(predict_drugs, key=lambda x: x[1])  # (drugName, geneCount)
        # name_dict.append((disease, top1_drug[0], top1_drug[1]))
    for i,a in number_dict.items():
        s = sum(number_dict[i])
        number_dict[i].append(sum(number_dict[i]) / len(number_dict[i]))
        print(f'{i}, {number_dict[i][-1]}, {s}')
    print(number_dict)

    print(name_dict)

    # 假设 disease_numbers 列表初始化
    disease_numbers = []

    # 添加数据到列表
    disease_numbers.append(['All diseases in database A', len(disease_df)])
    disease_numbers.append(['Known disease-drug pairs in database B', len(list(efo_to_name.keys()))])
    disease_numbers.append(['Randomly selected diseases', 100])
    disease_numbers.append(['Number of diseases with associated genes', len(list(disease_id_name.keys()))])
    disease_numbers.append(['Number of diseases with Mesh ID', len(list(disease_meshid_name.keys())) - 1])
    disease_numbers.append(
        ['Number of diseases with Mesh ID and available therapeutic drugs', len(disease_mm_list)])
    # 将 disease_numbers 写入文本文件
    with open("disease_numbers.txt", "w") as file:
        for item in disease_numbers:
            file.write(", ".join(map(str, item)) + "\n")
    print()
    for i, ddd in enumerate(disease_mm_list):
        # 输出结果
        if (i+1) % 5 ==0:
            print()
        print(ddd,end=' ')
    # 输出结果
    # print("Top 5 Accuracy:", sum(top5_acc) / len(top5_acc) if top5_acc else 0)
    # print("Top 10 Accuracy:", sum(top10_acc) / len(top10_acc) if top10_acc else 0)
    # print("Best Drugs:", best_durg)


# 运行主逻辑
if __name__ == '__main__':
    get_top_drug_main()
