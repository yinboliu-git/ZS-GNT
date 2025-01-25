import csv
import os

import joblib
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader

# 读取数据
# data = pd.read_excel('result_with_drugs.xlsx')  # 更改为实际的文件路径

import pandas as pd
import torch
from torch_geometric.data import Data
import numpy as np

import numpy as np
from models import GCN
os.environ["CUDA_VISIBLE_DEVICES"] = str(1)

def fl_edge(edge_index, fl_edge_index):
    # 使用 astype 将数字转换为统一格式的字符串
    source_nodes = edge_index[0].astype(str)
    target_nodes = edge_index[1].astype(str)
    fl_source_nodes = fl_edge_index[0].astype(str)
    fl_target_nodes = fl_edge_index[1].astype(str)

    # 使用 np.char.add 来连接节点编号和连字符
    neg_edge_set = np.char.add(source_nodes, np.char.add('-', target_nodes))
    fl_neg_edge_set = np.char.add(fl_source_nodes, np.char.add('-', fl_target_nodes))

    # 用 np.isin 检查是否存在于 fl_neg_edge_set 中，取反得到 mask
    mask = ~np.isin(neg_edge_set, fl_neg_edge_set)
    return edge_index[:, mask]


def generate_negative_edges(node_type_map, positive_edges, num_neg_samples):
    print('---选择负边---')
    # Randomly select nodes according to their type

    source_nodes = np.random.choice(node_type_map['source'], size=num_neg_samples * 2, )  # 大量过采样以减少循环次数
    target_nodes = np.random.choice(node_type_map['target'], size=num_neg_samples * 2, )
    print('--选择source and target--')
    # 生成潜在的负边
    potential_neg_edges = np.vstack((source_nodes, target_nodes)).T

    # 过滤本身的正边
    filtered_neg_edges = fl_edge(potential_neg_edges.T, positive_edges.T)

    print('--选择完成--')
    return filtered_neg_edges

def replace_negative_edges(data, train_data_edge, neg_edge_index):
    # Determine the number of positive edges
    num_pos_edges = int(data.edge_label.sum().item())
    num_neg_edges = len(data.edge_label) - num_pos_edges

    # 过滤掉训练集中出现的边
    neg_edge_index = fl_edge(neg_edge_index.detach().numpy(), train_data_edge)

    # Random selection of negative edges
    indices = np.random.choice(neg_edge_index.shape[1], num_neg_edges, replace=False)
    selected_neg_edges = torch.tensor(neg_edge_index[:, indices])

    # Combine positive and new negative edges
    edge_label_index = torch.cat([data.edge_label_index[:, data.edge_label == 1], selected_neg_edges], dim=1)

    # Update edge labels: 1 for positive, 0 for negative
    new_edge_label = torch.cat([torch.ones(num_pos_edges), torch.zeros(num_neg_edges)])

    # Update data object
    data.edge_label_index = edge_label_index
    data.edge_label = new_edge_label

    return data


def prepare_graph_data(filepath, save_dir):
    data = pd.read_csv(filepath)
    print('p2: prepare_graph_data')
    print(data.head())
    data['drug_concept_id'] = data['drug_concept_id'].replace('', pd.NA)  # 将空字符串替换为Pandas的NA

    # 为节点创建唯一编码并创建映射表
    disease_id_map = {nid: i for i, nid in enumerate(data['diseaseId'].unique())}
    gene_id_map = {nid: i + max(disease_id_map.values()) + 1 for i, nid in enumerate(data['geneId'].unique())}
    drug_id_map = {drug: i + max(gene_id_map.values()) + 1 for i, drug in enumerate(data['drug_concept_id'].unique()) if
                   pd.notna(drug)}

    # 映射节点到新的唯一编码
    disease_ids = data['diseaseId'].map(disease_id_map).values
    gene_ids = data['geneId'].map(gene_id_map).values
    drug_ids = data['drug_concept_id'].map(drug_id_map).fillna(-1).astype(int).values
    # 选择有效的边
    valid_edges = drug_ids != -1
    disease_gene_edges = np.array([disease_ids, gene_ids])
    gene_drug_edges = np.array([gene_ids[valid_edges], drug_ids[valid_edges]])

    # 合并边
    edge_index = torch.tensor(np.concatenate([disease_gene_edges, gene_drug_edges], axis=1), dtype=torch.long)
    edge_index = torch.cat([edge_index, edge_index[[1, 0], :]], dim=1)

    # 创建节点特征，这里使用简单的one-hot编码
    num_nodes = max(drug_id_map.values()) + 1
    x = torch.eye(num_nodes)

    # 创建图数据
    graph_data = Data(x=x, edge_index=edge_index)

    # 保存 ID 映射表到 CSV
    pd.DataFrame(list(disease_id_map.items()), columns=['Original_ID', 'Encoded_ID']).to_csv(save_dir+'disease_id_map.csv',
                                                                                             index=False)
    pd.DataFrame(list(gene_id_map.items()), columns=['Original_ID', 'Encoded_ID']).to_csv(save_dir+'gene_id_map.csv',
                                                                                          index=False)
    pd.DataFrame(list(drug_id_map.items()), columns=['Original_ID', 'Encoded_ID']).to_csv(save_dir+'drug_id_map.csv',
                                                                                          index=False)

    # Generating negative edges respecting the node type constraints
    disease_gene_neg_edges = generate_negative_edges(
        {'source': list(disease_id_map.values()), 'target': list(gene_id_map.values())},
        disease_gene_edges.T,
        disease_gene_edges.shape[1]
    )

    gene_drug_neg_edges = generate_negative_edges(
        {'source': list(gene_id_map.values()), 'target': list(drug_id_map.values())},
        gene_drug_edges.T,
        gene_drug_edges.shape[1]
    )

    edge_index_neg = torch.tensor(np.concatenate([disease_gene_neg_edges, gene_drug_neg_edges], axis=1),
                                  dtype=torch.long)
    edge_index_neg = torch.cat([edge_index_neg, edge_index_neg[[1, 0], :]], dim=1)

    return graph_data, edge_index_neg, disease_id_map, gene_id_map, drug_id_map, [disease_ids, gene_ids, drug_ids]


def get_metrics(real_score, predict_score):
    real_score, predict_score = real_score.flatten(), predict_score.flatten()
    sorted_predict_score = np.array(
        sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(
        sorted_predict_score_num * np.arange(1, 1000) / 1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1) - TP
    FN = real_score.sum() - TP
    TN = len(real_score.T) - TP - FP - FN

    fpr = FP / (FP + TN)
    tpr = TP / (TP + FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]

    # np.savetxt(roc_path.format(i), ROC_dot_matrix)

    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5 * (x_ROC[1:] - x_ROC[:-1]).T * (y_ROC[:-1] + y_ROC[1:])

    recall_list = tpr
    precision_list = TP / (TP + FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack(
        (recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]

    # np.savetxt(pr_path.format(i), PR_dot_matrix)

    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:])

    f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
    accuracy_list = (TP + TN) / len(real_score.T)
    specificity_list = TN / (TN + FP)

    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    # print(
    #     ' auc:{:.4f} ,aupr:{:.4f},f1_score:{:.4f}, accuracy:{:.4f}, recall:{:.4f}, specificity:{:.4f}, precision:{:.4f}'.format(
    #         auc[0, 0], aupr[0, 0], f1_score, accuracy, recall, specificity, precision))
    return [real_score, predict_score, auc[0, 0], aupr[0, 0], f1_score, accuracy, recall, specificity, precision], \
        ['y_true', 'y_score', 'auc', 'prc', 'f1_score', 'acc', 'recall', 'specificity', 'precision']


def create_batches(data, batch_size):
    # 创建边标签的 minibatches
    total_edges = data.edge_label_index.size(1)
    permutation = torch.randperm(total_edges)
    for i in range(0, total_edges, batch_size):
        indices = permutation[i:i+batch_size]
        batch_edge_label_index = data.edge_label_index[:, indices]
        batch_edge_label = data.edge_label[indices]
        yield batch_edge_label_index, batch_edge_label


from tqdm import tqdm  # 导入tqdm

# 训练模型
def train(data, model, optimizer, criterion, batch_size=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    model.train()
    if batch_size is None:
        batch_size = len(data.edge_label) // 6
    loss_all = 0
    # 创建批处理数据，并使用tqdm显示进度条
    batches = create_batches(data, batch_size)
    total_batches = len(data.edge_label) // batch_size + (len(data.edge_label) % batch_size != 0)  # 计算总批次数
    progress_bar = tqdm(total=total_batches, desc='Training', unit='batch')  # 初始化进度条
    for batch_edge_label_index, batch_edge_label in batches:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        pred = model.predict_link(out, batch_edge_label_index.to(device))
        loss = criterion(pred, batch_edge_label.type_as(pred).reshape(pred.shape))
        loss.backward()
        optimizer.step()
        loss_all += loss.item()
        # progress_bar.update(1)  # 更新进度条
        # progress_bar.set_postfix({'Batch loss': f'{loss.item():.4f}'})  # 显示额外的损失信息

    progress_bar.close()  # 完成后关闭进度条
    return loss_all


def test(data, train_data, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)
    train_data = train_data.to(device)
    data.x = data.x.to(device)
    train_data.x = train_data.x.to(device)
    model.eval()
    with torch.no_grad():
        out = model(train_data.x, train_data.edge_index)
        pred = model.predict_link(out, data.edge_label_index).sigmoid()
        # 获取真实的标签和预测的概率
        label = data.edge_label

        return label.cpu().numpy(), pred.cpu().numpy()


from tqdm import tqdm
import numpy as np


def generate_full_edge_index(num_nodes_tuple, edge_nodes_tuple):
    for start in range(num_nodes_tuple[0], num_nodes_tuple[1]):
        yield torch.tensor([start], dtype=torch.long), torch.arange(edge_nodes_tuple[0], edge_nodes_tuple[1],
                                                                    dtype=torch.long)


def generate_gene_drug_edge_index(num_nodes_list):
    for start in num_nodes_list:
        yield torch.tensor([start], dtype=torch.long), torch.arange(min(ids[2]), max(ids[2]), 1, dtype=torch.long)


def generate_disease_gene_edge_index(num_nodes):
    yield torch.tensor([num_nodes], dtype=torch.long), torch.arange(min(ids[1]), max(ids[1]), 1, dtype=torch.long)


def val_all(data, model, s_index_range, t_index_range, id_lookup_array, name='disease_gene', save_dir=None):
    print(f'正常预测{name}...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)
    data.x = data.x.to(device)

    out = model(data.x, data.edge_index)

    # 为了避免一次性生成过大的 edge_index，我们使用生成器
    edge_generator = generate_full_edge_index(s_index_range, t_index_range)

    # 准备 CSV 文件写入
    columns = ['Source', 'Target', 'Prediction']
    with open(save_dir + f'{name}_scores.csv', 'w') as f:
        f.write(','.join(columns) + '\n')  # 写入头部

    # 处理每一批数据
    for src, dst in tqdm(edge_generator, total=s_index_range[-1]):
        edge_index = torch.stack([src.repeat_interleave(dst.size(0)), dst.repeat(src.size(0))], dim=0)
        pred = model.predict_link(out, edge_index.to(device))
        pdata = torch.cat([edge_index.t().cpu(), pred.view(-1, 1).cpu()], dim=1)
        # Replace indices with original IDs using the lookup array
        pdata = pdata.detach().cpu().numpy()
        pdata = np.array(pdata, dtype=object)
        pdata[:, 0] = id_lookup_array[pdata[:, 0].astype(int)]
        pdata[:, 1] = id_lookup_array[pdata[:, 1].astype(int)]

        # 追加到 CSV
        df = pd.DataFrame(pdata, columns=columns)
        # 格式化 'Prediction' 列为四位小数
        df['Prediction'] = df['Prediction'] = df['Prediction'].astype(float).round(5)
        df.to_csv(save_dir+ f'{name}_scores.csv', mode='a', header=False, index=False)

    print(f"{name}All data has been processed and saved.")

def save_epoch_loss_auc(epoch, train_loss, auc_value, save_dir):
    # Define the file path
    file_path = os.path.join(save_dir, 'epoch_loss_auc.csv')

    # Check if the file exists
    if epoch==0 and os.path.exists(file_path):
        os.remove(file_path)
    file_exists = os.path.isfile(file_path)
    # Open the file in append mode
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        # If the file does not exist, write the header
        if not file_exists:
            writer.writerow(['epoch', 'loss', 'auc'])

        # Write the data
        writer.writerow([epoch, train_loss, auc_value])

def p2_main(input_diseaseCID, epochs=500,lr=0.01,hc=32):
    save_dir = f'./savedata/{input_diseaseCID}/'
    print('p2: 开始运行....')

    result_with_drugs_flie = save_dir + 'result_with_drugs.csv'

    graph_data, edge_index_neg, disease_id_map, gene_id_map, drug_id_map, ids = prepare_graph_data(result_with_drugs_flie, save_dir)

    transform = RandomLinkSplit(num_val=0.05, num_test=0.05, is_undirected=False, add_negative_train_samples=True,
                                neg_sampling_ratio=1)

    train_data, val_data, test_data = transform(graph_data)
    train_data_edge = train_data.edge_label_index.detach().numpy()

    # Replace negative edges in validation and test data
    val_data = replace_negative_edges(val_data, train_data_edge, edge_index_neg)
    test_data = replace_negative_edges(test_data, train_data_edge, edge_index_neg)

    print(graph_data)

    print('p2: 开始模型训练....')

    # 初始化模型
    model = GCN(num_features=train_data.num_features, hidden_channels=hc)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,)
    criterion = torch.nn.BCEWithLogitsLoss()

    # 训练模型
    save_metrics_list = []
    # save_loss_auc_list = []
    for epoch in range(epochs):
        train_loss = train(train_data, model, optimizer,criterion)
        # if epoch % 10 == 0:
            # l, p = test(val_data, train_data, model)
            # values, v_names = get_metrics(l, p)
            # save_epoch_loss_auc(epoch, train_loss, values[2], save_dir)
            # save_metrics_list.append([epoch, train_loss, v_names, values])
            # print(f'Epoch {epoch}: Loss={train_loss:.4f},')

    print('--1--')
    # 在测试集上测试模型
    print('--2--')

    # label, pred_score = test(test_data, train_data, model)
    l, p = test(test_data, train_data, model)
    values, v_names = get_metrics(l, p)
    save_epoch_loss_auc(epoch, train_loss, values[2], save_dir)
    save_metrics_list = [epoch, train_loss, v_names, values]
    print(f'Epoch {epoch}: Loss={train_loss:.4f},')
    joblib.dump(save_metrics_list,save_dir+'save_metrics.list')
    print('--3--')

    # get_metrics(label, pred_score)

    # ---------------------- pridcit --------------- #
    print('p2: 开始使用模型进行预测....')

    disease_index_range = (0, max(disease_id_map.values()) + 1)
    gene_index_range = (max(disease_id_map.values()) + 1, max(gene_id_map.values()) + 1)
    drug_index_range = (max(gene_id_map.values()) + 1, max(drug_id_map.values()) + 1)

    # Maximum index in id_map could be derived from the highest value in drug_id_map.
    max_index = max(drug_id_map.values())
    id_lookup_array = np.empty(max_index + 1, dtype=object)  # Creating an empty object array

    # Fill the array with the original IDs based on the encoded indices
    for original_id, encoded_index in {**disease_id_map, **gene_id_map, **drug_id_map}.items():
        id_lookup_array[encoded_index] = original_id

    val_all(val_data, model, disease_index_range, gene_index_range, id_lookup_array, name='disease_gene',save_dir=save_dir)
    val_all(val_data, model, gene_index_range, drug_index_range, id_lookup_array, name='gene_drug',save_dir=save_dir)

if __name__ == '__main__':
    p2_main('C1333977')
