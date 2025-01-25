import csv
import os
from itertools import product

import joblib
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader

# 读取数据
# data = pd.read_excel('result_with_drugs.xlsx')  # 更改为实际的文件路径
import setproctitle
setproctitle.setproctitle("ybliu_drug")

import pandas as pd
import torch
from torch_geometric.data import Data
import numpy as np
from utils import get_metrics

import numpy as np
from models import GCN

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

    # if batch_size==None:
    #     optimizer.zero_grad()
    #     train_data = data
    #     out = model(train_data.x, train_data.edge_index)
    #     pred = model.predict_link(out, train_data.edge_label_index)
    #     loss = criterion(pred, train_data.edge_label.type_as(pred).reshape(pred.shape))
    #     loss.backward()
    #     optimizer.step()
    #     return loss.item()
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
        progress_bar.update(1)  # 更新进度条
        progress_bar.set_postfix({'Batch loss': f'{loss.item():.4f}'})  # 显示额外的损失信息

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


def val_all(data, model, s_index_range, t_index_range, id_lookup_array, name='disease_gene', save_dir=None,batch_size=None):
    print(f'正常预测{name}...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)
    data.x = data.x.to(device)
    if batch_size is None:
        batch_size = len(data.edge_label) // 6
    batches = create_batches(data, batch_size)
    for batch_edge_label_index, batch_edge_label, batch_train_edge in batches:
        out = model(data.x, batch_train_edge)

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

def func(data_tuple):
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(1)  # Set GPU ID for the process
    # torch.cuda.set_device(1)
    model_class, train_data, val_data, epochs, save_dir_init,repeats, params = data_tuple
    args, _ = params
    for repeat in range(repeats):
        save_dir = save_dir_init
        param_string = f"hc{args['hidden_channels']}_layers{args['gcn_layers']}_dropout{args['drop_out']}_act{args['activation']}"
        model = model_class(num_features=train_data.num_features,
                            hidden_channels=args['hidden_channels'],
                            gcn_layers=args['gcn_layers'],
                            dropout=args['drop_out'],
                            activation=args['activation'])
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.BCEWithLogitsLoss()
        # 训练模型
        model_name = model_class.__name__
        print(f'正在进行{model_name}')
        save_dir = os.path.join(save_dir, model_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_dir = os.path.join(save_dir, f'repeat{repeat}_{param_string}')

        save_metrics_list = []
        save_loss_auc_list = [['epoch', 'loss', 'auc']]
        for epoch in range(epochs):
            train_loss = train(train_data, model, optimizer,criterion)
            if epoch % 10 == 0 :
                l, p = test(val_data, train_data, model)
                values, v_names = get_metrics(l, p)
                save_loss_auc_list.append([epoch, train_loss, values[2]])
                save_metrics_list.append([epoch, train_loss, v_names, values])
                print(f'Epoch {epoch}: Loss={train_loss:.4f}, AUC={values[2]}, AUPR={values[3]}')

        print('--1--')
        joblib.dump(save_metrics_list,save_dir+'save_metrics.list')

        # 保存损失和AUC到CSV文件
        df_loss_auc = pd.DataFrame(save_loss_auc_list[1:], columns=save_loss_auc_list[0])  # 创建DataFrame，指定列名
        df_loss_auc.to_csv(save_dir + 'loss_auc.csv', index=False)  # 保存到CSV，不包括行索引


import multiprocessing
from multiprocessing import Process, Queue
from mprocessing import get_available_gpus

def worker(gpu_id,lock, task_queue):
    """
    The worker function to be executed by each process. This function will be responsible for
    processing tasks assigned to a specific GPU.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # Set GPU ID for the process
    torch.cuda.set_device(gpu_id)  # Set the device to use in PyTorch
    setproctitle.setproctitle(f"ybliu_drug_GPU{gpu_id}")
    while True:
        with lock:
            task = task_queue.get()  # Get a task from the queue
            if task is None:
                break  # If None, shutdown this worker
        print(f'开始运行: {task[-1]}')
        try:
            func(task)
        except Exception as e:  # 捕获异常
            print("发生错误:", e)

        # try:
        #     func(task)
        # except Exception as e:  # 捕获异常
        #     print("发生错误:", e)

def manage_gpus(tasks, excluded_gpus=[]):
    """
    Setup and manage GPUs and tasks.
    """
    if multiprocessing.get_start_method(allow_none=True) is None:
        multiprocessing.set_start_method('spawn')

    gpu_list = get_available_gpus(100, excluded_gpus=excluded_gpus)  # 找到显存大于14G的显卡
    print(f'最大并行数: {gpu_list.__len__()}')
    task_queue = multiprocessing.Queue()
    lock = multiprocessing.Lock()


    # Put all tasks into the task queue
    for task in tasks:
        task_queue.put(task)
    # Add a None task for each GPU to signal them to stop
    for _ in gpu_list:
        task_queue.put(None)

    # Start a separate process for each GPU
    processes = []
    for gpu_id in gpu_list:
        p = Process(target=worker, args=(gpu_id,lock, task_queue))
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()


def p2_main(input_diseaseCID, epochs=500, repeats=1, search_param = False):
    if search_param == True:
        params = {
            'activation': ['leaky_relu',],
            'hidden_channels': [4],
            'gcn_layers': [4],
            'drop_out': [0.2],
        }

        # 使用 product 生成所有可能的参数组合
        param_combinations = [dict(zip(params.keys(), values)) for values in product(*params.values())]

        print(param_combinations[:10])  # 显示前十个组合，以便检查
    save_dir = f'./savedata/{input_diseaseCID}/'
    print('p2: 开始运行....')

    result_with_drugs_flie = save_dir + 'result_with_drugs.csv'

    graph_data, edge_index_neg, disease_id_map, gene_id_map, drug_id_map, ids = prepare_graph_data(result_with_drugs_flie, save_dir)

    transform = RandomLinkSplit(num_val=0.1, num_test=0.2, is_undirected=False, add_negative_train_samples=True,
                                neg_sampling_ratio=1)

    train_data, val_data, test_data = transform(graph_data)
    train_data_edge = train_data.edge_label_index.detach().numpy()

    # Replace negative edges in validation and test data
    val_data = replace_negative_edges(val_data, train_data_edge, edge_index_neg)
    test_data = replace_negative_edges(test_data, train_data_edge, edge_index_neg)

    print(graph_data)

    print('p2: 开始模型训练....')
    from models import ChebNet, GCN, GAT, GraphSAGE, GIN, GeneralGNN,GCRN,EdgeCNN,TopoNN,TransGCN
    model_list = [GraphSAGE] # , GIN, GAT, GCN, GraphSAGE
    task_list = []
    multi = False
    if search_param == True:
        if multi == True:
            for model_class in model_list:
                for args in param_combinations:
                    # 回收显存
                    torch.cuda.empty_cache()
                    print("显存已回收。")
                    task_list.append((model_class, train_data,val_data, epochs, save_dir,repeats, [args,None]))
            print(f'合计任务数: {task_list.__len__()}')
            manage_gpus(task_list,excluded_gpus=[])
        else:
            for model_class in model_list:
                for args in param_combinations:
                    task_list.append((model_class, train_data, val_data, epochs, save_dir, repeats, [args,None]))
            for task_tuple in task_list:
                try:
                    func(task_tuple)
                except Exception as e:  # 捕获异常
                    print("发生错误:", e)



if __name__ == '__main__':
    input_diseaseCID_list = []
    sampled_disease_cid_list = ['C0033860', 'C0279672', 'C1333977']
    for input_diseaseCID in sampled_disease_cid_list:
        p2_main(input_diseaseCID,
                epochs=500,
                repeats=5,
                search_param=True)