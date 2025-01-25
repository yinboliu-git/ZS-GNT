import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os

def get_mesh_id(term):
    # 对搜索词进行URL编码
    encoded_term = requests.utils.quote(term)
    url = f"https://www.ncbi.nlm.nih.gov/mesh/?term={encoded_term}"
    
    try:
        # 发送请求
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 首先尝试直接在当前页面查找 MeSH Unique ID
        mesh_id_element = soup.find(string=lambda text: 'MeSH Unique ID:' in str(text))
        if mesh_id_element:
            mesh_id = mesh_id_element.strip().split(': ')[1]
            return mesh_id
            
        # 如果当前页面没有找到，尝试查找第一个结果的链接
        # 修改选择器以匹配正确的链接
        first_result = soup.select_one("#maincontent > div > div:nth-child(5) > div:nth-child(1) > div.rslt > p > a")
        if first_result:
            detail_url = "https://www.ncbi.nlm.nih.gov" + first_result['href']
            detail_response = requests.get(detail_url)
            detail_soup = BeautifulSoup(detail_response.text, 'html.parser')
            
            mesh_id_element = detail_soup.find(string=lambda text: 'MeSH Unique ID:' in str(text))
            if mesh_id_element:
                mesh_id = mesh_id_element.strip().split(': ')[1]
                return mesh_id
                
        return "未找到"
    
    except Exception as e:
        print(f"处理 {term} 时出错: {str(e)}")
        return "错误"

def mesh_id_main(terms = ['Aortic rupture', 'Prolapse']):
    # 示例名称列表
    
    # 存储结果的列表
    results = []
    
    # 遍历每个术语
    for term in terms:
        print(f"正在处理: {term}")
        mesh_id = get_mesh_id(term)
        results.append({'Name': term, 'MeSH_ID': mesh_id})
        # 添加延时以避免过快请求
        print(results[-1])
        time.sleep(1)
    
    # 创建DataFrame并保存为CSV
    df = pd.DataFrame(results)
    return df

if __name__ == "__main__":
    # 读取savedata目录下的所有文件名（CID）
    cids = [f for f in os.listdir('./savedata') if os.path.isdir(os.path.join('./savedata', f))]
    
    # 读取映射文件
    disease_map = {}
    with open('../MedGenIDMappings.txt', 'r') as f:
        next(f)  # 跳过标题行
        for line in f:
            parts = line.strip().split('|')
            if len(parts) >= 2:
                cid = parts[0]
                disease_name = parts[1]
                disease_map[cid] = disease_name
    
    # 获取疾病名称列表
    disease_names = [disease_map[cid] for cid in cids if cid in disease_map]
    
    # 获取MeSH ID
    results_df = mesh_id_main(disease_names)
    
    # 添加CID列
    final_results = []
    for cid in cids:
        if cid in disease_map:
            disease_name = disease_map[cid]
            mesh_id = results_df[results_df['Name'] == disease_name]['MeSH_ID'].iloc[0] if not results_df[results_df['Name'] == disease_name].empty else "未找到"
            final_results.append({
                'CID': cid,
                'Disease_Name': disease_name,
                'MeSH_ID': mesh_id
            })
    
    # 保存结果
    final_df = pd.DataFrame(final_results)
    final_df.to_csv('./savedata/mesh_id.csv', index=False)