import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def get_filenames_from_directory(url):
    # if not os.path.exists(target_folder):
    #     os.makedirs(target_folder)
    urls=[]
    # 获取目录页面的内容
    response = requests.get(url)
    if response.ok:
        # 使用BeautifulSoup解析HTML内容
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 遍历所有的<a>标签
        for link in soup.find_all('a'):
            href = link.get('href')
            if href and not href.startswith('?') and not href.startswith('/'):
                # print(urljoin(url, href))
                urls.append(urljoin(url, href))

    else:
        print("失败")
    return urls

import random
def get_all_files_urls(url):
    urls=get_filenames_from_directory(url)
    all_h5_urls=[]
    for url in urls:
        print(url)
        h5_urls=get_filenames_from_directory(url)
        all_h5_urls.extend(h5_urls)
    random.shuffle(all_h5_urls)
    return all_h5_urls


# def write_paths_to_file(url,file_path):
#     urls=get_filenames_from_directory(url)
#     all_h5_urls=[]
#     for url in urls:
#         print(url)
#         h5_urls=get_filenames_from_directory(url)
#         all_h5_urls.extend(h5_urls)



url = 'http://datasets.lids.mit.edu/nyudepthv2/nyudepthv2_noskip/val_full/'  # Apache服务器目录的URL
file_to_write = './nyudepth_hdf5_train_new.csv'

with open(file_to_write,"w") as file:
    file.write("Name\n")

with open(file_to_write,"w") as file:
    for url in get_all_files_urls(url):
        content='data/nyudepth_hdf5/train/'+url.split('/')[-3]+'/'+url.split('/')[-2]
        file.write(content+'\n')
        print(content)