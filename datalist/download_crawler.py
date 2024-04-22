import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def get_urls_from_url(url):
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
    urls=get_urls_from_url(url)
    all_h5_urls=[]
    for url in urls:
        print(url)
        h5_urls=get_urls_from_url(url)
        all_h5_urls.extend(h5_urls)
    # random.shuffle(all_h5_urls)
    return all_h5_urls


# def write_paths_to_file(url,file_path):
#     urls=get_filenames_from_directory(url)
#     all_h5_urls=[]
#     for url in urls:
#         print(url)
#         h5_urls=get_filenames_from_directory(url)
#         all_h5_urls.extend(h5_urls)
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from pathlib import Path
count_downloaded=0
count_failed=0
total=0
lock=Lock()
def download_file(link, save_path):
    global count_downloaded,count_failed,total
    filename = link.split('/')[-2]+'/'+link.split('/')[-1]
    response = requests.get(link)
    if not os.path.exists(Path(save_path).parent):
        os.makedirs(Path(save_path).parent)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
    with lock:
        if response.status_code==200:
            count_downloaded+=1
        else:
            count_failed+=1
        print(f"\rDownloaded/Failed/Total:{count_downloaded}/{count_failed}/{total},{filename}",end='        ')
def download_files_using_threads(links, save_dir):
    if not os.path.exists(save_dir):
        print('路径不对')
        return
    with ThreadPoolExecutor(max_workers=10) as executor:  # 控制同时进行下载的线程数
        for link in links:
            save_path=Path(save_dir)/link.split('/')[-2]/link.split('/')[-1]
            executor.submit(download_file, link, save_path)


url = 'http://datasets.lids.mit.edu/kitti/rgb/train/'  # Apache服务器目录的URL
file_to_write = './all_train_test.csv'
save_path_image02='D:\\dataset\\data\\all_in_one\\image_02\\data'
save_path_image03='D:\\dataset\\data\\all_in_one\\image_03\\data'

urls=get_urls_from_url(url)
for one_url in urls:
    if one_url[-1]!='/':continue
    date_name=one_url.split('/')[-2]
    image02url=urljoin(one_url,'image_02/data/')
    image03url=urljoin(one_url,'image_03/data/')
    image02urls=get_urls_from_url(image02url)
    with ThreadPoolExecutor(max_workers=10) as executor:
        for fileurl in image02urls:
            file_name=fileurl.split('/')[-1]
            print(date_name+'-'+file_name)
            executor.submit(download_file,fileurl,os.path.join(save_path_image02,date_name+'-'+file_name))
        for fileurl in get_urls_from_url(image03url):
            file_name=fileurl.split('/')[-1]
            print(date_name+'-'+file_name)
            executor.submit(download_file,fileurl,os.path.join(save_path_image03,date_name+'-'+file_name))
    print(date_name)
    # break
# with open(file_to_write,"w") as file:
#     file.write("Name\n")

# with open(file_to_write,"w") as file:
#     for url in get_all_files_urls(url):
        # content='data/nyudepth_hdf5/train/'+url.split('/')[-3]+'/'+url.split('/')[-2]
        # file.write(content+'\n')
        # print(content)