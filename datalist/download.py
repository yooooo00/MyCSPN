import os
from pathlib import Path
import requests
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

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
    with ThreadPoolExecutor(max_workers=100) as executor:  # 控制同时进行下载的线程数
        for link in links:
            save_path=Path(save_dir)/link.split('/')[-2]/link.split('/')[-1]
            executor.submit(download_file, link, save_path)

def url_to_filepath(url):
    filepath = 'data/nyudepth_hdf5/train/'+url.split('/')[-2]+'/'+url.split('/')[-1]
    return filepath

def filepath_to_url(filepath):
    # link=base_url+line.strip().split(sep='/')[-2]+'/'+line.strip().split('/')[-1]
    url ='http://datasets.lids.mit.edu/nyudepthv2/nyudepthv2_noskip/val_full/'+Path(filepath).parent.name+'/'+Path(filepath).name
    return url


def download_csv_dataset(csv_path):
    global total,count_downloaded,count_failed
    count_downloaded,count_failed=0,0
    total=len(open(csv_path,'r').readlines())-1
    with open(csv_path,'r') as file:
        next(file)
        links=[]
        for line in file:
            links.append(filepath_to_url(line.strip()))
        download_files_using_threads(links=links,save_dir=Path(__file__).parent.parent/'data/nyudepth_hdf5/train')
        print(f"\n完成")


base_url='http://datasets.lids.mit.edu/nyudepthv2/nyudepthv2_noskip/val_full/'

download_csv_dataset(Path(__file__).parent/'nyudepth_hdf5_train_test.csv')
download_csv_dataset(Path(__file__).parent/'nyudepth_hdf5_val_test.csv')


