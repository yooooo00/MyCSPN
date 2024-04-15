import os
from pathlib import Path
import requests
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

# total=len(open(new_file_path,'r').readlines())-1
count_true=0
count_false=0
lock=Lock()
def check_single_url(url):
    response=requests.head(url)
    global count_false,count_true
    with lock:
        if response.status_code!=200:
            count_false+=1
            # print(f"{count}/{total}:{line.strip()},False        ",end='\r')
        else:
            count_true+=1
        print(f"\rTrue/False:{count_true}/{count_false}",end='')

def check_if_dataset_file_exists_online(filepath):
    global count_true,count_false
    count_true=0
    count_false=0
    with open(filepath,'r') as file:
        next(file)
        print('\r')
        with ThreadPoolExecutor(max_workers=100) as executor:
            for line in file:
                link=base_url+line.strip().split(sep='/')[-2]+'/'+line.strip().split('/')[-1]
                executor.submit(check_single_url,link)

def choose_dataset_part(full_path,target_path):
    with open(full_path,'r') as full_file,open(target_path,'w')as target_file:
        next(full_file)
        target_file.write('Name\n')
        count_line=0
        for i in range(80000):next(full_file)
        for line in full_file:
            target_file.write(line)
            count_line+=1
            if count_line>=100:break


base_url='http://datasets.lids.mit.edu/nyudepthv2/nyudepthv2_noskip/val_full/'
test_file_path=Path(__file__).parent/'nyudepth_hdf5_train_test.csv'
new_file_path=Path(__file__).parent/'nyudepth_hdf5_train_new.csv'
val_file_path=Path(__file__).parent/'nyudepth_hdf5_val_test.csv'
mini_file_path=Path(__file__).parent/'nyudepth_hdf5_train_mini.csv'
mini_val_file_path=Path(__file__).parent/'nyudepth_hdf5_val_mini.csv'
# check_if_dataset_file_exists_online(new_file_path)
choose_dataset_part(new_file_path,mini_val_file_path)