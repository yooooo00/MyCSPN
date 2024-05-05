import os
import random

print(os.path.exists("/content/drive"))
if os.path.exists("/content/drive"):
    count=0
    with open('/content/CSPN/cspn_pytorch/datalist/png_train_test.csv','w') as file:
        file.write("Name\n")
        namelist=os.listdir('/content/drive/MyDrive/Colab Notebooks/data/kitti/2011_10_03_drive_0027_sync/image_center/Depth-Anything_image_02')
        random.shuffle(namelist)
        for name in namelist:
            name=name.split('_')[0]+'.png'
            if os.path.exists(os.path.join('/content/drive/MyDrive/Colab Notebooks/data/kitti/2011_10_03_drive_0027_sync/output_CREStereo',name)):
                file.write(f"{name}\n")
                count+=1
            if count>=1000:break

    # with open('/content/CSPN/cspn_pytorch/datalist/png_val_test.csv','w') as file:
    #     file.write("Name\n")
    #     for name in namelist:
    #         if os.path.exists(os.path.join('/content/drive/MyDrive/Colab Notebooks/data/kitti/2011_10_03_drive_0027_sync/output_CREStereo',name)):
    #             file.write(f"{name}\n")
    with open('/content/CSPN/cspn_pytorch/datalist/png_val_test.csv','w') as file:
        file.write("Name\n")
        namelist=os.listdir('/content/drive/MyDrive/Colab Notebooks/data/kitti/2011_10_03_drive_0027_sync/image_center/Depth-Anything_image_02')
        # random.shuffle(namelist)
        count=0
        for name in namelist:
            name=name.split('_')[0]+'.png'
            if os.path.exists(os.path.join('/content/drive/MyDrive/Colab Notebooks/data/kitti/2011_10_03_drive_0027_sync/output_CREStereo',name)):
                file.write(f"{name}\n")
                count+=1
            if count>=10:break
# 以上在colab中才会运行


count=0
failed=0
failed2=0
with open('D:\\projects\\MyCSPN\\datalist\\FallingThings_train.csv','w') as file:
    file.write("Name\n")
    namelist=os.listdir('D:\\dataset\\data\\FallingThings\\kitchen_0_result\\left')
    # random.shuffle(namelist)
    for name in sorted(namelist):
        name_right=name.replace('left', 'right')
        if os.path.exists(os.path.join('D:\\dataset\\data\\FallingThings\\kitchen_0_result\\right',name_right)):
            name2=name.split('.')[0]+'_depth.png'
            # print(name)
            if os.path.exists(os.path.join('D:\\dataset\\data\\FallingThings\\kitchen_0_result\\output_CREStereo_full',name)):
                file.write(f"{name}\n")
                count+=1
                
            else:failed2+=1
        else:failed+=1
        # if count>=100:break
        print(f"\r{count}/{failed}/{failed2}",end='        ')