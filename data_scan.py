import os 
import random
import json
import gzip

from tqdm import trange
# from azure.storage.blob import generate_container_sas, ContainerSasPermissions
# from azure.storage.blob import ContainerClient
random.seed(56)

def save_to_json(data, json_file):
    with open(json_file, 'w', encoding='utf-8') as jsonfile:
        json.dump(data, jsonfile, ensure_ascii=False, indent=4)


def image2txtfile_json(train_sums,val_sums,image_dir,save_dir,scan_all=False):
    random.seed(None)
    index = random.randint(100,999)  #用于标识它与那个测试数据txt文件
    random.seed(56)

    if train_sums>=1000 and train_sums<1000000:
        k= "_"+str(int(train_sums/1000))+"k"
    elif train_sums>=1000000:
        k="_"+str(int(train_sums/1000000))+"m"
    else:
        k="_"+str(train_sums)
    if scan_all:
        k="_all"

    sums=val_sums+train_sums
    nums_images=[]


    if ".json" in image_dir: 
        print("使用json文件")
        if scan_all:
            print("使用全部数据")
            with open(image_dir,"r") as f:
                nums_images=json.load(f)
        else:
            with open(image_dir,"r") as f:
                all_data=json.load(f)
                if (val_sums+train_sums)<=len(all_data):
                    print(f"数据采集成功，json文件内共{len(all_data)}条数据，满足大小")
                    random.shuffle(all_data)
                    nums_images=all_data[0:sums]
                else:
                    print(f"注意:数据不够。你需要{sums}条数据，但是该json只有{len(all_data)}")
                    return 

    else:
        print("从文件夹的多个json文件夹中读取数据")
        for root, dirs, files in os.walk(image_dir):
            num_files=len(files)
            per_file=int(sums/num_files)
            for j in trange(1,num_files+1):
                filename="dataset_part_"+str(j)+".json"
                print("剩余采集：",sums,"采集数据从文件：",filename)
                with open(f""+image_dir+"/"+filename,"r") as f:
                    list_images=json.load(f)
                    random.shuffle(list_images)
                    if j==num_files:
                        nums_images=nums_images+list_images[0:sums]
                        break
                    per_catch_data_i=list_images[0:per_file]
                    nums_images=nums_images+per_catch_data_i
                    sums-= len(per_catch_data_i)


    random.shuffle(nums_images)
    if scan_all:
        train_images=nums_images[0:len(nums_images)-val_sums]
        val_image= nums_images[len(nums_images)-val_sums:]
    else:
        train_images=nums_images[0:train_sums]
        val_image= nums_images[train_sums:]

    # save
    train_filename = "train"+"_index"+str(index)+ k +".json"
    val_filename = "val_index"+str(index)+".json"
    save_to_json(train_images,save_dir+train_filename)
    save_to_json(val_image,save_dir+val_filename)
    print("dataset create successful, index is "+str(index))


if __name__ == "__main__":

    # # pretrain dataset 
    # train_sums=10000000 # 10000000=10M
    # val_sums=200
    # input_dir="/data01/luog/pmllm/zz_file/data/pretrained_data"
    # save_dir="/data01/luog/pmllm/data/pretrain/"
    # os.makedirs(save_dir,exist_ok=True)
    # image2txtfile_json(train_sums,val_sums,input_dir,save_dir)

    # # DTA dataset : DTA_bindingdb_kd / DTA_davis / DTA_kiba
    # train_sums=1000
    # val_sums=200
    # image_dir="/data01/luog/pmllm/zz_file/data/DTA_data/DTA_kiba.json"  # DTA_bindingdb_kd / DTA_davis / DTA_kiba
    # save_dir="/data01/luog/pmllm/data/DTA/DTA_kiba/"
    # os.makedirs(save_dir,exist_ok=True)
    # image2txtfile_json(train_sums,val_sums,image_dir,save_dir) # using all dataste: True or False

    # # DTI dataset 
    # train_sums=1000
    # val_sums=200
    # image_dir="/data01/luog/pmllm/zz_file/data/DTI_data/DTI_all.json"
    # save_dir="/data01/luog/pmllm/data/DTI/"
    # os.makedirs(save_dir,exist_ok=True)
    # image2txtfile_json(train_sums,val_sums,image_dir,save_dir)

    
    # protein with pocket dataset
    train_sums=1000
    val_sums=20
    image_dir="/root/private_data/luog/AbAgker/data/origin_json/skempi_AbAg_Kall237.json"  
    save_dir="/root/private_data/luog/AbAgker/data/"
    os.makedirs(save_dir,exist_ok=True)
    image2txtfile_json(train_sums,val_sums,image_dir,save_dir,True) # using all dataste: True or False