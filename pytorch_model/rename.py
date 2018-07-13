import os
import re
import tarfile
import shutil
def rename():
    root='/disk/yaoyang/data/ILSVRC2012_img_train/'
    files=os.listdir(root)
    for filename in files:
        portion=os.path.splitext(filename)
        if portion[1]=='.JPEG':
            newname=portion[0]+'.jpeg'
            print(filename+'--->'+newname)
            os.rename(root+filename,root+newname)
def mkdir():
    root = '/disk2/yaoyang/data/ILSVRC2012_img/val/'
    #root='./data/'
    files=os.listdir(root)
    for filename in files:
        portion = os.path.splitext(filename)
        tar = tarfile.open(root+filename)
        if portion[1] == '.tar':
            if os.path.isdir(root+portion[0]):
                pass
            else:
                print('untar:'+portion[0])
                os.makedirs(root+portion[0])
                src=root+portion[0]
                tar.extractall(src)
                for f in os.listdir(root+portion[0]):
                    print(f)

def combin(txt1,txt2):
    #root = '/disk2/yaoyang/data/ILSVRC2012_img/val/'
    root='./data/'
    name_path=os.path.join(root,txt2)
    print(name_path)
    label_path=os.path.join(root,txt1)
    f=open(label_path,'r')
    label_list=[]
    for i in f.readlines():
        s=i.strip()
        label_list.append(s)
    f.close()
    num=50000
    f=open(name_path,'r')
    o=open('./a.txt','w')
    name_list=[]
    for i in f.readlines():
        s=i.strip()
        name_list.append(s)
    f.close()
    for i in range(50000):
        s=name_list[i]+' '+label_list[i]
        o.write(s+'\n')
    o.close()

def rm_val():
    label_path='/disk2/yaoyang/data/ILSVRC2012_img/val_label.txt'
    #label_path='./a.txt'
    root='/disk2/yaoyang/data/ILSVRC2012_img/val/'
    f=open(label_path,'r')
    for i in f.readlines():
        portion = i.split()
        path=portion[0]
        name=path.split('/')[-1]
        print(name)
        label=portion[1]
        label_dir=os.path.join(root,label)
        if os.path.isdir(label_dir):
            pass
        else:
            os.mkdir(label_dir)

        dir_name=os.path.join(label_dir,name)
        print('copy ' + path + 'to ' + dir_name)
        shutil.copyfile(path, dir_name)


if __name__ == '__main__':
    txt1='ILSVRC2012_validation_ground_truth.txt'
    txt2='val.txt'
    #combin(txt1,txt2)
    rm_val()