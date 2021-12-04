import os
import re

'''
批量替换文件名
'''

# 扫描的入口文件夹
path = 'C:/Users/dcs/Desktop/文件夹/notes/study/MD/'
old_name = '、'
new_name = '.'
def change(p):
    path_list=os.listdir(p)
    for item in path_list:
        item = os.path.join(p,item)

        # 如果当前文件时文件夹则递归
        print(f'p=>{p}, item=>{item}')
        if os.path.isdir(item):
            change(item)
        # if(item[0] == '.'): 
        #     continue
        
        newitem = item.replace(old_name, new_name )
        try:
            os.rename(f'{item}', f'{newitem}')
        except Exception as e:
            print(f'============={e}==================')
        print(f'{item}=>{newitem}')


if __name__ == '__main__':
    change(path)
    print('finish')