"""
1. 訓練、検証のイメージとアノテーションのファイルパスのリストを作成する関数
"""

import os.path as osp

def make_filepath_list(rootpath):
    """
    データのパスを格納したリストを作成する
    Parameters
    ----------
    rootpath : str 
        データセットのルートパス

    Returns
    -------
    train_img_list : list
        訓練用の画像ファイルパスのリスト
    train_anno_list : list
        訓練用のアノテーションファイルパスのリスト
    val_img_list : list
        検証用の画像ファイルパスのリスト
    val_anno_list : list
        検証用のアノテーションファイルパスのリスト
    """
    imgpath_template = osp.join(rootpath, 'JPEGImages', '%s.jpg')
    annopath_template = osp.join(rootpath, 'Annotations', '%s.xml')
    
    train_id_names = osp.join(rootpath + 'ImageSets/Main/train.txt')
    val_id_names = osp.join(rootpath + 'ImageSets/Main/val.txt')
    
    # 画像とアノテーションのファイルパスを保存するリスト
    train_img_list = list()
    train_anno_list = list()
    
    for line in open(train_id_names):
        file_id = line.strip() # 空白文字を削除
        img_path = (imgpath_template % file_id) # %sにfile_idを代入　こういう書き方できるんだ
        anno_path = (annopath_template % file_id)
        
        train_img_list.append(img_path)
        train_anno_list.append(anno_path)
    
    val_img_list = list()
    val_anno_list = list()
    
    for line in open(val_id_names):
        file_id = line.strip() # 空白文字を削除
        img_path = (imgpath_template % file_id) # %sにfile_idを代入　こういう書き方できるんだ
        anno_path = (annopath_template % file_id)
        
        val_img_list.append(img_path)
        val_anno_list.append(anno_path)
        
    return train_img_list, train_anno_list, val_img_list, val_anno_list


