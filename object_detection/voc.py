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
    
    # + は文字列の結合してからjoin(, で結合されない限りpathの結合はない)
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


"""
2. バウンディングボックスの座標と正解ラベルをリスト化するclass
"""
import xml.etree.ElementTree as ElementTree # xmlファイルを読み込むためのライブラリ
import numpy as np

class GetBBoxAndLabel(object):
    """
    1枚の画像のアノテーション(座標とラベル)をnumpy配列で返す
    
    Attributes
    ----------
    classes : list
        VOCのクラス名を格納したリスト
    """
        
    def __init__(self, classes):
        """
        Parameters
        ----------
        classes : list
            VOCのクラス名を格納したリスト
        """
        self.classes = classes
        
    def __call__(self, xml_path, width, height):
        """
        インスタンスから呼び出すときに実行される関数
        
        Parameters
        ----------
        xml_path : str
            xmlファイルのパス
        width : int
            画像の幅 正規化に必要
        height : int
            画像の高さ 正規化に必要
            
        Returns(ndarray)
        -------
        [[xmin, ymin, xmax, ymax, label_ind], ... ]
        要素数は画像内に存在する物体数と同じ
        """
        
        annotation = []
    
        xml = ElementTree.parse(xml_path).getroot() # xmlファイルを読み込む .getroot()でxmlのルート要素を取得
        
        for obj in xml.iter('object'):
            # アノテーションで検知がdifficultに設定されているものは除外
            difficult = int(obj.find('difficult').text)
            if difficult == 1:
                continue
            
            # 1つの物体に対するアノテーションを格納するリスト
            bndbox = []
            
            # バウンディングボックスのラベルをリストに格納
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            
            grid = ['xmin', 'ymin', 'xmax', 'ymax']
            
            for gr in (grid):
                axis_value = int(bbox.find(gr).text) -1
                
                if gr == 'xmin' or gr == 'xmax':
                    axis_value /= width
                
                else:
                    axis_value /= height
                    
                bndbox.append(axis_value)
            
            # 物体名のインデックスを格納
            label_idx = self.classes.index(name)
            bndbox.append(label_idx)
            
            annotation += [bndbox]
            
        return np.array(annotation)

'''
3. イメージのアノテーションの前処理を行うDataTransformクラス

'''
from augmentations import Compose, ConvertFromInts, ToAbsoluteCoords, PhotometricDistort, Expand, RandomSampleCrop, RandomMirror, ToPercentCoords, Resize, SubtractMeans

class DataTransform(object):
    ''' データの前処理クラス
    イメージのサイズを300x300にする
    訓練時は拡張処理を行う
    
    Attributes
    ----------
    data_transform(dict): 前処理メソッドを格納した辞書
    '''
    
    
