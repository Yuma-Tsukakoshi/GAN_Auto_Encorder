'''
1. データの拡張処理を行うクラス
'''
from typing import Any
import cv2
import numpy as np
from numpy import random

class Compose(object):
    def __init__(self, transforms):
        ''''
        Args:
            transforms (List[Transform]): List of transforms to compose. 変換処理のリスト
        Example:
            transforms.Compose([
                transforms.CenterCrop(10),
                transforms.ToTensor(),
            ])
        '''
        
        self.transforms = transforms
    
    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        
        return img, boxes, labels

'''
2. ピクセルデータのint型をfloat型に変換するクラス
'''
class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels

'''
3. アノテーションデータの正規化を元に戻すクラス
'''
class ToAbsoluteCoords(object):
    # boxes: [xmin, ymin, xmax, ymax] widthとheightで割って正規化された座標を用いていたのでかけて戻す
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height
        
        return image, boxes, labels

'''
4. 輝度(明るさ)をランダムに変化させるクラス
'''
class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        # 条件を満たさない場合はエラーを出す
        self.delta = delta
    
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2): #50%の確率(0or1出力)で実行
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels
    
'''
5. コントラスト(明るいところは明るく、暗いところは暗く)をランダムに変化させるクラス
'''
class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        # 式の後に文字列を指定することで、式が偽のときにエラーメッセージを表示できる。
        assert self.upper >= self.lower , 'contrast uppper must be >= lower.'
        assert self.lower >= 0, 'contrast lower must be non-negative.'
    
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        
        return image, boxes, labels
    
'''
6. BGRとHSVを相互に変換するクラス
'''
class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.current = current
        self.transform = transform
    
    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            # 特定の例外を手動で発生させる
            raise NotImplementedError
        
        return image, boxes, labels

'''
7. 彩度をランダムに変化させるクラス HSV : 色相(Hue)[:,:,0]、彩度(Saturation)[:,:,1]、明度(Value)[:,:,2]
'''
class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower , 'contrast uppper must be >= lower.'
        assert self.lower >= 0, 'contrast lower must be non-negative.'
        
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:,:, 1] *= random.uniform(self.lower, self.upper)
        
        return image, boxes, labels

'''
8. ランダムに色相を変化させるクラス
'''
class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta
    
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:,:,0] += random.uniform(-self.delta, self.delta)
            image[:,:,0][image[:,:,0] > 360.0 ] -= 360.0 # 一周するから360を超えたら360引く
            image[:,:,0][image[:,:,0] < 0.0 ] += 360.0
        
        return image, boxes, labels
    
'''
9. 測光に歪みを加えるクラス → 光の当たり方を変える
'''
class RandomLightingNoise(object):
    def __init__(self):
        self.perms = (
            (0,1,2), (0,2,1),
            (1,0,2), (1,2,0),
            (2,0,1), (2,1,0)
        )
        
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)
            image = shuffle(image)
        
        return swap, boxes, labels
    

'''
10. 色チャネルの並び順を変えるクラス
'''
class SwapChannels(object):
    def __init__(self, swaps):
        '''
        Args:
            swaps (int triple): final order of channels
                ex) (2, 1, 0)
        '''
        self.swaps = swaps
    
    def __call__(self, image):
        '''
        Args:
            image (Tensor): image tensor to be transformed
        
        Return:
            a tensor with channels swapped according to swap
        '''
        # if torch.is_tensor(image):
        # tensor型からnumpy型に変換
        #     image = image.data.cpu().numpy() 
        # else:
        #     image = np.array(image)
            
        image = image[:,:,self.swaps]
        return image
    
'''
11. 輝度(明るさ)、彩度、色相、コントラストを変化させ、歪みを加えるクラス
'''
class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            #コントラスト(BGRに適用)
            RandomContrast(),
            #カラーモデルをHSVにコンバート
            ConvertColor(transform='HSV'),
            #彩度の変化(HSVに適用
            RandomSaturation(),
            #色相の変化(HSVに適用)
            RandomHue(),
            # カラーモデルをHSVからBGRにコンバート
            ConvertColor(current='HSV', transform='BGR'),
            #コントラストの変化(BGRに適用)
            RandomContrast()
        ]
    
        # 輝度
        self.rand_brightness = RandomBrightness() 
        # 測光の歪み
        self.rand_light_noise = RandomLightingNoise() 
    
    def __call__(self, image, boxes, labels):
        im = image.copy()
        # 輝度の変化
        im, boxes, labels = self.rand_brightness(im, boxes, labels) 

        # データオーギュメンテーションでよく使われる手法(コントラストの順序をあえて変える)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        
        # 彩度、色相、コントラストの適用
        im, boxes, labels = distort(im, boxes, labels)
        
        return self.rand_light_noise(im, boxes, labels)

'''
12. イメージをランダムに拡大するクラス
'''
class Expand(object):
    def __init__(self, mean):
        self.mean = mean
    
    def __call__(self, image, boxes, labels):
        if random.randint(2):
            return image, boxes, labels

        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        # 計算方法 → 画像のサイズをratio倍に拡大し、最後に元の画像サイズを除くことで拡大した画像の余白を計算
        # 背景 : 仮にキャンパス内にランダムで座標を確定した後に、横幅や縦幅が元の画像サイズを超えると切れてしまう
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)
        
        expand_image = np.zeros(
            (int(height*ratio), (int)(width*ratio), depth),
            dtype = image.dtype) # dtypeは元の画像と同じにする
        
        expand_image[:, :, :] = self.mean # 色味の平均値で拡大されたキャンバスを埋める
        expand_image[int(top):int(top + height),
                    int(left):int(left + width)] = image
        image = expand_image
        
        boxes = boxes.copy()
        boxes[:,:2] += (int(left), int(top))  # xmin, ymin
        boxes[:,2:] += (int(left), int(top))  # xmax, ymax
        
        return image, boxes, labels

'''
13. イメージの左右をランダムに反転させるクラス
'''
class RandomMirror(object):
    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1] # ::-1: スライスステップを -1 に設定することで、画像の各行を反転順序で取得 チャネルは保持
            boxes = boxes.copy()
            # boxes[:, 2::-2]: バウンディングボックスの右端と左端の座標を反転順序で取得 0:xmin,2:xmax → 2:xmax,0:xmin
            boxes[:, 0::2] = width - boxes[:, 2::-2] # 画像の幅からの距離を計算 → 反転させる
        
        return image, boxes, classes
    
'''
14. アノテーションデータを0~1.0の範囲に正規化するクラス
'''
class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        boxes[:, 0] /= width # xmin 
        boxes[:, 2] /= width # xmax
        boxes[:, 1] /= height # ymin
        boxes[:, 3] /= height # ymax
        
        return image, boxes, labels
    
'''
15. イメージのサイズをinput_sizeにリサイズするクラス
'''
class Resize(object):
    def __init__(self, size=300):
        self.size = size
        
    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size, self.size))
        return image, boxes, labels

'''
16. 色情報(RGB値)から平均値を引き算するクラス 
'''
class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)
    
    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels
    
'''
17. 2セットのBBoxの重なる部分を検出する
'''
def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:]) # 右上の座標で小さい方を採用
    min_xy = np.maximum(box_a[:, :2], box_b[:2]) # 左下で座標の大きい方を採用
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf) # np.clip: 配列の要素を指定した範囲内に収める
    return inter[:,0] * inter[:,1] # 共通部分の面積を計算

'''
18. 2セットのBBoxの類似度を示すジャッカード係数を計算する
'''
def jaccard_numpy(box_a, box_b):
    '''
    Args:
        box_a : Multiple BBox, shape: [num_boxes, 4] → BBoxの座標
        box_b : Single BBox, shape: [4] →  トリミング用の長方形

    Returns:
        jaccard overlap : shape: [box_a.shape[0], box_a.shape[1]]
    '''
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1]))
    area_b = ((box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    union = area_a + area_b - inter 
    return inter / union

'''
19. イメージの特定の領域をランダムに切り出すクラス 
'''
class RandomSampleCrop(object):
    # TODO: サンプリング後のサイズの調整を理解する
    '''
    イメージの切り出しに合わせてバウンディングボックスも変形させる
    
    Arguments:
        img (Image): トレーニング中に入力されるイメージ
        boxes (Tensor): オリジナルのバウンディングボックス
        labels (Tensor): バウンディングボックスのラベル
        mode; (float tuple): 2セットのBBoxの類似度を示すジャッカード係数
        
    Return:
        (img, boxes, classes)
            img (Image) : トリミングされたイメージ
            boxes (Tensor): 調整後のバウンディングボックス
            labels (Tensor): バウンディングボックスのラベル
    '''
    def __init__(self):
        self.sample_options = (
            # 元の入力イメージをそのまま使用
            None, 
            # 0.1 ~ 1.0の範囲でランダムにサンプリング
            (0.1, None), 
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # バッチをランダムにサンプリング
            (None, None)
        )
        # 警告を無視するためにobject型のndarrayに変換
        self.sample_options = np.array(self.sample_options, dtype=object)
        
    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            # ランダムにサンプリング
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels
            
            #iou : 2セットのBBoxの類似度を示すジャッカード係数
            min_iou, max_iou = mode
            # float型にinfinityを設定する
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')
            
            # トリミングの試行回数
            for _ in range(50):
                current_image = image
                
                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)
                
                # アスペクト比が1/2 ~ 2の範囲でランダムにサンプリング
                if h / w < 0.5 or h / w > 2:
                    continue
                
                left = random.uniform(width - w)
                top = random.uniform(height - h)
                
                # イメージからトリミングする領域を作る
                # x1, y1, x2, y2を整数(int)に変換
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])
                overlap = jaccard_numpy(boxes, rect)
                
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue
                
                # イメージから切り抜く : height, widthの順
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]
                
                # box: xmin, ymin, xmax, ymax
                # gt boxの中心座標がトリミング後の画像の中にあるかどうかを確認
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
                
                # 左上側にある全てのgt boxをマスクする
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])
                
                # 右下側にある全てのgt boxをマスクする
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])
                
                mask = m1 * m2
                
                # 1つ以上のgt boxが残っているかどうかを確認
                if not mask.any():
                    continue
                
                # gt boxを取得
                current_boxes = boxes[mask, :].copy()
                
                # gt boxのラベルを取得
                current_labels = labels[mask]
                
                # ボックスの左上隅を利用する
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2])
                
                # トリミングされた状態に合わせる
                current_boxes[:, :2] -= rect[:2]
                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:], rect[2:])
                
                current_boxes[:, 2:] -= rect[:2]
                return current_image, current_boxes, current_labels
                