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
5. コントラストをランダムに変化させるクラス
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
7. 彩度をランダムに変化させるクラス HSVのSに相当
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
        # andで条件を結合
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
class RandomLightNoise(object):
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
class SwapChannesls(object):
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
    