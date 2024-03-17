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
        self.rand_brightness = RandomBrightness() # instanceの生成
        # 測光の歪み
        self.rand_light_noise = RandomLightNoise() 
    
    def __call__(self, image, boxes, labels):
        im = image.copy()
        #明るさの変化
        im, boxes, labels = self.rand_brightness(im, boxes, labels) # callメソッドの実行
        # 彩度、色相、コントラストの適用は上限と下限の間でランダムに
        # 歪みオフセットを選択することにより、確率0.5で適用
        if random.randint(2): # コントラストの変化を始めに適用するか後に適用するか
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
        boxes[:,:2] += (int(left), int(top))  
        boxes[:,2:] += (int(left), int(top))
        
        return image, boxes, labels

'''
13. イメージの左右をランダムに反転させるクラス
'''