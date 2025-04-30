import cv2
import os
osp = os.path

def cv2_cvt(image_path, save=False):
    im = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    # cv2.imshow('image', im)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if save:
        cv2.imwrite(osp.join(osp.dirname(__file__),'Asset', image_path.split('/')[-1]), im)

if __name__ == '__main__':
    image_path = osp.join(osp.dirname(__file__), '..', 'CtcSDF', 'data', 'images', 'test', '6_20200918_120513_839512060362_55.png')
    cv2_cvt(image_path, save=True)