import json
import cv2
import numpy as np
import os
# from PIL import Image
# import base64
# import io

def read_mask_from_labelme(path):
    with open(path, "r",encoding="utf-8") as f: # load json
        obj = json.load(f)
    shapes = obj['shapes']
    size = [obj['imageHeight'], obj['imageWidth']]
    labels = np.empty(len(shapes), dtype=object)
    masks = np.zeros((len(shapes), *size), dtype=np.float32)
    for i, entry in enumerate(shapes):
        labels[i] = entry['label']
        
        # draw mask
        pts = np.array(entry['points']).astype(np.int32)
        pts = pts.reshape((-1, 1, 2))
        mask = cv2.fillPoly(np.zeros(size, dtype=np.float32), [pts], color=1)
        masks[i] = mask
    return masks, labels

def subimages_from_labelme(json_dir, img_dir, window_size=(224, 224), stride_size=(112, 112)):
    json_fn = os.listdir(json_dir)
    json_paths = [os.path.join(json_dir, fn) for fn in json_fn]
    img_paths = [os.path.join(img_dir, fn) for fn in txt_fn]
    img_paths = [os.path.splitext(path)[0] for path in img_paths] # exclude extension
    ret = []
    for json_path, img_path in zip(json_dir, img_paths):
        # check image ext type
        if os.path.isfile(img_path+".jpg"):
            img_path = img_path+".jpg"
        elif os.path.isfile(img_path+".png"):
            img_path = img_path+".png"
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8),-1) # read image
        masks, str_labels = read_mask_from_labelme(json_path)
        
        # find number of class and encode them into integer
        cls_name, clss, labels = np.unique(str_labels, return_index=True, return_inverse=True)
        one_masks = np.zeros((len(clss)+1, *masks.shape[1:]), dtype=bool) # semantics mask of classes
        for i in range(len(labels)):
            one_masks[labels[i]] |= masks[i]
        one_masks[-1] = ~one_masks.any(axis=0) # set background mask
        
        one_masks = generate_sliding_windows(one_masks, window_size=window_size, stride_size=stride_size, axis=(1, 2))
        subimgs = generate_sliding_windows(img, window_size=window_size, stride_size=stride_size)
        one_labels = one_masks.any(axis=(3, 4)) # if any pixel belongs to class, label it to that class
        
        # flatten height, width -> (n_class, w*h)
        one_labels = one_labels.reshape((one_labels.shape[0], -1))
        # flatten height, width -> (w*h, window_h, window_w, n_channel)
        subimgs = subimgs.reshape((subimgs.shape[0]*subimgs.shape[1], *windows_size, subimgs.shape[-1]))
        
        cls_name = (*cls_name, "background")
        labeled_subimgs = dict()
        # foreach class
        for i in range(len(one_labels)):
            labeled_subimgs[cls_name[i]] = subimgs[one_labels[i]]
        ret.append(labeled_subimgs)
    return ret
        

def labelme2yolofmt(json_dir, output_dir):
    '''Transform labelme format into yolo bound box format
    # Args
        json_dir: The directory include labelme format .json file.
        output_dir: The directory that yolo bound box format .txt file will be store.
    '''
    def read_rabelme(path):
        '''read json and transform into dict'''
        with open(path, "r",encoding="utf-8") as f:
            obj = json.load(f)
        shapes = obj['shapes']  
        cls_dict = dict()
        size = [obj['imageHeight'], obj['imageWidth']]
        for entry in shapes:
            e = cls_dict.get(entry['label'], [])
            pts = np.array(entry['points']).astype(np.float32)
            pts[:, 0] /= size[1]
            pts[:, 1] /= size[0]
            e.append(pts)
            cls_dict[entry['label']] = e        
        return cls_dict
    def poly2rect(polygon):
        '''Transform polygon [(x, y), ...] to rectange (left, top, wifth, height)'''
        left = polygon[:, 0].min()
        right = polygon[:, 0].max()
        top = polygon[:, 1].min()
        bottom = polygon[:, 1].max()
        width = right - left
        height = bottom - top
        return (left, top, width, height)
    
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    
    json_fn = os.listdir(json_dir)
    json_paths = [os.path.join(json_dir, fn) for fn in json_fn]
    polygons_list = [read_rabelme(path) for path in json_paths]
    file_txt_list = []
    for polygons, fn in zip(polygons_list, json_fn):
        counter = 0
        cls_lines = []
        for pts_list in polygons.values():
            rects = [poly2rect(pts) for pts in pts_list]
            lines = [[counter, *rect] for rect in rects]
            cls_lines += lines
            counter += 1
        
        cls_lines = ["{} {:.6f} {:.6f} {:.6f} {:.6f}".format(*line) for line in cls_lines]  
        cls_lines = "\n".join(cls_lines)
        fn, _ = os.path.splitext(fn)
        
        with open(os.path.join(output_dir, fn) + ".txt", "w",encoding="utf-8") as f:
            f.write(cls_lines)
        
def read_yolofmt(txt_path):
    with open(txt_path, "r",encoding="utf-8") as f:
        rects = f.read().split("\n")
    rects = [np.array(rect.split(" "), dtype=str).astype(np.float32) for rect in rects]
    rects = np.stack(rects, axis=0)
    label = rects[:, 0]
    rects = rects[:, 1:]
    n_cls, order = np.unique(label, return_inverse=True)
    ret = dict()
    for i, c in enumerate(n_cls):
        ret[int(c)] = rects[order==i]
    return ret
    
def crop_by_yolofmt(txt_path, img_path, to_rect=True):
    ret = {}
    rects_list = read_yolofmt(txt_path)
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8),-1)
    size = img.shape[0:2]
    for label, rects in rects_list.items():
        sub_imgs = []
        lefts = (rects[:, 0]*size[1]).astype(np.int32)
        tops = (rects[:, 1]*size[0]).astype(np.int32)
        widths = (rects[:, 2]*size[1]).astype(np.int32)
        heights = (rects[:, 3]*size[0]).astype(np.int32)
        rights = widths + lefts
        bottoms = heights + tops
        if to_rect:
            sw = widths -  heights
            tops[sw>0] -= sw[sw>0]//2
            bottoms[sw>0] += sw[sw>0]//2 + (sw[sw>0]%2)
            lefts[sw<0] -= -sw[sw<0]//2
            rights[sw<0] += -sw[sw<0]//2 + (-sw[sw<0]%2)
            bo = bottoms - size[0]
            ro = rights - size[1]
            tops[bo>0] -= bo[bo>0]
            lefts[ro>0] -= ro[ro>0]
            bottoms[tops<0] += -tops[tops<0]
            rights[lefts<0] += -lefts[lefts<0]
            lefts[lefts<0] = 0
            tops[tops<0] = 0
            bottoms[bo>0] = size[0]
            rights[ro>0] = size[1]
            
        for l, t, r, b in zip(lefts, tops, rights, bottoms):
            sub_imgs.append(img[t:b, l:r])
        ret[label] = sub_imgs
    return ret
            
def save_cropped_by_yolofmt(txt_dir, img_dir, output_dir):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
        
    txt_fn = os.listdir(txt_dir)
    txt_paths = [os.path.join(txt_dir, fn) for fn in txt_fn]
    img_paths = [os.path.join(img_dir, fn) for fn in txt_fn]
    img_paths = [os.path.splitext(path)[0] for path in img_paths]
    cls_rect = []
    for txt_path, img_path, fn in zip(txt_paths, img_paths, txt_fn):
        if os.path.isfile(img_path+".jpg"):
            img_path = img_path+".jpg"
        elif os.path.isfile(img_path+".png"):
            img_path = img_path+".png"
        
        sub_imgs_dict = crop_by_yolofmt(txt_path, img_path) 
        for label, sub_imgs in sub_imgs_dict.items():
            for i, sub_img in enumerate(sub_imgs):
                fn = os.path.splitext(fn)[0]
                path = os.path.join(output_dir, fn)
                cv2.imencode('.jpg', sub_img, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tofile('{}-{}-{}.jpg'.format(path, label, i))

def generate_sliding_windows(I, window_size=3, stride_size=1, axis=None, copy=True):
    if axis is None:
        axis = (0, 1)
    if isinstance(window_size, int):
        window_size = (window_size, window_size)
    windows = np.lib.stride_tricks.sliding_window_view(I, window_size, axis=axis)
    if copy:
        windows = windows.copy()
    if isinstance(stride_size, int):
        stride_size = (stride_size, stride_size)
    return windows[::stride_size[0], ::stride_size[1]]
  
  
  
  
  

    
