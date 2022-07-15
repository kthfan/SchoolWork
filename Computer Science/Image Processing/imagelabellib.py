import json
import cv2
import numpy as np
import os

def read_mask_from_labelme(path):
    '''Read labelme .json file and transform into masks
    # Args
        path: Path of the .json file.
    # Returns
        masks: Masks in json file
        labels: Corresponding labels of masks.
    '''
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

def save_cropped_subimages_from_labelme(json_dir, img_dir, output_dir, window_size=(224, 224), stride_size=(112, 112)):
    
    # check whether output_dir exists
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    
    
    json_fn = os.listdir(json_dir) # get .txt file name
    json_paths = [os.path.join(json_dir, fn) for fn in json_fn] # file name to json path
    img_paths = [os.path.join(img_dir, fn) for fn in json_fn] # file name to image path
    img_paths = [os.path.splitext(path)[0] for path in img_paths] # exclude extension
    
    # foreach file
    for img_path, json_path, fn in zip(img_paths, json_paths, json_fn):
        # check image extension type
        if os.path.isfile(img_path+".jpg"):
            img_path = img_path+".jpg"
        elif os.path.isfile(img_path+".png"):
            img_path = img_path+".png"
            
        fn = os.path.splitext(fn)[0] # only filename
            
        labeled_subimgs = subimages_from_labelme(json_path, img_path, window_size, stride_size)
        # foreach class
        for label, subimgs in labeled_subimgs.items():
            class_dir = os.path.join(output_dir, str(label))
            # check whether output_dir exists
            if not os.path.isdir(class_dir):
                os.mkdir(class_dir)
                
            # foreach image
            for i in range(subimgs.shape[0]):
                # save subimage, "dir/label/fn-i.jpg"
                path = os.path.join(class_dir, "{}-{}.jpg".format(fn, i))
                cv2.imencode('.jpg', subimgs[i], [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tofile(path)
        
def subimages_from_labelme(json_path, img_path, window_size=(224, 224), stride_size=(112, 112)):
    '''Read json file and image, then give the subimages and corresponding label.
    # Args
        json_path: Path of json file.
        img_path: Path of image.
        window_size: Size (height, width) of subimages.
        stride_size: Stride while performing sliding window.
    # Returns
        labeled_subimgs: dict(), Subimages(values) and corresponding label(keys).
    '''
    
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8),-1) # read image
    masks, str_labels = read_mask_from_labelme(json_path)
    masks = masks.astype(bool)
        
    # find number of class and encode them into integer
    cls_name, clss, labels = np.unique(str_labels, return_index=True, return_inverse=True)
    
    # semantics mask of classes
    one_masks = np.zeros((len(clss)+1, *masks.shape[1:]), dtype=bool)
    for i in range(len(labels)):
        one_masks[labels[i]] |= masks[i]
    one_masks[-1] = ~one_masks.any(axis=0) # set background mask
    
    one_masks = np.transpose(one_masks, (1, 2, 0)) # transpose class to channel
    one_masks = generate_sliding_windows(one_masks, window_size=window_size, stride_size=stride_size)
    one_masks = np.transpose(one_masks, (2, 0, 1, 3, 4)) # transpose (n_class, n_y, n_x, window_size[0], window_size[1])
    
    subimgs = generate_sliding_windows(img, window_size=window_size, stride_size=stride_size)
    subimgs = np.transpose(subimgs, (0, 1, 3, 4, 2)) # transpose to (n_y, n_x, window_size[0], window_size[1], n_channel)
    
    # if any pixel belongs to class, label it to that class
    one_labels = one_masks.any(axis=(3, 4))
    one_labels[-1] = one_masks[-1].all(axis=(2, 3)) # background using all
    
    # flatten height, width -> (n_class, w*h)
    one_labels = one_labels.reshape((one_labels.shape[0], -1))
    # flatten height, width -> (w*h, window_h, window_w, n_channel)
    
    subimgs = subimgs.reshape((subimgs.shape[0]*subimgs.shape[1], *window_size, subimgs.shape[-1]))
        
    cls_name = (*cls_name, "background")
    labeled_subimgs = dict()
    
    # foreach class
    for i in range(len(one_labels)):
        labeled_subimgs[cls_name[i]] = subimgs[one_labels[i]]
    return labeled_subimgs
        

def labelme2yolofmt(json_dir, output_dir):
    '''Transform labelme format into yolo bounding box format
    # Args
        json_dir: The directory including labelme format .json file.
        output_dir: The directory that yolo bounding box format .txt file will be store.
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
    
    # check whether output_dir exists
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    
    json_fn = os.listdir(json_dir) # find file name
    json_paths = [os.path.join(json_dir, fn) for fn in json_fn] # file name to path
    polygons_list = [read_rabelme(path) for path in json_paths] # read json and get polygons
    file_txt_list = []
    
    # foreach file
    for polygons, fn in zip(polygons_list, json_fn):
        counter = 0
        cls_lines = []
        # foreach class
        for pts_list in polygons.values():
            rects = [poly2rect(pts) for pts in pts_list] # compute bounding box via polygon
            lines = [[counter, *rect] for rect in rects] # add label of bounding box
            cls_lines += lines
            counter += 1
        
        cls_lines = ["{} {:.6f} {:.6f} {:.6f} {:.6f}".format(*line) for line in cls_lines] # to text
        cls_lines = "\n".join(cls_lines) # add line
        fn, _ = os.path.splitext(fn)
        
        # write .txt
        with open(os.path.join(output_dir, fn) + ".txt", "w",encoding="utf-8") as f:
            f.write(cls_lines)
        
def read_yolofmt(txt_path):
    '''Read bounding box in yolo format .txt file.
    # Args
        txt_path: Path of txt file.
    # Returns
        ret: Labels of bounding boxes (key) and bounding boxes (value)
    '''
    
    # Read txt file 
    with open(txt_path, "r",encoding="utf-8") as f:
        rects = f.read().split("\n")
    
    # Text to float
    rects = [np.array(rect.split(" "), dtype=str).astype(np.float32) for rect in rects]
    rects = np.stack(rects, axis=0)
    
    label = rects[:, 0] # read label 
    rects = rects[:, 1:] # read rect
    n_cls, order = np.unique(label, return_inverse=True) # number of classes
    ret = dict()
    # foreach class
    for i, c in enumerate(n_cls):
        ret[int(c)] = rects[order==i] # bounding boxes rects[order==i] belongs to label c
    return ret
    
def crop_by_yolofmt(txt_path, img_path, to_rect=True):
    '''Crop image into subimages by bounding boxes.
    # Args
        txt_path: Path of yolo format bounding boxes.
        img_path: Path of image.
        to_rect:  Crop subimages in rectangle shape.
    # Returns
        ret: dict(). Cropped subimages (value) and corresponding label (key). 
    '''
    ret = {}
    rects_list = read_yolofmt(txt_path) # read bounding boxes
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8),-1) # read image
    size = img.shape[0:2]
    
    # foreach classes
    for label, rects in rects_list.items():
        # compute positions of left, right, top ,bottom
        lefts = (rects[:, 0]*size[1]).astype(np.int32)
        tops = (rects[:, 1]*size[0]).astype(np.int32)
        widths = (rects[:, 2]*size[1]).astype(np.int32)
        heights = (rects[:, 3]*size[0]).astype(np.int32)
        rights = widths + lefts
        bottoms = heights + tops
        if to_rect: # if cropping into rect
            # ensuring width == height
            sw = widths -  heights
            tops[sw>0] -= sw[sw>0]//2
            bottoms[sw>0] += sw[sw>0]//2 + (sw[sw>0]%2)
            lefts[sw<0] -= -sw[sw<0]//2
            rights[sw<0] += -sw[sw<0]//2 + (-sw[sw<0]%2)
            
            # ensuring 0 < widths <= size[1] and 0 < heights <= size[0]
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
            
        sub_imgs = []
        # foreach bounding box
        for l, t, r, b in zip(lefts, tops, rights, bottoms):
            sub_imgs.append(img[t:b, l:r])
        ret[label] = sub_imgs
    return ret

def save_cropped_by_yolofmt(txt_dir, img_dir, output_dir, to_rect=True):
    '''Read yolo bounding boxes .txt and images, then save cropped subimages.
    # Args
        txt_dir: The directory including bounding boxes .txt file.
        img_dir: The directory including images
        output_dir: The directory that cropped subimages will be store.
    '''
    if not os.path.isdir(output_dir): # check if path exists
        os.mkdir(output_dir)
        
    txt_fn = os.listdir(txt_dir) # get .txt file name
    txt_paths = [os.path.join(txt_dir, fn) for fn in txt_fn] # file name to txt path
    img_paths = [os.path.join(img_dir, fn) for fn in txt_fn] # file name to image path
    img_paths = [os.path.splitext(path)[0] for path in img_paths] # exclude extension
    cls_rect = []
    # foreach file
    for txt_path, img_path, fn in zip(txt_paths, img_paths, txt_fn):
        # check image extension type
        if os.path.isfile(img_path+".jpg"):
            img_path = img_path+".jpg"
        elif os.path.isfile(img_path+".png"):
            img_path = img_path+".png"
            
        fn = os.path.splitext(fn)[0] # only filename
        
        sub_imgs_dict = crop_by_yolofmt(txt_path, img_path, to_rect=to_rect) 
        # foreach class
        for label, sub_imgs in sub_imgs_dict.items():
            # foreach subimage
            for i, sub_img in enumerate(sub_imgs):
                # save subimage, "dir/fn-label-i.jpg"
                path = os.path.join(output_dir, '{}-{}-{}.jpg'.format(fn, label, i))
                cv2.imencode('.jpg', sub_img, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tofile(path)

def generate_sliding_windows(I, window_size=3, stride_size=1, copy=True):
    '''Generate sliding windows
    # Args
        I: Input image
        window_size: Size of windows
        stride_size: Size of stride
        copy: If False, sliding windows are read-only.
    # Returns
        windows: Returned sliding windows with shape (n_y, n_x, n_channel, window_size[0], window_size[1]).
    '''
    
    if isinstance(window_size, int):
        window_size = (window_size, window_size)
    windows = np.lib.stride_tricks.sliding_window_view(I, window_size, axis=(0, 1))
    windows = windows[::stride_size[0], ::stride_size[1]]
    
    if copy:
        windows = windows.copy()
    if isinstance(stride_size, int):
        stride_size = (stride_size, stride_size)
    
    return windows



if __name__ == '__main__':
    # read directory
    main_dir = input()
    to_rect = True
    
    # sub-directory
    json_dir = os.path.join(main_dir, "json")
    txt_dir = os.path.join(main_dir, "txt")
    img_dir = os.path.join(main_dir, "原圖")
    crop_dir = os.path.join(main_dir, "crop")
    labeled_crop_dir = os.path.join(main_dir, "labeled crop")
    
    
    labelme2yolofmt(json_dir, txt_dir) # create bounding boxex .txt files
    save_cropped_by_yolofmt(txt_dir, img_dir, crop_dir, to_rect=to_rect) # create cropped cracked images
    save_cropped_subimages_from_labelme(json_dir, img_dir, labeled_crop_dir) # create labeled images (cracked and non-cracked)
    
