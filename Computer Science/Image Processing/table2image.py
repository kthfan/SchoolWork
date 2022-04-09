import cv2
import numpy as np
import requests

class Table:
    def __init__(self, line_width=1, border_color=(0,0,0), img_shape=(256, 256), font_scale=0.5, line_height=0.8, align='center'):
        self.line_width = line_width
        self.border_color = border_color
        self.table_entries = dict()
        self.img_shape = img_shape
        self.line_height = line_height
        self.font_scale = font_scale
        self.align = align
    def put_item(self, row, col, item):
        self.table_entries["{}:{}".format(row, col)] = item
    def get_item(self, row, col):
        return self.table_entries.get("{}:{}".format(row, col))
    def get_tabel_dim(self):
        return
    def draw_entry(self, item, input_img, y1, y2, x1, x2):
        if isinstance(item, Table):
            input_img[y1:y2, x1:x2] = item.draw_table(input_img=input_img[y1:y2, x1:x2])
        elif isinstance(item, np.ndarray):
            self.draw_image(item, input_img, y1, y2, x1, x2)
        else:
            self.draw_text(str(item), input_img, y1, y2, x1, x2, self.font_scale, self.line_height, self.align)
        return input_img
    def draw_image(self, img, input_img, y1, y2, x1, x2):
        height, width = img.shape[0], img.shape[1]
        center_y = y1 + (y2 - y1 - height) // 2
        center_x = x1 + (x2 - x1 - width) // 2
        alpha = 1
        if img.shape[-1] == 4:
            alpha = img[:, :, 3:4]
            img = img[:, :, 0:3]

        input_img[center_y:center_y+height, center_x:center_x+width] = alpha*img + (1-alpha)*input_img[center_y:center_y+height, center_x:center_x+width]
        Table.debug = [alpha, img, input_img[center_y:center_y+height, center_x:center_x+width]]
        return input_img
    def draw_text(self, line_text, input_img, y1, y2, x1, x2, font_scale, line_height, align):
        line_text = line_text.split('\n')
        width, height = 0, 0
        dim_list = np.zeros((2, len(line_text)), dtype=np.int32)
        for line, text in enumerate(line_text):
            tw, th = cv2.getTextSize(text=text, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=font_scale, thickness=1)[0]
            height += th;
            dim_list[1, line] = int(height);
            left = x1 + (x2 - x1 - tw) // 2
            dim_list[0, line] = left; 
            height += line_height*th;
        center_y = y2 - (y2 - y1 + int(height)) // 2
        dim_list[1] += center_y
        for line, text in enumerate(line_text):
            left = None
            if align=='center': left = dim_list[0, line]
            elif align=='left': left = dim_list[0].min()
            input_img = cv2.putText(input_img, text, (left, dim_list[1, line]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
            
class LineTable(Table):
    def __init__(self, row_size_list=[1], col_size_list=[1], **kwds):
        super().__init__(**kwds)
        self.regrid(row_size_list, col_size_list)
    def _compute_grid(self, row_size_list, col_size_list, line_width):
        row_size_list, col_size_list = np.array(row_size_list), np.array(col_size_list)
        row_size_list, col_size_list = row_size_list*(1-line_width) / row_size_list.sum(), col_size_list*(1-line_width) / col_size_list.sum()
        row_size_list, col_size_list = np.concatenate([np.zeros(1), row_size_list.cumsum()]), np.concatenate([np.zeros(1), col_size_list.cumsum()])
        row_size_list, col_size_list = row_size_list.astype(np.int32), col_size_list.astype(np.int32)
        return row_size_list, col_size_list
    def set_line_width(self, line_width):
        self.line_width = line_width
        self.row, self.col = self._compute_grid(self.row_size_list, self.col_size_list, line_width)
    def regrid(self, row_size_list=[1], col_size_list=[1]):
        self.row_size_list = row_size_list
        self.col_size_list = col_size_list
        self.row, self.col = self._compute_grid(self.row_size_list, self.col_size_list, self.line_width)
    def grid(self, row_size_list=[1], col_size_list=[1]):
        row, col =  self._compute_grid(row_size_list, col_size_list, self.line_width)
        self.row, self.col = np.concatenate([self.row, row[1:-1]]), np.concatenate([self.col, col[1:-1]])
        self.row, self.col = np.sort(self.row), np.sort(self.col)
        self.row_size_list, self.col_size_list = self.row.diff(), self.col.diff()
    def get_tabel_dim(self, row=None, col=None, iput_shape=None):
        if iput_shape is None: iput_shape = self.img_shape
        if row is None: row = np.arange(1, len(self.row_size_list))
        if col is None: col = np.arange(1, len(self.col_size_list))
        if isinstance(row, int): row = np.array([row])
        if isinstance(col, int): col = np.array([col])
        
        row_size_list, col_size_list = np.array(self.row_size_list), np.array(self.col_size_list)
        row_size_list, col_size_list = row_size_list*iput_shape[0] / row_size_list.sum(), col_size_list*iput_shape[1] / col_size_list.sum()
        return row_size_list[row].sum(), col_size_list[col].sum()
    
    def draw_table(self, input_img=None):
        if input_img is None: input_img = np.ones((*self.img_shape, 3), dtype=np.float32)
        M, N = input_img.shape[0], input_img.shape[1]
        row_size_list, col_size_list, line_width, border_color = self.row_size_list, self.col_size_list, self.line_width, self.border_color
        
        row_size_list, col_size_list = np.array(row_size_list), np.array(col_size_list)
        row_size_list, col_size_list = row_size_list*(M-line_width) / row_size_list.sum(), col_size_list*(N-line_width) / col_size_list.sum()
        row_size_list, col_size_list = np.concatenate([np.zeros(1), row_size_list.cumsum()]), np.concatenate([np.zeros(1), col_size_list.cumsum()])
        row_size_list, col_size_list = row_size_list.astype(np.int32), col_size_list.astype(np.int32)
        border_color = np.array(border_color)

        for y in row_size_list: input_img[y:y+line_width, :, :] = border_color
        for x in col_size_list: input_img[:, x:x+line_width, :] = border_color
        
        for yx, item in self.table_entries.items():
            [y, x] = [int(i) for i in yx.split(':')]
            y1, y2 = row_size_list[y-1], row_size_list[y]+line_width
            x1, x2 = col_size_list[x-1], col_size_list[x]+line_width
            self.draw_entry(item, input_img, y1, y2, x1, x2)
            
        return input_img
    
class TransparentTable(LineTable):
    def draw_table(self, input_img=None):
        if input_img is None: input_img = np.ones((*self.img_shape, 3), dtype=np.float32)
        M, N = input_img.shape[0], input_img.shape[1]
        row_size_list, col_size_list, line_width, border_color = self.row_size_list, self.col_size_list, self.line_width, self.border_color
        
        row_size_list, col_size_list = np.array(row_size_list), np.array(col_size_list)
        row_size_list, col_size_list = row_size_list*(M-line_width) / row_size_list.sum(), col_size_list*(N-line_width) / col_size_list.sum()
        row_size_list, col_size_list = np.concatenate([np.zeros(1), row_size_list.cumsum()]), np.concatenate([np.zeros(1), col_size_list.cumsum()])
        row_size_list, col_size_list = row_size_list.astype(np.int32), col_size_list.astype(np.int32)
        border_color = np.array(border_color)

        for yx, item in self.table_entries.items():
            [y, x] = [int(i) for i in yx.split(':')]
            y1, y2 = row_size_list[y-1], row_size_list[y]+line_width
            x1, x2 = col_size_list[x-1], col_size_list[x]+line_width
            self.draw_entry(item, input_img, y1, y2, x1, x2)
            
        return input_img
    
class FlexTable(Table):
    def __init__(self, rows=1, cols=1, **kwds):
        super().__init__(**kwds)
        self.regrid(rows, cols)
    def _compute_dim(self, elem):
        if isinstance(elem, Table):
            return elem.get_tabel_dim()
        elif isinstance(elem, np.ndarray):
            return self._compute_image_dim(elem)
        return self._compute_text_dim(str(elem))
    def _compute_image_dim(self, img):
        tw, th = cv2.getTextSize(text='W', fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=self.font_scale, thickness=1)[0]
        return img.shape[0] // th, img.shape[1] // tw
    def _compute_text_dim(self, text):
        if text == '': return (0, 0)
        h = text.count('\n') + 1
        w = max([len(t) for t in text.split('\n')])
        return (h, w)
    def set_line_width(self, line_width):
        self.line_width = line_width
    def regrid(self, rows=1, cols=1):
        self.rows, self.cols = rows, cols
    def grid(self, rows=1, cols=1):
        self.regrid(rows, cols)
    def get_tabel_dim(self, row=None, col=None):
        if row is None: row = np.arange(1, self.rows+1)
        if col is None: col = np.arange(1, self.cols+1)
        if isinstance(row, int): row = [row]
        if isinstance(col, int): col = [col]
        
        result = np.zeros((2, len(row), len(col)), dtype=np.int32)
        
        for i,r in enumerate(row):
            for j,c in enumerate(col):
                elem = self.get_item(r, c)
                if elem is None: continue
                h, w  = self._compute_dim(elem)
                result[0, i, j] = h; result[1, i, j] = w
        h, w = result[0].max(axis=1).sum(), result[1].max(axis=0).sum()
        return (h, w)
    

class FlexLineTable(FlexTable):
    def draw_table(self, input_img=None):
        if input_img is None: input_img = np.ones((*self.img_shape, 3), dtype=np.float32)
        M, N = input_img.shape[0], input_img.shape[1]
        line_width = self.line_width
        
        row_size_list = np.array([self.get_tabel_dim(row=row)[0] for row in range(1, self.rows+1)]).astype(np.int32)
        col_size_list = np.array([self.get_tabel_dim(col=col)[1] for col in range(1, self.cols+1)]).astype(np.int32)
        row_size_list, col_size_list = row_size_list*(M-line_width) / row_size_list.sum(), col_size_list*(N-line_width) / col_size_list.sum()
        row_size_list, col_size_list = np.concatenate([np.zeros(1), row_size_list.cumsum()]), np.concatenate([np.zeros(1), col_size_list.cumsum()])
        row_size_list, col_size_list = row_size_list.astype(np.int32), col_size_list.astype(np.int32)
        border_color = np.array(self.border_color)
        
        for y in row_size_list: input_img[y:y+line_width, :, :] = border_color
        for x in col_size_list: input_img[:, x:x+line_width, :] = border_color
        
        for yx, item in self.table_entries.items():
            [y, x] = [int(i) for i in yx.split(':')]
            y1, y2 = row_size_list[y-1], row_size_list[y]+line_width
            x1, x2 = col_size_list[x-1], col_size_list[x]+line_width
            self.draw_entry(item, input_img, y1, y2, x1, x2)
            
        return input_img

class FlexTransparentTable(FlexTable):
    def draw_table(self, input_img=None):
        if input_img is None: input_img = np.ones((*self.img_shape, 3), dtype=np.float32)
        M, N = input_img.shape[0], input_img.shape[1]
        line_width = self.line_width
        
        row_size_list = np.array([self.get_tabel_dim(row=row)[0] for row in range(1, self.rows+1)]).astype(np.int32)
        col_size_list = np.array([self.get_tabel_dim(col=col)[1] for col in range(1, self.cols+1)]).astype(np.int32)
        row_size_list, col_size_list = row_size_list*(M-line_width) / row_size_list.sum(), col_size_list*(N-line_width) / col_size_list.sum()
        row_size_list, col_size_list = np.concatenate([np.zeros(1), row_size_list.cumsum()]), np.concatenate([np.zeros(1), col_size_list.cumsum()])
        row_size_list, col_size_list = row_size_list.astype(np.int32), col_size_list.astype(np.int32)
        border_color = np.array(self.border_color)

        for yx, item in self.table_entries.items():
            [y, x] = [int(i) for i in yx.split(':')]
            y1, y2 = row_size_list[y-1], row_size_list[y]+line_width
            x1, x2 = col_size_list[x-1], col_size_list[x]+line_width
            self.draw_entry(item, input_img, y1, y2, x1, x2)
        return input_img
    
def latex2png(latex, height=None, width=None):
    url = r'https://latex.codecogs.com/png.latex?\dpi{300} \huge ' + latex
    session = requests.Session()
    retry = requests.packages.urllib3.util.retry.Retry(connect=3, backoff_factor=0.5)
    adapter = requests.adapters.HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter); session.mount('https://', adapter)
    req = session.get(url)
    img = cv2.imdecode(np.asarray(bytearray(req.content), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    
    M, N = img.shape[0], img.shape[1]
    if height is None and width is None: return img
    elif height is None: height = M * width // N
    elif width is None: width = N * height // M
    img = cv2.resize(img, (width, height))
    return img
def from_matrix(mat, transp=False):
    table = FlexTransparentTable(*mat.shape) if transp else FlexLineTable(*mat.shape)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            table.put_item(i+1, j+1, mat[i, j])
    return table

def validation_matrix(true_y, pred_y, threshold=0.5, beta=1):
    pred_y = pred_y.ravel()
    if threshold is not None: pred_y = pred_y > threshold
    true_y = np.array(true_y).astype(bool).ravel()
    
    TP = np.count_nonzero(true_y & pred_y)
    TN = np.count_nonzero((~true_y) & (~pred_y))
    FP = np.count_nonzero((~true_y) & pred_y)
    FN = np.count_nonzero(true_y & (~pred_y))

    T = true_y.size
    P = TP + FN
    N = TN + FP
    
    entry2 = lambda a,b: from_matrix(np.array([[a], [b]], dtype=object), True)
    entry3 = lambda a,b,c: from_matrix(np.array([[a], [b], [c]], dtype=object), True)
    f2s = lambda f: str(round(f, 2))
    fracs = 50
    
    cm = np.array([
        [entry3('True Positive\n(TP)', '='+f2s(TP/T), ' '), entry3('False Positive\n(FP)', '='+f2s(FP/T), 'Type I error')],
        [entry3('False Negative\n(FN)', '='+f2s(FN/T), 'Type II error'), entry3('True Negative\n(TN)', '='+f2s(TN/T), ' ')]
    ]) 
    cm_ylabel = np.array([['Positive\n= '+f2s((TP+FN)/T)], ['Negative\n= '+f2s((TN+FP)/T)]])
    cm_xlabel = np.array([['Positive (P)\n= '+f2s(P/T), 'Negative (N)\n= '+f2s(N/T)]])
    cm, cm_ylabel, cm_xlabel = from_matrix(cm), from_matrix(cm_ylabel), from_matrix(cm_xlabel)
    rt = np.array([
        [entry2('Positive predictive value\n(PPV); Precision', latex2png(r'\frac{TP}{TP+FP} = ' + f2s(TP / (TP + FP)), fracs)/255), 
         entry2('False discovery rate\n(FDR)', latex2png(r'\frac{FP}{TN+FP} = ' + f2s(FP / (TN + FP)), fracs)/255)],
        [entry2('False omission rate\n(FOR)', latex2png(r'\frac{FN}{TP+FN} = ' + f2s(FN / (TP + FN)), fracs)/255), 
         entry2('Negative predictive value\n(NPV)', latex2png(r'\frac{TN}{TP+FN} = ' + f2s(TN / (TP + FN)), fracs)/255)]
    ])
    lb = np.array([
        [entry2('True Positive Rate (TPR)\nSensitivity, Recall', latex2png(r'\frac{TP}{TP+FN} = ' + f2s(TP / (TP + FN)), fracs)/255), 
         entry2('False Positive Rate (FPR)\nFall-out', latex2png(r'\frac{TP}{TP+FN} = ' + f2s(TP / (TP + FN)), fracs)/255)],
        [entry2('False Negative Rate (FNR)\nMiss rate', latex2png(r'\frac{TP}{TP+FN} = ' + f2s(TP / (TP + FN)), fracs)/255), 
         entry2('True Negative Rate (TNR)\nSpecificity', latex2png(r'\frac{TP}{TP+FN} = ' + f2s(TP / (TP + FN)), fracs)/255)]
    ])
    acc = np.array([['Accuracy = ' + f2s((TP+TN)/T)]])
    rt, lb, acc = from_matrix(rt), from_matrix(lb), from_matrix(acc)
    ext = FlexTransparentTable(1, 1, line_height=0.8, align='left')
    ext.put_item(1, 1, latex2png(
        r'\begin{align*}'+\
        r'& \text{Positive likelihood ratio (LR+)} = ' + f2s(TP/FP*(FP+TN)/(TP+FN)) + r'\\'+\
        r'& \text{Negative likelihood ratio (LR-)} = ' + f2s(FN/TN*(FP+TN)/(TP+FN)) + r'\\'+\
        r'& \text{Diagnostic odds ratio (DOR)} = ' + f2s(TP*TN/(FP*FN)) + r'\\'+\
        r'& F_{'+str(beta)+r'}\text{-score} = ' + f2s(TP / ((1+beta**2)*TP + beta**2*FN + FP)) + r'\\'+\
        r'& \text{G-measure} = ' + f2s(TP / ((TP+FP)*(TP+FN))**0.5) + r'\\'+\
        r'\end{align*}', 200)/255)
    
    table_mat = np.array([
        ['', '', 'True Condition', FlexLineTable(1, 2)],
        ['', 'Total (T)\n= '+f2s(T), cm_xlabel, acc],
        ['Predicted\noutcome', cm_ylabel, cm, rt],
        [FlexLineTable(2, 1), '', lb, ext]
    ], dtype=object)
    table = from_matrix(table_mat)
    return table
