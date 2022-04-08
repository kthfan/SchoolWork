class Table:
    def __init__(self):
        return
    def draw_text(input_img=None):
        return
    def put_table(self, row, col, table):
        return
    def put_text(self, row, col, item):
        return
    def get_tabel_dim(self):
        return
    def draw_text(self, line_text, input_img, y1, y2, x1, x2):
        line_text = line_text.split('\n')
        width, height = 0, 0
        dim_list = np.zeros((2, len(line_text)), dtype=np.int32)
        for line, text in enumerate(line_text):
            tw, th = cv2.getTextSize(text=text, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, thickness=1)[0]
            height += th;
            dim_list[1, line] = height;
            left = x1 + (x2 - x1 - tw) // 2
            dim_list[0, line] = left; 
        center_y = y2 - (y2 - y1 + height) // 2
        dim_list[1] += center_y
        for line, text in enumerate(line_text):
            input_img = cv2.putText(input_img, text, (dim_list[0, line], dim_list[1, line]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
            
class LineTable(Table):
    def __init__(self, row_size_list=[1], col_size_list=[1], line_width=1, border_color=(0,0,0), img_shape=(256, 256)):
        super().__init__()
        self.line_width = line_width
        self.border_color = border_color
        self.regrid(row_size_list, col_size_list)
        self.sub_tables = dict()
        self.table_entries = dict()
        self.img_shape = img_shape
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
    def put_table(self, row, col, table):
        self.sub_tables["{}:{}".format(row, col)] = table
    def put_text(self, row, col, item):
        self.table_entries["{}:{}".format(row, col)] = str(item)
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
        if input_img is None: input_img = np.ones((*img_shape, 3), dtype=np.float32)
        M, N = input_img.shape[0], input_img.shape[1]
        row_size_list, col_size_list, line_width, border_color = self.row_size_list, self.col_size_list, self.line_width, self.border_color
        
#         line_width = int(line_width * (M*N)**0.5)
        row_size_list, col_size_list = np.array(row_size_list), np.array(col_size_list)
        row_size_list, col_size_list = row_size_list*(M-line_width) / row_size_list.sum(), col_size_list*(N-line_width) / col_size_list.sum()
        row_size_list, col_size_list = np.concatenate([np.zeros(1), row_size_list.cumsum()]), np.concatenate([np.zeros(1), col_size_list.cumsum()])
        row_size_list, col_size_list = row_size_list.astype(np.int32), col_size_list.astype(np.int32)
        border_color = np.array(border_color)

        for y in row_size_list: input_img[y:y+line_width, :, :] = border_color
        for x in col_size_list: input_img[:, x:x+line_width, :] = border_color
        
        for yx, table in self.sub_tables.items():
            [y, x] = [int(i) for i in yx.split(':')]
            y1, y2 = row_size_list[y-1], row_size_list[y]+line_width
            x1, x2 = col_size_list[x-1], col_size_list[x]+line_width
            input_img[y1:y2, x1:x2] = table.draw_table(input_img=input_img[y1:y2, x1:x2])
        
        for yx, text in self.table_entries.items():
            [y, x] = [int(i) for i in yx.split(':')]
            y1, y2 = row_size_list[y-1], row_size_list[y]+line_width
            x1, x2 = col_size_list[x-1], col_size_list[x]+line_width
            self.draw_text(text, input_img, y1, y2, x1, x2)
        return input_img
    
class TransparentTable(LineTable):
    def draw_table(self, input_img=None):
        if input_img is None: input_img = np.ones((255, 255, 3), dtype=np.float32)
        M, N = input_img.shape[0], input_img.shape[1]
        row_size_list, col_size_list, line_width, border_color = self.row_size_list, self.col_size_list, self.line_width, self.border_color
        
        row_size_list, col_size_list = np.array(row_size_list), np.array(col_size_list)
        row_size_list, col_size_list = row_size_list*(M-line_width) / row_size_list.sum(), col_size_list*(N-line_width) / col_size_list.sum()
        row_size_list, col_size_list = np.concatenate([np.zeros(1), row_size_list.cumsum()]), np.concatenate([np.zeros(1), col_size_list.cumsum()])
        row_size_list, col_size_list = row_size_list.astype(np.int32), col_size_list.astype(np.int32)
        border_color = np.array(border_color)

        for yx, table in self.sub_tables.items():
            [y, x] = [int(i) for i in yx.split(':')]
            y1, y2 = row_size_list[y-1], row_size_list[y]+line_width
            x1, x2 = col_size_list[x-1], col_size_list[x]+line_width
            input_img[y1:y2, x1:x2] = table.draw_table(input_img=input_img[y1:y2, x1:x2])
        
        for yx, text in self.table_entries.items():
            [y, x] = [int(i) for i in yx.split(':')]
            y1, y2 = row_size_list[y-1], row_size_list[y]+line_width
            x1, x2 = col_size_list[x-1], col_size_list[x]+line_width
            self.draw_text(text, input_img, y1, y2, x1, x2)
        return input_img
    
class FlexTable(Table):
    def __init__(self, rows=1, cols=1, line_width=1, border_color=(0,0,0)):
        super().__init__()
        self.line_width = line_width
        self.border_color = border_color
        self.regrid(rows, cols)
        self.sub_tables = dict()
        self.table_entries = dict()
    def _compute_dim(self, elem):
        if isinstance(elem, Table):
            return elem.get_tabel_dim()
        return self._compute_text_dim(elem)
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
    def put_table(self, row, col, table):
        self.sub_tables["{}:{}".format(row, col)] = table
    def put_text(self, row, col, item):
        self.table_entries["{}:{}".format(row, col)] = str(item)
    def get_tabel_dim(self, row=None, col=None):
        if row is None: row = np.arange(1, self.rows)
        if col is None: col = np.arange(1, self.cols)
        if isinstance(row, int): row = [row]
        if isinstance(col, int): col = [col]
        
        result = np.zeros((2, len(row), len(col)), dtype=np.int32)
        
        for i,r in enumerate(row):
            for j,c in enumerate(col):
                yx = "{}:{}".format(r, c)
                elem = self.sub_tables.get(yx) or self.table_entries.get(yx)
                if elem is None: continue
                h, w  = self._compute_dim(elem)
                result[0, i, j] = h; result[1, i, j] = w
        h, w = result[0].max(axis=1).sum(), result[1].max(axis=0).sum()
        return (h, w)
    
    
    def draw_table(self, input_img=None):
        if input_img is None: input_img = np.ones((255, 255, 3), dtype=np.float32)
        M, N = input_img.shape[0], input_img.shape[1]
        line_width = self.line_width
        
        row_size_list = np.array([self.get_tabel_dim(row=row)[0] for row in range(self.rows)]).astype(np.int32)
        col_size_list = np.array([self.get_tabel_dim(col=col)[1] for col in range(self.cols)]).astype(np.int32)
        
        border_color = np.array(self.border_color)

        for y in row_size_list: input_img[y:y+line_width, :, :] = border_color
        for x in col_size_list: input_img[:, x:x+line_width, :] = border_color
        
        for yx, table in self.sub_tables.items():
            [y, x] = [int(i) for i in yx.split(':')]
            y1, y2 = row_size_list[y-1], row_size_list[y]+line_width
            x1, x2 = col_size_list[x-1], col_size_list[x]+line_width
            input_img[y1:y2, x1:x2] = table.draw_table(input_img=input_img[y1:y2, x1:x2])
        
        for yx, text in self.table_entries.items():
            [y, x] = [int(i) for i in yx.split(':')]
            y1, y2 = row_size_list[y-1], row_size_list[y]+line_width
            x1, x2 = col_size_list[x-1], col_size_list[x]+line_width
            self.draw_text(text, input_img, y1, y2, x1, x2)
        return input_img
        
