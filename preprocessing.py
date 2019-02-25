'''
Final proj - Team #7
part-related attribute(4)
'''
import os
import argparse
from PIL import Image
import numpy as np
import csv
from keras import utils

#datapath
path_list_attr_img = './data/list_attr_img.csv'
path_list_attr_cloth = './data/list_attr_cloth.csv'
train_data_dir = './Dataset/Train'
test_data_dir = './Dataset/Test'
label_file = './data/img_attr_part.csv'

# dimensions of our images
img_width, img_height = 224, 224

parser = argparse.ArgumentParser(description='Clothes Part Attribute')
parser.add_argument('-c', type=int, default=1000, help='number of read-in data')

# 列出要分類的類別數量(category組:50，attribute組則依attribute的個數自訂)
nb_classes = 216

# 可以定義每個class所代表的名稱，For Displaying
class_part = {
    0: 'arro_ollar',
    1: 'asymmetrica_em',
    2: 'bac_ow',
    3: 'bac_utout',
    4: 'bac_nit',
    5: 'bac_ace',
    6: 'bac_triped',
    7: 'backless',
    8: 'batwing',
    9: 'beade_ollar',
    10: 'bell',
    11: 'bell-sleeve',
    12: 'belted',
    13: 'belte_hiffon',
    14: 'belte_loral',
    15: 'belte_lora_rint',
    16: 'belte_ace',
    17: 'belte_axi',
    18: 'belte_laid',
    19: 'boa_eck',
    20: 'bow',
    21: 'bow-back',
    22: 'bow-front',
    23: 'box_ocket',
    24: 'braided',
    25: 'button',
    26: 'button-front',
    27: 'buttoned',
    28: 'cap-sleeve',
    29: 'chiffo_urplice',
    30: 'cinched',
    31: 'classi_rew',
    32: 'classi_re_eck',
    33: 'classi_ocket',
    34: 'classi_-neck',
    35: 'collar',
    36: 'colla_ace',
    37: 'collared',
    38: 'collarless',
    39: 'collarles_aux',
    40: 'colorbloc_ocket',
    41: 'contrast',
    42: 'contras_rim',
    43: 'contrast-trimmed',
    44: 'convertible',
    45: 'cowl',
    46: 'cow_eck',
    47: 'crew',
    48: 'cre_eck',
    49: 'crisscross',
    50: 'crisscross-back',
    51: 'croche_ringe',
    52: 'crochet-trimmed',
    53: 'cross-back',
    54: 'crossback',
    55: 'cuffed',
    56: 'cuffed-sleeve',
    57: 'curved',
    58: 'curve_em',
    59: 'cutout-back',
    60: 'dee_-neck',
    61: 'deep-v',
    62: 'dolman',
    63: 'dolma_leeve',
    64: 'dolman-sleeve',
    65: 'dolphin',
    66: 'dolphi_em',
    67: 'double-breasted',
    68: 'drape-front',
    69: 'draped',
    70: 'drape_pen-front',
    71: 'drape_hawl',
    72: 'drape_urplice',
    73: 'drawstring',
    74: 'dro_aist',
    75: 'drop-sleeve',
    76: 'drop-waist',
    77: 'dropped',
    78: 'elephant',
    79: 'elephan_rint',
    80: 'fau_eather-trimmed',
    81: 'fitte_-neck',
    82: 'flat',
    83: 'fla_ront',
    84: 'flat-front',
    85: 'flora_rin_trapless',
    86: 'flora_rin_urplice',
    87: 'flora_urplice',
    88: 'flounce',
    89: 'flounced',
    90: 'fluted',
    91: 'flutter',
    92: 'flutte_leeve',
    93: 'flutter-sleeve',
    94: 'fringe',
    95: 'fringed',
    96: 'gathere_aistline',
    97: 'graphi_acerback',
    98: 'heathere_-neck',
    99: 'hem',
    100: 'high-neck',
    101: 'high-slit',
    102: 'high-sli_axi',
    103: 'high-waist',
    104: 'high-waisted',
    105: 'hood',
    106: 'hooded',
    107: 'hoode_axi',
    108: 'hoode_tility',
    109: 'illusion',
    110: 'illusio_eckline',
    111: 'kangaroo',
    112: 'kangaro_ocket',
    113: 'keyhole',
    114: 'kni_pen',
    115: 'kni_ocket',
    116: 'kni_aglan',
    117: 'kni_hawl',
    118: 'kni_-neck',
    119: 'knotted',
    120: 'lac_eplum',
    121: 'lac_leeve',
    122: 'lac_rim',
    123: 'lace-trim',
    124: 'lace-trimmed',
    125: 'lace-up',
    126: 'ladder-back',
    127: 'lapel',
    128: 'leathe_eplum',
    129: 'leathe_rimmed',
    130: 'leather-trimmed',
    131: 'lon_leeve',
    132: 'long-sleeve',
    133: 'long-sleeved',
    134: 'm-slit',
    135: 'm-sli_axi',
    136: 'mes_acerback',
    137: 'mesh-trimmed',
    138: 'mock',
    139: 'moc_eck',
    140: 'mock-neck',
    141: 'nec_ibbed',
    142: 'nec_kater',
    143: 'nec_triped',
    144: 'neckline',
    145: 'notche_ollar',
    146: 'off-the-shoulder',
    147: 'one-button',
    148: 'one-shoulder',
    149: 'open-back',
    150: 'open-front',
    151: 'open-knit',
    152: 'open-shoulder',
    153: 'peplum',
    154: 'pin',
    155: 'pocket',
    156: 'prin_acerback',
    157: 'prin_trapless',
    158: 'prin_trappy',
    159: 'prin_urplice',
    160: 'prin_-neck',
    161: 'racerback',
    162: 'raglan',
    163: 'ragla_leeve',
    164: 'ruffl_rim',
    165: 'scallop',
    166: 'scalloped',
    167: 'scoop',
    168: 'scoop_neck',
    169: 'self-tie',
    170: 'shawl',
    171: 'shoulder',
    172: 'sid_lit',
    173: 'side-slit',
    174: 'single-button',
    175: 'sleeve',
    176: 'sleeveless',
    177: 'slit',
    178: 'split',
    179: 'split-back',
    180: 'split-neck',
    181: 'strap',
    182: 'strapless',
    183: 'straples_ribal',
    184: 'strappy',
    185: 'stripe_-neck',
    186: 'surplice',
    187: 'suspender',
    188: 't-back',
    189: 'tassel',
    190: 'tasseled',
    191: 'tie-back',
    192: 'tie-front',
    193: 'tie-neck',
    194: 'toggle',
    195: 'topstitched',
    196: 'trim',
    197: 'trimmed',
    198: 'tulip-back',
    199: 'turtle-neck',
    200: 'twist-front',
    201: 'twisted',
    202: 'two-button',
    203: 'v-back',
    204: 'v-cut',
    205: 'v-neck',
    206: 'vent',
    207: 'vente_em',
    208: 'y-back',
    209: 'zip',
    210: 'zip-front',
    211: 'zip-pocket',
    212: 'zip-up',
    213: 'zipped',
    214: 'zipper',
    215: 'zippered'
}

def DataPreprocessing(datacnt):

    print('Number of read data : ', datacnt)
    rows = csv.reader(open(label_file, newline=''))

    # Declare list for training and testing data and its' label
    train_data = []
    train_label = []

    # Simple example for loading training data.
    # loading testing data by yourselfi
    inow = 0
    for row in rows:
        if inow >= datacnt:
            break
        for root, dirs, files in os.walk(train_data_dir):
            for f in files:    
                fpath = os.path.join(root,f)
                if inow >= datacnt:
                    break

                if fpath.find(row[0]) != -1: #check filename ; windows -> \\ ; linux -> /
                    img = Image.open(fpath)
                    img = img.resize((img_width, img_height))
                    img = np.array(img)
                    imgr = np.resize(img,(img_width,img_height,3)) 
                    train_data.append(imgr)
                    tmp = np.zeros(216)
                    for i in range(1,len(list(row))):
                        if row[i].isdigit():
                            tmp[int(row[i])] = 1
                    train_label.append(tmp)
                    print(inow, imgr.shape)
                    inow = inow + 1

    # 將data type轉換成 numpy array
    train_data = np.array(train_data)
    train_label = np.array(train_label)

    # data shape(影像數量, 長, 寬, 深度) RGB深度=3、灰階=1
    print('training data shape : ', train_data.shape)
    print('training label shape :', train_label.shape)

    # 儲存處理好的data
    np.save('./data/train_data_'+str(datacnt)+'.npy',train_data)
    np.save('./data/train_label_'+str(datacnt)+'.npy', train_label)
    return train_data, train_label

def LoadCSV():
    img_attr = []
    cloth_attr = []
    index = []
    i=0

    imgrows = csv.reader(open(path_list_attr_img, newline=''))
    clothrows = csv.DictReader(open(path_list_attr_cloth, newline=''))

    for row in clothrows:
        if int(row['attribute_type']) == 4:
            cloth_attr.append(row)
            index.append(i)
            #print(i)
        i = i+1
    
    index = np.asarray(index)

    for row in imgrows:
        tmp = []
        imgname = str(row[0]).replace("img/","")
        tmp.append(imgname)
        check = 0
        for k in range(0,index.shape[0]-1):
            if int(row[index[k]+1]) == 1:
                tmp.append(k) #記錄第i個label
                check = 1
        #print(tmp)
        if check > 0:
            img_attr.append(tmp)
    
    with open(label_file, 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerows(img_attr)
    
    #print(img_attr[4])

def predResult(pred):
    print('Label\tpredict')
    for idx, tag in class_part.items():
        print(tag,'\t', pred[0][idx])

if __name__ == '__main__':
    args = parser.parse_args()

    if not os.path.isfile(label_file):
        print('read in csv label...')
        LoadCSV()
        print('finish')

    print('images reading...')
    train_data, train_label = DataPreprocessing(args.c)
    print('Data preprocessing finish')
