from genericpath import exists
import os
from importlib_metadata import distribution
from tqdm import tqdm
import shutil

main_dir = '../original_stem_after_lemmatize_stage_4/'
dirs = ['conversation/','fiction/','formal/','journalistic/','science/']
distr_dirs = ['./train/','./test/', './val/']
distribution=[#train     test    val
             [.65,      .25,    .10], #conversation
             [.80,      .10,    .10], #fiction
             [.70,      .15,    .15], #formal
             [.60,      .30,    .10], #journalistic
             [.75,      .20,    .5], #science
             ]
#создаем директории test train и val
for ddir in distr_dirs:   
    if not os.path.isdir(ddir):
        os.mkdir(ddir)

for id, dir in enumerate(dirs):
    print('dir ' + dir + ' in progress')

    for _, _, filenames in os.walk(main_dir+dir):
            original_fn = filenames

    original_fn.sort(key=len) #сортировка для порядка от 0 до 200
    train_am, test_am, val_am = int(len(original_fn)*distribution[id][0]), int(len(original_fn)*distribution[id][1]), int(len(original_fn)*distribution[id][2])
    #создаем поддиректории с классами
    for ddir in distr_dirs:   
        if not os.path.isdir(ddir+dir):
            os.mkdir(ddir+dir)

    for id, fn in tqdm(enumerate(original_fn)):
        if id < train_am:
            shutil.copy(main_dir+dir+fn, distr_dirs[0]+dir+fn)
        elif id >=train_am and id < train_am+test_am:
            shutil.copy(main_dir+dir+fn, distr_dirs[1]+dir+fn)
        elif id >= train_am+test_am:
            shutil.copy(main_dir+dir+fn, distr_dirs[2]+dir+fn)
