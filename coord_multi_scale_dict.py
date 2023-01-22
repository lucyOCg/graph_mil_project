#file to create graph dictionary for multi-scale graph

#import pandas as pd
import numpy as np
#import os
import h5py
from PIL import Image

#272197,

img_10x = '/nobackup/sclg/LMC_patches_10x_real/patches/272288.h5'
img_20x = '/nobackup/sclg/patches_20x_real/patches/272288.h5'
img_40x = '/nobackup/sclg/CLAM_master/patches_40x/256_pix/patches/272288.h5'


def get_patches_from_path(path):
    #print('path:', path)
    patch_file = h5py.File(path, "r")
    #print(patch_file['features'])
    return patch_file
    
def get_coord_tuple_list(h5_file):
    img_coords = h5_file['coords']
    img_coords_list = [] 
    img_coords_tuples = []
    for coord in img_coords:
        img_coords_list.append(coord)
        img_coords_tuples.append(tuple(coord))
    return img_coords_list, img_coords_tuples 

def coords_adjacent2(coords, dictionary):
    "Function to check if patches are adjacent, using patch coordinates as input," 
    "output gives list of ids tuples of adjacent patches"
    "e.g input = coords of patch idx, output = [(idx, idx2), (idx, idx3)]"
    "Where idx2 and idx3 are adjacent to patch idx"
    "diagonal patches are not included"
    edge_indices_1 = []
    x1, y1 = coords
    x2 = x1 + 256
    patch = dictionary.get((x2, y1))
    if patch != None:
        edge_indices_1 += [(dictionary[(x1,y1)],patch)]
    x2 = x1 - 256
    patch = dictionary.get((x2, y1))
    if patch != None:
        edge_indices_1 += [(dictionary[(x1,y1)],patch)]
    y2 = y1 + 256
    patch = dictionary.get((x1, y2))
    if patch != None:
        edge_indices_1 += [(dictionary[(x1,y1)],patch)]
    y2 = y1 - 256
    patch = dictionary.get((x1, y2))
    if patch != None:
        edge_indices_1 += [(dictionary[(x1,y1)],patch)]
    return edge_indices_1
    


def transform_coords(coord_list):
    img_coords_list_2 = np.multiply(coord_list, 2)
    img_coords_tuples = []
    for coord in img_coords_list_2:
        img_coords_tuples.append(tuple(coord))
    idx_list = list(range(len(img_coords_tuples)))
    #print(idx_list)
    coord_dict_1 = dict(zip(img_coords_tuples,idx_list))

    #Getting right adjacent patch coordinates
    x_coords = list(img_coords_list_2[:,0]+ 256)
    y_coords = list(img_coords_list_2[:,1])
    img_coords_list_3 = list(zip(x_coords, y_coords))
    #print('rh coord list:', img_coords_list_3[:10])
    coord_dict_2 = dict(zip(img_coords_list_3,idx_list))
    

    #Getting patch coords below top left patch coords
    x_coords = list(img_coords_list_2[:,0])
    y_coords = list(img_coords_list_2[:,1]-256)
    img_coords_list_4 = list(zip(x_coords, y_coords))
    #print('bottom coord list:', img_coords_list_4[:10])
    coord_dict_3 = dict(zip(img_coords_list_4,idx_list))

    #Getting patch coords for the diagonal patch coords
    x_coords = list(img_coords_list_2[:,0]+256)
    y_coords = list(img_coords_list_2[:,1]-256)
    img_coords_list_5 = list(zip(x_coords, y_coords))
    #print('diagonal coord list:', img_coords_list_5[:10])
    coord_dict_4 = dict(zip(img_coords_list_5,idx_list))

    final_coords = img_coords_tuples+img_coords_list_3+img_coords_list_4+img_coords_list_5
    coord_dict_1.update(coord_dict_2)#.update(coord_dict_3).update(coord_dict_4)
    coord_dict_1.update(coord_dict_3)
    coord_dict_1.update(coord_dict_4)
    
    return final_coords, coord_dict_1
    
#Getting top left patch coordinates    
def skewed_transform(coord_list, n, m):
    img_coords_list_2 = np.multiply(coord_list, 2)
    img_coords_tuples = []
    for coord in img_coords_list_2:
        img_coords_tuples.append(tuple(coord))
    idx_list = list(range(len(img_coords_tuples)))
    #print(idx_list)
    #coord_dict_1 = dict(zip(img_coords_tuples,idx_list))

    #Getting right adjacent patch coordinates
    x_coords = list(img_coords_list_2[:,0]+ 256+n)
    y_coords = list(img_coords_list_2[:,1]+m)
    img_coords_list_3 = list(zip(x_coords, y_coords))
    #print('rh coord list:', img_coords_list_3[:10])
    #print(idx_list)
    coord_dict_1 = dict(zip(img_coords_list_3,idx_list))

    #Getting patch coords below top left patch coords
    x_coords = list(img_coords_list_2[:,0]+n)
    y_coords = list(img_coords_list_2[:,1]-256+m)
    img_coords_list_4 = list(zip(x_coords, y_coords))
    #print('bottom coord list:', img_coords_list_4[:10])
    coord_dict_2 = dict(zip(img_coords_list_4,idx_list))

    #Getting patch coords for the diagonal patch coords
    x_coords = list(img_coords_list_2[:,0]+256 +n)
    y_coords = list(img_coords_list_2[:,1]-256+m)
    img_coords_list_5 = list(zip(x_coords, y_coords))
    #print('diagonal coord list:', img_coords_list_5[:10])
    coord_dict_3 = dict(zip(img_coords_list_5,idx_list))

    #Getting patch coords for the skewed patch coords
    x_coords = list(img_coords_list_2[:,0]+n)
    y_coords = list(img_coords_list_2[:,1]+m)
    img_coords_list_6 = list(zip(x_coords, y_coords))
    #print('diagonal coord list:', img_coords_list_5[:10])
    coord_dict_4 = dict(zip(img_coords_list_6,idx_list))

    #final_coords = img_coords_tuples_20x+img_coords_list_3+img_coords_list_4+img_coords_list_5
    final_coords = img_coords_list_3+img_coords_list_4+img_coords_list_5+img_coords_list_6
    coord_dict_1.update(coord_dict_2)#.update(coord_dict_3).update(coord_dict_4)
    coord_dict_1.update(coord_dict_3)
    coord_dict_1.update(coord_dict_4)
    return final_coords, coord_dict_1

 

def coords_scale(coords, transfomed_dict, coords_dict):
    "function that outputs the the ids of patches from different resolutions that are connected"
    edge_indices_1 = []
    patch = transfomed_dict.get(coords)
    #print(coords, patch)
    if patch != None:
        #print(coords_dict[coords])
        #print(transformed_dict[coords])
        edge_indices_1 += [(patch,coords_dict[coords])]
        #edge_indices_1 += [(transformed_dict[coords],coords_dict[coords])]
    return edge_indices_1


def coords_scale_2(coords, transfomed_dict, coords_dict):
    "function that outputs the the ids of patches from different resolutions that are connected"
    edge_indices_1 = []
    patch = transfomed_dict.get(coords)
    print('patch:',patch)
    #print(coords, patch)
    if patch != None:
        #print('x:',transformed_dict[coords])
        #print('y:',coords_dict[coords])
        #edge_indices_1 += [(patch,coords_dict[coords])]
        edge_indices_1 += [(transformed_dict[coords],coords_dict[coords])]
    return edge_indices_1


    
h5_10x = get_patches_from_path(img_10x)
h5_20x = get_patches_from_path(img_20x)
h5_40x = get_patches_from_path(img_40x)

h5_files = [h5_10x, h5_20x, h5_40x]


list_10x, tuple_10x = get_coord_tuple_list(h5_10x)
list_20x, tuple_20x = get_coord_tuple_list(h5_20x)
list_40x, tuple_40x = get_coord_tuple_list(h5_40x)

print('10x patches', len(list_10x))
print('20x patches', len(list_20x))
print('40x patches', len(list_40x))

#********************************* 10x -> 20x connections ***********************************
print('**************** 10x -> 20x connections ******************')

low_value = -2 #value of skew
high_value = 2 #value of skew


transformed_10x_coords, transformed_dict = transform_coords(list_10x)

skew_10x_list = []
skew_10x_dict = {}
for i in range(low_value,high_value):
    for j in range(low_value,high_value):
       skew_10x_coords, skew_dict = skewed_transform(list_10x, i, j)
       skew_10x_list += skew_10x_coords
       skew_10x_dict.update(skew_dict)
       

#print('transformed_dict:', len(transformed_dict))
#print('skew_dict:', len(skew_dict))
#print(skew_10x_list)
#print('skew_list:', len(skew_10x_list))
#print('skew_10x_dict:', len(skew_10x_dict))

#print(transformed_10x_coords[:10])
#print(skew_10x_coords[:10])
#print(tuple_20x[:10])
#skew_10x_dict = dict(skew_10x_dict)

#print(skew_10x_dict)
#print(final_coords)
#keys = [k for k, v in transformed_dict.items() if v == 706]
#print(keys)


idx_list_20x = list(range(len(tuple_10x),(len(tuple_10x)+len(tuple_20x))))
coord_dict_20x = dict(zip(tuple_20x,idx_list_20x))


edge_indices = []
for coord in tuple_20x:
    edge_index = coords_scale(coord, transformed_dict, coord_dict_20x)
    if edge_index != []:
        edge_indices.append(edge_index)
#
#print(len(edge_indices))
#print('skew_edge_indices', edge_indices)



#print(skew_dict)
skew_edge_indices = []
for coord in tuple_20x:
    edge_index = coords_scale(coord, skew_10x_dict, coord_dict_20x)
    if edge_index != []:
        skew_edge_indices.append(edge_index)


print('edge_indices:',len(edge_indices))
print('skew_edge_indices:',len(skew_edge_indices))
#print('skew_edge_indices', skew_edge_indices)


#********************************* 20x -> 40x connections ***********************************
print('**************** 20x -> 40x connections ******************')

transformed_20x_coords, transformed_dict_20x = transform_coords(list_20x)

skew_20x_list = []
skew_20x_dict = {}
for i in range(low_value,high_value):
    for j in range(low_value,high_value):
       skew_20x_coords, skew_dict = skewed_transform(list_20x, i, j)
       #print('skew_20x_coords',len(skew_20x_coords))
       skew_20x_list += skew_20x_coords
       skew_20x_dict.update(skew_dict)


idx_list_40x = list(range(len(tuple_20x),(len(tuple_20x)+len(tuple_40x))))
#print('idx_list_40x[:10]',idx_list_40x[:10])
#print('idx_list_40x[:-10]',idx_list_40x[-10:])
#print('len(tuple_40x)',len(tuple_40x))
#print('len(idx_list_40x)',len(idx_list_40x))
#print('len(skew_20x_list)',len(skew_20x_list))
#print('skew_20x_list[:10]',skew_20x_list[:10])
coord_dict_40x = dict(zip(tuple_40x,idx_list_40x))



edge_indices = []
for coord in tuple_40x:
    edge_index = coords_scale(coord, transformed_dict_20x, coord_dict_40x)
    if edge_index != []:
        edge_indices.append(edge_index)
#
print('number of indices:',len(edge_indices))
#print('edge_indices_40x', edge_indices)



#print(skew_dict)
skew_edge_indices = []
for coord in tuple_40x:
    edge_index = coords_scale(coord, skew_20x_dict, coord_dict_40x)
    if edge_index != []:
        skew_edge_indices.append(edge_index)

print('number of skewed indices:', len(skew_edge_indices))
#print('skew_edge_indices_40x', skew_edge_indices)



#print(transformed_20x_coords[:10])
#print(skew_20x_list[:10])
#print(tuple_40x[:10])
#print(len(transformed_20x_coords))
#print(len(skew_20x_coords))
#print(len(tuple_40x))
