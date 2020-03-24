# -*- coding: utf-8 -*-
import pdb
from skimage.morphology import watershed
import numpy as np
from skimage.morphology import reconstruction, dilation, erosion, disk, diamond, square
from skimage import img_as_ubyte
from skimage.color import label2rgb
from skimage.morphology import remove_small_objects
from scipy import ndimage as ndi
from skimage.morphology import label
from skimage.feature import peak_local_max

import skimage.morphology as morphology

def PrepareProb(img, convertuint8=True, inverse=True):
    if convertuint8:
        try:
            img = img_as_ubyte(img)
        except:
            pdb.set_trace()
    if inverse:
        img = 255 - img
    return img


def HreconstructionErosion(prob_img, h):

    def making_top_mask(x, lamb=h):
       return min(255, x + lamb)

    f = np.vectorize(making_top_mask)
    shift_prob_img = f(prob_img)

    seed = shift_prob_img
    mask = prob_img
    recons = reconstruction(
        seed, mask, method='erosion').astype(np.dtype('ubyte'))
    return recons


def find_maxima(img, convertuint8=False, inverse=False, mask=None):
    img = PrepareProb(img, convertuint8=convertuint8, inverse=inverse)
    recons = HreconstructionErosion(img, 1)
    if mask is None:
        return recons - img
    else:
        res = recons - img
        res[mask==0] = 0
        return res
def GetContours(img):
    """
    The image has to be a binary image 
    """
    img[img > 0] = 1
    return dilation(img, disk(2)) - erosion(img, disk(2))


def generate_wsl(ws):
    se = square(3)
    ero = ws.copy()
    ero[ero == 0] = ero.max() + 1
    ero = erosion(ero, se)
    ero[ws == 0] = 0

    grad = dilation(ws, se) - ero
    grad[ws == 0] = 0
    grad[grad > 0] = 255
    # pdb.set_trace()
    #try:
    #    return img_as_ubyte(grad)
    #except:
    grad = grad.astype(np.uint8)
    return grad

def DynamicWatershedAlias(p_img, lamb, p_thresh = 0.5):
    #pdb.set_trace()
    b_img = (p_img > p_thresh) + 0
    Probs_inv = PrepareProb(p_img)


    Hrecons = HreconstructionErosion(Probs_inv, lamb)
    markers_Probs_inv = find_maxima(Hrecons, mask = b_img)
    markers_Probs_inv = label(markers_Probs_inv)
    ws_labels = watershed(Hrecons, markers_Probs_inv, mask=b_img)
    arrange_label = ArrangeLabel(ws_labels)
    wsl = generate_wsl(arrange_label)
    arrange_label[wsl > 0] = 0
    

    return arrange_label

def ArrangeLabel(mat):
    val, counts = np.unique(mat, return_counts=True)
    background_val = val[np.argmax(counts)]
    mat = label(mat, background = background_val)
    if np.min(mat) < 0:
        mat += np.min(mat)
        mat = ArrangeLabel(mat)
    return mat


def Watershed_Dynamic(prob_image, param=7, thresh = 0.5):
    segmentation_mask = DynamicWatershedAlias(prob_image, param, thresh)
    segmentation_rgb = label2rgb(segmentation_mask,bg_label = 0, bg_color=(0.2, 0.5, 0.6))
    return segmentation_mask,segmentation_rgb
    
    
    
def Watershed_Proposed(mask,maskers):
     maskers = maskers & mask
     maskers = label(maskers)
     distance = ndi.distance_transform_edt(mask)
     labels = watershed(-distance, markers=maskers, mask=mask)
     labels = remove_small_objects(labels, 100)
     labelsrgb = label2rgb(labels,bg_label = 0, bg_color=(0.2, 0.5, 0.6))
     return labels, labelsrgb

def Watershed_Classical(mask):
    distance = ndi.distance_transform_edt(mask)
    local_maxi = peak_local_max(distance, indices=False,footprint=np.ones((2, 2)), 
                            labels=mask)
    markers = ndi.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask=mask)
    labels = remove_small_objects(labels, 100)
    labelsrgb = label2rgb(labels,bg_label = 0, bg_color=(0.2, 0.5, 0.6))
    return labels, labelsrgb

def condition_erosion(mask,erosion_structure,threshold):
    mask_process = np.zeros(np.shape(mask))
    mask_label,N = label(mask,return_num=True)
    for i in range(1,N+1):
        mask_temp = (mask_label==i)
        while np.sum(mask_temp)>=threshold:
            mask_temp = erosion(mask_temp,erosion_structure)
        mask_process = mask_process + mask_temp
    return mask_process
        
def Watershed_Condition_erosion(mask):
    fine_structure = morphology.diamond(1)
    coarse_structure = morphology.diamond(3)
    coarse_structure[3,0]=0
    coarse_structure[3,6]=0

    #==========step1 coarse erosion=============
    seed_mask = condition_erosion(mask,coarse_structure,200)
    #==========step2 fine erosion=============
    seed_mask = condition_erosion(seed_mask,fine_structure,50)
    
    distance = ndi.distance_transform_edt(mask)
    markers = label(seed_mask)
    labels = watershed(-distance, markers, mask=mask)
    labelsrgb = label2rgb(labels,bg_label = 0, bg_color=(0.2, 0.5, 0.6))
#    markersrgb = label2rgb(markers,bg_label = 0, bg_color=(0.2, 0.5, 0.6))
    return labels, labelsrgb
