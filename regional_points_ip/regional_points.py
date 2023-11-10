import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.colors import LinearSegmentedColormap
from skimage import data, img_as_float
from skimage.segmentation import active_contour
from skimage import measure
from skimage.measure import label,regionprops
import skimage
import nibabel as nib
import os
import numpy as np
from scipy.interpolate import interp1d
import itertools
from math import sqrt
from scipy.spatial import distance
import itertools

def curve_interpolate(ctr, param_list = [0.0, 0.5, 1.0] ,nsample=200):
    dist = np.cumsum(np.sqrt(np.sum(np.diff(ctr.T, axis=0)**2, axis=1 )) )
    dist = np.insert(dist, 0, 0)/dist[-1]
    
    i_function = interp1d(dist, ctr.T, kind='slinear', axis=0)
    i_pts = i_function(param_list)
    
    return i_pts.T

def septalwall_points(LVM, RV , threshold=2):
    intersection_points = []
    for p1 in RV:
        for p2 in LVM:
            dist = sqrt((p2[1] - p1[1])**2 + (p2[0]  - p1[0])**2)
            if dist<=threshold:
                intersection_points.append(p2)          
    return intersection_points

#Getting the contour points in order
def orderpoints(ip):
    ip2 = ip.tolist()
    for i in range(len(ip2)-1):
        xcord1 = ip2[i][0]
        ycord1 = ip2[i][1]
        xcord2 = ip2[i+1][0]
        ycord2 = ip2[i+1][1]
        dist_temp = sqrt(( xcord1 -  xcord2)**2 + ( ycord1  -  ycord2)**2)
        if dist_temp > 25: #threshold
            ip = np.roll(ip,-(i+1),axis=0)
            return ip            
    return ip

def getsegments(contourLVM,contourRV,param_list1,param_list2):

    c1_list = contourRV[0].tolist()
    c2_list = contourLVM[0].tolist()
   
    
    #Get points of the septal wall - i.e points common between LVM and RV contours
    #Threshold can be set to adjust septal wall point boundaries.
    intersection_points = septalwall_points(c2_list,c1_list,3)
    """intersection_points = []
    for c in c1_list:
        if c in c2_list:
            intersection_points.append(c)"""
    
    isp = sorted(intersection_points, key = lambda k : c2_list.index(k))
    ip = np.asarray(isp, dtype=np.float64)
    
    
    # Condition in case there is no common points or just one point
    if ip.size != 0 and ip.shape[0]!=1:
        
        #find_contours of skimage returns contour points as (y,x). Switch them to (x,y)
        ip[:,[0, 1]] = ip[:,[1, 0]]

        #Remove any duplicate points and preserve order
        _,idx=np.unique(ip, axis=0,return_index=True)
        ip = ip[np.sort(idx)]
        
        #Getting points in order
        ip = orderpoints(ip)
        
        x = [points[0] for points in ip]
        y = [points[1] for points in ip]
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        ctr = np.array([x, y])

        plt.plot(ctr[0],ctr[1])
    
        p_curve = curve_interpolate(ctr, param_list=param_list1)
        
        #Getting anterior, anterolateral, inferolateral and inferior
        
        A = [i for i in c2_list if i not in intersection_points]   
        c2_points = sorted(A, key = lambda k : c2_list.index(k))

        rp = np.asarray(c2_points, dtype=np.float64)
        _,idx = np.unique(rp, axis=0,return_index=True)
        rp = rp[np.sort(idx)]

        d1 = sqrt((p_curve[0,0] - p_curve[0,1])**2 + (p_curve[1,0] - p_curve[1,1])**2)
        d2 = sqrt((p_curve[0,2] - p_curve[0,1])**2 + (p_curve[1,2]  - p_curve[1,1])**2)

        mid_point = np.array([[ p_curve[1,1], p_curve[0,1]]])
        distances = distance.cdist(rp,mid_point,'euclidean')


        indices = np.argwhere(distances<min([d1,d2]))

        rp1 = np.delete(rp,indices, axis=0)

        rp1[:,[0, 1]] = rp1[:,[1, 0]]

        end_point = np.array([[p_curve[0,2], p_curve[1,2]]])
        dist2 = distance.cdist(rp1,end_point,'euclidean')
        index_min = np.where(dist2 == np.amin(dist2))[0][0]
        rp2 = np.roll(rp1,-index_min,axis=0)
        
        x2 = [points[0] for points in rp2]
        y2 = [points[1] for points in rp2]

        x2 = np.asarray(x2, dtype=np.float64)
        y2 = np.asarray(y2, dtype=np.float64) 
        ctr2 = np.array([x2, y2])

        #param_list2=[0, 0.25, 0.5, 0.75, 1]
        p_curve2 = curve_interpolate(ctr2, param_list=param_list2)
        
        return p_curve,p_curve2
    else:
        return np.array([]),np.array([])

def display_im_label(im, gt, titlef=""): # display the MR images along with ground truth labels
    
    # Set color maps for ground truth labels
    cm = matplotlib.colors.ListedColormap(['black', '#762a83', '#af8dc3', '#1b7837'])
    norm = matplotlib.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cm.N, clip=True)   
    color1 = itertools.cycle(["red", "blue", "green"])
    color2 = itertools.cycle(["darkviolet", "magenta", "yellow","orange", "cyan"])
    
    for i_s in range(im.shape[2]):
        
        #print(titlef+ " Frame at begin: "+str(i_s))
        #display.clear_output(wait=True) # clear figure
        
        im2d = im[:,:,i_s] # get 2D MR image
        gt2d = gt[:,:,i_s] # get 2D label

        plt.figure(figsize=(20,20))                 
        
        # show MR image
        fig, ax = plt.subplots()
        ax.imshow(im2d, cmap='gray') 
        ax.imshow(gt2d, cmap=cm, norm=norm, interpolation='none', alpha=0.3)
        
        #1 - for LV
        #2 - for LVM
        #3 - for RV
        contours1 = measure.find_contours(gt2d==1, 0.5)
        contours2 = measure.find_contours(gt2d==2, 0.5)
        contours3 = measure.find_contours(gt2d==3, 0.5)
        
        if len(contours3)!=0 and len(contours2)!=0:
            
            # Use this to get the end points of the septal wall - uses maximum distance method
            #End points of septal wall can also be found using function getsegments_septalwall() by passing param_list = [0,1]
            """x1,y1,x2,y2 = get_points(contours2,contours3)
            if not(x1==0 and y1==0 and x2 ==0 and y2 ==0):
                ax.plot(x1, y1, '.b', markersize=5)
                ax.plot(x2, y2, '.r', markersize=5)"""
            
            
            #Funtion below gets the segments for basal and and mid-cavity
            #Points returned in clockwise direction
            
            #Use param_list1 to get points for setal wall
            # between points 0 - 0.5 : infersospetal
            # between points 0.5 - 1 : anterosospetal
            
            #Use param_list2 to get anterior, anterolateral, inferolateral and inferior
            #Clockwise
            # between points 0 - 0.25 : anterior
            # between points 0.25 - 0.5 : anterolateral
            # between points 0.5 - 0.75 : inferolateral
            # between points 0.75 - 1: inferior
            
            param_list1=[0.0, 0.5, 1]
            param_list2=[0.0, 0.25, 0.5, 0.75, 1]
       
            septal_points, segment_points2 = getsegments(contours2,contours3,param_list1,param_list2)
            if septal_points.size != 0:
                for i in range(len(param_list1)):
                    #ax.plot(septal_points[0,i], septal_points[1,i], '.r') 
                    ax.scatter(septal_points[0,i], septal_points[1,i], color=next(color1), s = 5)
            
            if segment_points2.size != 0:
                for i in range(len(param_list2)):
                    #ax.plot(segment_points2[0,i], segment_points2[1,i], '.b')
                    ax.scatter(segment_points2[0,i], segment_points2[1,i], color=next(color2), s = 5)
               
            
        #Uncomment code below to plot contour borders. 
        #contours1 - LV
        #contours2 - LVM
        #contours3 - RV
        
        """
        for contour in contours1:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
        
        for contour in contours2:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
            
        for contour in contours3:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
        """   
        

        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        image_filename = titlef[:-7]+ "_Frame_"+str(i_s) + ".png"
        plt.axis('off')
        plt.title(titlef+ " Frame: "+str(i_s))
        patient_folder_name = os.path.join('/home/shreya/projects/rrg-punithak/shreya/Regional-Cardiac/regional_points_ip/figs',titlef[:-7])
        full_path = os.path.join(patient_folder_name,image_filename)
        os.makedirs(patient_folder_name, exist_ok=True)
        plt.savefig(full_path, transparent=True)
        plt.close()
        #plt.show()
        
        
        #plt.axis('off')
        #print(titlef+ " Frame: "+str(i_s))
        #plt.title(titlef+ " Frame: "+str(i_s))

        #plt.pause(3) 

def get_contours(directory_mri,directory_labels):
    for filename in os.listdir(directory_labels):
        if filename.endswith(".nii.gz"): 
            filename_mri = filename[:-7] + "_0000" + filename[-7:]
            img_pred_path = os.path.join(directory_labels, filename)
            img_mri_path = os.path.join(directory_mri, filename_mri)
            
            #import the nifti files
            img_pred = nib.load(img_pred_path).get_fdata()
            img_mri = nib.load(img_mri_path).get_fdata()
            
            #display contours
            display_im_label(img_mri,img_pred,filename)
        else:
            continue

def get_points(contour_LVM,contour_RV):
    c1_list = contour_LVM[0].tolist()
    c2_list = contour_RV[0].tolist()
    
    #Getting the points common between the two contours
    intersection_points = []
    for c in c1_list:
        if c in c2_list:
            intersection_points.append(c)
    ip = np.asarray(intersection_points, dtype=np.float64)
    
    if ip.size != 0:
        ip[:,[0, 1]] = ip[:,[1, 0]]
        max_distance = 0
        #dist = []
        x1,y1,x2,y2 = ip[0][0],ip[0][1],ip[0][0],ip[0][1]
        for a in ip:
            for b in ip:
                distance = sqrt( (a[0] - b[0])**2 + (a[1] - b[1])**2 )
                if distance > max_distance:
                    #print("caught")
                    max_distance = distance
                    #dist.append(max_distance)
                    x1,y1,x2,y2 = a[0], a[1], b[0], b[1]
        return x1,y1,x2,y2
    else:
        return 0,0,0,0

def main():
    directory_mri = "/home/shreya/projects/rrg-punithak/shreya/MM/nnUNet_raw_data_base/nnUNet_raw_data/Task114_heart_MNMs/imagesTs"
    directory_labels = "/home/shreya/projects/rrg-punithak/shreya/MNM_predictions/Predictions_3D"
    get_contours(directory_mri,directory_labels)

if __name__ == "__main__":
    main()