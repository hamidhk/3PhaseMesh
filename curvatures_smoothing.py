   
from sys import version
print('python version is,', version)
# This code is written in Python 3.7 installed using Conda package manager!
from skimage import io, measure
import numpy as np
from mayavi import mlab
from scipy import ndimage, spatial
from math import log10
from os import listdir
from time import time
import concurrent.futures
import pickle
from numba import jit


################################################################
################ ADJUST GLOBAL VARIABLES BELOW #################
################################################################
## pixel size in meters
pixel = 8.4*1E-6
## size of CPU l2-cache in byte,
## required for splitting large arrays, for example @smoothingParallel
l2_cache = 256000
## max number of threads (CPUs) @ multi-threading
num_workers = 4
################################################################
################################################################



def main():
    # lists to store name, total interfacial area & average mean curvature of images
    img_lst, Awn_lst, Hwn_lst = [], [], [] 
    path = '/media/Datas/sdb2/000-data/Schluter_wrr16_data/2-original-unzipped/test/'  # wrr2016 - MD
    entries = listdir(path)
    for entry in entries:
        if entry.endswith('.mhd'):
            print('Reading image', entry, 'as a numpy array')
            # open .raw image  as np.array (z,y,x) using simpleitk plugin of skimage
            img = io.imread(path + entry, plugin='simpleitk')
            #img = io.imread('/home/hamidh/000_img/26_QS_MI_b_B_nlm.tif') # tif image
            print('\nimage size & dimensions:', img.size, img.shape, '\n')
            # io.imshow(img[300,:,:])
            # io.show()
            print('{} nonwetting, {} wetting, voxels:'\
                   .format(np.sum(img == 0), np.sum(img == 1), '\n'))
            N = np.where(img == 0, img, -1) + 1     # contains only nonwetting (1); rest 0
            W = np.where(img == 1, img, 0)          # contains only wetting (1); rest 0
            # S = np.where(img == 2, img, 1) -1       # contains solid plus boundary (1); rest 0
            del img
            # W=W[0:50, 0:151, 0:153] #creating a small test volume
            # N=N[0:50, 0:151, 0:153]
            lbld_W, nr_blb_W = ndimage.measurements.label(W)
            lbld_N, nr_blb_N = ndimage.measurements.label(N)
            print(nr_blb_W, 'isolated wetting blob(s)')
            print(nr_blb_N, 'isolated nonwetting blob(s)', '\n')
            # returns which wetting and nonwetting blobs are neighbors
            print('finding neighbor blobs of vol. A & vol. B, eg. wetting & nonwetting neighbor blobs:')
            
#            nbr_blobs = labeledVolSetA_labeledVolSetB_neighborBlobs(lbld_W, lbld_N)

            # ################################# testing ###################################
            blb_W = np.where(lbld_W == 502, lbld_W, 0) # wetting (W) blob
            blb_N = np.where(lbld_N == 1, lbld_N, 0) # nonwetting (N) blob
            vertsW, facesW, normsW, valsW = measure.marching_cubes_lewiner(blb_W)
            vertsN, facesN, normsN, valsN = measure.marching_cubes_lewiner(blb_N)
            print('num. verts & faces in wetting mesh:', len(vertsW), len(facesW))
            print('num. verts & faces in nonwetting mesh: ', len(vertsN), len(facesN), '\n')
            print('extracting the interface...')         
            vertsWN, facesWN = meshA_meshB_commonSurface(vertsW, facesW, vertsN, facesN)
            if 1 == 1:
                print('num. verts & faces @ interface mesh: ', len(vertsWN), len(facesWN), '\n')
                nbr1 = verticesLaplaceBeltramiNeighborhoodParallel(facesWN, vertsWN)
                vertsWN2 = smoothingParallel(vertsWN, facesWN, nbr1, verts_constraint=1.7)
                mlab.figure(figure='smoothing', bgcolor=(0.95,0.95,0.95), size=(1000, 800))
                notsmooth = mlab.triangular_mesh(vertsWN[:,0], vertsWN[:,1], vertsWN[:,2], facesWN, representation='wireframe', color=(0,0,1))
                smooth  = mlab.triangular_mesh(vertsWN2[:,0], vertsWN2[:,1], vertsWN2[:,2], facesWN, color=(1,0,0)) 
                mlab.show()
            # #############################################################################

            Awn_all, kHwn_all = [], [] # lists to store areas and mean curvatures for an image
            for item in nbr_blobs:
                # A large droplet has often several smaller neighbors
                # outer if-else is to avoid creation of mesh every time when finding interfaces with the neighbors 
                if nr_blb_W > nr_blb_N:
                    # creates arrays where the rest of labels are masked out.
                    # masked arrays are passed to marching cube for triangulation.
                    blb_N = np.where(lbld_N == item[0], lbld_N, 0) # nonwetting (N) blob
                    vertsN, facesN, normsN, valsN = measure.marching_cubes_lewiner(blb_N) # mesh on N blob
                    for blb in item[1]:
                        blb_W = np.where(lbld_W == blb, lbld_W, 0) # wetting (W) blob
                        vertsW, facesW, normsW, valsW = measure.marching_cubes_lewiner(blb_W) # mesh on W blob   
                        # wet or nonwet mesh is not water-tight if it touches img boundaries
                        print('\n\n############### wetting blob ', blb, ' and nonwetting blob ', item[0], ' ###############')
                        print('num. verts & faces in nonwetting mesh: ', len(vertsN), len(facesN)) 
                        print('num. verts & faces in wetting mesh:', len(vertsW), len(facesW), '\n')
                        # Very small mesh may be simply noise. It also has little or no impact in area/curvature measurement.
                        # We eliminate them from smoothing process to speed up & avoid possible errors.
                        if len(vertsW) > 22 and len(vertsN) > 22:
                            # mainInner takes a wetting/nonwetting meshes, extracts interface, smooths
                            # & returns interface area (meter-squared) and mean curvature (1/meter)                     
                            # pixel_size in meter
                            aWN, kH = mainInner(vertsW, facesW, vertsN, facesN, pixel)
                            Awn_all.append(aWN)
                            kHwn_all.append(kH)
                        else:
                            # marks area & curvature arrays with np.inf (just something that does not happen in calc.)
                            Awn_all.append(np.inf)
                            kHwn_all.append(np.inf)
                else:
                    # creates arrays where the rest of labels are masked out.
                    # masked arrays are passed to marching cube for triangulation.
                    blb_W = np.where(lbld_W == item[0], lbld_W, 0) # wetting (W) blob
                    vertsW, facesW, normsW, valsW = measure.marching_cubes_lewiner(blb_W) # mesh on W blob
                    for blb in item[1]:
                        blb_N = np.where(lbld_N == blb, lbld_N, 0) # nonwetting (N) blob
                        vertsN, facesN, normsN, valsN = measure.marching_cubes_lewiner(blb_N) # mesh on N blob   
                        # wet or nonwet mesh is not water-tight if it touches img boundaries
                        print('\n\n############### wetting blob ', item[0], ' and nonwetting blob ', blb, ' ###############')
                        print('num. verts & faces in wetting mesh:', len(vertsW), len(facesW))
                        print('num. verts & faces in nonwetting mesh: ', len(vertsN), len(facesN), '\n') 
                        # Very small mesh may be simply noise. It also has little or no impact in area/curvature measurement.
                        # We eliminate them from smoothing process to speed up & avoid possible errors.
                        if len(vertsW) > 22 and len(vertsN) > 22:
                            # mainInner takes a wetting/nonwetting meshes, extracts interface, smooths
                            # & returns interface area (meter-squared) and mean curvature (1/meter)                     
                            # pixel_size in meter
                            aWN, kH = mainInner(vertsW, facesW, vertsN, facesN, pixel)
                            Awn_all.append(aWN)
                            kHwn_all.append(kH)
                        else:
                            # marks area & curvature arrays with np.inf (just something that does not happen in calc.)
                            Awn_all.append(np.inf)
                            kHwn_all.append(np.inf)

            Awn_all = np.array(Awn_all)        
            kHwn_all = np.array(kHwn_all)
            Awn_all = Awn_all[Awn_all!=np.inf]
            kHwn_all = kHwn_all[kHwn_all!=np.inf]
            print('\n\n####################################################')
            print('Summary for image: ', entry, '\n')
            print('\n array of interfacial areas:', Awn_all)
            print('\n array of mean curvatures:', kHwn_all)
            # saving results of one image
            # saves two arrays in one file
            np.savez(entry + '_res', Awn_all, kHwn_all)
            Awn = np.sum(Awn_all)               # in squared meter
            Hwn = np.sum(kHwn_all*Awn_all)/Awn  # in 1/meter
            print('\nTotal interfacial area, Awn (m**2) & average of mean curvatures, Hwn (1/m) for image ',\
                    entry, 'are: ', Awn, 'and ', Hwn)

            Awn_lst.append(Awn)
            Hwn_lst.append(Hwn)
            img_lst.append(entry)
            print('####################################################\n\n')

            ###### saving results after calc. for every image
            with open('Awn_lst.txt', 'wb') as fp:   # pickle
                pickle.dump(Awn_lst, fp)
            with open('Hwn_lst.txt', 'wb') as fp:   # pickle
                pickle.dump(Hwn_lst, fp)
            with open('img_lst.txt', 'wb') as fp:   # pickle
                pickle.dump(img_lst, fp)
            #########################################################################


            #     # kH, kG, k1, k2 = meanGaussianPrincipalCurvatures(vertsWN2, nV2, nbrWN)
            #     # sgn_kH = np.sign(kH)
            #     # mlab.figure(figure='smoothing', bgcolor=(0.95,0.95,0.95), size=(1000, 800))
            #     # surf1 = mlab.triangular_mesh(vertsWN2[:,0], vertsWN2[:,1], vertsWN2[:,2], facesWN, scalars=sgn_kH)
            #     # mlab.show()


            #     # visualizations
            #     # mlab.figure(figure='before & after smoothing', bgcolor=(0.95,0.95,0.95), size=(800, 700))
            #     # nl3wni = mlab.quiver3d(vl3WN[:,0], vl3WN[:,1], vl3WN[:,2], nwn[:,0], nwn[:,1], nwn[:,2], line_width=2, scale_factor=1, color=(1,0,0))
            #     # nl3si = mlab.quiver3d(vl3S[:,0], vl3S[:,1], vl3S[:,2], ns[:,0], ns[:,1], ns[:,2], line_width=2, scale_factor=1, color=(0.45,0.45,0.45))
            #     # # l3wni = mlab.points3d(vl3WN[:,0], vl3WN[:,1], vl3WN[:,2], np.ones(len(vl3WN)), scale_factor=0.2, color=(1,0,0))
            #     # # l3si = mlab.points3d(vl3S[:,0], vl3S[:,1], vl3S[:,2], np.ones(len(vl3S)), scale_factor=0.2, color=(0.45,0.45,0.45))
            #     # nl3wnf = mlab.quiver3d(vl3WN2[:,0], vl3WN2[:,1], vl3WN2[:,2], nwn2[:,0], nwn2[:,1], nwn2[:,2], line_width=2, scale_factor=1.8, color=(0,0,1))
            #     # nl3sf = mlab.quiver3d(vl3S2[:,0], vl3S2[:,1], vl3S2[:,2], ns2[:,0], ns2[:,1], ns2[:,2], line_width=2, scale_factor=1.8, color=(0.25,0.25,0.25))
            #     # l3wnf = mlab.points3d(vl3WN2[:,0], vl3WN2[:,1], vl3WN2[:,2], np.ones(len(vl3WN2)), scale_factor=0.2, color=(0,0,1))
            #     # l3sf = mlab.points3d(vl3S[:,0], vl3S[:,1], vl3S[:,2], np.ones(len(vl3S)), scale_factor=0.2, color=(0.25,0.25,0.25))
            #     # surfwni = mlab.triangular_mesh(vertsWN[:,0], vertsWN[:,1], vertsWN[:,2], facesWN, representation='wireframe', color=(1,0,0))
            #     # surfwnf = mlab.triangular_mesh(vertsWN2[:,0], vertsWN2[:,1], vertsWN2[:,2], facesWN, color=(0,0,1))
            #     # # nwni = mlab.quiver3d(vertsWN[:,0], vertsWN[:,1], vertsWN[:,2], nWN[:,0], nWN[:,1], nWN[:,2], line_width=1.5, scale_factor=1.1, color=(1,0,0))
            #     # # nwnf = mlab.quiver3d(vertsWN2[:,0], vertsWN2[:,1], vertsWN2[:,2], nWN2[:,0], nWN2[:,1], nWN2[:,2], line_width=1.5, scale_factor=1.1, color=(0,0,1))
            #     # # mlab.show()
            #     # surfsi = mlab.triangular_mesh(vertsSb[:,0], vertsSb[:,1], vertsSb[:,2], facesSb, representation='wireframe', color=(0.75,0.75,0.75))
            #     # surfsf = mlab.triangular_mesh(vertsSb2[:,0], vertsSb2[:,1], vertsSb2[:,2], facesSb, color=(0,1,0))
            #     # mlab.show()



def mainInner(vertsA, facesA, vertsB, facesB, pixel_size):
    print('Interface extraction...')
    # extracts verts/faces common in W & N meshes (WN interface)
    t0 = time()
    vertsAB, facesAB = meshA_meshB_commonSurface(vertsA, facesA, vertsB, facesB)
    if vertsAB is not None:
        print('num. verts & faces @ interface: ', len(vertsAB), len(facesAB), '\n')
        # # constraining solid mesh to qq = 5 pixel(s) vicinity of WN interface 
        # # (used in contact angle measurement)
        # zmin, zmax = np.min(vertsAB[:,0]), np.max(vertsAB[:,0])
        # ymin, ymax = np.min(vertsAB[:,1]), np.max(vertsAB[:,1])
        # xmin, xmax = np.min(vertsAB[:,2]), np.max(vertsAB[:,2])
        # vertsSb, facesSb = bracketMesh(vertsS, facesS, 5,\
        #                                  zmin, zmax, ymin, ymax, xmin, xmax)
        # # verts neighborhood map for WN - required for smoothing
        t1 = time()
        print('Constructing neighborhood map for interface...') 
        nbrAB = verticesLaplaceBeltramiNeighborhoodParallel(facesAB, vertsAB)
        # # isotropic smoothing, calc. surface area and mean curvature at WN interface
        t2 = time()
        print('Isotropic smoothing of interface...')
        vertsAB2 = smoothingParallel(vertsAB, facesAB, nbrAB, verts_constraint=1.7)     
        t3 = time()
        aAB = measure.mesh_surface_area(vertsAB2, facesAB)*(pixel_size**2)  # in squared meter
        nV = verticesUnitNormals(vertsAB2, facesAB)
        kH, aa = meanGaussianPrincipalCurvatures(vertsAB2, nV, nbrAB)
        kH = np.mean(kH)/pixel_size                                        # in 1/meter
        print('\nnum verts & faces at interface:', len(vertsAB), len(facesAB))
        print('Interface extraction in', t1-t0, 'sec')
        print('Neighborhood map construction in', t2-t1, 'sec')
        print('Smoothing in', t3-t2, 'sec\n')
        # mlab.figure(figure='smoothing', bgcolor=(0.95,0.95,0.95), size=(1000, 800))
        # notsmooth = mlab.triangular_mesh(vertsAB[:,0], vertsAB[:,1], vertsAB[:,2], facesAB, representation='wireframe', color=(0,0,1))
        # smooth  = mlab.triangular_mesh(vertsAB2[:,0], vertsAB2[:,1], vertsAB2[:,2], facesAB, color=(1,0,0)) 
        # mlab.show()
    else:
        # marks area & curvature arrays with np.inf which does not happen in calc.
        aAB = np.inf
        kH = np.inf
    # returns interfacial area (squared meter) & mean curvatures (1/meter) 
    return aAB, kH


def wrapper(func, *args, **kwargs):
    # to time a func. with arguments(example below)
    #wrapped = wrapper(func, *args)
    #print(timeit.timeit(wrapped, number=10))
     def wrapped():
         return func(*args, **kwargs)
     return wrapped


def arraySplitter(arr, **kwargs):
    # splits an arr into chunks which fit L2 cache of CPU
    # number of chunks can also be given as kwarg
    mm_default = max(1, int(arr.size*arr.itemsize/l2_cache))
    mm = kwargs.get('num_chunks', mm_default)
    d = int(len(arr)/mm)
    chunk_idx = []
    if mm == 1:
        chunk_idx.append([0, len(arr)])
    else:
        chunk_idx.append([0, d])
        for k in range(1,mm):
            lo = chunk_idx[k-1][1]
            if k < mm-1:
                hi = lo + d
            else:
                hi = len(arr)
            chunk_idx.append([lo, hi])
    return chunk_idx


def nbrLBSplitter(nbrLB):
    # similar to arraySplitter, but specific to neighborhood
    # map array! Splits this array for parallel smoothing
    # sorting (below) is necessary for correct nbrLB splitting
    nbrLB = nbrLB[nbrLB[:,0].argsort()]
    subnbr = arraySplitter(nbrLB)
    if len(subnbr) == 1:
        return subnbr
    else:
        hi0 = subnbr[0][1]
        hi0 = nbrLB[:,0][hi0]
        hi0 = max(np.where(nbrLB[:,0]==hi0)[0]) + 1
        subnbr[0] = [0, hi0]
        for i in range(1,len(subnbr)):
            lo = subnbr[i-1][1]
            hi = nbrLB[:,0][subnbr[i][1]-1]
            hi = max(np.where(nbrLB[:,0]==hi)[0]) + 1
            subnbr[i] = [lo, hi]
    return subnbr


def labeledVolSetA_labeledVolSetB_neighborBlobs(lA, lB):
    # receives two sets of labeled volumes (lA & lB) as imgs (np.arrays) where
    # the isolated blobs are labeled by scipy.ndimage.label for individual sets
    # A vols have common surface with B vols (but no common vols)
    # lA & lB must be 3d imgs & broadcastable (the same shape and size)
    # returns a list of labels of A and B which have common contact surface (AB)
    nghbr_ab = []
    # splits lA & lB to m*m*m sub-arrays for speed and memory efficiency
    m = 20
    zz, yy, xx = lA.shape
    zr, yr, xr = zz%m, yy%m, xx%m
    AA = np.zeros(shape=(zz+m-zr+1, yy+m-yr+1, xx+m-xr+1), dtype=lA.dtype)
    BB = np.zeros(shape=(zz+m-zr+1, yy+m-yr+1, xx+m-xr+1), dtype=lB.dtype)
    AA[0:zz, 0:yy, 0:xx] = lA
    BB[0:zz, 0:yy, 0:xx] = lB
    z,y,x = (AA.shape[0]-1)/m, (AA.shape[1]-1)/m, (AA.shape[2]-1)/m
    z,y,x = int(z), int(y), int(x)
    for i in range(1, m+1):
        for j in range(1, m+1):
            for k in range(1, m+1):
                # (+1)'s in array slicing below is to ensure neighbor blocks 
                # have a layer of common pixels. Neighborhood of blob X in lA
                # & blob Y in lB will not be identified, if X is at boarder
                # of a block, and Y is at boarder of the next block and if 
                # the two blocks have no common layer of pixels.
                aa = AA[(i-1)*z : i*z+1, (j-1)*y : j*y+1, (k-1)*x : k*x+1]
                bb = BB[(i-1)*z : i*z+1, (j-1)*y : j*y+1, (k-1)*x : k*x+1]
                la = np.unique(aa)
                la = la[la!=0]
                nr_a = len(la)
                lb = np.unique(bb)
                lb = lb[lb!=0]
                nr_b = len(lb)
                if nr_a!=0 and nr_b!=0:
                    # if-else below puts the set with least number of
                    # labels at the outer for loop for speedup
                    if nr_a <= nr_b:
                        for ii in la:
                            blb_aa = np.where(aa==ii, aa, 0)
                            blb_aa_dlt = ndimage.binary_dilation(blb_aa) # bool           
                            for jj in lb:
                                blb_bb = np.where(bb==jj, bb, 0)          
                                if np.logical_and(blb_aa_dlt, blb_bb).any():
                                # True if the two have intersection
                                    nghbr_ab.append((ii,jj))
                    else:
                        for jj in lb:
                            blb_bb = np.where(bb==jj, bb, 0)
                            blb_bb_dlt = ndimage.binary_dilation(blb_bb)           
                            for ii in la:
                                blb_aa = np.where(aa==ii, aa, 0)
                                if np.logical_and(blb_bb_dlt, blb_aa).any():
                                    nghbr_ab.append((ii,jj))
    
    ####### Alt #1 ######
    # returns all (i,j)'s where i is a blob in A and j is a blob in B
    # removing repeated items from nghbr_ab
    # lst = []
    # arr = np.array(nghbr_ab)
    # arr = arr[arr[:,0].argsort()]
    # ax0 = np.unique(arr[:,0])
    # for ss in ax0:
    #     crit = arr[:,0]==ss
    #     ax1 = np.unique(arr[:,1][crit])
    #     for tt in ax1:
    #         lst.append((ss,tt))
    # nghbr_ab = np.array(lst)
    # del lst
    # print(nghbr_ab)      
    # return nghbr_ab

    ####### Alt #2 ######
    # returns (i,j1,j2,j3...)'s where i is a blob in A and j1,j2,j3 are neighbors of i in B 
    lst = []
    arr = np.array(nghbr_ab)
    arr = arr[arr[:,0].argsort()]
    axA = np.unique(arr[:,0])
    axB = np.unique(arr[:,1])
    if len(axA)>len(axB):
        for ss in axB:
            crit = arr[:,1]==ss
            ax0 = np.unique(arr[:,0][crit])
            ax0=ax0.tolist()
            lst.append((ss,ax0))
    else:
        for ss in axA:
            crit = arr[:,0]==ss
            ax1 = np.unique(arr[:,1][crit])
            ax1=ax1.tolist()
            lst.append((ss,ax1))
    nghbr_ab = lst
    del lst
    print(nghbr_ab)      
    return nghbr_ab


def meshA_meshB_commonSurface(vertsA, facesA, vertsB, facesB):
    # receives verts/faces of mesh A & mesh B for two neighbor volumes
    # returns mesh (verts/faces) for  AB common surface

    def facesUnitNormalAndFilter(verts, faces):
        # returns the unit normals of faces
        tris = verts[faces]      
        nF = np.cross(tris[:,1] - tris[:,0], tris[:,2] - tris[:,0])
        del tris
        le = np.sqrt(nF[:,0]**2 + nF[:,1]**2 + nF[:,2]**2)
        le[le==0] = 2.2250738585072014e-308
        nF[:,0] /= le # length if nF becomes unity
        nF[:,1] /= le
        nF[:,2] /= le
        # le is also two-times the tirangle area; mask below removes
        # the triangle where the area is too large 
        # triangle with 3 equal edges of sqrt(3) is considered a limit
        le = le < 2.6
        faces = faces[le]
        nF = nF[le]
        return faces, nF


    def findFaces(idx, faces):
        nbrFc = [] # i-th element is the faces, vertex idx[i] is @
        numfc = []  # i-th element is number of those faces
        for i in range(len(idx)):
            fc = []
            a = faces[ faces[:,0] == idx[i] ]
            b = faces[ faces[:,1] == idx[i] ]
            c = faces[ faces[:,2] == idx[i] ]
            for x in (a,b,c):
                if len(x)>=1:
                    fc.extend(x)
            nbrFc.append(np.array(fc))
        for f in nbrFc:
            numfc.append(len(f))
        numfc = np.array(numfc)
        res = (nbrFc, numfc)
        return res


    def large_mesh_and_small_mesh(vertsA, facesA, vertsB, facesB):
        # this is helpful if two mesh have large difference in size,
        # bracketMesh removes a large part of larger mesh. Later, CPU time to
        # search for common surface, will be proportional to the number 
        # of verts in smaller mesh
        # if-else is to ensure bracketMesh acts on the larger mesh
        if len(facesA) <= len(facesB):
            zmin, zmax = np.min(vertsA[:,0]), np.max(vertsA[:,0])
            ymin, ymax = np.min(vertsA[:,1]), np.max(vertsA[:,1])
            xmin, xmax = np.min(vertsA[:,2]), np.max(vertsA[:,2])
            vertsB, facesB = bracketMesh(vertsB, facesB, 1, \
                                          zmin, zmax, ymin, ymax, xmin, xmax)
        else:
            zmin, zmax = np.min(vertsB[:,0]), np.max(vertsB[:,0])
            ymin, ymax = np.min(vertsB[:,1]), np.max(vertsB[:,1])
            xmin, xmax = np.min(vertsB[:,2]), np.max(vertsB[:,2])
            vertsA, facesA = bracketMesh(vertsA, facesA, 1, \
                                          zmin, zmax, ymin, ymax, xmin, xmax) 
        return  vertsA, facesA, vertsB, facesB


    def intersectAB_maskA_maskB_basic(vertsA, facesA, vertsB, facesB):
        print('Basic interface extraction func.')
        # below, if all 3 verts of A are also in B, the face in A is appended to facesAB
        # these faces also masked off (-2) in facesA.
        # in addition, faces in A with no common verts with B are masked off.
        facesAB = []
        triA = vertsA[facesA]
        triA0, triA1, triA2  = triA[:,0].tolist(), triA[:,1].tolist(), triA[:,2].tolist()
        vertsBl = vertsB.tolist()
        facesA_cp = facesA[:,0].copy()
        for i in range(len(triA)):
            c0 = triA0[i] in vertsBl
            c1 = triA1[i] in vertsBl
            c2 = triA2[i] in vertsBl
            if (c0 and c1 and c2): 
                facesAB.append(facesA[i])
                facesA_cp[i] = -2
            elif (not c0 and not c1 and not c2): 
                facesA_cp[i] = -2
        # what is left @ facesA, are faces with 1 or 2 common verts with mesh B.
        facesA = facesA[facesA_cp!=-2]
        del triA, triA0, triA1, triA2, vertsBl, facesA_cp
        # constructs facesAB, vertsAB for mesh AB (the common surface)
        facesAB = np.array(facesAB)
        vertsAB, facesAB = tidyUpMesh(vertsA, facesAB)
        # masking for mesh B, the same as above 
        triB = vertsB[facesB]
        triB0, triB1, triB2  = triB[:,0].tolist(), triB[:,1].tolist(), triB[:,2].tolist()
        facesB_cp = facesB[:,0].copy()
        vertsABl = vertsAB.tolist()
        for j in range(len(triB)):
            c0 = triB0[j] in vertsABl
            c1 = triB1[j] in vertsABl
            c2 = triB2[j] in vertsABl
            if (c0 and c1 and c2):
                facesB_cp[j] = -2    
            elif (not c0 and not c1 and not c2):
                facesB_cp[j] = -2
        # what is left @ facesB, are faces with 1 or 2 common verts with mesh A (or mesh AB).
        facesB = facesB[facesB_cp!=-2]
        del triB, triB0, triB1, triB2, vertsABl, facesB_cp # del large lists/array from memory
        vertsB, facesB = tidyUpMesh(vertsB, facesB)
        vertsA, facesA = tidyUpMesh(vertsA, facesA)
        return vertsAB, facesAB, vertsA, facesA, vertsB, facesB


    def intersectAB_maskA_advanced_parallel(vertsA, facesA, vertsB, facesB):
        
        def sequential(U):
            low, high = U[0], U[1]
            vertsB2, facesB2 = vertsB.copy(), facesB.copy()
            fa = facesA[low : high]
            ww = np.concatenate((fa[:,0], fa[:,1], fa[:,2]))
            ww = np.unique(ww)
            ww = ww[ww.argsort()]
            va = vertsA[ww]
            triA = vertsA[fa]
            triA0, triA1, triA2  = triA[:,0].tolist(), triA[:,1].tolist(), triA[:,2].tolist()
            legB2_i = len(facesB2)
            va, fa, vertsB2, facesB2 = large_mesh_and_small_mesh(va, fa, vertsB2, facesB2)
            legB2_f = len(facesB2)
            if legB2_f != legB2_i:
                vertsBl = vertsB2.tolist()
                for i in range(len(triA)):
                    c0 = triA0[i] in vertsBl
                    c1 = triA1[i] in vertsBl
                    c2 = triA2[i] in vertsBl
                    if (c0 and c1 and c2): 
                        facesAB.append(fa[i])
                        facesA2[low:high][i] = -2
                    elif (not c0 and not c1 and not c2): 
                        facesA2[low:high][i] = -2
            del triA, triA0, triA1, triA2, vertsB2, facesB2
            # end of sequential func.
        print('Advanced/parallel interface extraction for large meshes')
        facesAB = []
        facesA2 = facesA[:,0].copy()         
        # calculating the boundaries of sub-arrays of facesA
        n = int((10*1.152/15/8000)*len(facesA)) + 5  # rough num. of sub-sample of A for optimization
        d = int(len(facesA)/n)
        while d >= len(facesB):
            n +=1
            d = int(len(facesA)/n) 
        lst = []
        for k in range(n):
            low = k*d
            if k < n-1:
                high = (k+1)*d
            else:
                high = (k+1)*d + len(facesA)%n
            lst.append([low,high])

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            for outp  in executor.map(sequential, lst):
                pass

        facesA = facesA[facesA2!=-2]
        facesAB = np.array(facesAB)
        vertsAB, facesAB = tidyUpMesh(vertsA, facesAB)
        vertsA, facesA = tidyUpMesh(vertsA, facesA)
        return vertsAB, facesAB, vertsA, facesA


    def maskA_basic(vertsA, facesA, vertsB):
        triA = vertsA[facesA]
        triA0, triA1, triA2  = triA[:,0].tolist(), triA[:,1].tolist(), triA[:,2].tolist()
        facesA_cp = facesA[:,0].copy()
        vertsBl = vertsB.tolist()
        for j in range(len(triA)):
            c0 = triA0[j] in vertsBl
            c1 = triA1[j] in vertsBl
            c2 = triA2[j] in vertsBl
            if (c0 and c1 and c2):
                facesA_cp[j] = -2    
            elif (not c0 and not c1 and not c2):
                facesA_cp[j] = -2
        # what is left @ facesA, are faces with 1 or 2 common verts with mesh B
        facesA = facesA[facesA_cp!=-2]
        del triA, triA0, triA1, triA2, vertsBl, facesA_cp # del large lists/array from memory
        vertsA, facesA = tidyUpMesh(vertsA, facesA)
        return  vertsA, facesA


    def maskA_advanced_parallel(vertsA, facesA, vertsB, facesB):         

        def sequential(U):
            low, high = U[0], U[1]
            vertsB2, facesB2 = vertsB.copy(), facesB.copy()
            fa = facesA[low : high]
            ww = np.concatenate((fa[:,0], fa[:,1], fa[:,2]))
            ww = np.unique(ww)
            ww = ww[ww.argsort()]
            va = vertsA[ww]
            triA = vertsA[fa]
            triA0, triA1, triA2  = triA[:,0].tolist(), triA[:,1].tolist(), triA[:,2].tolist()
            legB2_i = len(facesB2)
            if len(fa)<len(facesB2):
                va, fa, vertsB2, facesB2 = large_mesh_and_small_mesh(va, fa, vertsB2, facesB2)
                legB2_f = len(facesB2)
                if legB2_f != legB2_i:
                    vertsBl = vertsB2.tolist()
                    for i in range(len(triA)):
                        c0 = triA0[i] in vertsBl
                        c1 = triA1[i] in vertsBl
                        c2 = triA2[i] in vertsBl
                        if (c0 and c1 and c2):
                            facesA2[low:high][i] = -2
                        elif (not c0 and not c1 and not c2): 
                            facesA2[low:high][i] = -2
            else:
                vertsBl = vertsB2.tolist()
                for i in range(len(triA)):
                    c0 = triA0[i] in vertsBl
                    c1 = triA1[i] in vertsBl
                    c2 = triA2[i] in vertsBl
                    if (c0 and c1 and c2):
                        facesA2[low:high][i] = -2
                    elif (not c0 and not c1 and not c2): 
                        facesA2[low:high][i] = -2
            del triA, triA0, triA1, triA2, vertsB2, facesB2 #,vertsBl,


        n = int((10*1.152/15/8000)*len(facesA)) + 5  # rough num. of sub-samples of A for optimization
        d = int(len(facesA)/n)
        while d >= len(facesB):
            n +=1
            d = int(len(facesA)/n)          
        facesA2 = facesA[:,0].copy()
        lst = []
        for k in range(n):
            low = k*d
            if k < n-1:
                high = (k+1)*d
            else: # k = n-1
                high = (k+1)*d + len(facesA)%n
            lst.append([low,high])
        # multi-threading
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            for inp, outp  in zip(lst, executor.map(sequential, lst)):
                pass
        facesA = facesA[facesA2!=-2]
        vertsA, facesA = tidyUpMesh(vertsA, facesA)
        return vertsA, facesA


    def condition_manager(vertsA, facesA, vertsB, facesB):
        if min(len(facesA), len(facesB))<=10000:
            vertsAB, facesAB, vertsA, facesA, vertsB, facesB = \
                intersectAB_maskA_maskB_basic(vertsA, facesA, vertsB, facesB)
        else:
            vertsAB, facesAB, vertsA, facesA = \
                intersectAB_maskA_advanced_parallel(vertsA, facesA, vertsB, facesB)
            vertsB, facesB = \
                maskA_advanced_parallel(vertsB, facesB, vertsAB, facesAB) # masks B using AB
        return vertsAB, facesAB, vertsA, facesA, vertsB, facesB


    if len(facesA)/len(facesB)>50 or len(facesB)/len(facesA)>50:
        vertsA, facesA, vertsB, facesB = \
            large_mesh_and_small_mesh(vertsA, facesA, vertsB, facesB)
        vertsAB, facesAB, vertsA, facesA, vertsB, facesB = \
            condition_manager(vertsA, facesA, vertsB, facesB)
    else:
        vertsAB, facesAB, vertsA, facesA, vertsB, facesB = \
            condition_manager(vertsA, facesA, vertsB, facesB)

    if len(facesAB)==0:
        print('none/insignificant intersection between A & B meshes')
        return None,None 
    else:
        # finds edges at border(s) of AB mesh
        # AB mesh may have several holes in addition to major
        # surrounding border; func. returns edges in all borders
        edgeAB = meshBorderEdges(facesAB)
        # @ AB mesh, patching missing single triangles which
        # touch borders (holes - happens seldom!)  
        idx = np.unique(np.concatenate((edgeAB[:, 0], edgeAB[:, 1])))
        qq = []
        for i in range(len(idx)):
            e0 = edgeAB[edgeAB[:,0]==idx[i]]
            e1 = edgeAB[edgeAB[:,1]==idx[i]]
            if len(e0) + len(e1) > 2:
                if len(e0)!=0 and len(e1)!=0:
                    yy = np.unique(np.concatenate((e0,e1)))
                elif len(e0)==0 and len(e1)!=0:
                    yy = np.unique(e1.flatten())
                elif len(e0)!=0 and len(e1)==0:
                    yy = np.unique(e0.flatten())
                yy = yy[yy!=idx[i]]
                qq.append((idx[i],yy))
        if len(qq)>0:
            for item in qq:
                x, lst = item[0], item[1]
                for j in lst:
                    e2 = edgeAB[edgeAB[:,0]==j]
                    e3 = edgeAB[edgeAB[:,1]==j]
                    if len(e2)==1 and len(e3)==1:
                        zz = np.unique(np.concatenate((e2,e3)))
                    elif len(e2)==0 and len(e3)==2:
                        zz = np.unique(e3.flatten())
                    elif len(e2)==2 and len(e3)==0:
                        zz = np.unique(e2.flatten())
                    zz = zz[zz!=x]
                    zz = zz[zz!=j]
                    e4 = edgeAB[edgeAB[:,0]==zz]
                    e5 = edgeAB[edgeAB[:,1]==zz]
                    tt = []
                    if len(e4)==1 and len(e5)==1:
                        tt = np.unique(np.concatenate((e4,e5)))
                    elif len(e4)==0 and len(e5)==2:
                        tt = np.unique(e1.flatten())
                    elif len(e4)==2 and len(e5)==0:
                        tt = np.unique(e4.flatten())
                    if x in tt:
                        fc = np.array([[x,j,zz]])
                        fc.sort()
                        facesAB = np.concatenate((facesAB, fc))
                        # new face (fc) added, now deleting edges of fc from edgeAB
                        edgeAB = edgeAB.tolist()
                        edgeAB.remove([fc[0][0], fc[0][1]])
                        edgeAB.remove([fc[0][1], fc[0][2]])
                        edgeAB.remove([fc[0][0], fc[0][2]])
                        edgeAB = np.array(edgeAB)
                        break
            del qq
        # creates  new faces at AB border where a border edge 
        # is common in both A and B meshes    
        facesA.sort()
        facesB.sort()
        facesA_cp = facesA.copy()
        facesB_cp = facesB.copy()
        eAB = vertsAB[edgeAB].tolist()
        vertsAB = vertsAB.tolist()
        facesAB_new = []   
        edgeAB_new = []
        idx_ab = []

        eA01 = np.vstack((facesA[:,0], facesA[:,1])).T
        eA01 = vertsA[eA01].tolist()
        eA12 = np.vstack((facesA[:,1], facesA[:,2])).T
        eA12 = vertsA[eA12].tolist()
        eA02 = np.vstack((facesA[:,0], facesA[:,2])).T
        eA02 = vertsA[eA02].tolist()

        eB01 = np.vstack((facesB[:,0], facesB[:,1])).T
        eB01 = vertsB[eB01].tolist()
        eB12 = np.vstack((facesB[:,1], facesB[:,2])).T
        eB12 = vertsB[eB12].tolist()
        eB02 = np.vstack((facesB[:,0], facesB[:,2])).T
        eB02 = vertsB[eB02].tolist()

        for i in range(len(eAB)):
            idxA, idxB = [], []
            if eAB[i] in eA01:
                j = eA01.index(eAB[i])
                idxA.append(facesA[j,2])
                facesA_cp[j] = -2
            elif eAB[i] in eA12:
                j = eA12.index(eAB[i])
                idxA.append(facesA[j,0])
                facesA_cp[j] = -2
            elif eAB[i] in eA02:
                j = eA02.index(eAB[i])
                idxA.append(facesA[j,1])
                facesA_cp[j] = -2

            if eAB[i] in eB01:
                k = eB01.index(eAB[i])
                idxB.append(facesB[k,2])
                facesB_cp[k] = -2
            elif eAB[i] in eB12:
                k = eB12.index(eAB[i])
                idxB.append(facesB[k,0])
                facesB_cp[k] = -2
            elif eAB[i] in eB02:
                k = eB02.index(eAB[i])
                idxB.append(facesB[k,1])
                facesB_cp[k] = -2

            if len(idxA) > 0 and len(idxB) > 0:
                f0, f1 = edgeAB[i]
                edgeAB[i] = -3
                vrtx = (vertsA[idxA[0]] + vertsB[idxB[0]])/2
                if [idxA[0], idxB[0]] not in idx_ab:
                    idx_ab.append([idxA[0], idxB[0]])
                    # updating vertsAB & facesAB_new
                    vertsAB.append(vrtx.tolist())
                    f2 = len(vertsAB) - 1
                    facesAB_new.append([f0, f1, f2])
                    edgeAB_new.append([f0, f2])
                    edgeAB_new.append([f1, f2])
                else:
                    f2 = vertsAB.index(vrtx.tolist())
                    facesAB_new.append([f0, f1, f2])
                    e0=[min(f0, f2), max(f0, f2)]
                    e1=[min(f1, f2), max(f1, f2)]
                    for e in [e0, e1]:
                        if e in edgeAB_new:
                            edgeAB_new.remove(e)
                        else:
                            edgeAB_new.append(e)    
        del facesA, facesB, facesA_cp, facesB_cp, idx_ab
        del eAB, eA01, eA12, eA02, eB01, eB12, eB02
        edgeAB_new = np.array(edgeAB_new)
        edgeAB = edgeAB[edgeAB[:,0]!=-3]
        if len(edgeAB_new)>0:
            edgeAB = np.concatenate((edgeAB, edgeAB_new))
        facesAB_new = np.array(facesAB_new)
        vertsAB = np.array(vertsAB)
        # finds norms for faces/verts for new faces at border 
        # if  dot prod. is negative, readjusts verts in faces
        # so the faces normals point the correct direction
        normF1 = facesUnitNormals(vertsAB, facesAB_new)
        subfaces = arraySplitter(facesAB)
        subverts = arraySplitter(vertsAB)
        nV = verticesUnitNormalsParallel(vertsAB, facesAB, subfaces, subverts)
        for i in range(len(facesAB_new)):
            f0,f1,f2 = sorted(facesAB_new[i])
            n0 = nV[f0] # or f1
            dotp = n0[0]*normF1[i,0] + n0[1]*normF1[i,1] + n0[2]*normF1[i,2]
            if dotp < 0:
                facesAB_new[i] = f2, f1, f0
        if len(facesAB_new)>0:
            facesAB = np.concatenate((facesAB, facesAB_new))
            del facesAB_new, edgeAB_new

        # segments of 3 connected verts at border
        idx = np.unique(np.concatenate((edgeAB[:, 0], edgeAB[:, 1])))
        seg = -np.ones(shape=(len(idx), 3), dtype=edgeAB.dtype)
        qq = []
        for i in range(len(idx)):
            e0 = edgeAB[edgeAB[:,0]==idx[i]]
            e1 = edgeAB[edgeAB[:,1]==idx[i]]
            #if len(e0)+len(e1)==2: (not necessary)
            if len(e0)==1 and len(e1)==1:
                xx = np.unique(np.concatenate((e0,e1)))
            elif len(e0)==0 and len(e1)==2:
                xx = np.unique(e1.flatten())
            elif len(e0)==2 and len(e1)==0:
                xx = np.unique(e0.flatten())                                 
            xx = xx[xx!=idx[i]]
            if len(xx) == 2:
                seg[i] = xx[0], idx[i], xx[1]
            # elif len(e0)+len(e1)==4: happens seldom! but possible to code!
        seg = seg[seg[:,0]!=-1]
        # not all idx's appear in seg. This causes error! 
        # line below simply updates idx & prevents errors in expense of ignoring a few points!
        idx = seg[:,1]

        #### to add or not to add the triangles in seg to facesAB 
        u,v,w = vertsAB[seg[:,0]], vertsAB[seg[:,1]], vertsAB[seg[:,2]]
        uv = v - u
        vw = w - v
        uv = unitVector(uv)
        vw = unitVector(vw)

        # using the arctan2 func. (block below) seems not to be the solution here!
        dot = uv[:,0]*vw[:,0] + uv[:,1]*vw[:,1] + uv[:,2]*vw[:,2]
        crs = np.cross(uv, vw)
        nrm = unitVector(crs)
        det = nrm[:,0]*(uv[:,1]*vw[:,2] - uv[:,2]*vw[:,1]) - nrm[:,1]*(uv[:,0]*vw[:,2] - uv[:,2]*vw[:,0]) + nrm[:,2]*(uv[:,0]*vw[:,1] - uv[:,1]*vw[:,0])
        angle3 = np.arctan2(det, dot)*180/np.pi
        angle3[angle3<0] += 360
        # some more corrections
        nbrFc, numfc = findFaces(idx, facesAB)
        seg1 = seg.copy()
        seg1[numfc!=1] = -1
        idx1 = seg1[:,2][seg1[:,2]!=-1]
        idx2 = idx[angle3 < 180]
        idx3 = np.intersect1d(idx1, idx2) # verts only in 1 triangle
        seg2 = seg.copy()
        if len(idx3)>0:
            fcs1 = []
            for item in idx3:
                if item in seg2[:,1]:
                    fc = seg2[seg2[:,1]==item]
                    fcs1.append(fc[0])
                    seg2[seg2[:,1] == fc[0,0]] = -1
                    seg2[seg2[:,1] == fc[0,1]] = -1
                    seg2[seg2[:,1] == fc[0,2]] = -1
            fcs1 = np.array(fcs1)

        seg2[numfc==1] = -1
        idx1 = seg2[:,1][seg2[:,1]!=-1]
        idx3 = np.intersect1d(idx1, idx2) # verts in more than 1 triangle
        if len(idx3)>0:
            fcs2 = []
            for item in idx3:
                if item in seg2[:,1]:
                    fc = seg2[seg2[:,1]==item]
                    fcs2.append(fc[0])
                    seg2[seg2[:,1] == fc[0,0]] = -1
                    seg2[seg2[:,1] == fc[0,1]] = -1
                    seg2[seg2[:,1] == fc[0,2]] = -1
            fcs2 = np.array(fcs2)   

        if len(fcs1)>0:
            fcs1, nfcs1 = facesUnitNormalAndFilter(vertsAB, fcs1)
            if len(fcs1)>0:
                nV1 = nV[fcs1[:,1]]  # nV (verts normals ) calculated above
                dotp1 = nV1[:,0]*nfcs1[:,0] + nV1[:,1]*nfcs1[:,1] + nV1[:,2]*nfcs1[:,2]
                dotp1 = np.sign(dotp1)
                fcs1 = fcs1[dotp1>0]
                facesAB = np.concatenate((facesAB, fcs1))

        if len(fcs2)>0:
            fcs2, nfcs2 = facesUnitNormalAndFilter(vertsAB, fcs2)
            if len(fcs2)>0:
                nV2 = nV[fcs2[:,1]]
                dotp2 = nV2[:,0]*nfcs2[:,0] + nV2[:,1]*nfcs2[:,1] + nV2[:,2]*nfcs2[:,2]
                dotp2 = np.sign(dotp2)
                fcs2 = fcs2[dotp2>0]
                facesAB = np.concatenate((facesAB, fcs2))

        del seg, seg1, seg2, idx, fcs1, fcs2, dotp1, dotp2, nfcs1, nfcs2
    return vertsAB, facesAB


def meshA_meshB_commonSurface_2_new_incomplete(vertsA, facesA, vertsB, facesB):

    def large_mesh_and_small_mesh(vertsA, facesA, vertsB, facesB):
        # this is helpful if two mesh have large difference in size,
        # bracketMesh removes a large part of larger mesh. Later, CPU time to
        # search for common surface, will be proportional to the number 
        # of verts in smaller mesh
        # if-else is to ensure bracketMesh acts on the larger mesh
        if len(facesA) <= len(facesB):
            zmin, zmax = np.min(vertsA[:,0]), np.max(vertsA[:,0])
            ymin, ymax = np.min(vertsA[:,1]), np.max(vertsA[:,1])
            xmin, xmax = np.min(vertsA[:,2]), np.max(vertsA[:,2])
            vertsB, facesB = bracketMesh(vertsB, facesB, 1, \
                                          zmin, zmax, ymin, ymax, xmin, xmax)
        else:
            zmin, zmax = np.min(vertsB[:,0]), np.max(vertsB[:,0])
            ymin, ymax = np.min(vertsB[:,1]), np.max(vertsB[:,1])
            xmin, xmax = np.min(vertsB[:,2]), np.max(vertsB[:,2])
            vertsA, facesA = bracketMesh(vertsA, facesA, 1, \
                                          zmin, zmax, ymin, ymax, xmin, xmax) 
        return  vertsA, facesA, vertsB, facesB


    vertsA, facesA, vertsB, facesB = large_mesh_and_small_mesh(vertsA, facesA, vertsB, facesB)
    mfacesA = (vertsA[facesA[:,0]] + vertsA[facesA[:,1]] + vertsA[facesA[:,2]])/2
    mfacesB = (vertsB[facesB[:,0]] + vertsB[facesB[:,1]] + vertsB[facesB[:,2]])/2
    lst = []
    if len(mfacesA) <= len(mfacesB):
        for i in range(len(mfacesA)):
            if min(np.sqrt(sum(((mfacesA[i] - mfacesB)**2).T))) <= 0.8:
                lst.append(i)
        facesAB = facesA[lst]
        vertsAB, facesAB = tidyUpMesh(vertsA, facesAB)
    else:
        for i in range(len(mfacesB)):
            if min(np.sqrt(sum(((mfacesB[i] - mfacesA)**2).T))) <= 0.8:
                lst.append(i)
        facesAB = facesB[lst]
        vertsAB, facesAB = tidyUpMesh(vertsB, facesAB)
    print('num verts and faces at interface:', len(vertsAB), len(facesAB))    
    mlab.figure(figure='interface test', bgcolor=(0.95,0.95,0.95), size=(1000, 800))
    s = mlab.triangular_mesh(vertsAB[:,0], vertsAB[:,1], vertsAB[:,2], facesAB, color=(0,0,1))
    mlab.show()
    return vertsAB, facesAB


def meshA_meshB_meshC_commonLine_incomplete(vertsA, facesA, vertsB, facesB, vertsC, facesC):
    # search for verts common in A & B & S
    vAB, vAC, vBC, vABC = [], [], [], []
    # if-else is for further speedup and memory efficiency
    #if len(facesA) <= len(facesB):
    vAidx = np.concatenate((facesA[:,0], facesA[:,1], facesA[:,2]))
    vAidx = np.unique(vAidx)
    vA = vertsA[vAidx]
    vBidx = np.concatenate((facesB[:,0], facesB[:,1], facesB[:,2]))
    vBidx = np.unique(vBidx)
    vB = vertsB[vBidx]
    vCidx = np.concatenate((facesC[:,0], facesC[:,1], facesC[:,2]))
    vCidx = np.unique(vCidx)
    vC = vertsC[vCidx]
    vAl= vA.tolist()
    vBl= vB.tolist()
    vCl= vC.tolist()
    for item in vAl:
        if item in vCl:
            vAC.append(item)
        if item in vBl:
            vAB.append(item)
            if item in vCl:
                vABC.append(item)
    for item in vBl:
        if item in vCl:
            vBC.append(item)    
    vAB, vAC, vBC = np.array(vAB), np.array(vAC), np.array(vBC)
    vABS = np.array(vABC)
    xx = vAB, vAC, vBC, vABS
    return xx


def meshAB_meshC_commonLine_incomplete(nbrLB_AB, vertsAB, vertsC, facesC):
    
    crit = nbrLB_AB[:,3]==-1 # if True, axes 0 & 1 are @ mesh boundary 
    l0 = nbrLB_AB[:,0][crit]    
    l1 = nbrLB_AB[:,1][crit]
    idxAB = np.concatenate((l0, l1))
    idxAB = np.unique(idxAB) # idxAB indices in AB mesh
    vAB = vertsAB[idxAB]     # verts in AB mesh
    idxC = np.concatenate((facesC[:,0], facesC[:,1], facesC[:,2]))
    idxC = np.unique(idxC)
    vC = vertsC[idxC]
    # search for verts common in AB & C 
    ABCinAB = [] # indices of ABC verts in AB
    ABCinC = []  # indices of ABC verts in C
    vAB= vAB.tolist()
    vC= vC.tolist()
    if len(idxAB) <= len(idxC):
        for ii in range(len(idxAB)):
            if vAB[ii] in vC:
                ABCinAB.append(idxAB[ii])
                ABCinC.append(vC.index(vAB[ii]))
    else:
        for jj in range(len(idxC)):
            if vC[jj] in vAB:
                ABCinC.append(idxC[jj])
                ABCinAB.append(vAB.index(vC[jj]))
    ABCinAB = np.array(ABCinAB)
    ABCinC = np.array(ABCinC)
    return ABCinAB, ABCinC


def tidyUpMesh(verts, faces):
    # inputs are np.arrays
    # when a number of faces & their corresponding verts are removed from 
    # a mesh, faces may contain intermittent verts indices.
    # This happens also when faces are subsampled from another mesh,
    # eg in search for common surface between two mesh.
    # func. receives faces changed in such a manner, together with vertices
    # of the original mesh, & returns faces with tidy indexing, keeping
    # their original connection with verts
    if len(faces)==0:
        print('Warning! faces array empty')
    else: 
        faces = faces[faces[:,2].argsort()] # sorts axes (2 first , 0 last) 
        faces = faces[faces[:,1].argsort()]
        faces = faces[faces[:,0].argsort()]
        # constructing np.array for verts
        idx = np.concatenate((faces[:,0], faces[:,1], faces[:,2]))
        idx = np.unique(idx)
        idx = idx[idx.argsort()]
        verts = verts[idx]
        # tidying faces
        for i in range(len(idx)):
            faces[faces == idx[i]] = i
       
        # # tidying faces (alternative 2)
        # # (vectorized) - one instruction, four several data points @ once 
        # i=0
        # while i+4 <= len(idx):
        #     faces[faces == idx[i]] = i
        #     faces[faces == idx[i+1]] = i+1
        #     faces[faces == idx[i+2]] = i+2
        #     faces[faces == idx[i+3]] = i+3
        #     i += 4
        # faces[faces == idx[-1]] = len(idx)-1
        # faces[faces == idx[-2]] = len(idx)-2 
        # faces[faces == idx[-3]] = len(idx)-3
    return verts, faces 


def bracketMesh(verts, faces, qq, zmin, zmax, ymin, ymax, xmin, xmax):
    # receives a mesh (verts and faces)
    # returns the mesh constrained by a box of coordinates as in func. args
    # constraining box will be away from mesh boarders by a distance of qq
    tri = verts[faces]
    zmin, ymin, xmin = zmin - qq, ymin - qq, xmin - qq
    zmax, ymax, xmax = zmax + qq, ymax + qq, xmax + qq
    idx1 = np.where(tri[:,:,0] < zmin)
    idx2 = np.where(tri[:,:,0] > zmax)
    idx3 = np.where(tri[:,:,1] < ymin)
    idx4 = np.where(tri[:,:,1] > ymax)
    idx5 = np.where(tri[:,:,2] < xmin)
    idx6 = np.where(tri[:,:,2] > xmax)
    # masking off out-of-box coordinates by setting them to -1
    for item in (idx1, idx2, idx3, idx4, idx5, idx6):
        tri[item[0]] = -1 # item[0] returns indices of zeroth-dim. of tri
    idx = np.where(tri!=-1)
    idx = np.unique(idx[0])
    if len(idx) > 0:
        faces = faces[idx]
        verts, faces = tidyUpMesh(verts, faces)
        return verts, faces
    else:
        print('The mesh is not in vicinity of the given coordinates. Proceeding with the rest of the code.')
        return verts, faces


def meshBorderEdges(faces):
    # returns the edges at the boundary of the mesh
    # an edge(i-j) @ boundary is in 1 face only
    # an edge inside mesh is in 2 faces
    # i & j in edge [i j] are indexes of verts
    fc = faces.copy()
    fc.sort()
    e01 = np.vstack((fc[:,0], fc[:,1])).T
    e12 = np.vstack((fc[:,1], fc[:,2])).T
    e02 = np.vstack((fc[:,0], fc[:,2])).T
    edge = np.concatenate((e01, e12, e02))
    del fc, e01, e12, e02  
    edge = edge[edge[:,0].argsort()]
    ls = []
    for i in range(len(edge)):
        u,v = edge[i]
        e0 = edge[edge[:,0]==u]
        e1 = e0[e0[:,1]==v]
        if len(e1) == 1:
            ls.append([u,v])
    border = np.array(ls)
    del edge
    if len(border) > 0:
        return border
    else:
        return print('Mesh has no boundary (closed surface).')


@jit(nopython=True)
def unitVector(vec):
    # returns array of unit vectors of a np.array with shape=(n,3)
    leng = np.sqrt(vec[:,0]**2 + vec[:,1]**2 + vec[:,2]**2)
    leng[leng==0]= 2.2250738585072014e-308 # avoids devision by zero if vector is for ex. (0,0,0)
    vec[:,0] /= leng
    vec[:,1] /= leng
    vec[:,2] /= leng                
    return vec


def facesUnitNormals(verts, faces):
    # returns the unit normals of faces
    tris = verts[faces]      
    nFace = np.cross(tris[:,1] - tris[:,0], tris[:,2] - tris[:,0])
    del tris
    nFace = unitVector(nFace) # normalizing (length=1)
    return nFace


def verticesUnitNormals(verts, faces):
    # returns the unit normals of vertices
    tris = verts[faces]
    # normals of faces         
    nFace = np.cross(tris[:,1] - tris[:,0], tris[:,2] - tris[:,0])
    del tris
    nFace = unitVector(nFace) # normalizing (length=1)
    nVerts = np.zeros(verts.shape, dtype=verts.dtype)
    # norms of a vertex found by adding norms of faces surrounding vertex
    nVerts[faces[:,0]] += nFace
    nVerts[faces[:,1]] += nFace
    nVerts[faces[:,2]] += nFace
    nVerts = unitVector(nVerts) # normalizing (length=1)
    return nVerts


def verticesUnitNormalsParallel(verts, faces, subfaces, subverts):
    # subverts is output of arrSplitter(verts) which splits
    # verts into smaller chunks for parallel computation
    # subfaces is the same for faces array
    def verticesUnitNormalsSmallArray(verts, faces):
        # returns the unit normals of vertices
        tris = verts[faces]
        # normals of faces         
        nF = np.cross(tris[:,1] - tris[:,0], tris[:,2] - tris[:,0])
        del tris
        nF = unitVector(nF) # normalizing (length=1)
        nV = np.zeros(verts.shape, dtype=verts.dtype)
        # norms of a vertex found by adding norms of faces surrounding vertex
        nV[faces[:,0]] += nF
        nV[faces[:,1]] += nF
        nV[faces[:,2]] += nF
        nV = unitVector(nV) # normalizing (length=1)
        return nV

    def verticesUnitNormalsLargeArray(verts, faces, subfaces, subverts):
        
        def facesUnitNormalsSequential(U):
            # returns the unit normals of faces
            l, h = U[0], U[1]
            tris = verts[faces[l:h]]      
            nF[l:h] = np.cross(tris[:,1] - tris[:,0], tris[:,2] - tris[:,0])
            nF[l:h] = unitVector(nF[l:h]) # normalizing (length=1)

        # def verticesNormalsSequential(U):     # this func has bug!
        ## returns the unit normals of vertices
        ## norms of a vertex found by adding norms of faces surrounding vertex
        #     l, h = U[0], U[1]
        #     nV[faces[l:h][:,0]] += nF[l:h]
        #     nV[faces[l:h][:,1]] += nF[l:h]
        #     nV[faces[l:h][:,2]] += nF[l:h]
        #     # nV not normalized yet!
        
        #@jit(nopython=True)
        def unitVectorSequential(U):
            # returns array of unit vectors of a np.array with shape=(n,3)
            l, h = U[0], U[1]
            leng = np.sqrt(nV[l:h][:,0]**2 + nV[l:h][:,1]**2 + nV[l:h][:,2]**2)
            leng[leng==0]= 2.2250738585072014e-308 # avoids devision by zero if vector is for ex. (0,0,0)
            nV[l:h][:,0] /= leng
            nV[l:h][:,1] /= leng
            nV[l:h][:,2] /= leng

        ## defining nF (norm faces) and nV (norm vertices) for entire domain
        nF = np.zeros(faces.shape, dtype=verts.dtype) # dtype the same as verts
        nV = np.zeros(verts.shape, dtype=verts.dtype)
        ## finding normals at faces (parallel)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            for stuff in executor.map(facesUnitNormalsSequential, subfaces):
                pass
        # finding normals at verts (parallel) - nV not yet normalized (leng!=1)
        # this part is just sequential (parallel has a bug)
        nV[faces[:,0]] += nF
        nV[faces[:,1]] += nF
        nV[faces[:,2]] += nF
        # normalizing (length = 1) nV's (parallel)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            for stuff in executor.map(unitVectorSequential, subverts):
                pass
        del nF
        return nV

    if len(subfaces) == 1: # sequential
        nV = verticesUnitNormalsSmallArray(verts, faces)
    else:                  # parallel    
        nV = verticesUnitNormalsLargeArray(verts, faces, subfaces, subverts)
    return nV
        

@jit(nopython=True)
def averageAllDotProducts(nbrLB, nV):
    # returns the sum of dot products of vectors n0 and n1
    # n0 & n1 symbolize unit normals for all pairs of neighbor verts
    # the sum is normalized with the number of all pairs
    # & returned value approaches to 1 when surfaces becoming smoother
    n0 = nV[nbrLB[:,0]]
    n1 = nV[nbrLB[:,1]]
    dotprd = n0[:,0]*n1[:,0] + n0[:,1]*n1[:,1] + n0[:,2]*n1[:,2]
    return np.sum(dotprd)/len(dotprd)


def averageAllDotProductsParallel(nV, nbrLB, subnbr):
    # subnbr is output of nbrLBSplitter(nbrLB) which splits
    # nbrLB into smaller chunks for parallel computation
    lst1, lst2 = [], []
    def averageAllDotProductsSequential(U):
        # returns the sum of dot products of vectors n0 and n1
        # n0 & n1 symbolize unit normals for all pairs of neighbor verts
        # the sum is normalized with the number of all pairs
        # & returned value approaches to 1 when surfaces becoming smoother
        nbr = nbrLB[U[0]: U[1]]     # nbr is a chunk of nbrLB
        n0 = nV[nbr[:,0]]
        n1 = nV[nbr[:,1]]
        dotprd = n0[:,0]*n1[:,0] + n0[:,1]*n1[:,1] + n0[:,2]*n1[:,2]
        lst1.append(np.sum(dotprd))
        lst2.append(len(dotprd))
    if len(subnbr) == 1:
        averageAllDotProductsSequential(subnbr[0])
        return lst1[0]/lst2[0]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            for stuff in executor.map(averageAllDotProductsSequential, subnbr):
                pass
        return sum(lst1)/sum(lst2)        


def verticesLaplaceBeltramiNeighborhood(faces, numVerts):
    # for every vertex, finds all faces at the 1-ring
    # neighborhood of vertex. Then, using this,
    # returns also vertices neighbor with vertex;
    # returns also a np.array with 4 axes, where
    # axes 0, 1 are all (i,j) vertices making an edge,
    # & axes 2,3 are the two vertices on opposite sides
    # of the (i,j) edge. If edge in the mesh boundary,
    # axes 3 will be -1. This 4D array is used in cotangent
    # discretization of Laplace-Beltrami operator for mean curvature.
    nbrFc = [] # i-th element is faces vertex i is at
    for i in range(numVerts):
        fc = []
        a = faces[faces[:,0]==i]
        b = faces[faces[:,1]==i]
        c = faces[faces[:,2]==i]
        for x in (a,b,c):
            if len(x)>=1:
                fc.extend(x)
        nbrFc.append(np.array(fc))

    nbrV = [] # i-th element is vertices neighbor with vertex i
    for ii in range(len(nbrFc)):
        lst=[]
        for y in nbrFc[ii]:
            lst.extend(y.flatten())
        lst = sorted(set(lst))
        lst.remove(ii)
        nbrV.append(lst)

    nbrLB = [] # construction of 4D np.array described above
    for i in range(len(nbrV)):
        for j in nbrV[i]:
            ls=[]
            r = nbrFc[i][nbrFc[i][:,0]==j]
            s = nbrFc[i][nbrFc[i][:,1]==j]
            t = nbrFc[i][nbrFc[i][:,2]==j]
            for x in (r,s,t):
                if len(x)>=1:
                    ls.extend(x.flatten())
            ls = sorted(set(ls))
            ls.remove(i)
            ls.remove(j)
            if len(ls) == 2: # edge i-j not in boundary of mesh
                nbrLB.append([i, j, ls[0], ls[1]])
            elif len(ls) == 1: # edge i-j at boundary of mesh
                nbrLB.append([i, j, ls[0], -1]) # -1 is just an identifier of boundary
    nbrLB = np.array(nbrLB)
    nbrLB = nbrLB[nbrLB[:,0].argsort()] # this sorting is necessary @smoothingParallel
    return nbrLB


def verticesLaplaceBeltramiNeighborhoodParallel(faces, verts):
    

    def verticesNeighborsA(faces, numVerts):
        # for every vertex, finds all faces at the 1-ring
        # neighborhood of vertex. Then, using this,
        # returns also vertices neighbor with vertex;
        # returns also a np.array with 4 axes, where
        # axes 0, 1 are all (i,j) vertices making an edge,
        # & axes 2,3 are the two vertices on opposite sides
        # of the (i,j) edge. If edge in the mesh boundary,
        # axes 3 will be -1. This 4D array is used in cotangent
        # discretization of Laplace-Beltrami operator for mean curvature.
        nbrFc = [] # i-th element is faces vertex i is at
        for i in range(numVerts):
            fc = []
            a = faces[faces[:,0]==i]
            b = faces[faces[:,1]==i]
            c = faces[faces[:,2]==i]
            for x in (a,b,c):
                if len(x)>=1:
                    fc.extend(x)
            nbrFc.append(np.array(fc))

        nbrV = [] # i-th element is vertices neighbor with vertex i
        for ii in range(len(nbrFc)):
            lst=[]
            for y in nbrFc[ii]:
                lst.extend(y.flatten())
            lst = sorted(set(lst))
            lst.remove(ii)
            nbrV.append(lst)

        nbrLB = [] # construction of 4D np.array described above
        for i in range(len(nbrV)):
            for j in nbrV[i]:
                ls=[]
                r = nbrFc[i][nbrFc[i][:,0]==j]
                s = nbrFc[i][nbrFc[i][:,1]==j]
                t = nbrFc[i][nbrFc[i][:,2]==j]
                for x in (r,s,t):
                    if len(x)>=1:
                        ls.extend(x.flatten())
                ls = sorted(set(ls))
                ls.remove(i)
                ls.remove(j)
                if len(ls) == 2: # edge i-j not in boundary of mesh
                    nbrLB.append([i, j, ls[0], ls[1]])
                if len(ls) == 1: # edge i-j at boundary of mesh
                    nbrLB.append([i, j, ls[0], -1]) # -1 is just an identifier of boundary
        nbrLB = np.array(nbrLB)
        return nbrLB
    

    def verticesNeighborsB(facesS, idxVt):
        nbrFc = [] # i-th element is faces vertex idxVt[i] is at
        for i in range(len(idxVt)):
            fc = []
            a = facesS[ facesS[:,0] == idxVt[i] ]
            b = facesS[ facesS[:,1] == idxVt[i] ]
            c = facesS[ facesS[:,2] == idxVt[i] ]
            for x in (a,b,c):
                if len(x)>=1:
                    fc.extend(x)
            nbrFc.append(np.array(fc))

        nbrV = [] # i-th element is vertices neighbor with vertex idxVt[i]
        for ii in range(len(nbrFc)):
            lst=[]
            for y in nbrFc[ii]:
                lst.extend(y.flatten())
            lst = sorted(set(lst))
            lst.remove(idxVt[ii])
            nbrV.append(lst)

        nbrLB = [] # construction of 4D np.array described above
        for i in range(len(nbrV)):
            for j in nbrV[i]:
                ls=[]
                r = nbrFc[i][nbrFc[i][:,0]==j]
                s = nbrFc[i][nbrFc[i][:,1]==j]
                t = nbrFc[i][nbrFc[i][:,2]==j]
                for x in (r,s,t):
                    if len(x)>=1:
                        ls.extend(x.flatten())
                ls = sorted(set(ls))
                ls.remove(idxVt[i])
                ls.remove(j)
                if len(ls) == 2: # edge i-j not in boundary of mesh
                    nbrLB.append([idxVt[i], j, ls[0], ls[1]])
                if len(ls) == 1: # edge i-j at boundary of mesh
                    nbrLB.append([idxVt[i], j, ls[0], -1]) # -1 is just an identifier of boundary
        nbrLB = np.array(nbrLB)
        return nbrLB


    def sequential(U):
        zit, zft, yit, yft, xit, xft = U[0], U[1], U[2], U[3], U[4], U[5]
        facest = faces.copy()
        idx1t = np.where(tri[:,:,0] < zit)
        idx2t = np.where(tri[:,:,0] > zft)
        idx3t = np.where(tri[:,:,1] < yit)
        idx4t = np.where(tri[:,:,1] > yft)
        idx5t = np.where(tri[:,:,2] < xit)
        idx6t = np.where(tri[:,:,2] > xft)
        for item in (idx1t, idx2t, idx3t, idx4t, idx5t, idx6t):
            facest[item[0]] = -1
        idx = np.where(facest!=-1)
        idx = np.unique(idx[0])
        facest = facest[idx]
        if len(facest) != 0:
            idxVt = np.concatenate((facest[:,0], facest[:,1], facest[:,2]))                
            idxVt = np.unique(idxVt)
            # search box (2 pixels larger than target box in every direction)
            facess = faces.copy()
            zis, zfs = zit - 2, zft + 2
            yis, yfs = yit - 2, yft + 2
            xis, xfs = xit - 2, xft + 2
            idx1s = np.where(tri[:,:,0] <= zis)
            idx2s = np.where(tri[:,:,0] >= zfs)
            idx3s = np.where(tri[:,:,1] <= yis)
            idx4s = np.where(tri[:,:,1] >= yfs)
            idx5s = np.where(tri[:,:,2] <= xis)
            idx6s = np.where(tri[:,:,2] >= xfs)
            for item in (idx1s, idx2s, idx3s, idx4s, idx5s, idx6s):
                facess[item[0]] = -1
            idx = np.where(facess!=-1)
            idx = np.unique(idx[0])
            facess = facess[idx]
            nbr = verticesNeighborsB(facess, idxVt)
            lst.append(nbr)


    def sequential_cleaning(U):
        # removing the repeated results
        for i in range(U[0],U[1]):
            crit = nbrLB[:,0] == i
            ax1 = nbrLB[:,1][crit]
            for j in range(len(ax1)-1):
                for k in range(j+1, len(ax1)):
                    if ax1[k] == ax1[j]:
                        ax1[k] = -2
            nbrLB[:,1][crit] = ax1


    nn = max(1, int(log10(len(verts))-3))
    # no data splitting for small arrays (sequential)
    if nn == 1:
        nbrLB = verticesNeighborsA(faces, len(verts))
    # spliting large array to nn subarrays
    else:
        zmin, zmax = np.min(verts[:,0]), np.max(verts[:,0])
        ymin, ymax = np.min(verts[:,1]), np.max(verts[:,1])
        xmin, xmax = np.min(verts[:,2]), np.max(verts[:,2])
        dz, dy, dx = (zmax-zmin)/nn, (ymax-ymin)/nn, (xmax-xmin)/nn
        lst = []
        tri = verts[faces]
        zyx = []
        for ii in range(1, nn+1):
            zit, zft = zmin + (ii-1)*dz, zmin + ii*dz + 1 # target box (z)
            for jj in range(1, nn+1):
                yit, yft = ymin + (jj-1)*dy, ymin + jj*dy + 1 # target box (y)
                for kk in range(1, nn+1):
                    xit, xft = xmin + (kk-1)*dx, xmin + kk*dx + 1 # target box (x)
                    zyx.append([zit, zft, yit, yft, xit, xft])
        # multi-threading on subarrays
        # and neighbors' search on individual parts
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            for outp  in executor.map(sequential, zyx):
                pass
        # assembling the results in nbrLB
        leng = 0
        leng_arr = []
        leng_arr.append(0)
        for arr in lst:
            leng += len(arr)
            leng_arr.append(leng)
        nbrLB = np.zeros(shape=(leng,4), dtype=faces.dtype)
        for i,W in zip(range(len(leng_arr)-1), lst):
            nbrLB[leng_arr[i] : leng_arr[i+1]] = W
        # calc. sub-intervals to split nbrLB into subarrays 
        sub = []
        mm = max(1, int((10*1.152/15/6000)*len(nbrLB)/5))   # rough num. of sub-arrays
        d = int(len(nbrLB)/mm)
        for k in range(mm):
            low = k*d
            if k < mm-1:
                high = (k+1)*d
            else:
                high = (k+1)*d + len(nbrLB)%mm
            sub.append([low,high])
        # removing repeated results in nbrLB (multi-threading)
        t0 = time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            for outp  in executor.map(sequential_cleaning, sub):
                pass
        nbrLB = nbrLB[nbrLB[:,1]!=-2]
    nbrLB = nbrLB[nbrLB[:,0].argsort()]   # this sorting is necessary @smoothingParallel
    return nbrLB


def meanGaussianPrincipalCurvatures(verts, nV, nbrLB, **kwargs):
    # returns a weight func.(wf) for smoothing triangle mesh data
    # returns also max triangle area
    # defualt wf is the mean curvature as in isotropic diffusion smoothing.
    # wf is calculated by anisotropic diffusion method, only 
    # if 'anis_diff' if given as args. In this method, Gaussian and 
    # principle curvatures are also calculated to find wf.

    # For details of the method, check:
    # "Discrete Differential-Geometry Operators for Triangulated 2-Manifolds"
    # by Meyer, Desbrun, Schroderl, Barr, 2003, Springer
    # @ "Visualization and Mathematics III" book, pp 35-57,  

    method = kwargs.get('method')

    # verts inside mesh have both alpha and beta angles
    # verts on the boundaries have only alpha angle
    # a for alpha-side and b for beta-side of an edge (i-j)
    vr0 = verts[nbrLB[:,0]] # vertices
    vr1 = verts[nbrLB[:,1]] # vertices
    vr2 = verts[nbrLB[:,2]] # vertices
    vr3 = verts[nbrLB[:,3]] # vertices
    # triangle sides (edges)
    ua = vr0 - vr2 # vectors for edges
    va = vr1 - vr2
    del vr2
    wab = vr0 - vr1
    ub = vr0 - vr3
    del vr0
    vb = vr1 - vr3
    del vr1, vr3
    withOutBeta = nbrLB[:,3]==-1
    ub[withOutBeta] = 0 # setting side ub to zero when beta doesn't exist
    vb[withOutBeta] = 0 # setting side vb to zero when beta doesn't exist

    uava = ua[:,0]*va[:,0] + ua[:,1]*va[:,1] + ua[:,2]*va[:,2] # dot prods
    uawab = ua[:,0]*wab[:,0] + ua[:,1]*wab[:,1] + ua[:,2]*wab[:,2]
    vawab = va[:,0]*wab[:,0] + va[:,1]*wab[:,1] + va[:,2]*wab[:,2]
    ubvb = ub[:,0]*vb[:,0] + ub[:,1]*vb[:,1] + ub[:,2]*vb[:,2]
    ubwab = ub[:,0]*wab[:,0] + ub[:,1]*wab[:,1] + ub[:,2]*wab[:,2]
    vbwab = vb[:,0]*wab[:,0] + vb[:,1]*wab[:,1] + vb[:,2]*wab[:,2]

    l2ua = ua[:,0]**2 + ua[:,1]**2 + ua[:,2]**2 # squared of lengths
    del ua
    l2va = va[:,0]**2 + va[:,1]**2 + va[:,2]**2
    del va
    l2wab = wab[:,0]**2 + wab[:,1]**2 + wab[:,2]**2
    l2ub = ub[:,0]**2 + ub[:,1]**2 + ub[:,2]**2
    del ub
    l2vb = vb[:,0]**2 + vb[:,1]**2 + vb[:,2]**2
    del vb

    # 2x Triangle area on alpha and beta sides of i,j edge
    areaTa = np.sqrt(l2ua*l2va-(uava**2))
    del l2va
    areaTb = np.sqrt(l2ub*l2vb-(ubvb**2))
    del l2ub, l2vb
    # smoothing may sometimes squeeze all 3 verts of a face so close
    # that the area becomes zero. This causes zero-division warning
    # & potentially errors. The two lines below is to prevent this!
    areaTa[areaTa==0] = 2.2250738585072014e-308
    areaTb[areaTb==0] = 2.2250738585072014e-308
    # max triangle area in the mesh
    maxa = 0.5*max(np.max(areaTa), np.max(areaTb[areaTb>0]))
    
    cota = uava/areaTa  # cot(alpha)
    cotb = np.zeros(shape=cota.shape, dtype=cota.dtype)
    # cot(beta), when beta exists
    withBeta = nbrLB[:,3]!=-1
    cotb[withBeta] = ubvb[withBeta]/areaTb[withBeta]
    del withBeta
    cotb[withOutBeta] = 0  # when beta doesn't exist (edge in boundary)
    del withOutBeta
    
    # three dot products to see if a triangle is obtuse
    # axis0 (alpha & beta); axis1 (angle by vert i); axis2 (the other angle)
    aa = np.vstack((uava, uawab, -vawab)).T
    del uava, vawab
    bb = np.vstack((ubvb, ubwab, -vbwab)).T
    del ubvb, ubwab, vbwab  
    
    # Av (A_voroni) stores areas of alpha and beta sides in its axes 0, 1
    Av = np.zeros(shape=(len(nbrLB),2), dtype=verts.dtype)
    # True if all three are positive (all angles>=90)
    aaAllPos = np.logical_and(aa[:,0]>=0, aa[:,1]>=0, aa[:,2]>=0) == True
    # voroni area where triangle in alpha-side is not obtuse
    Av[:,0][aaAllPos] = cota[aaAllPos]*l2wab[aaAllPos]/8
    del aaAllPos
    # True if all three are positive (all angles>=90)
    bbAllPos = np.logical_and(bb[:,0]>=0, bb[:,1]>=0, bb[:,2]>=0) == True
    # voroni area where triangle in beta-side is not obtuse
    Av[:,1][bbAllPos] = cotb[bbAllPos]*l2wab[bbAllPos]/8
    del bbAllPos
    # voroni area at alpha-side when triangle is obtuse at i-angle
    aa1Neg = aa[:,1]<0
    Av[:,0][aa1Neg] = areaTa[aa1Neg]/4
    del aa1Neg
    # voroni area at beta-side when triangle is obtuse at i-angle
    bb1Neg = bb[:,1]<0
    Av[:,1][bb1Neg] = areaTb[bb1Neg]/4
    del bb1Neg
    # voroni area at alpha-side when triangle is obtuse but not in i-angle
    aa0Neg = aa[:,0]<0
    Av[:,0][aa0Neg] = areaTa[aa0Neg]/8
    del aa0Neg
    aa2Neg = aa[:,2]<0
    del aa
    Av[:,0][aa2Neg] = areaTa[aa2Neg]/8
    del aa2Neg, areaTa
    # voroni area at beta-side when triangle is obtuse but not in i-angle
    bb0Neg = bb[:,0]<0
    Av[:,1][bb0Neg] = areaTb[bb0Neg]/8
    del bb0Neg
    bb2Neg = bb[:,2]<0
    del bb
    Av[:,1][bb2Neg] = areaTb[bb2Neg]/8
    del bb2Neg, areaTb

    # calc. Area mixed (Amxd) and mean curvature (kH) per vertex
    Amxd = np.zeros(len(verts), dtype=verts.dtype)
    kH = np.zeros(len(verts), dtype=verts.dtype)
    norm = nV[nbrLB[:,0]]
    del nV
    dotprd = wab[:,0]*norm[:,0] + wab[:,1]*norm[:,1] + wab[:,2]*norm[:,2]
    del norm, wab
    kk = (cota + cotb) * dotprd # per edge (i,j)
    del cota, cotb, dotprd
    for i in range(len(verts)):
        # Av's are voroni area per edge (i,j)
        # Av[:,0] & Av[:,1] for alpha & beta sides respec.
        # Amxd's are voroni area per vertex (i)
        crit = nbrLB[:,0]==i
        Amxd[i] = np.sum(Av[:,0][crit]) + np.sum(Av[:,1][crit])
        if Amxd[i] == 0:    # to prevent devision-by-zero error
            Amxd[i] = 2.2250738585072014e-308
        kH[i] = 0.25*np.sum(kk[crit])/Amxd[i] # for vertex i
    del Av, kk


    # wieght func. (wf) for anisotropic diffusion
    if method == 'aniso_diff':
        kH[kH==0] = 2.2250738585072014e-308  # to prevent devision-by-zero error
        # Gaussian curvature (kG)
        kG = np.zeros(len(verts), dtype=verts.dtype)
        tasum = np.zeros(len(verts), dtype=verts.dtype)
        # uawab is dotprod of two edges making theta (ta) angle at vertex i
        l2 = np.sqrt(l2ua*l2wab)
        del l2ua, l2wab
        l2[l2==0] = 2.2250738585072014e-308 # to prevent devision-by-zero error
        # costa = uawab/l2 # cos(ta)
        ta = np.arccos(uawab/l2)
        del uawab, l2
        for i in range(len(tasum)):
            tasum[i] = np.sum(ta[nbrLB[:,0]==i])
            kG[i] = (2*np.pi - tasum[i])/Amxd[i]
        del tasum, ta, Amxd, nbrLB

        # principal curvatures (k1, k2)
        dlta = kH**2 - kG
        dlta[dlta<0] = 0
        dlta = np.sqrt(dlta)
        k1 = kH + dlta
        k2 = kH - dlta
        del dlta

        # weight function (wf) for smoothing by aniso. diff.
        wf = np.ones(len(verts), dtype=verts.dtype)
        TT = 0.7 # user-defined parameter, feature-noise threshold
        kHabs = np.absolute(kH)
        k1abs = np.absolute(k1)
        k2abs = np.absolute(k2)
        # corners will not move in smoothing (wf=0)
        msk1 = np.logical_and(k1abs>TT, k2abs>TT, k1*k2>0)
        wf[msk1] = 0    
        # msk2 below is not initialized by np.zeros to avoid devision
        # by zero, below @ wf=k1/kH or k2/kH
        # initialized neither by np.ones, because k1,k2,kH can be 1
        # initialization by values larger than all curvatures
        mx = 1.1*max(np.max(kHabs), np.max(k1abs), np.max(k2abs))
        msk2 = mx*np.ones(len(verts), dtype=verts.dtype)
        del mx, verts
        for i in range(len(msk2)):
            if kHabs[i] != 0: # to avoid devision by zero @ wf=k1/kH or k2/kH
                msk2[i] = min(k1abs[i], k2abs[i], kHabs[i])
        del kHabs
        # for geometric or feature edges (not mesh edges), 
        # smoothing speed proportional to min curvature
        crit1 = k1abs==msk2
        del k1abs
        crit2 = k2abs==msk2
        del k2abs, msk2
        wf[crit1] = k1[crit1]/kH[crit1]
        del crit1
        wf[crit2] = k2[crit2]/kH[crit2]   
        del crit2
        # 3 lines below commented out; as wf is initialized by np.ones 
        #msk3 = np.logical_and(k1abs<=TT, k2abs<=TT)
        #wf[msk3] = 1    # isotropic smoothing for noisy regions
        #wf[kHabs==msk2] = 1 # isotropic smoothing for noisy regions
        wf[wf<-0.1] = -0.1 # stability thresh. to avoid strong inverse diff.
        wf = wf*kH
        #del kH
        # in smoothing by anis. diff. the verts are moved along
        # their unit normal vectors by wf*kH (x_new = x_old -(wf*kH)*normal)
        # in isotropic diffusion wf's simply mean curvature. kH (below)
    else:
        wf = kH
        del kH, nbrLB, verts, Amxd, uawab, l2ua, l2wab
    # return kH, kG, k1, k2      # returns all curvatures
    return wf, maxa


def meanGaussianPrincipalCurvaturesParallel(verts, nV, nbrLB, subnbr, **kwargs):
    # subnbr is output of nbrLBSplitter(nbrLB) which splits
    # nbrLB into smaller chunks for parallel computation
    method = kwargs.get('method')

    def meanGaussianPrincipalCurvaturesSequential(U):
        # returns a weight func.(wf) for smoothing triangle mesh data
        # returns also max triangle area
        # defualt wf is the mean curvature as in isotropic diffusion smoothing.
        # wf is calculated by anisotropic diffusion method, only 
        # if 'anis_diff' if given as args. In this method, Gaussian and 
        # principle curvatures are also calculated to find wf.

        # For details of the method, check:
        # "Discrete Differential-Geometry Operators for Triangulated 2-Manifolds"
        # by Meyer, Desbrun, Schroderl, Barr, 2003, Springer
        # @ "Visualization and Mathematics III" book, pp 35-57,  

        # nbr is a chunk of nbrLB
        nbr = nbrLB[U[0]: U[1]]
        vrt_idx = np.unique(nbr[:,0])   # verts idx in nbr

        # verts inside mesh have both alpha and beta angles
        # verts on the boundaries have only alpha angle
        # a for alpha-side and b for beta-side of an edge (i-j)
        vr0 = verts[nbr[:,0]] # vertices
        vr1 = verts[nbr[:,1]] # vertices
        vr2 = verts[nbr[:,2]] # vertices
        vr3 = verts[nbr[:,3]] # vertices
        # triangle sides (edges)
        ua = vr0 - vr2 # vectors for edges
        va = vr1 - vr2
        del vr2
        wab = vr0 - vr1
        ub = vr0 - vr3
        del vr0
        vb = vr1 - vr3
        del vr1, vr3
        withOutBeta = nbr[:,3]==-1
        ub[withOutBeta] = 0 # setting side ub to zero when beta doesn't exist
        vb[withOutBeta] = 0 # setting side vb to zero when beta doesn't exist

        uava = ua[:,0]*va[:,0] + ua[:,1]*va[:,1] + ua[:,2]*va[:,2] # dot prods
        uawab = ua[:,0]*wab[:,0] + ua[:,1]*wab[:,1] + ua[:,2]*wab[:,2]
        vawab = va[:,0]*wab[:,0] + va[:,1]*wab[:,1] + va[:,2]*wab[:,2]
        ubvb = ub[:,0]*vb[:,0] + ub[:,1]*vb[:,1] + ub[:,2]*vb[:,2]
        ubwab = ub[:,0]*wab[:,0] + ub[:,1]*wab[:,1] + ub[:,2]*wab[:,2]
        vbwab = vb[:,0]*wab[:,0] + vb[:,1]*wab[:,1] + vb[:,2]*wab[:,2]

        l2ua = ua[:,0]**2 + ua[:,1]**2 + ua[:,2]**2 # squared of lengths
        del ua
        l2va = va[:,0]**2 + va[:,1]**2 + va[:,2]**2
        del va
        l2wab = wab[:,0]**2 + wab[:,1]**2 + wab[:,2]**2
        l2ub = ub[:,0]**2 + ub[:,1]**2 + ub[:,2]**2
        del ub
        l2vb = vb[:,0]**2 + vb[:,1]**2 + vb[:,2]**2
        del vb

        # 2x Triangle area on alpha and beta sides of i,j edge
        areaTa = np.sqrt(l2ua*l2va-(uava**2))
        del l2va
        areaTb = np.sqrt(l2ub*l2vb-(ubvb**2))
        del l2ub, l2vb
        # smoothing may sometimes squeeze all 3 verts of a face so close
        # that the area becomes zero. This causes zero-division warning
        # & potentially errors. The two lines below is to prevent this!
        areaTa[areaTa==0] = 2.2250738585072014e-308
        areaTb[areaTb==0] = 2.2250738585072014e-308
        # max triangle area in the mesh
        maxa = 0.5*max(np.max(areaTa), np.max(areaTb[areaTb>0]))
     
        cota = uava/areaTa  # cot(alpha)
        cotb = np.zeros(shape=cota.shape, dtype=cota.dtype)
        # cot(beta), when beta exists
        withBeta = nbr[:,3]!=-1
        cotb[withBeta] = ubvb[withBeta]/areaTb[withBeta]
        del withBeta
        cotb[withOutBeta] = 0  # when beta doesn't exist (edge in boundary)
        del withOutBeta
        
        # three dot products to see if a triangle is obtuse
        # axis0 (alpha & beta); axis1 (angle by vert i); axis2 (the other angle)
        aa = np.vstack((uava, uawab, -vawab)).T
        del uava, vawab
        bb = np.vstack((ubvb, ubwab, -vbwab)).T
        del ubvb, ubwab, vbwab  
            
        # Av (A_voroni) stores areas of alpha and beta sides in its axes 0, 1
        Av = np.zeros(shape=(len(nbr),2), dtype=verts.dtype)   ################################
        # True if all three are positive (all angles>=90)
        aaAllPos = np.logical_and(aa[:,0]>=0, aa[:,1]>=0, aa[:,2]>=0) == True
        # voroni area where triangle in alpha-side is not obtuse
        Av[:,0][aaAllPos] = cota[aaAllPos]*l2wab[aaAllPos]/8
        del aaAllPos
        # True if all three are positive (all angles>=90)
        bbAllPos = np.logical_and(bb[:,0]>=0, bb[:,1]>=0, bb[:,2]>=0) == True
        # voroni area where triangle in beta-side is not obtuse
        Av[:,1][bbAllPos] = cotb[bbAllPos]*l2wab[bbAllPos]/8
        del bbAllPos
        # voroni area at alpha-side when triangle is obtuse at i-angle
        aa1Neg = aa[:,1]<0
        Av[:,0][aa1Neg] = areaTa[aa1Neg]/4
        del aa1Neg
        # voroni area at beta-side when triangle is obtuse at i-angle
        bb1Neg = bb[:,1]<0
        Av[:,1][bb1Neg] = areaTb[bb1Neg]/4
        del bb1Neg
        # voroni area at alpha-side when triangle is obtuse but not in i-angle
        aa0Neg = aa[:,0]<0
        Av[:,0][aa0Neg] = areaTa[aa0Neg]/8
        del aa0Neg
        aa2Neg = aa[:,2]<0
        del aa
        Av[:,0][aa2Neg] = areaTa[aa2Neg]/8
        del aa2Neg, areaTa
        # voroni area at beta-side when triangle is obtuse but not in i-angle
        bb0Neg = bb[:,0]<0
        Av[:,1][bb0Neg] = areaTb[bb0Neg]/8
        del bb0Neg
        bb2Neg = bb[:,2]<0
        del bb
        Av[:,1][bb2Neg] = areaTb[bb2Neg]/8
        del bb2Neg, areaTb

        # calc. Area mixed (Amxd) and mean curvature (kH) per vertex
        Amxd = np.zeros(len(vrt_idx), dtype=verts.dtype)
        kH = np.zeros(len(vrt_idx), dtype=verts.dtype)
        norm = nV[nbr[:,0]]

        dotprd = wab[:,0]*norm[:,0] + wab[:,1]*norm[:,1] + wab[:,2]*norm[:,2]
        del norm, wab
        kk = (cota + cotb) * dotprd # per edge (i,j)
        del cota, cotb, dotprd
        for i in range(len(vrt_idx)):
            # Av's are voroni area per edge (i,j)
            # Av[:,0] & Av[:,1] for alpha & beta sides respec.
            # Amxd's are voroni area per vertex (i)
            crit = nbr[:,0]==vrt_idx[i]
            Amxd[i] = np.sum(Av[:,0][crit]) + np.sum(Av[:,1][crit])
            if Amxd[i] == 0:    # to prevent devision-by-zero error
                Amxd[i] = 2.2250738585072014e-308
            kH[i] = 0.25*np.sum(kk[crit])/Amxd[i] # for vertex i

        del Av, kk

        # wieght func. (wf) for anisotropic diffusion
        # if 'aniso_diff' in args:
        if method == 'aniso_diff':
            kH[kH==0] = 2.2250738585072014e-308  # to prevent devision-by-zero error
            # Gaussian curvature (kG)
            kG = np.zeros(len(vrt_idx), dtype=verts.dtype)
            tasum = np.zeros(len(vrt_idx), dtype=verts.dtype)
            # uawab is dotprod of two edges making theta (ta) angle at vertex i
            l2 = np.sqrt(l2ua*l2wab)
            del l2ua, l2wab
            l2[l2==0] = 2.2250738585072014e-308 # to prevent devision-by-zero error
            # costa = uawab/l2 # cos(ta)
            ta = np.arccos(uawab/l2)
            del uawab, l2
            for i in range(len(tasum)):
                tasum[i] = np.sum(ta[nbr[:,0]==vrt_idx[i]])
                kG[i] = (2*np.pi - tasum[i])/Amxd[i]
            del tasum, ta, Amxd, nbr

            # principal curvatures (k1, k2)
            dlta = kH**2 - kG
            dlta[dlta<0] = 0
            dlta = np.sqrt(dlta)
            k1 = kH + dlta
            k2 = kH - dlta
            del dlta

            # weight function (wf) for smoothing by aniso. diff.
            wf = np.ones(len(vrt_idx), dtype=verts.dtype)
            TT = 0.7 # user-defined parameter, feature-noise threshold
            kHabs = np.absolute(kH)
            k1abs = np.absolute(k1)
            k2abs = np.absolute(k2)
            # corners will not move in smoothing (wf=0)
            msk1 = np.logical_and(k1abs>TT, k2abs>TT, k1*k2>0)
            wf[msk1] = 0    
            # msk2 below is not initialized by np.zeros to avoid devision
            # by zero, below @ wf=k1/kH or k2/kH
            # initialized neither by np.ones, because k1,k2,kH can be 1
            # initialization by values larger than all curvatures
            mx = 1.1*max(np.max(kHabs), np.max(k1abs), np.max(k2abs))
            msk2 = mx*np.ones(len(vrt_idx), dtype=verts.dtype)
            del mx
            for i in range(len(msk2)):
                if kHabs[i] != 0: # to avoid devision by zero @ wf=k1/kH or k2/kH
                    msk2[i] = min(k1abs[i], k2abs[i], kHabs[i])
            del kHabs
            # for geometric or feature edges (not mesh edges), 
            # smoothing speed proportional to min curvature
            crit1 = k1abs==msk2
            del k1abs
            crit2 = k2abs==msk2
            del k2abs, msk2
            wf[crit1] = k1[crit1]/kH[crit1]
            del crit1
            wf[crit2] = k2[crit2]/kH[crit2]   
            del crit2
            # 3 lines below commented out; as wf is initialized by np.ones 
            #msk3 = np.logical_and(k1abs<=TT, k2abs<=TT)
            #wf[msk3] = 1    # isotropic smoothing for noisy regions
            #wf[kHabs==msk2] = 1 # isotropic smoothing for noisy regions
            wf[wf<-0.1] = -0.1 # stability thresh. to avoid strong inverse diff.
            wf = wf*kH
            del kH
            # in smoothing by anis. diff. the verts are moved along
            # their unit normal vectors by wf*kH (x_new = x_old -(wf*kH)*normal)
            # in isotropic diffusion wf's simply mean curvature. kH (below)
        else:
            wf = kH
            del kH, nbr, Amxd, uawab, l2ua, l2wab   # del vrts not verts
        weight[vrt_idx] = wf         # smoothing weights (or curvatures) at vrt_idx chunk
        max_a.append(maxa)           # max tri. area at vrt_idx
        # end of sequentil func.


    weight = np.zeros(len(verts), dtype=verts.dtype)
    max_a = [] # stores max triangle area
    if len(subnbr) == 1: # sequential
        meanGaussianPrincipalCurvaturesSequential(subnbr[0])
        max_a = max_a[0]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # func called to calc. max triangle area; wt unimportant here
            for stuff in executor.map(meanGaussianPrincipalCurvaturesSequential, subnbr):
                pass
        max_a = max(max_a)                 
    return weight, max_a


def smoothing(verts, faces, nbrLB, **kwargs):
    # smooths a triangulated mesh e.g.the WN-interface
    # receives verts, faces, & Laplace-Beltrami neighborhood map (nbrLB) of verts
    # returns smoothed verts
    print('\nsmoothing iteration num.')
    nV = verticesUnitNormals(verts, faces) # unit normals of verts
    # weight called to calc. min/max triangle area; wt unimportant here
    wt, max_A = meanGaussianPrincipalCurvatures(verts, nV, nbrLB)
    del wt
    # new verts, smoothing criteria, verts distance from originals, 
    # & min/max face area @ iterations 
    # averageAllDotProducts returns average for dot products of all 
    # neighbors' unit normals
    # smoother surface will have an average approaching unity
    VV, smooth, constr, maxa = [], [], [], []
    smooth.append(np.float64(averageAllDotProducts(nbrLB, nV))) # at original verts
    constr.append(np.float64(0))
    maxa.append(max_A)
    VV.append(verts)
    del verts
    # mm is iter. counter @ while loop (must start @ 1)
    # the convergence is checked every nn iters.
    condition, mm, nn = True, 1, 50
    # DD max distance of each vertex from its original value
    DD_default = 1.7 # default DD is sqrt(3) - longest diam. in a voxel
    method = kwargs.get('method')
    DD = kwargs.get('verts_constraint', DD_default)
    verts_original = np.copy(VV[0])

    while condition:    # smoothing loop
        print(mm, end="  ")
        # verts tuned by moving along their unit normals
        # movement has a weight function (func. weight)
        # @ isotropic diff. weights are mean curvatures
        # @ aniso. diff. weights have feature/noise detection
        # by a thresholding variable (see TT @ wieght func.)
        # weights are multiplied by 0.1 to ensure not-to-fast
        # changes. This seems to be necessary in complex shapes.
        # new_vert = vert - 0.1*(weight)*(unit normal at vert)
        if method == 'aniso_diff':
            tune, max_a = meanGaussianPrincipalCurvatures(\
                                    VV[mm-1], nV, nbrLB, method='aniso_diff')
        else:
            tune, max_a = meanGaussianPrincipalCurvatures(\
                                    VV[mm-1], nV, nbrLB)
        verts_itr = VV[mm-1] - 0.1*(nV.T*tune).T
        del tune, nV
        # comparing verts_itr with the originals & correcting the jumped ones
        # constraint is to prevent displacement more than DD at every vert
        # if disp. is more than DD, vertex goes back to value @ previous iter.
        dd = sum(((verts_original - verts_itr)**2).T) # squared of distances
        jump = dd >= DD**2 # binary mask, True if a vertex jumped over constraint
        dd = sum(dd)
        verts_itr[jump] = VV[mm-1][jump]
        del jump

        # sum of squared of distance difference of updated and original verts
        constr.append(dd) #constr.append(sum(np.sqrt(dd)))
        VV.append(verts_itr) # save new verts
        del verts_itr
        maxa.append(max_a)
        nV = verticesUnitNormals(VV[-1], faces) # update norms with new verts
        smooth.append(np.float64(averageAllDotProducts(nbrLB, nV)))

        if  mm >= nn: 
            # checks to stop iter. @ step 50 & every single iter. after 50 
            # criteria 1
            # true when max smooth in the last nn iterations
            # happens before the very last iteration.
            kk = np.argmax(smooth)
            crit1 = kk < mm
            # criteria 2
            # true when the very last two iter's have a diff. less than 1e-06
            crit2 = abs(smooth[-1] - smooth[-2]) <= 1e-06

            if crit1 or crit2:
                # update verts with the ones gave max avg. dot prods. & stop iter.
                del nbrLB, faces, nV
                verts = VV[kk]
                VV[kk] = -1
                condition = False
                print('\n##############   Smoothing summary   ###############\n')
                print('printing initial (iter. 0) & last ', nn, 'iter.\n' )
                print('iter. nr.', 2*' ','ave. dot prods', 2*' ',\
                        'sum squared dist. from initial',\
                        3*' ', 'max triangle area')
                # printing 1st iter.
                print('0', 14*' ', smooth[0].round(7), 11*' ',\
                           constr[0].round(4), 22*' ', maxa[0].round(4))
                # printing last nn iter.
                for ii in range(mm+1-nn, mm+1):
                    print(ii, 14*' ', smooth[ii].round(7), 11*' ',\
                            constr[ii].round(4), 22*' ', maxa[ii].round(4))        
                print('\naverage for dot products of all neighbour unit', \
                        'normals is max at iter.', kk)
                print('the max of avg. dot products is', smooth[kk].round(5), \
                        'and sum of squared distance of verts from their originals is',\
                        constr[kk].round(2),'\n')
            if kk == mm:
                # smooth may still increase, so iter. does not stop
                VV[mm-nn:mm] = [-1]*nn
                # replaces unnecessary & large elements of VV with an integer
                # only last element is needed for further iter.
        mm += 1
    del VV, smooth, constr, maxa       
    return verts # smoothed verts


def smoothingParallel(verts, faces, nbrLB, **kwargs):
    # parallization of smoothing func. by domain decomposition 
    # i.e. splitting a mesh into smaller chunks
    # The parallelization is not trivial because the values of normals, voroni
    # area, curvature etc. at boundaries of chunks requires values from other chunks!

    # smooths a triangulated mesh e.g.the WN-interface
    # receives verts, faces, & Laplace-Beltrami neighborhood map (nbrLB) of verts
    # returns smoothed verts
    
    method = kwargs.get('method')   # to use anisotropic diffusion ('aniso_diff')

    def verticesUpdateParallel(verts, nV, tune, subverts):
        verts_itr = np.zeros(shape=verts.shape, dtype=verts.dtype)
        dd_lst = []
        def verticesUpdateSequential(U):
            verts_slice = verts[U[0]: U[1]]            # slice verts
            tune_slice = tune[U[0]: U[1]]              # slice tune
            nV_slice = nV[U[0]: U[1]]                  # slice nV
            verts_original_slice = verts_original[U[0]: U[1]]  # slice verts_original
            # calc. verts_itr 
            verts_itr[U[0]: U[1]] = verts_slice - 0.1*(nV_slice.T*tune_slice).T  
            del nV_slice, tune_slice
            # comparing verts_itr with the originals & correcting the jumped ones
            # constraint is to prevent displacement more than DD at every vert
            # if disp. is more than DD, vertex goes back to value @ previous iter.
            dd = sum(((verts_original_slice - verts_itr[U[0]: U[1]])**2).T) # squared of distances
            del verts_original_slice
            jump = dd >= DD**2
            dd = sum(dd)
            dd_lst.append(dd)
            verts_itr[U[0]:U[1]][jump] = verts_slice[jump]
            del jump, verts_slice
            # end of sequential func
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            for stuff in executor.map(verticesUpdateSequential, subverts):
                pass
        dd = sum(dd_lst)
        return verts_itr, dd


    # splitting faces, verts, nbrLB to smaller chunks for parallel computation
    subfaces = arraySplitter(faces)
    subverts = arraySplitter(verts)
    subnbr = nbrLBSplitter(nbrLB)

    nV = verticesUnitNormalsParallel(verts, faces, subfaces, subverts) # computing normals @ verts
    if method == 'aniso_diff':
        wt, max_A = meanGaussianPrincipalCurvaturesParallel(verts, nV, nbrLB, subnbr, method='aniso_diff')
    else:
        wt, max_A = meanGaussianPrincipalCurvaturesParallel(verts, nV, nbrLB, subnbr)
    del wt
    # # new verts, smoothing criteria, verts distance from originals, 
    # # & min/max face area @ iterations 
    # # averageAllDotProducts returns average for dot products of all 
    # # neighbors' unit normals
    # # smoother surface will have an average approaching unity
    VV, smooth, constr, maxa = [], [], [], []
    smooth.append(np.float64(averageAllDotProductsParallel(nV, nbrLB, subnbr))) # at original verts
    constr.append(np.float64(0))
    maxa.append(max_A)
    VV.append(verts)
    del verts
    # mm is iter. counter @ while loop (must start @ 1)
    # the convergence is checked every nn iters.
    condition, mm, nn = True, 1, 50
    # DD max distance of each vertex from its original value
    DD_default = 1.7 # default DD is sqrt(3) - longest diam. in a voxel
    DD = kwargs.get('verts_constraint', DD_default)
    verts_original = np.copy(VV[0])

    print('\nsmoothing iteration num.')
    while condition:    # smoothing loop
        print(mm, end="  ")
        # verts tuned by moving along their unit normals
        # movement has a weight function (func. weight)
        # @ isotropic diff. weights are mean curvatures
        # @ aniso. diff. weights have feature/noise detection
        # by a thresholding variable (see TT @ wieght func.)
        # weights are multiplied by 0.1 to ensure not-to-fast
        # changes. This seems to be necessary in complex shapes.
        # new_vert = vert - 0.1*(weight)*(unit normal at vert)
        if method == 'aniso_diff':
            tune, max_a = meanGaussianPrincipalCurvaturesParallel(VV[mm-1], nV, nbrLB, subnbr, method='aniso_diff')
        else:
            tune, max_a = meanGaussianPrincipalCurvaturesParallel(VV[mm-1], nV, nbrLB, subnbr)       
        verts_itr, dd = verticesUpdateParallel(VV[mm-1], nV, tune, subverts)

        constr.append(dd) #constr.append(sum(np.sqrt(dd)))
        VV.append(verts_itr) # save new verts
        del verts_itr
        maxa.append(max_a)
        nV = verticesUnitNormalsParallel(VV[-1], faces, subfaces, subverts) # update norms with new verts
        smooth.append(np.float64(averageAllDotProductsParallel(nV, nbrLB, subnbr)))

        if  mm >= nn: 
            # checks to stop iter. @ step 50 & every single iter. after 50 
            # checks if iteration should be ended
            # criteria 1
            # true when max smooth in the last nn iterations
            # happens before the very last iteration.
            kk = np.argmax(smooth)
            crit1 = kk < mm
            # criteria 2
            # true when the very last two iter's have a diff. less than 1e-06
            crit2 = abs(smooth[-1] - smooth[-2]) <= 1e-06

            if crit1 or crit2:
                # update verts with the ones gave max avg. dot prods. & stop iter.
                verts = VV[kk]
                condition = False
                print('\n##############   Smoothing summary   ###############\n')
                print('printing initial (iter. 0) & last ', nn, 'iter.\n' )
                print('iter. nr.', 2*' ','ave. dot prods', 2*' ',\
                        'sum squared dist. from initial',\
                        3*' ', 'max triangle area')
                # printing 1st iter.
                print('0', 14*' ', smooth[0].round(7), 11*' ',\
                           constr[0].round(4), 22*' ', maxa[0].round(4))
                # printing last nn iter.
                for ii in range(mm+1-nn, mm+1):
                    print(ii, 14*' ', smooth[ii].round(7), 11*' ',\
                            constr[ii].round(4), 22*' ', maxa[ii].round(4))        
                print('\naverage for dot products of all neighbour unit', \
                        'normals is max at iter.', kk)
                print('the max of avg. dot products is', smooth[kk].round(5), \
                        'and sum of squared distance of verts from their originals is',\
                        constr[kk].round(2),'\n')
            if kk == mm:
                # smooth may still increase, so iter. does not stop
                VV[mm-nn:mm] = [-1]*nn
                # replaces unnecessary & large elements of VV with an integer
                # only last element is needed for further iter.
        mm += 1
    del VV, tune, smooth, constr, maxa       
    return verts # smoothed verts


def smoothingStDev(verts, faces, nbrLB, **kwargs):
    # smooths a triangulated mesh e.g.the WN-interface
    # receives verts, faces, & Laplace-Beltrami neighborhood map (nbrLB) of verts
    # returns smoothed verts
    print('\nsmoothing iteration num.')
    verts_original = np.copy(verts)
    nV = verticesUnitNormals(verts, faces) # unit normals of verts
    # weight called to calc. min/max triangle area; wt unimportant here
    wt, max_A = meanGaussianPrincipalCurvatures(verts, nV, nbrLB)
    # new verts, smoothing criteria, verts distance from originals, 
    # & min/max face area @ iterations 
    # averageAllDotProducts returns average for dot products of all 
    # neighbors' unit normals
    # smoother surface will have a average approaching unity
    VV, smooth, constr,  maxa = [], [], [], []
    smooth.append(np.std(wt)) # at original verts
    del wt
    constr.append(np.float64(0))
    maxa.append(max_A)
    VV.append(verts)
    del verts
    # mm is iter. counter @ while loop (must start @ 1)
    # the convergence is checked every nn iters.
    condition, mm, nn = True, 1, 50
    # DD max distance of each vertex from its original value
    DD_default = 1.7 # default DD is sqrt(3) - longest diam. in a voxel
    method = kwargs.get('method')
    DD = kwargs.get('verts_constraint', DD_default)

    while condition:    # smoothing loop
        print(mm, end="  ")
        # verts tuned by moving along their unit normals
        # movement has a weight function (func. weight)
        # @ isotropic diff. weights are mean curvatures
        # @ aniso. diff. weights have feature/noise detection
        # by a thresholding variable (see TT @ weight func.)
        # weights are multiplied by 0.1 to ensure not-to-fast
        # changes. This seems to be necessary in complex shapes.
        # new_vert = vert - 0.1*(weight)*(unit normal at vert)
        if method == 'aniso_diff':
            tune, max_a = meanGaussianPrincipalCurvatures(\
                                    VV[mm-1], nV, nbrLB, method='aniso_diff')
        else:
            tune, max_a = meanGaussianPrincipalCurvatures(\
                                    VV[mm-1], nV, nbrLB)
        verts_itr = VV[mm-1] - 0.1*(nV.T*tune).T
        smooth.append(np.std(tune))
        del tune
        # comparing verts_itr with the originals & correcting the jumped ones
        # constraint is to prevent displacement more than DD at every vert
        # if disp. is more than DD, vertex goes back to value @ previous iter.
        dd = sum(((verts_original - verts_itr)**2).T) # squared of distances
        jump = dd >= DD**2 # binary mask, True if a vertex jumped over constraint
        dd = sum(dd)
        verts_itr[jump] = VV[mm-1][jump]
        del jump
        VV.append(verts_itr) # save new verts
        del verts_itr
        maxa.append(max_a)
        # sum of squared of distance difference of updated and original verts
        constr.append(dd) #constr.append(sum(np.sqrt(dd)))

        nV = verticesUnitNormals(VV[-1], faces) # update norms with new verts

        if  mm >= nn: 
            # checks to stop iter. @ step 50 & every single iter. after 50 
            # criteria 1
            # true when min smooth in the last nn iterations
            # happens before the very last iteration.
            kk = np.argmin(smooth)
            crit1 = kk < mm
            # criteria 2
            # true when the very last two iter's have a diff. less than 1e-06
            crit2 = abs(smooth[-1] - smooth[-2]) <= 1e-06

            if crit1 or crit2:
                # update verts with the ones gave max avg. dot prods. & stop iter.
                del nbrLB, faces, nV
                verts = VV[kk-1]
                VV[kk-1] = -1
                condition = False
                print('\n##############   Smoothing summary   ###############\n')
                print('printing initial (iter. 0) & last ', nn, 'iter.\n' )
                print('iter. nr.', 2*' ','std dev mean curv', 2*' ',\
                        'sum squared dist. from initial',\
                        2*' ', 'max triangle area')
                # printing 1st iter.
                print('0', 14*' ', smooth[0].round(7), 11*' ',\
                           constr[0].round(4), 22*' ', maxa[0].round(4))
                # printing last nn iter.
                for ii in range(mm+1-nn, mm+1):
                    print(ii, 14*' ', smooth[ii].round(7), 11*' ',\
                            constr[ii].round(4), 22*' ', maxa[ii].round(4))         
                print('\naverage for dot products of all neighbor unit', \
                        'normals is max at iter.', kk)
                print('the min standard dev. of mean curvatures is', smooth[kk], \
                        'and sum of squared distance of verts from their originals is',\
                        constr[kk].round(2),'\n')
            if kk == mm:
                # smooth may still increase, so iter. does not stop
                VV[mm-nn:mm] = [-1]*nn
                # replaces unnecessary & large elements of VV with an integer
                # only last element is needed for further iter.
        mm += 1
    del VV, smooth, constr, maxa       
    return verts # smoothed verts


def smoothingBall(r, **kwargs):
    method = kwargs.get('method')
    # tests the smoothing method by
    # generating and smoothing a ball of radius (r)
    rr=r+1
    z,y,x = np.ogrid[-rr:rr+1, -rr:rr+1, -rr:rr+1]
    # ball of radius r where the ball doesn't touch boundaries of the array
    ball = z**2+y**2+x**2 <= (rr-1)**2
    verts, faces, norms, vals = measure.marching_cubes_lewiner(ball)
    print('\nBall with radius {} has {} verts and {} faces.'\
            .format(r, len(verts), len(faces)))
    # mlab.triangular_mesh(verts[:,0], verts[:,1], verts[:,2], faces)
    # mlab.show()
    nbrLB = verticesLaplaceBeltramiNeighborhoodParallel(faces, verts)
    if method == 'aniso_diff':
        verts2 = smoothingParallel(verts, faces, nbrLB, verts_constraint=1.7, method='aniso_diff')
    else:
        verts2 = smoothingParallel(verts, faces, nbrLB, verts_constraint=1.7)
    mlab.triangular_mesh(verts2[:,0], verts2[:,1], verts2[:,2], faces)
    mlab.show()


def smoothingDoubleTorus(**kwargs):
    # source for equation: https://www.mathcurve.com/surfaces.gb/tore/tn.shtml
    method = kwargs.get('method')
    n = 100  # increase for finer result
    r = 0.1
    x = np.linspace(0, 2, n)
    y = np.linspace(-1, 1, n)
    z = np.linspace(-0.7, 0.7, n)
    x, y, z = np.meshgrid(x,y,z)
    
    fx = x*(x-1)**2*(x-2)
    gxy = fx + y**2
    # T is a vol. contained by a double torus
    T = z**2 + gxy**2 - r**2 <= 0 # doube torus & the vol. inside
    ii,jj,kk = T.shape
    Torus = np.zeros(shape=(ii + 2, jj + 2, kk + 2))
    Torus[1 : ii + 1, 1 : jj + 1, 1 : kk + 1] = T
    verts, faces, norms, vals = measure.marching_cubes_lewiner(Torus)
    print(len(verts), len(faces))
    mlab.triangular_mesh(verts[:,0], verts[:,1], verts[:,2], faces)
    mlab.show()
    nbrLB = verticesLaplaceBeltramiNeighborhoodParallel(faces, verts)
    if method == 'aniso_diff':
        verts2 = smoothingParallel(verts, faces, nbrLB, verts_constraint=1, method='aniso_diff')
    else:
        verts2 = smoothingParallel(verts, faces, nbrLB, verts_constraint=1)
    mlab.triangular_mesh(verts2[:,0], verts2[:,1], verts2[:,2], faces)
    mlab.show()


def smoothingTripleTorus(**kwargs):
    # source for equation: https://www.mathcurve.com/surfaces.gb/tore/tn.shtml
    method = kwargs.get('method')
    n = 100 # increase for finer result
    R = 1.05
    x = np.linspace(-R, R, n)
    y = np.linspace(-R, R, n)
    z = np.linspace(-0.8, 0.8, n)
    x, y, z = np.meshgrid(x,y, z)

    ee = 0.003  # a very small number
    # T is a vol. contained by a triple torus
    T = z**2 - (ee - ((x**2 + y**2)**2 - x*(x**2 - 3*y**2))**2) <=0 
    ii,jj,kk = T.shape
    Torus = np.zeros(shape=(ii + 2, jj + 2, kk + 2))
    Torus[1 : ii + 1, 1 : jj + 1, 1 : kk + 1] = T
    verts, faces, norms, vals = measure.marching_cubes_lewiner(Torus)
    print(len(verts), len(faces))
    mlab.triangular_mesh(verts[:,0], verts[:,1], verts[:,2], faces)
    mlab.show()
    nbrLB = verticesLaplaceBeltramiNeighborhoodParallel(faces, verts)
    if method == 'aniso_diff':
        verts2 = smoothingParallel(verts, faces, nbrLB, verts_constraint=1, method='aniso_diff')
    else:
        verts2 = smoothingParallel(verts, faces, nbrLB, verts_constraint=1)
    mlab.triangular_mesh(verts2[:,0], verts2[:,1], verts2[:,2], faces)
    mlab.show()



if __name__ == '__main__':
    main()



# ########### test smoothing a ball/torus ############
# smoothingBall(5)               # smooths a ball (r=5) by isotropic diff.
# smoothingBall(5, method='aniso_diff') # smooths a ball (r=5) by anisotropic diff.
# smoothingTripleTorus()       # smooths triple torus by isotropic diff.
# smoothingTripleTorus(method='aniso_diff')
# smoothingDoubleTorus(method='aniso_diff') # smooths double torus by anisotropic diff.
# smoothingDoubleTorus()        # smooths double torus by isotropic diff.



# ################ timing functions #################
# from timeit import timeit
# wrappedA = wrapper(verticesLaplaceBeltramiNeighborhood,facesW, len(vertsW))
# wrappedB = wrapper(verticesLaplaceBeltramiNeighborhood2,facesW, vertsW)
# print('{0:5.3f} {1:5.3f}'.format(timeit(wrappedA, number=1),\
#         timeit(wrappedB, number=1)))



# ####### how to save/load list & numpy array #######
# lst = [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1)]
# # list of tuples
# import pickle
# with open('lst.txt', 'wb') as fp:   # pickle
#     pickle.dump(lst, fp)
# with open('lst.txt', 'rb') as fp:   # unpickle
#     b = pickle.load(fp)
# ###### how to save/load several np.array(s) #######
# # np.save for 1 arr only -- np.savez for several
# np.savez('name', vertsW, facesW)
# npzfile = np.load('name.npz')
# vertsW = npzfile[npzfile.files[0]]
# facesW = npzfile[npzfile.files[1]]
# ###################################################