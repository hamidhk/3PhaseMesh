   
print('\nThis code is written in Python 3.7!', flush=True)
from sys import version
print('Python version in your system is ', version, flush=True)
import os  # used to go through multiple 3D image files in a folder
from skimage import io  # used to read '.raw'/'.mhd' image into numpy array
import numpy as np      # used for computations
from scipy import ndimage # used to label 3 phases in an image
from mayavi import mlab # used only for visualization
from time import time   # used to measure runtime of certain functions
from stl import mesh    # used only to save stl files
import pickle           # used to save a part of results
import matplotlib.pyplot as plt # used to create/save histogram
import seaborn as sbn   # used to create/save histogram
from statsmodels.tsa.stattools import acf

################################################################
################ ADJUST GLOBAL VARIABLES BELOW #################
################################################################
## path of the folder containing segmented 3-phase images
path = '/home/usr/folder/'
## values for each phase in segmented image
## image must contain 3 phases
# wetting phase (A), nonwetting phase (B), solid (S)
Aval, Bval, Sval = 1, 0, 2
################################################################
################################################################

# create result directory
if not os.path.isdir(path + '0output'):
    os.mkdir(path + '0output')
path_out = path + '0output/'

def main():
    for filename in os.listdir(path):
        if filename.endswith('.mhd'):
            if not os.path.isdir(path_out + filename):
                os.mkdir(path_out + filename)
            path_ = path_out + '/' + filename + '/'
            print('\n\nReading image', filename, 'as a numpy array', flush=True)
            # open .raw image  as np.array (z,y,x) using simpleitk plugin of skimage
            img = io.imread(path + filename, plugin='simpleitk')
            # img = io.imread(path + filename) # tif image
            # io.imshow(img[300,:,:])
            # io.show()
            # # # # Resclice image to a smaller volume! # # # #
            # img=img[150:250, 150:250, 150:250] # small test volume
            print('Image size & dimensions:', img.size, img.shape, flush=True)
            print(f'Image has {np.sum(img == Bval)} phase B (nonwetting), {np.sum(img == Aval)} phase A (wetting) voxels.\n', flush=True)

            ###################################################################
            ## find 2 & 3 phase intersections, create mesh & data structure for 3-phase smoothing
            ## without kwarg return_smoothing_structures, only basic mesh info returned
            # verts, facesi, facess, labc = ThreePhaseIntersection(img)
    
            ## with the kwarg, all structural info required for smoothing is returend
            verts, facesi, facess, labc, nbrfci, nbrfcs, msk_nbrfci, msk_nbrfcs, \
            nbri, nbrs, ind_nbri, ind_nbrs, msk_nbri, msk_nbrs, interface, cluster \
            = ThreePhaseIntersection(img, return_smoothing_structures=True)
            print(f'\n{len(interface)} individual of fluid-fluid interface(s), & {len(cluster)} individual fluid cluster(s) not in contact with image boundaries\n')
            ###################################################################
            ## 3-phase mesh smoothing
            verts_ = smoothingThreePhase(verts, facesi, facess, nbrfci, nbrfcs, msk_nbrfci, \
                msk_nbrfcs, nbri, nbrs, ind_nbri, ind_nbrs, msk_nbri, msk_nbrs, interface, verts_constraint=1.7) #, method='aniso_diff')
            # verts_ = verts
            ###################################################################
            ## calc. interfacial area, curvature, contact angle after smoothing
            t = time()
            nVi = verticesUnitNormals(verts_, facesi, nbrfci, msk_nbrfci)
            # kHi, kGi, Avori, maxa = meanGaussianPrincipalCurvatures(verts_,facesi,nbrfci,msk_nbrfci, nVi, nbri, ind_nbri, msk_nbri, return_Gaussian=True, vorArea='DMSB99')
            kHi, kGi, Avori, maxa = meanGaussianPrincipalCurvatures(verts_,facesi,nbrfci,msk_nbrfci, nVi, nbri, ind_nbri, msk_nbri, return_Gaussian=True)
            nVs = verticesUnitNormals(verts_, facess, nbrfcs, msk_nbrfcs)
            ang = (180/np.pi)*np.arccos(nVi[:,0]*nVs[:,0] + nVi[:,1]*nVs[:,1] + nVi[:,2]*nVs[:,2])

            # kHs, kGs, Avors, maxa = meanGaussianPrincipalCurvatures(verts_,facess,nbrfcs,msk_nbrfcs, nVs, nbrs, ind_nbrs, msk_nbrs, return_Gaussian=True, vorArea='DMSB99')
            kHs, kGs, Avors, maxa = meanGaussianPrincipalCurvatures(verts_,facess,nbrfcs,msk_nbrfcs, nVs, nbrs, ind_nbrs, msk_nbrs, return_Gaussian=True)

            del nbrfci, nbrfcs, msk_nbrfci, msk_nbrfcs, ind_nbri, ind_nbrs, msk_nbri, msk_nbrs

            # # # # # calc. for individual fluid-fluid interfaces # # # # #
            # add mean curvature, interfacial area, mean contact angle to 'interface' list
            # & remove interfaces with less than 100 vertices (small interfaces, too uncertain)
            # also create a similar summary called interf which is an array
            ls_ = []
            interf = np.zeros(shape=(len(interface),7), dtype=verts.dtype)
            for i, el in enumerate(interface):
                if len(el[5])>100:
                    ind = el[5][labc[el[5],2]==-4]   # interfacial verts on 3-phase line
                    ca = sum(ang[ind])/len(ang[ind]) # mean contact angle for interface
                    Ai = verts_[facesi[el[6]]]
                    Ai = np.cross(Ai[:,1] - Ai[:,0], Ai[:,2] - Ai[:,0])
                    Ai = np.sum(0.5*np.sqrt(Ai[:,0]**2 + Ai[:,1]**2 + Ai[:,2]**2))  ## interfacial area                  
                    khm = sum(kHi[el[5]]*Avori[el[5]])/Ai # average mean curvature for itnerface
                    kHi[el[5]] = kHi[el[5]]*(np.sum(Avori[el[5]])/Ai) # adjusted point-wise curvatures
                    # # append labelA, labelB, meancurv, Area, meanAngle, indexes of verts, indexes of facesi
                    ls_.append([el[0], el[1], khm, Ai, ca, el[5], el[6]])
                    interf[i] = el[0], el[1], khm, Ai, ca, len(el[5]), len(el[6])      
            interface = ls_
            interface.insert(0, 'labelPhaseA, labelPhaseB, meanCurv(1/pixel), Area(pixel**2), meanAngle(deg), indicesOfABVertices, indicesOfABFaces')
            interf = interf[interf[:,0]!=0]
            
            # # # # # calc. for individual fluid clusters (a watertight mesh) # # # # #
            cluster, clust = clusterBasedContactCalculations(verts, verts_, facesi, nVi, ang, kGi, kGs, Avori, Avors, labc, cluster, path_, filename)
            # # for state-of-the-art calculations/visualizations uncomment code blocks in the cluster function
            # calc. mean contact angle for individual fluid clusters using Gauss-Bonnet theorem (instead direct local measurement)
            #    2*pi*X_of_2D_Manifold = integral_over_entire_surface_of_cluster(kG*dA1) + kd  # kd is deficit curvature, kG is Gaussian curvature
            # or 4*pi*X_of_3D_Cluster  = integral_over_entire_surface_of_cluster(kG*dA1) + kd
            # --> find/replace kd in below equation for mean contact angle
            # --> mean_contact_angle_for_cluster = kd/(4*num_3phase_lines_on_cluster)
            
            print(f'\nInterface-based and cluster-based calculations finished in {round(time()-t, 4)} sec!', flush=True)
            
            ###################################################################
            ################# save output (arrays and lists) ##################
            # # save all arrays (arr_0.npy, arr_1.npy ...) in a file with the 
            # # same name as img # # - UNCOMMENT BELOW - # #
            t=time()
            np.savez(path_ + filename + '_init', verts, labc, facesi, facess) # intial
            np.savez(path_ + filename + '_final', verts_, nVi, nVs, ang, kHi, kGi, Avori, kHs, kGs, Avors) # final
            np.savez(path_ + filename + '_interface_summary', interf)         # final
            np.savez(path_ + filename + '_cluster_summary', clust)         # final
            with open(path_ + filename + '_interface_summary', 'wb') as fp:   # interface (label, verts_index, finals)
                pickle.dump(interface, fp)
            with open(path_ + filename + '_cluster_summary', 'wb') as fp:   # cluster (... finals)
                pickle.dump(cluster, fp)
            ## save interface - not an array - use pickle!
            print(f'Output arrays/list saved in {round(time()-t,4)} sec!', flush=True)       
            
            # ###################################################################
            # # # construct/save histogram of contact angles
            t = time()
            ang = ang[labc[:,2]==-4] # ang array only on 3-phase common line
            fig, ax = plt.subplots()
            # sbn.set(color_codes=True)
            fig_ = sbn.distplot(ang, bins=100)
            plt.xlabel('Contact angle (Deg)')
            plt.ylabel('Relative frequency')
            plt.title('Contact angle histogram')
            ax.grid(linestyle=':', linewidth=0.5)
            plt.savefig(path_ + filename + "_hist.png")
            # plt.show()
            np.savetxt(path_+ filename + '_CONTACT_ANGLES.csv', ang, delimiter=',')
            print(f'Histogram & csv text file of contact angles are created/saved in {round(time()-t,4)} sec!', flush=True)

            ###################################################################
            # # print a summary!
            print('\n\n######################      Summary for fluid-fluid interfaces      ########################', flush=True)
            print('\nInterface, labelPhaseA, labelPhaseB, meanCurv(1/pixel), Area(pixel**2), meanContactAngle(deg), num_verts, num_faces', flush=True)
            for i, x in enumerate(interface[1::]):
                print(f'{i}             {x[0]},           {x[1]},         {round(x[2],5)},         {round(x[3],1)},         {round(x[4],1)},                  {len(x[5])},         {len(x[6])}')
    
            print('\n\n######################      Summary for fluid clusters      ########################', flush=True)
            print('\nPhaseA -3_PhaseB -2, cluster label, meanContactAngle(deg), meanDeficitAngle(deg), 2DManifoldEulerChar, NumberOf3PhaseLines, num_verts, num_fluid_faces, num_solid_faces', flush=True)            
            for i, x in enumerate(cluster[1::]):
                print(f'      {x[0]},           {x[1]},              {round(x[2],1)},              {round(x[3],1)},                 {x[4]},                    {len(x[5][2])},               {len(x[6])},            {len(x[7])},            {len(x[8])}', flush=True)
                if len(x[5][0])>0:
                    ls = [round(x[5][0][i],1) for i in range(len(x[5][0]))]
                    print(f'sum of singed rotation angles per line: {ls}', flush=True)
                    ls = [round(x[5][1][i],1) for i in range(len(x[5][1]))]
                    print(f'sum of NOT-singed rotation angles per line: {ls}',  '\n', flush=True)

            del ang, kHi, kGi, kHs, kGs, interf, interface, Avori, Avors, maxa, ls_, clust
            ###################################################################
            # # visualize individual clusters - and save mesh stil files for clusters
            # # the saved stl files can be loaded in for instance Paraview
            # # - UNCOMMENT BELOW - # #
            if len(cluster)>1:
                for ii,x in enumerate(cluster[1::]):
                    fci = facesi[x[7]]
                    fcs = facess[x[8]]
                    # save two stl (fluid, solid) for cluster
                    obj = mesh.Mesh(np.zeros(fcs.shape[0], dtype=mesh.Mesh.dtype))
                    for i, f in enumerate(fcs):
                        for j in range(3):
                            obj.vectors[i][j] = verts_[f[j],:]
                    obj.save(path_+filename+'_'+str(int(ii))+'_p'+str(int(x[0]))+'_clus_'+str(int(x[1]))+'_ang_mean_'+str(int(x[2]))+'_ang_defi_'+str(int(x[3]))+'_Euler_'+str(int(x[4]))+'_Nlines_'+str(len(x[5][2]))+'_s.stl') # write into file

                    obj = mesh.Mesh(np.zeros(fci.shape[0], dtype=mesh.Mesh.dtype))
                    for i, f in enumerate(fci):
                        for j in range(3):
                            obj.vectors[i][j] = verts_[f[j],:]
                    obj.save(path_+filename+'_'+str(int(ii))+'_p'+str(int(x[0]))+'_clus_'+str(int(x[1]))+'_ang_mean_'+str(int(x[2]))+'_ang_defi_'+str(int(x[3]))+'_Euler_'+str(int(x[4]))+'_Nlines_'+str(len(x[5][2]))+'_f.stl') # write into file
                    
                    # visualize cluster 
                    # nVs_ = nVs[x[6]]
                    # nVi_ = nVi[x[6]]
                    # vr = verts_[x[6]]
                    # line = verts_[x[6]][labc[x[6],2]==-4] # verts @ 3-phase line
                    # # border = verts_[np.concatenate((nbrs[:,0][nbrs[:,3]==-1], nbrs[:,1][nbrs[:,3]==-1], nbri[:,0][nbri[:,3]==-1], nbri[:,1][nbri[:,3]==-1]))]
                    # fig = mlab.figure(figure='smoothed clusters', bgcolor=(0.95,0.95,0.95), size=(1200, 1000))
                    # # s_init = mlab.triangular_mesh(verts[:,0], verts[:,1], verts[:,2], facess, color=(0.85, 0.85, 0.85))
                    # s0 = mlab.triangular_mesh(verts_[:,0], verts_[:,1], verts_[:,2], fcs, representation='wireframe', color=(0.7, 0.7, 0.7))
                    # s1 = mlab.triangular_mesh(verts_[:,0], verts_[:,1], verts_[:,2], fcs,  color=(0.53, 0.84, 0.98))
                    # # i_init = mlab.triangular_mesh(verts[:,0], verts[:,1], verts[:,2], facesi, color=(0.65, 0.65, 0.65))
                    # i0 = mlab.triangular_mesh(verts_[:,0], verts_[:,1], verts_[:,2], fci, representation='wireframe', color=(0.7, 0.7, 0.7))
                    # i1 = mlab.triangular_mesh(verts_[:,0], verts_[:,1], verts_[:,2], fci,  color=(0.8, 0.33, 0.5))
                    # # border = mlab.points3d(border[:,0], border[:,1], border[:,2], np.ones(len(border)), line_width=0.5, scale_factor=0.2, color=(0,0,0))
                    # line = mlab.points3d(line[:,0], line[:,1], line[:,2], np.ones(len(line)), line_width=0.5, scale_factor=0.25, color=(0,1,0))
                    # nVs_ = mlab.quiver3d(vr[:,0], vr[:,1], vr[:,2], nVs_[:,0], nVs_[:,1], nVs_[:,2], line_width=2, scale_factor=0.3, color=(0,0,1))
                    # nVi_ = mlab.quiver3d(vr[:,0], vr[:,1], vr[:,2], nVi_[:,0], nVi_[:,1], nVi_[:,2], line_width=2, scale_factor=0.3, color=(0.3,0.3,0.3))
                    # mlab.savefig(path_+filename+'_'+str(int(ii))+'_p'+str(int(x[0]))+'_clus_'+str(int(x[1]))+'_ang_mean_'+str(int(x[2]))+'_ang_defi_'+str(int(x[3]))+'_Euler_'+str(int(x[4]))+'_Nlines_'+str(int(x[5]))+'.png')
                    # # mlab.show()
                    # mlab.close()
                del obj
            # # visualize original/smoothed mesh
            # # - UNCOMMENT BELOW - # #
            # line = verts_[labc[:,2]==-4] # verts @ 3-phase line
            # border = verts_[np.concatenate((nbrs[:,0][nbrs[:,3]==-1], nbrs[:,1][nbrs[:,3]==-1], nbri[:,0][nbri[:,3]==-1], nbri[:,1][nbri[:,3]==-1]))]
            # mlab.figure(figure='smoothed 3-phase interfaces', bgcolor=(0.95,0.95,0.95), size=(1200, 1000))
            # # s_init = mlab.triangular_mesh(verts[:,0], verts[:,1], verts[:,2], facess, color=(0.85, 0.85, 0.85))
            # s0 = mlab.triangular_mesh(verts_[:,0], verts_[:,1], verts_[:,2], facess, representation='wireframe', color=(0.7, 0.7, 0.7))
            # s1 = mlab.triangular_mesh(verts_[:,0], verts_[:,1], verts_[:,2], facess,  color=(0.53, 0.84, 0.98))
            # # i_init = mlab.triangular_mesh(verts[:,0], verts[:,1], verts[:,2], facesi, color=(0.45, 0.45, 0.45))
            # i0 = mlab.triangular_mesh(verts_[:,0], verts_[:,1], verts_[:,2], facesi, representation='wireframe', color=(0.7, 0.7, 0.7))
            # i1 = mlab.triangular_mesh(verts_[:,0], verts_[:,1], verts_[:,2], facesi,  color=(0.8, 0.33, 0.5))
            # border = mlab.points3d(border[:,0], border[:,1], border[:,2], np.ones(len(border)), line_width=0.5, scale_factor=0.2, color=(0,0,0))
            # line = mlab.points3d(line[:,0], line[:,1], line[:,2], np.ones(len(line)), line_width=0.5, scale_factor=0.25, color=(0,1,0))
            # # nVs = mlab.quiver3d(verts_[:,0], verts_[:,1], verts_[:,2], nVs[:,0], nVs[:,1], nVs[:,2], line_width=2, scale_factor=0.3, color=(0,0,1))
            # # nVi = mlab.quiver3d(verts_[:,0], verts_[:,1], verts_[:,2], nVi[:,0], nVi[:,1], nVi[:,2], line_width=2, scale_factor=0.3, color=(0.3,0.3,0.3))
            # mlab.savefig(filename+'_.png')
            # mlab.show()


            ###################################################################
            ################## create/save mesh stl file ######################
            # # create/save two mesh stl files; for AB interface & solid  meshes
            # #  - UNCOMMENT BELOW - # #
            # t=time()
            # obj = mesh.Mesh(np.zeros(facesi.shape[0], dtype=mesh.Mesh.dtype))
            # for i, f in enumerate(facesi):
            #     for j in range(3):
            #         obj.vectors[i][j] = verts[f[j],:]
            # obj.save(filename + '_init_i.stl') # write into file
            # # solid stl file
            # obj = mesh.Mesh(np.zeros(facess.shape[0], dtype=mesh.Mesh.dtype))
            # for i, f in enumerate(facess):
            #     for j in range(3):
            #         obj.vectors[i][j] = verts[f[j],:]
            # obj.save(filename + '_init_s.stl') # write into file        
            # print(f'\nCreated/saved initial & final fluid-fluid mesh stl in {round(time()-t,4)} sec!', flush=True)

            # t=time()
            # obj = mesh.Mesh(np.zeros(facesi.shape[0], dtype=mesh.Mesh.dtype))
            # for i, f in enumerate(facesi):
            #     for j in range(3):
            #         obj.vectors[i][j] = verts_[f[j],:]
            # obj.save(filename + '_final_i.stl') # write into file
            # # solid stl file
            # obj = mesh.Mesh(np.zeros(facess.shape[0], dtype=mesh.Mesh.dtype))
            # for i, f in enumerate(facess):
            #     for j in range(3):
            #         obj.vectors[i][j] = verts_[f[j],:]
            # obj.save(filename + '_final_s.stl') # write into file
            # del  obj 
            # print(f'Created/saved initial & final solid-fluid mesh stl in {round(time()-t,4)} sec!\n', flush=True)

            ##################################################################
            # # visualize sign of kH
            # #  - UNCOMMENT BELOW - # #
            # sgn_kH = np.sign(kH)
            # mlab.figure(figure='sign of interfacial mean curvature', bgcolor=(0.95,0.95,0.95), size=(1000, 800))
            # mlab.triangular_mesh(verts_[:,0], verts_[:,1], verts_[:,2], facesi, scalars=sgn_kH)
            # mlab.show()
            
            ##################################################################
            # # visualize all interfaces, each with a different color!
            # #  - UNCOMMENT BELOW - # #
            # mlab.figure(figure='mesh', bgcolor=(0.95,0.95,0.95), size=(1500, 1200))
            # for i, el in enumerate(interface):
            #     if i>0:
            #         col = (float(np.random.rand(1)), float(np.random.rand(1)), float(np.random.rand(1)))
            #         fc = facesi[el[6]]
            #         mlab.triangular_mesh(verts_[:,0], verts_[:,1], verts_[:,2], fc,  color=col)
            # mlab.triangular_mesh(verts_[:,0], verts_[:,1], verts_[:,2], facesi, representation='wireframe', color=(0.7, 0.7, 0.7))                    
            # mlab.triangular_mesh(verts_[:,0], verts_[:,1], verts_[:,2], facess,  color=(0.53, 0.84, 0.98))
            # mlab.triangular_mesh(verts_[:,0], verts_[:,1], verts_[:,2], facess, representation='wireframe', color=(0.7, 0.7, 0.7)) 
            # mlab.show()

            ##################################################################

            ######################################################
            # # how to save/load list
            # lst = [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1)]
            # # list of tuples
            # import pickle
            # with open('lst.txt', 'wb') as fp:   # pickle
            #     pickle.dump(lst, fp)
            # with open('lst.txt', 'rb') as fp:   # unpickle
            #     b = pickle.load(fp)

            ######################################################
            # # how to save/load np.array(s) 
            # # np.save for 1 arr only -- np.savez for several
            # np.savez('name', verts, faces)
            # npzfile = np.load('name.npz')
            # verts = npzfile[npzfile.files[0]]
            # faces = npzfile[npzfile.files[1]]

            #####################################################
            # # how to time a functions with arguments
            # from timeit import timeit
            # wrappedA = wrapper(func, arg1, arg2)
            # wrappedB = wrapper(func_, arg1_, arg2_)
            # print('{0:5.3f} {1:5.3f}'.format(timeit(wrappedA, number=1),\
            #         timeit(wrappedB, number=1)))
            #####################################################
            del nbri, nbrs, verts, labc, facesi, facess, verts_, nVi, nVs, cluster


def ThreePhaseIntersection(img, **kwargs):
    print('The ThreePhaseIntersection function finds two & three phase intersections! creates triangular meshes!', flush=True)
    print('Corrects non-orientable vertices on mesh!', flush=True)
    print('Creates structure info for mesh smoothing!', flush=True)
    smoothing_struct = kwargs.get('return_smoothing_structures')

    def intersectAlongAllAxes_slower_but_needs_less_memory():
        def intersectAlongAxis__(lA, lB):
            # cen (center vertices)--> (z,y,x,n); n is 1 or -1 to create consistent mesh with normal vectors from A to B 
            # shape0 -> z-axis, shape1 -> y-axis, shape2 -> x-axis
            shp0, shp1, shp2 = lA.shape[0], lA.shape[1], lA.shape[2]
            y_,x_ = np.mgrid[0:shp1, 0:shp2]

            # search in first layer of lA
            msk = np.logical_and(lA[0]!=0, lB[1]!=0)
            y,x = y_[msk], x_[msk]
            cen = np.vstack((0.5*np.ones(len(y)), y, x, np.ones(len(y)))).T
            # search in layers of lA between first & last
            for i in range(1, shp0-1):
                msk = np.logical_and(lA[i]!=0, lB[i-1]!=0)
                y = y_[msk]
                if len(y)>0:
                    x = x_[msk]
                    cen = np.concatenate((cen, np.vstack(((i-0.5)*np.ones(len(y)), y, x, -np.ones(len(y)))).T))
                
                msk = np.logical_and(lA[i]!=0, lB[i+1]!=0)
                y= y_[msk]
                if len(y)>0:
                    x = x_[msk]
                    cen = np.concatenate((cen, np.vstack(((i+0.5)*np.ones(len(y)), y, x,  np.ones(len(y)))).T))
            # search in last layer of lA
            msk = np.logical_and(lA[shp0-1]!=0, lB[shp0-2]!=0)
            y= y_[msk]
            if len(y)>0:
                x = x_[msk]
                cen = np.concatenate((cen, np.vstack(((shp0-1.5)*np.ones(len(y)), y, x, -np.ones(len(y)))).T))
            # create 4 corner vertices for 1 center vertex
            l = len(cen)
            cor = np.zeros(shape=(4*l,3), dtype=cen.dtype)
            for i, el in enumerate([[-0.5,-0.5], [0.5,-0.5], [0.5,0.5], [-0.5,0.5]]):
                cor[i*l: (i+1)*l,0] = cen[:,0]
                cor[i*l: (i+1)*l,1] = cen[:,1] + el[0]
                cor[i*l: (i+1)*l,2] = cen[:,2] + el[1]
            # # record which cen each cor belog to (indexes of cen's)
            # co_id = np.zeros(4*l, dtype=np.int64)
            # for i in range(4):
            #     co_id[i*l: (i+1)*l] = np.arange(l)
            return cen, cor#, co_id

        for la_,lb_ in (lA,lB),(lS, lS_not):
            # find center of common squares along z-axis
            cen0, cor0 = intersectAlongAxis__(la_,lb_)    
            # find center of common squares along y-axis
            # swap axes of arrays for faster looping; z_new = x_old, y_new = z_old, x_new = y_old 
            cen1, cor1 = intersectAlongAxis__(la_.swapaxes(1,0).swapaxes(2,1), lb_.swapaxes(1,0).swapaxes(2,1))
            # correct order of coordinates
            cen1 = np.vstack((cen1[:,2],cen1[:,0],cen1[:,1],cen1[:,3])).T
            cor1 = np.vstack((cor1[:,2],cor1[:,0],cor1[:,1])).T
            # find center of common squares along x-axis
            # swap axes of arrays for faster looping; z_new = y_old, y_new = x_old, x_new = z_old  
            cen2, cor2 = intersectAlongAxis__(la_.swapaxes(2,0).swapaxes(2,1), lb_.swapaxes(2,0).swapaxes(2,1))
            # correct order of coordinates
            cen2 = np.vstack((cen2[:,1],cen2[:,2],cen2[:,0],cen2[:,3])).T
            cor2 = np.vstack((cor2[:,1],cor2[:,2],cor2[:,0])).T
            # record which cen vertex each cor vertex belog to (indexes of cen's)
            l0, l1, l2 = len(cen0), len(cen1), len(cen2)
            co_id = np.zeros(4*(l0+l1+l2), dtype=np.int64)
            for i in range(4):
                co_id[i*l0 : (i+1)*l0] = np.arange(l0)
            for i in range(4):
                co_id[4*l0 + i*l1 : 4*l0 + (i+1)*l1] = np.arange(4*l0, 4*l0 + l1, 1)
            for i in range(4):
                co_id[4*l0 + 4*l1 + i*l2 : 4*l0 + 4*l1 + (i+1)*l2] = np.arange(4*l0 + 4*l1, 4*l0 + 4*l1 + l2, 1)

            cor0 = np.concatenate((cor0, cor1, cor2)) # some vertice here are repeated
            del cor1, cor2
            cen0 = np.concatenate((cen0, cen1, cen2)) # all vertice here are unique
            del cen1, cen2
            # unique indexes for verts before creating faces
            # unq, idx = np.unique(cor0, axis=0, return_inverse=True) # idx is indices of cor0 in unq

    def intersectAlongAllAxes(lA, lB, _B, z_,y_,x_):
        def intersectAlongAxis0(lA, lB, _B, z_,y_,x_):
            ## cen (center vertices)--> (z,y,x); all unique
            ## cor (corner vertices)--> (z,y,x); some repetitions (correction later)
            ## norm; normal vectors @ cen - to be used in triangle definition
            # shape0 -> z-axis, shape1 -> y-axis, shape2 -> x-axis
            shp0, shp1, shp2 = lA.shape[0], lA.shape[1], lA.shape[2]

            # take first layer of lB to its end; therefor lB_[i+1] corresponds to lA[i]
            lB_ = np.concatenate((lB[1:shp0],[lB[0]]))
            _B_ = np.concatenate((_B[1:shp0],[_B[0]]))
            msk = np.logical_and(lA!=0,lB_!=0)
            msk[-1] = False # last index (layer) in lA corresponds to 1st one in lB
            z,y,x = z_[msk]+0.5, y_[msk], x_[msk]
            cen = np.vstack((z,y,x)).T # coordinate of center vertices between pixels
            norm = np.zeros(shape=(len(z),3), dtype=cen.dtype) # normal vector @ cen
            norm[:,0] = 1 # all norms facing from lA to lB
            lab_cen = np.vstack((lA[msk], lB_[msk], _B_[msk])).T # store labels @ cen

            # take last layer of lB to its beginning; therefor lB_[i-1] corresponds to lA[i]
            lB_ = np.concatenate(([[lB[shp0-1]], lB[0:shp0-1]]))
            _B_ = np.concatenate(([[_B[shp0-1]], _B[0:shp0-1]]))
            msk = np.logical_and(lA!=0,lB_!=0)
            msk[0] = False # first index (layer) in lA corresponds to last one in lB
            z,y,x = z_[msk]-0.5, y_[msk], x_[msk]
            cen = np.concatenate((cen, np.vstack((z,y,x)).T)) # create & concatenate
            norm_ = np.zeros(shape=(len(z),3), dtype=cen.dtype) # normal vector @ cen
            norm_[:,0] = -1 # all norms facing from lA to lB
            norm = np.concatenate((norm, norm_))
            lab_cen = np.concatenate((lab_cen, np.vstack((lA[msk], lB_[msk], _B_[msk])).T)) # store labels @ cen
            del z_,y_,x_, norm_, lA, lB, lB_, _B_, msk
            
            # create 4 corner vertices for 1 center vertex 
            l = len(cen)
            cor = np.zeros(shape=(4*l,3), dtype=cen.dtype)
            # cor_id[:,0] couple cor's with indeces of corresponding cen's
            # cor_id[:,1] identify if a cor around a cen is 0th, 1st, 2nd or 3rd
            cor_id = np.zeros(shape=(4*l,2), dtype=np.int64)
            for i, el in enumerate( [[-0.5,-0.5], [0.5,-0.5], [0.5,0.5], [-0.5,0.5]] ):
                cor[i*l: (i+1)*l,0] = cen[:,0] # the same as cen in z-dir
                cor[i*l: (i+1)*l,1] = cen[:,1] + el[0] # half more/less in y-dir
                cor[i*l: (i+1)*l,2] = cen[:,2] + el[1] # half more/less in x-dir
                cor_id[i*l: (i+1)*l, 0] = np.arange(l)
                cor_id[i*l: (i+1)*l, 1] = i
            lab_cor = np.zeros(shape=(4*l,3), dtype=np.int64) # labels @ cor
            for i in range(4):
                lab_cor[i*l: (i+1)*l] = lab_cen
            return cen, norm, cor, cor_id, lab_cen, lab_cor

        # find center of common squares (the square between two pixels) along z-axis
        cen0, norm0, cor0, cor_id0, lab_cen0, lab_cor0 = intersectAlongAxis0(lA, lB, _B, z_, y_, x_)   
        
        # find center of common squares along y-axis
        # swap axes of arrays for faster reading; z_new = x_old, y_new = z_old, x_new = y_old 
        cen1, norm1, cor1, cor_id1, lab_cen1, lab_cor1 = intersectAlongAxis0(lA.swapaxes(1,0).swapaxes(2,1),\
                                            lB.swapaxes(1,0).swapaxes(2,1), _B.swapaxes(1,0).swapaxes(2,1),\
                y_.swapaxes(1,0).swapaxes(2,1),x_.swapaxes(1,0).swapaxes(2,1),z_.swapaxes(1,0).swapaxes(2,1))
        # correct order of coordinates
        cen1 = np.vstack((cen1[:,2],cen1[:,0],cen1[:,1])).T
        cor1 = np.vstack((cor1[:,2],cor1[:,0],cor1[:,1])).T
        norm1 = np.vstack((norm1[:,2],norm1[:,0],norm1[:,1])).T
        
        # find center of common squares along x-axis
        # swap axes of arrays for faster reading; z_new = y_old, y_new = x_old, x_new = z_old  
        cen2, norm2, cor2, cor_id2, lab_cen2, lab_cor2  = intersectAlongAxis0(lA.swapaxes(2,0).swapaxes(2,1),\
                                            lB.swapaxes(2,0).swapaxes(2,1), _B.swapaxes(2,0).swapaxes(2,1),\
                x_.swapaxes(2,0).swapaxes(2,1),z_.swapaxes(2,0).swapaxes(2,1),y_.swapaxes(2,0).swapaxes(2,1))
        # correct order of coordinates
        cen2 = np.vstack((cen2[:,1],cen2[:,2],cen2[:,0])).T
        cor2 = np.vstack((cor2[:,1],cor2[:,2],cor2[:,0])).T
        norm2 = np.vstack((norm2[:,1],norm2[:,2],norm2[:,0])).T
        
        # merge arrays
        cor_id1[:,0] = cor_id1[:,0] + len(cen0)
        cor_id2[:,0] = cor_id2[:,0] + len(cen1) + len(cen0)
        cor_id0 = np.concatenate((cor_id0, cor_id1, cor_id2))

        cor0 = np.concatenate((cor0, cor1, cor2)) # some vertice here are repeated
        cen0 = np.concatenate((cen0, cen1, cen2)) # all vertice here are unique
        norm0 = np.concatenate((norm0, norm1, norm2)) # normal vectors for cen0
        lab_cen0 = np.concatenate((lab_cen0,lab_cen1,lab_cen2)) # labels @ cen0
        lab_cor0 = np.concatenate((lab_cor0,lab_cor1,lab_cor2)) # labels at cor0
        return cen0, norm0, cor0, cor_id0, lab_cen0, lab_cor0

    def orientationCorrectorSixCenters(u):
            # # when 6 center vertex around 1 corner vertex
            v = no[u[:,4]]
            c = ce[u[:,4]]
            px = c-0.5*v
            px_= c+0.5*v
            s, si, ct   = np.unique(px, return_inverse=True, return_counts=True, axis=0)
            s_, si_, ct_ = np.unique(px_, return_inverse=True, return_counts=True, axis=0)
            ls, ls_ = len(s), len(s_)
            if ls==3 and ls_==3:
                ## in 2phase is a saddle point; always orientable! 
                ## did not find any sample in 3-phase in an entire image!
                pass

            elif ls==4 and ls_==4:
                ## in 2-phase is an orientable saddle point; 
                ## in 3-phase, it is often non-orientable (but sometimes orientable) 
                if len(u[:,3][u[:,3]!=-1])==3: # in 3p
                    so_ = u[:,3]!=-1
                    ab_ = u[:,3]==-1
                    a_=len(np.unique(px[so_],axis=0))
                    b_=len(np.unique(px[ab_],axis=0))
                    a__=len(np.unique(px_[so_],axis=0))
                    b__=len(np.unique(px_[ab_],axis=0))
                    if (a_==1 and b_==3 and a__==3 and b__==1) \
                        or (a_==3 and b_==1 and a__==1 and b__==3):
                        return so_
            elif ls==4 and ls_==2:
                ## in 2phase, when for instace solid mesh folds, this is the folding point (orientable!)
                ## in 3phase, did not find any sample an entire image
                pass

            elif ls==2 and ls_==4: 
                ## in 2phase, when mesh folds - this is the point at the two sides of the folding point (non-oreintable)
                ## in 3phase when there is edge-to-edge contact (non-oreintable) - common source of nonorientation in 3p point -
                if -1 in u[:,3] and (-2 in u[:,3] or -3 in u[:,3]): # in 3p
                    msk = u[:,3]==-1
                    return msk
                else: # in 2p
                    return si==1

            elif ls==2 and ls_==6:
                ## corner-to-corner pixels (always non-orientable) - happens often in 2 & sometimes in 3 phase
                return si==1
            
            elif ls==6 and ls_==2:
                ## corner-to-corner pixels (always non-orientable) - happens seldom! happens in 2 phase
                return si_==1

    def orientationCorrectorSevenCenters(u):
            # # when 7 center vertex around 1 corner vertex
            v = no[u[:,4]]
            c = ce[u[:,4]]
            px = c-0.5*v
            px_= c+0.5*v
            s, si, ct   = np.unique(px, return_inverse=True, return_counts=True, axis=0)
            s_, si_, ct_ = np.unique(px_, return_inverse=True, return_counts=True, axis=0)
            ls, ls_ = len(s), len(s_)
            if ls==3 and ls_==5:
                if -1 in u[:,3] and (-2 in u[:,3] or -3 in u[:,3]): 
                    ## in 3phase when there is edge-to-edge contact (non-oreintable)
                    ## solution. take ces (center solid ) together & cei together
                    return u[:,3]==-1
                else:
                    ## 2p nonorientable
                    ## solution. the 3 px coming together should be seperated form the rest 
                    si__, ct__ = np.unique(si, return_counts=True)
                    return si==si__[ct__==3]
            
            elif ls==5 and ls_==5:
                ## 2p, none found in an entire image!
                ## 3phase, some orientable & some not; 
                ## 3phase nonorientables are pixel corners with either 3 ces or 3 cei 
                if len(u[:,3][u[:,3]!=-1])==3 or len(u[:,3][u[:,3]==-1])==3:
                    so_ = u[:,3]!=-1
                    ab_ = u[:,3]==-1
                    a_=len(np.unique(px[so_],axis=0))
                    b_=len(np.unique(px[ab_],axis=0))
                    a__=len(np.unique(px_[so_],axis=0))
                    b__=len(np.unique(px_[ab_],axis=0))
                    if (a_==1 and b_==4) or (a__==4 and b__==1):
                        return so_

            # elif: # # #    all states below, pass!
                ## possible tested states below!
                # elif ls==5 and ls_==3:
                #     ## orientable both in 2 phase (mesh floding point) & 3 phase (5 ces in touch with 2 cei)
                #     pass

                # elif ls==4 and ls_==3:
                #     ## 3phase, typical solid saddle in contact with one interface center (cei) - (orientable)
                #     ## 2p, none found in an entire image!
                #     pass

                # elif ls==3 and ls_==4:
                #     ## 3phase, typical saddle point with 4 cei & 3 ces (orientable)
                #     ## 2p, none found in an entire image!
                #     pass
                
                # elif ls==4 and ls_==4:
                #     ## 3phase, either typical saddle point with 6 ces & 1 cei; or structure with 5 ces & 2 cei (orientable)
                #     ## 2p, none found in an entire image!
                #     pass
                       
                # elif ls==5 and ls_==4:
                #     ## 3phase, a type of saddle (ls=4, ls_=4 @ len(u)=6) with 6 ces & 1 cei; or structure with 5 ces & 2 cei (orientable)
                #     ## 2p, none found in an entire image!
                #     pass
                
                # elif ls==4 and ls_==5:
                #     ## 3phase, different structures, cei+ces=3+4; 2+5; 4+3  (orientable)
                #     ## 2p, none found in an entire image!
                #     pass

                # elif ls==6 and ls_==3:
                #     ## 3phase, different structure example, cei+ces=2+5;  (orientable)
                #     ## 2p, none found in an entire image!
                #     pass

                # elif ls==3 and ls_==6:
                #     ## 3phase, ces+cei=3+4;  (orientable)
                #     ## 2p, none found in an entire image!
                #     pass

    def orientationCorrectorEightCenters(u):
        # # when 8 center vertex around 1 corner vertex
        v = no[u[:,4]]
        c = ce[u[:,4]]
        px = c-0.5*v
        px_= c+0.5*v
        s, si, ct   = np.unique(px, return_inverse=True, return_counts=True, axis=0)
        s_, si_, ct_ = np.unique(px_, return_inverse=True, return_counts=True, axis=0)
        ls, ls_ = len(s), len(s_)

        ## 3phase, both  orientable & nonorientable possible!
        if -1 in u[:,3] and (-2 in u[:,3] or -3 in u[:,3]):
            if ls==4 and ls_==5:
                ## either orientable, or type of nonorientable which is 
                ## corrected in label correction (duplication of a vertex common in two AB-interface)
                pass

            elif ls==5 and ls_==4:
                ## orientable! a typical saddle point on solid (6 ces) plus 2 cei
                pass
            
            elif ls==5 and ls_==6:
                if len(np.unique(u[:,1][u[:,3]==-1]))==1 and len(np.unique(u[:,2][u[:,3]==-1]))==1 and len(u[u[:,3]==-1])==5: 
                    # if 1st & 2nd conditions are False, it's a label issue (which is corrected before this step!)
                    # if 3rd condition is False (len(cei)=4), the vertex orientable & doesn't need correction!
                    return u[:,3]==-1

            elif ls==6 and ls_==5:
                # 3 cei pinned on 5 ces out of real 3-phase contact line!
                if len(np.unique(u[:,1][u[:,3]==-1]))==1 and len(np.unique(u[:,2][u[:,3]==-1]))==1:
                    # if both conditions are False, it's a label issue (which is corrected before this step!)
                    if 3 in ct or 3 in ct_:
                        # 3 px coming (@ AB) together, should be seperated!
                        return u[:,3]==-1
            
            elif ls==5 and ls_==5:
                # either a label issue (which is corrected before this step!), or orientable (3 cei on 5 ces)
                pass
            
            elif ls==6 and ls_==6:
                if len(np.unique(u[:,1][u[:,3]==-1]))==1 and len(np.unique(u[:,2][u[:,3]==-1]))==1:
                    # if both conditions are False, it's a label issue (which is corrected before this step!)
                    # otherwise a non-orientable! (4 cei pinned on 4 ces)
                    return u[:,3]==-1
            
            elif ls==4 and ls_==4:
                # very seldom! only 1 example found in an entire image
                # orientable! 2 cei with 6 ces - without label issue
                # 6-ces is of type (ls==2 and ls_==4) @ func orientationCorrectorSixCenters(u0)
                # after ces (2p) correction - 3p corrected below!
                if len(np.unique(u[:,1][u[:,3]==-1]))==1 and len(np.unique(u[:,2][u[:,3]==-1]))==1:
                    # this first if is to ensure the problem is not with labels
                    u0 = u[u[:,3]!=-1]
                    msk0 = None
                    if len(u0)==6:
                        msk0 = orientationCorrectorSixCenters(u0)
                    if msk0 is not None:
                        msk = np.zeros(len(px), dtype=np.bool)
                        msk[u[:,3]!=-1]=msk0
                        lb_,la_ = len(u0[u0[:,3]==-2]), len(u0[u0[:,3]==-3])
                        if lb_==1:
                            msk[u[:,3]==-1] = not msk0[u0[:,3]==-2]
                        elif la_==1:
                            msk[u[:,3]==-1] = not  msk0[u0[:,3]==-3]
                        return msk

        ## 2p nonorientable
        else:
            if 3 in ct:
                ## 2p nonorientable (type 1)
                ## solution. the 3 px coming together should be seperated form the rest 
                si__, ct__ = np.unique(si, return_counts=True)
                return si==si__[ct__==3]
            else:
                ## 2p nonorientable (type 2, the symmetric structure)
                ## solution. px with two similar coordinate components (z,y,x) grouped together
                msk = np.zeros(len(px), dtype=np.bool) # bool array (all False)
                dpx = px-px[0]
                for j in range(len(dpx)):
                    if len(dpx[j][dpx[j]==0])>=2:
                        msk[j] = True
                return msk

    def orientationCorrectorNineCenters(u):
        # # when 9 center vertex around 1 corner vertex
        # output is either boolean or array of (0s,1s,2s)
        v = no[u[:,4]]
        c = ce[u[:,4]]
        px = c-0.5*v
        px_= c+0.5*v
        s, si, ct   = np.unique(px, return_inverse=True, return_counts=True, axis=0)
        s_, si_, ct_ = np.unique(px_, return_inverse=True, return_counts=True, axis=0)
        ls, ls_ = len(s), len(s_)
        ## 3phase
        if -1 in u[:,3] and (-2 in u[:,3] or -3 in u[:,3]):
            px = px.astype(np.int64)
            bb = np.vstack((px[:,0], px[:,1], px[:,2], si, u[:,0], u[:,1], u[:,2], u[:,3], u[:,4])).T
            if len(np.unique(u[:,1][u[:,3]==-1]))==1 and len(np.unique(u[:,2][u[:,3]==-1]))==1:
                # this if is to ensure that the problem is not labels!
                msk = u[:,3]==-1
                if len(u[msk])==6: # always nonorientable! 3ces + 6cei
                    # 6 cei is often an AB-saddle (orientable) - but should be checked in general
                    msk0 = orientationCorrectorSixCenters(u[msk])
                    if msk0 is not None:
                        msk = 1*msk
                        msk[u[:,3]==-1] = msk0  # msk0 contains 0,1
                        msk[u[:,3]!=-1]= 2      # now msk contains 0,1,2 - three unique verts   
                    return msk   # msk contains 0,1 - two unique verts
                
                elif len(u[msk])==3: # sometimes nonorientable! 6ces + 3cei
                    # nonorientable, if 3 cei come together
                    s, ct   = np.unique(px[msk], return_counts=True, axis=0)
                    s_, ct_   = np.unique(px_[msk], return_counts=True, axis=0)
                    if 3 in ct or 3 in ct_:
                        # now before returning msk, 6 ces should be checked; it is often a solid-saddle (orientable) but not always
                        msk0 = orientationCorrectorSixCenters(u[u[:,3]!=-1])
                        if msk0 is not None:
                            msk = 1*msk
                            msk[u[:,3]!=-1] = msk0  # msk0 contains 0,1
                            msk[u[:,3]==-1]= 2      # now msk contains 0,1,2 - three unique verts
                        return msk  # msk contains 0,1 - two unique verts  
        ## 2phase
        else:
            # always nonorientable! when mesh folds & comes in contact with itself
            # the only type found in an entire image is when corner of a solid pixel touches a solid saddle (3ces + 6ces)
            # # example image, img = np.array([ [[2, 0], [0, 0]],  [[0, 2], [2, 0]]  ]) # 2p (9ces,  0cei)
            ## solution. the 3 px coming together should be seperated form the rest
            if ls==4 and ls_==3:
                si__, ct__ = np.unique(si, return_counts=True)
                msk = si==si__[ct__==3]
                return msk
            elif ls==3 and ls_==4:
                si__, ct__ = np.unique(si_, return_counts=True)
                msk = si_==si__[ct__==3] 
                return msk
            # # lssp.extend([u[0,0]+lces+lcei] + u[:,4].tolist())  # RØD
            # # lssp_.extend([u[0,0]+lces+lcei] + u[:,4].tolist())  # GRØNN
            # # print(np.unique(u[:,1][u[:,3]==-1]), np.unique(u[:,2][u[:,3]==-1]))
            # # px = px.astype(np.int64)
            # # bb = np.vstack((px[:,0], px[:,1], px[:,2], si, u[:,0], u[:,1], u[:,2], u[:,3], u[:,4])).T

    def orientationCorrectorMoreThanNineCenters(u):
        # # when more than 9 center vertex around 1 corner vertex
        # Extremely rare cases!
        # could happen in 3phase! No examples found in an entire image!
        # below are two aritifical examples where 9 ces 
        # (a solid pixel corner (3ces) in contact with a solid saddle (6ces) & saddle has which has 1 or 2 cei on itself
        # the 3ces & saddle should be seperated and the cei's go with the saddle
        # img = np.array([ [[2, 0], [0, 0]],  [[0, 2], [2, 1]]  ])   # (9ces + 1cei)
        # img = np.array([ [[2, 0], [0, 1]],  [[0, 2], [2, 1]]  ])   # (9ces + 2cei)
        if (len(u)==10 and len(u[u[:,3]!=-1])==9) or (len(u)==11 and len(u[u[:,3]!=-1])==9):
            msk0 = orientationCorrectorNineCenters(u[u[:,3]!=-1])
            if msk0 is not None:
                msk = u[:,3]==-1
                if len(msk0[msk0==True])==6:
                    msk[u[:,3]!=-1] = msk0
                    return msk
                elif len(msk0[msk0==False])==6:
                    ms0 = msk0==False
                    ms1 = msk0==True
                    msk0[ms0] = True
                    msk0[ms1] = False
                    msk[u[:,3]!=-1] = msk0
                    return msk # boolean output        
    
    #################################################
    ### define each phase, & label blobs in each
    t0 = time()
    _A = img == Aval    # contains only phase A - wetting True;  rest False - 
    _B = img == Bval    # contains only phase B - nonwetting True; rest False - 
    _S = img == Sval    # contains only phase S - solid plus boundary True; rest False - 

    _S_not = -3*_A -2*_B   # contains not_solid - -3 for _A, -2 for _B & 0 for solid - 
    s = ndimage.generate_binary_structure(3,1) # default structure!
    lA, num_A = ndimage.measurements.label(_A, structure=s)
    lB, num_B = ndimage.measurements.label(_B, structure=s)
    lS, num_S = ndimage.measurements.label(_S, structure=s)
    lS_not = lA + lB
    print(num_A, 'isolated phase A (wetting) blob(s)', flush=True)
    print(num_B, 'isolated phase B (nonwetting) blob(s)', flush=True)
    print(num_S, 'isolated phase S (solid) blob(s)', flush=True)
    if 0 in (num_A, num_B, num_S):
        return print('Error: Image has less than 3 phases!\n', flush=True)
        
    print(f'\nLabel individual blobs of all three phases in {round(time()-t0,4)} sec!', flush=True)
    
    #################################################
    ### create 3D gird (3 time the size of lA)
    t = time()
    shp0, shp1, shp2 = lA.shape[0], lA.shape[1], lA.shape[2]
    z_,y_,x_ = np.mgrid[0:img.shape[0], 0:img.shape[1], 0:img.shape[2]]
    ces, ns, cos, cos_, lab_ces, lab_cos  = intersectAlongAllAxes(lS, lS_not, _S_not, z_,y_,x_)
    cei, ni, coi, coi_, lab_cei, lab_coi = intersectAlongAllAxes(lA, lB, _B, z_,y_,x_)
    del img, _A, _S, _B, _S_not, z_,y_,x_

    ### merge solid & interface centers (ce) verts & corner (co) verts, & other arrays
    lces, lcei = len(ces), len(cei)

    ce = np.concatenate((ces,cei))
    no = np.concatenate((ns,ni))
    co = np.concatenate((cos,coi))
    co_ = np.concatenate((cos_[:,0],len(ces)+coi_[:,0]))
    co_1 = np.concatenate((cos_[:,1],coi_[:,1]))
    co_ = np.vstack((co_,co_1)).T
    lab_cei[:,2] = -lab_cei[:,2] # -1 for interface
    lab_coi[:,2] = -lab_coi[:,2] # -1 for interface
    labc_ce = np.concatenate((lab_ces, lab_cei)) # labels @ ce
    labc_co =  np.concatenate((lab_cos, lab_coi)) # labels @ co
    del ces, cei, ns, ni, cos, coi, cos_, coi_, co_1, lab_cei, lab_coi, lab_cos, lab_ces
    print(f'Find intersections & collect required info in {round(time()-t,4)} sec!', flush=True)
    
    #################################################  
    ### find unique indexes for verts (just for co; all ce are unique)
    t=time()
    # idx is indices of co in unq, idx_ is indices of unq in co (1st occurrence)
    unq, idx_, idx, cnt = np.unique(co, axis=0, return_index=True, return_inverse=True, return_counts=True)
    print(f'Find unique indexes for vertices in  {round(time()-t,4)} sec!', flush=True) # about 85 sec (nlogn)
    
    #################################################
    ### duplicate co's which are @ multiple intersections (interfaces); find/correct nonorientable co etc
    t=time()
    labc_unq = labc_co[idx_] # labels at unq (unique co's)
    arr = np.vstack((idx, labc_co[:,0],labc_co[:,1],labc_co[:,2], co_[:,0], co_[:,1])).T
    arr = arr[arr[:,0].argsort()]
    # lssp, lssp_, lssp__ = [], [], []
    unq_, labc_unq_, cnt_ = [], [], [] # to store newly-defined verts, labels & counts in corrections
    lenq = len(unq)
    for i in range(len(unq)):
        if i>0:
            _l = _h
            _h = _l + cnt[i]
        elif i==0:
            _l, _h = 0, cnt[0]
        l0 = arr[_l :  _h ]

        if 1<len(l0)<6:
            # # only labels check! (no need to check for mesh orientation) 
            # # check labels & duplicate unq[i] when it is associated with 
            # # more than 1 pair of labels (unq[i] is @ borders of two AB-interfaces)
            l0i =  l0[ l0[:,3]==-1 ]
            if len(l0i)>1:
                na = np.unique(l0i[:,1])
                nb = np.unique(l0i[:,2])
                if len(na)>1 or len(nb)>1:
                    l0sa = l0[ l0[:,3]==-3 ]
                    l0sb = l0[ l0[:,3]==-2 ]
                    for j, a in enumerate(na):
                        for k, b in enumerate(nb):
                            if j!=0 or k!=0:
                                unq_.append(unq[i])
                                labc_unq_.append([a,b,-1])
                                lenq_ = len(unq_) - 1
                                cnt_.append(0) # update cnt (needed later)
                                for l_ in l0i:
                                    if l_[1]==a and l_[2]==b:
                                        l_[0] = lenq + lenq_
                                        cnt[i] -=1      # update cnt (needed later)
                                        cnt_[-1] +=1
                                for l_ in l0sa:
                                    if l_[2]==a:
                                        l_[0] = lenq + lenq_
                                        cnt[i] -=1
                                        cnt_[-1] +=1
                                for l_ in l0sb:
                                    if l_[2]==b:
                                        l_[0] = lenq + lenq_
                                        cnt[i] -=1
                                        cnt_[-1] +=1
                    l0 = np.concatenate((l0i,l0sa,l0sb))
                    arr[_l :  _h ] = l0
                    # print('l0<6\n',arr[_l :  _h ],'\n\n')

        elif len(l0)>5:
            # # check both label & orientation
            # 1st, correction for label if needed
            l0i =  l0[ l0[:,3]==-1 ]
            if len(l0i)>1:
                na = np.unique(l0i[:,1])
                nb = np.unique(l0i[:,2])
                if len(na)>1 or len(nb)>1:
                    l0sa = l0[ l0[:,3]==-3 ]
                    l0sb = l0[ l0[:,3]==-2 ]
                    for j, a in enumerate(na):
                        for k, b in enumerate(nb):
                            if j!=0 or k!=0:
                                unq_.append(unq[i])
                                labc_unq_.append([a,b,-1])
                                lenq_ = len(unq_) - 1
                                cnt_.append(0) # update cnt (needed later)
                                for l_ in l0i:
                                    if l_[1]==a and l_[2]==b:
                                        l_[0] = lenq + lenq_
                                        cnt[i] -=1      # update cnt (needed later)
                                        cnt_[-1] +=1
                                for l_ in l0sa:
                                    if l_[2]==a:
                                        l_[0] = lenq + lenq_
                                        cnt[i] -=1
                                        cnt_[-1] +=1
                                for l_ in l0sb:
                                    if l_[2]==b:
                                        l_[0] = lenq + lenq_
                                        cnt[i] -=1
                                        cnt_[-1] +=1
                    l0 = np.concatenate((l0i,l0sa,l0sb))
                    arr[_l :  _h ] = l0
                    # print('l0>5\n',arr[_l :  _h ],'\n\n')
            # next, correct orientation, if mesh is nonorientable @ the point(s)
            ix_=np.unique(l0[:,0])
            for x_ in ix_:
                l0_ = l0[l0[:,0]==x_]
                msk = None
                if   len(l0_)<6:
                    pass
                elif len(l0_)==6:
                    msk = orientationCorrectorSixCenters(l0_)
                elif len(l0_)==7:
                    msk = orientationCorrectorSevenCenters(l0_)                
                elif len(l0_)==8:
                    msk = orientationCorrectorEightCenters(l0_)
                elif len(l0_)==9: # rare
                    msk = orientationCorrectorNineCenters(l0_)
                elif len(l0_)>9:  # extremely rare
                    msk = orientationCorrectorMoreThanNineCenters(l0_)
                
                if msk is not None:
                    if msk.dtype == bool: # msk is boolean
                        unq_.append(unq[i]) # update unq
                        lenq_ = len(unq_) - 1
                        l0__ = l0_[msk]
                        l0__[:,0] = lenq + lenq_
                        cnt_.append(len(l0__)) # update cnt
                        cnt[i] -= len(l0__)             # update cnt
                        arr[_l :  _h, 0][msk] = l0__[:,0] # update arr
                        labc_unq_.append([l0__[0,1],l0__[0,2],l0__[0,3]])
                        # lssp.extend([l0[0,0]+lces+lcei] + l0[:,4][msk].tolist() )  # RØD
                        # lssp_.extend([l0[0,0]+lces+lcei] + l0[:,4][msk==False].tolist() )  # GRØNN

                    elif msk.dtype != bool: # msk is (0s,1s,2s)
                        for o_ in range(3):
                            unq_.append(unq[i]) # update unq
                            lenq_ = len(unq_) - 1
                            l0__ = l0_[ msk == o_ ]
                            l0__[:,0] = lenq + lenq_
                            cnt_.append(len(l0__)) # update cnt
                            cnt[i] -= len(l0__)             # update cnt
                            arr[_l :  _h, 0][ msk == o_ ] = l0__[:,0] # update arr
                            labc_unq_.append([l0__[0,1],l0__[0,2],l0__[0,3]])
                        # lssp.extend([l0[0,0]+lces+lcei] + l0[:,4][msk==0].tolist() )  # RØD
                        # lssp_.extend([l0[0,0]+lces+lcei] + l0[:,4][msk==1].tolist() )  # GRØNN
                        # lssp__.extend([l0[0,0]+lces+lcei] + l0[:,4][msk==2].tolist() )  # BLÅ                
    # update unq, labc_unq & cnt
    if len(unq_)>0:
        unq = np.concatenate((unq, np.array(unq_)))
        labc_unq = np.concatenate((labc_unq, np.array(labc_unq_)))
        cnt = np.concatenate((cnt, np.array(cnt_)))
    del unq_, labc_unq_, cnt_
    print(f'\nCorrect mesh orientation/labels issues in {round(time()-t,4)} sec!', flush=True)
    
    #################################################
    ### merge verts & labels (centers & unique_corners)
    verts = np.concatenate((ce,unq))
    labc = np.concatenate((labc_ce, labc_unq))
    arr[:,0] += len(ce) # because ce and co are joined in verts
    
    #################################################
    ########### create faces (triangles) ############
    t=time()
    arr = np.vstack((arr[:,4],arr[:,5],arr[:,0],arr[:,3])).T # arr = co_[:,0], co_[:,1], idx, labc_[:,2]
    arr = arr[arr[:,0].argsort()]
    ce_ = arr[0::4, 0]         # center index
    co0 = arr[:,2][arr[:,1]==0] # corner index 0
    co1 = arr[:,2][arr[:,1]==1] # corner index 1
    co2 = arr[:,2][arr[:,1]==2] # corner index 2
    co3 = arr[:,2][arr[:,1]==3] # corner index 3
    fc0 = np.vstack((co0,ce_,co1)).T
    fc1 = np.vstack((co1,ce_,co2)).T
    fc2 = np.vstack((co2,ce_,co3)).T
    fc3 = np.vstack((co3,ce_,co0)).T
    faces = np.concatenate((fc0,fc1,fc2,fc3))
    del fc0, fc1, fc2, fc3, ce, labc_co, labc_ce, labc_unq
    # readjust faces, so normals are from phase A to B (or S to S_not)  
    dp = facesUnitNormals(verts, faces)
    nc = np.concatenate((no,no,no,no))
    dp = dp[:,0]*nc[:,0]+dp[:,1]*nc[:,1]+dp[:,2]*nc[:,2]
    fc = faces[dp<0]
    fc = np.vstack((fc[:,2],fc[:,1],fc[:,0])).T
    faces[dp<0] = fc
    # seperate solid and interface (AB) faces
    facess =faces[faces[:,1]<lces] 
    facesi =faces[faces[:,1]>=lces]
    del faces, fc, nc, dp, no
    print(f'Create faces (triangles) in {round(time()-t, 4)} sec!', flush=True)
    
    #################################################
    if smoothing_struct==True:
        ######################################################################
        ############# CREATE STRUCTURES NEEDED FOR MESH SMOOTHING ############
        ######################################################################
        #########  structures #1 (nbrs, nbri) required for smoothing #########
        print('\nCreate structures required for mesh smoothing!', flush=True)
        # nbrs (for solid mesh) & nbri (for AB interface mesh)
        # are  np.array with 4 columns, where
        # col 0, 1 are all (i,j) & (j,i) vertices making an edge,
        # & col 2,3 are the two vertices on opposite sides
        # of the (i,j) edge. If edge in the mesh boundary,
        # axes 3 will be -1. This array is used in cotangent
        # discretization of Laplace-Beltrami operator to find mean curvature.
        t = time()
        # structure between center verts (ce) & corner verts (co)
        ce__ = np.concatenate((\
                np.vstack((ce_, co0, co1, co3)).T,\
                np.vstack((ce_, co1, co0, co2)).T,\
                np.vstack((ce_, co2, co1, co3)).T,\
                np.vstack((ce_, co3, co0, co2)).T   ))
        # structure between corner verts (co) with each other
        bnd = -np.ones(lces+lcei, dtype=facess.dtype)
        co__ = np.concatenate((\
                np.vstack((co0, co1, ce_, bnd)).T,\
                np.vstack((co1, co2, ce_, bnd)).T,\
                np.vstack((co2, co3, ce_, bnd)).T,\
                np.vstack((co3, co0, ce_, bnd)).T   ))
        del co0, co1, co2, co3, bnd
        # use sort in 1st & 2nd col to have the smaller index of an edge @ 1st col.
        co___ = np.vstack((co__[:,0], co__[:,1])).T
        co___.sort() # smaller index @ first column
        co__ = np.vstack((co___[:,0], co___[:,1], co__[:,2], co__[:,3])).T
        # use of lexsort to have correct consecutive indices after use np.unique below
        co__ = co__[ np.lexsort((co__[:,1], co__[:,0])) ]
        co___ = np.vstack((co__[:,0], co__[:,1])).T
        co__unq,  co__idx_, co__cnt = np.unique(co___, axis=0,  return_index=True, return_counts=True)
        #  co__idx_ is indices of co__unq in co___ (1st occurrence)
        
        # values in counts (co__cnt) are either 
        # 1, edge @ boundary (not at 3-phase contact line)
        # 2, (edge @ middle of mesh)
        # 3, edge @ 3-phase line (boundary for AB mesh, not boundary for Solid mesh)
        
        # # co__cnt==1
        ind = co__idx_[co__cnt==1] # # @ boundary for solid mesh, @ boundary for AB mesh (not @ 3phase contact line)
        co__1 = np.vstack((co__[ind,0], co__[ind,1], co__[ind,2], -np.ones(len(ind), dtype=facesi.dtype))).T

        # # co__cnt==2
        ind = co__idx_[co__cnt==2]  # # @ mesh boundary
        # 2nd occurrence of unique items ( @ co__cnt==2 )
        # happens right after 1st occurrence in co__ (co__idx_ & co__idx_+1)
        co__2 = np.vstack((co__[ind,0], co__[ind,1], co__[ind,2], co__[ind + 1,2])).T

        # # co__cnt==3
        ind = co__idx_[co__cnt==3] # # not @ boundary for solid mesh - @ boundary for AB mesh
        # 2nd & 3rd occurrences of unique items ( @ co__cnt==3 )
        # happen consecutively right after 1st occurrence in co__ (co__idx_, co__idx_+1, co__idx_+2)
        ce0 = np.vstack((co__[ind,2], co__[ind + 1,2], co__[ind + 2,2])).T
        ce0i = ce0.copy()
        ce0[ce0>=lces] = -1 # for solid mesh
        ce0i[ce0i<lces] = -1 # for AB mesh
        ce0.sort() # -1's @ 1st column - ce's @ 2nd & 3rd col.
        ce0i.sort() # -1's @ 1st & 2nd column - ce's @ 2nd col.
        co__3 = np.vstack((co__[ind,0], co__[ind,1], ce0[:,1], ce0[:,2])).T
        co__3i = np.vstack((co__[ind,0], co__[ind,1], ce0i[:,2], -np.ones(len(ind), dtype=facesi.dtype) )).T

        # collect the 3 cases above
        co__ = np.concatenate((co__1, co__2, co__3, co__3i)) # # @ middle and boundary - all together
        
        # unify ce__ & co__ - & separate solid and interface (AB)  
        nbrs = np.concatenate(( ce__[ce__[:,0]<lces], co__[co__[:,2]<lces] ))
        nbri = np.concatenate(( ce__[ce__[:,0]>=lces], co__[co__[:,2]>=lces] ))

        nbrs = np.concatenate((nbrs, np.vstack((nbrs[:,1], nbrs[:,0], nbrs[:,2], nbrs[:,3])).T)) 
        nbri = np.concatenate((nbri, np.vstack((nbri[:,1], nbri[:,0], nbri[:,2], nbri[:,3])).T)) 
        # the latter adds up edge (v0-->v1) that is opposite of edge (v1-->v0). All needed!
        
        nbrs = nbrs[nbrs[:,0].argsort()] # this sorting is necessary @smoothingParallel
        nbri = nbri[nbri[:,0].argsort()] # this sorting is necessary @smoothingParallel

        del ce__, co__, co___, co__unq,  co__idx_, co__cnt, ind, ce0, ce0i, co__1, co__2, co__3, co__3i
        print(f'1) Laplace-Beltrami structures, nbrs & nbri, created in {round(time()-t, 4)} sec!', flush=True)
        
        # ######################################################################
        # ########  structures #2 (nbrfcs, nbrfci) required for smoothing ########  
        # ### find triangles, each vertex is @ -- &  verts @ 3-phase contact line
        # # first, find faces each ce (center) vertex is @
        t=time()
        cesf  = np.vstack((np.arange(lces), np.arange(lces,2*lces), np.arange(2*lces,3*lces), np.arange(3*lces,4*lces))).T # in solid
        ceif  = np.vstack((np.arange(lcei), np.arange(lcei,2*lcei), np.arange(2*lcei,3*lcei), np.arange(3*lcei,4*lcei))).T # in AB interface
        cef = np.concatenate((cesf, ceif)) # i-th element is faces around i-th ce
        del cesf
        # now, find which ce (centers) are around each co (corner) vertex
        arr = arr[arr[:,2].argsort()] # arr = co_[:,0], co_[:,1], idx, labc_[:,2]
        siz = np.max(cnt)
        # i-th element is ce's around co[i] (centers vertices around a corner vertex)
        nbrfcs = -np.ones(shape=(len(unq), 1+4*siz), dtype=facess.dtype)  # solid (SA & SB)
        nbrfci = -np.ones(shape=(len(unq), 1+4*siz), dtype=facesi.dtype)  # interface (AB)
        ind = []
        for i, cn in enumerate(cnt): # or len(unq) - unique co (corner) vertices -
            if i>0:
                _l = _h
                _h = _l + cn
            elif i==0:
                _l, _h = 0, cnt[0]
            l0 = arr[_l :  _h ]
            xi = l0[:,0][ l0[:,3] == -1 ] # fluid-fluid interface
            xs = l0[:,0][ l0[:,3] != -1 ] # solid-fluid
            if len(xi)>0:
                nbrfci[i,0] = l0[0,2]
                fi = cef[xi].flatten()
                ls = fi[np.where(facesi[fi]==l0[0,2])[0]]
                nbrfci[ i, 1:1+len(ls) ] = ls
                if len(xs)>0: # or (len(xs)==0 and len(xi)==2) or (len(xs)==0 and len(xi)==1): 
                    # # cond len(xs)>0 collects 3phase points on solid
                    # # 2nd/3rd cond collects them @ img boundary
                    # xi[0]  is corresponding center index for corner below
                    # l0[0,2] is index of verts (a corner vert) @ 3pahse contact line
                    ind.append([xi[0], l0[0,2]])
            if len(xs)>0:
                nbrfcs[i,0] = l0[0,2]
                fs = cef[xs].flatten()
                ls = fs[np.where(facess[fs]==l0[0,2])[0]]
                nbrfcs[ i, 1:1+len(ls) ] = ls
        # nbrfcs/nbrfci without ce's included
        nbrfcs = nbrfcs[ nbrfcs[:,0]!=-1 ]
        nbrfci = nbrfci[ nbrfci[:,0]!=-1 ]
        # nbrfcs/nbrfci with ce's included too (all verts)
        x_ = -np.ones(shape=(len(cef), 1+4*siz), dtype=nbrfcs.dtype )
        x_[:,0]=cef[:,0]
        x_[lces::,0]+= lces
        x_[:,1]=cef[:,0]
        x_[:,2]=cef[:,1]
        x_[:,3]=cef[:,2]
        x_[:,4]=cef[:,3]
        nbrfcs = np.concatenate((x_[0:lces], nbrfcs))
        nbrfci = np.concatenate((x_[lces::], nbrfci))

        if len(ind)>0:
            ind = np.array(ind)
            labc[ind[:,1]] = labc[ind[:,0]] # overriding labels by (phaseA, phaseB) - fluid-fluid - labels extracted from associated ce vert of a co vert in 3pahse line
            labc[ind[:,1],2] = -4       # mark verts @ 3phase intersection with -4 in column 2 of label array      
        del cef, x_, arr, ind

        # corresponding mask arrays for nbrfci & nbrfcs
        msk_nbrfci = np.zeros(shape=(nbrfci.shape[0], nbrfci.shape[1]-1), dtype=np.bool)
        msk_nbrfcs = np.zeros(shape=(nbrfcs.shape[0], nbrfcs.shape[1]-1), dtype=np.bool)
        msk_nbrfci[nbrfci[:,1::]!=-1] = True
        msk_nbrfcs[nbrfcs[:,1::]!=-1] = True
        print(f'2) nbrfcs & nbrfci structures - faces around vertices - & corresponding boolean masks created in {round(time()-t, 4)} sec!', flush=True)
        
        ######################################################################
        ####### structre#3 (msk_nbrs, msk_nbri) required for smoothing #######
        # retunrs indexing/masking structures for nbri & nbrs
        t=time()
        # solid mesh
        x_, ind, cnt = np.unique(nbrs[:,0], return_index=True, return_counts=True)
        siz = np.max(cnt)
        x_ = np.vstack((cnt, ind, ind + cnt)).T

        ind_nbrs = - np.ones(shape=(len(x_),siz), dtype=facess.dtype)
        for i in range(len(ind_nbrs)):
            ind_nbrs[i,0:x_[i,0]] = np.arange(x_[i,1],x_[i,2],1)
        msk_nbrs = np.zeros(shape=ind_nbrs.shape, dtype=np.bool)
        msk_nbrs[ind_nbrs!=-1] = True
        
        # interface (AB) mesh
        x_, ind, cnt = np.unique(nbri[:,0], return_index=True, return_counts=True)
        siz = np.max(cnt)
        x_ = np.vstack((cnt, ind, ind + cnt)).T
        ind_nbri = - np.ones(shape=(len(x_),siz), dtype=facesi.dtype)
        for i in range(len(ind_nbri)):
            ind_nbri[i,0:x_[i,0]] = np.arange(x_[i,1],x_[i,2],1)
        msk_nbri = np.zeros(shape=ind_nbri.shape, dtype=np.bool)
        msk_nbri[ind_nbri!=-1] = True

        del x_, ind, cnt
        print(f'3) Indexing/masking structures for nbri & nbrs - created in {round(time()-t, 4)} sec!', flush=True)

        #########################################################################
        ############  structure#4 (interface) required for smoothing #############  
        # interface is list of interfaces between phases A & B - 
        # each element is [ labelA, labelB, indices of verts @ this interface, indices of facesi @ this interface ]
        t=time()
        # # find indices of verts @ individual interfaces
        msk = np.logical_or(labc[:,2]==-1, labc[:,2]==-4)
        x_ = np.arange(len(labc))
        x_ = x_[ msk ]
        labc_ = labc[ msk ]
        x_ = np.vstack((labc_[:,0], labc_[:,1], x_)).T
        x_ = x_[ np.lexsort((x_[:,1], x_[:,0])) ] # sorts with 0th col first & 1st col. next
        x_unq, idx, cnt = np.unique( np.vstack((x_[:,0], x_[:,1])).T, axis=0, return_index=True, return_counts=True)
        x_unq = np.vstack((x_unq[:,0], x_unq[:,1], idx, idx + cnt)).T
        del msk, labc_, idx, cnt
        
        # # find indices of facesi @ individual interfaces
        labcf_ =labc[facesi[:,1]]
        xf_ = np.arange(len(facesi)) # indexes of facesi

        xf_ = np.vstack((labcf_[:,0], labcf_[:,1], xf_)).T
        xf_ = xf_[ np.lexsort((xf_[:,1], xf_[:,0])) ] # sorts with 0th col first & 1st col. next

        xf_unq, idxf, cntf = np.unique( np.vstack((xf_[:,0], xf_[:,1])).T, axis=0, return_index=True, return_counts=True)
        xf_unq = np.vstack((xf_unq[:,0], xf_unq[:,1], idxf, idxf + cntf)).T

        interface = []
        for q, qf in zip(x_unq, xf_unq):
            interface.append([q[0], q[1], np.nan, np.nan, np.nan, x_[ q[2]:q[3], 2 ], xf_[ qf[2]:qf[3], 2 ]])
            # q[0], q[1] are labels from A, B phase respectively;
            # the three np.nan are empty elements to store mean curvature, area and mean contact angle when calculated
            # x_[q[2]:q[3], 2 ] is indices of labc or verts which have q[0],q[1] as their labels - only for fluid-fluid mesh (AB)
            # xf_[qf[2]:q[f3], 2 ] is the same as above for facesi
        del x_, x_unq, xf_unq, labcf_, idxf, cntf
        print(f'4) Interface structure - intersection of blobs in phase A with blobs in phase B - created in {round(time()-t, 4)} sec!', flush=True)

        #########################################################################
        # find clusters (both phases A, B) not touching image boundary
        t = time()
        # 1st find a mask to set aside clusters in contact with borders
        # assumed a cylindrical mask here! adjust accordingly if otherwise
        ind = np.arange(len(verts))
        zdwn, zup = -0.5+2, (shp0-0.5)-2
        inside = np.logical_and(verts[:,0]>zdwn, verts[:,0]<zup)

        ymin, ymax = np.min(verts[:,1]), np.max(verts[:,1])
        xmin, xmax = np.min(verts[:,2]), np.max(verts[:,2])
        ycen, xcen = (ymax+ymin)/2, (xmax+xmin)/2
        r = min((ymax-ymin)/2, (xmax-xmin)/2)-0.5
        msk_radial = (verts[:,1]-ycen)**2 + (verts[:,2]-xcen)**2 < r**2
        inside = np.logical_and(inside, msk_radial)
        
        # use mask above to find labels in contact with boundary
        # this will be used to mask off clusters when creating the list 'cluster'
        l_out = labc[inside==False] # labels out !
        msk = np.logical_or(l_out[:,2]==-1, l_out[:,2]==-4)
        l0_out = l_out[msk]
        la_out = l_out[l_out[:,2] == -3]
        la_out = np.vstack((la_out[:,1], la_out[:,0], la_out[:,2])).T
        la_out = np.concatenate((l0_out, la_out)) # phase A labels at 1st col.
        la_out = np.unique(la_out[:,0])
        lb_out = l_out[l_out[:,2] == -2]
        lb_out = np.concatenate((l0_out, lb_out))  # phase B labels at 2nd col.
        lb_out = np.unique(lb_out[:,1])
        del msk, msk_radial, inside, l0_out

        # Find indices of faces (AB & solid) for individual clusters; & their Euler characteristic (X)
        # X(2D watertight mesh on cluster) = num_Verts - (3/2)*num_Faces + num_Faces
        # X(of 3D cluster as vol.) = X(2D watertight mesh on cluster)/2
        # also, number of 3-phase lines on individual clusters is needed in order to find cluster's deficit curvature
        # & eventually mean contact angle per cluster

        # # 1st - find indexes of verts for individual clusters - not needed anymore (alt. #2 in the for loop below)
        # ind = ind[inside]
        # l_ = labc[inside]
        # l_ = np.vstack((l_[:,0], l_[:,1], l_[:,2], ind)).T        
        # msk = np.logical_or(l_[:,2]==-1, l_[:,2]==-4)
        # l0_ = l_[msk]
        # la_ = l_[l_[:,2] == -3]
        # la_ = np.vstack((la_[:,1], la_[:,0], la_[:,2], la_[:,3])).T
        # la_ = np.concatenate((l0_, la_)) # phase A labels at 1st col.
        # la_ = la_[la_[:,0].argsort()]    # sort with 1st col.
        # lb_ = l_[l_[:,2] == -2]
        # lb_ = np.concatenate((l0_, lb_))  # phase B labels at 2nd col.
        # lb_ = lb_[lb_[:,1].argsort()]     # sort with 2nd col.
        # a_unq, idx, cnt = np.unique( la_[:,0], return_index=True, return_counts=True)
        # a_unq = np.vstack((a_unq, idx, idx + cnt)).T
        # b_unq, idx, cnt = np.unique( lb_[:,1], return_index=True, return_counts=True)
        # b_unq = np.vstack((b_unq, idx, idx + cnt)).T

        # 2nd - find indices of facesi @ individual clusters - faces @ fluid-fluid part
        lf_ =labc[facesi[:,1]]
        lfa_ = np.vstack((lf_[:,0], np.arange(len(facesi)))).T # 1st col is phase A labels, last col. is indexes of facesi
        lfa_ = lfa_[lfa_[:,0].argsort()]
        lfb_ = np.vstack((lf_[:,1], np.arange(len(facesi)))).T # 1st col is phase B labels, last col. is indexes of facesi
        lfb_ = lfb_[lfb_[:,0].argsort()]

        laf_unq, idx, cnt = np.unique( lfa_[:,0], return_index=True, return_counts=True)
        laf_unq = np.vstack((laf_unq, idx, idx + cnt)).T
        lbf_unq, idx, cnt = np.unique( lfb_[:,0], return_index=True, return_counts=True)
        lbf_unq = np.vstack((lbf_unq, idx, idx + cnt)).T

        # 3rd - find indices of facess @ individual clusters - faces @ solid-fluid part
        lf_ = labc[facess[:,1]]
        lf_ = np.vstack((lf_[:,1], lf_[:,2], np.arange(len(facess)))).T 
        lfas_= lf_[lf_[:,1]==-3] # 1st col is phase A labels, last col. is indexes of facess
        lfas_ = lfas_[lfas_[:,0].argsort()]
        lfbs_= lf_[lf_[:,1]==-2] # 1st col is phase B labels, last col. is indexes of facess
        lfbs_ = lfbs_[lfbs_[:,0].argsort()]
        
        lfas_unq, idx, cnt = np.unique( lfas_[:,0], return_index=True, return_counts=True)
        lfas_unq = np.vstack((lfas_unq, idx, idx + cnt)).T
        lfbs_unq, idx, cnt = np.unique( lfbs_[:,0], return_index=True, return_counts=True)
        lfbs_unq = np.vstack((lfbs_unq, idx, idx + cnt)).T

        # create list of clusters not in contact with image borders
        cluster = []
        # print('hereeeee', len(la_out), num_A, len(lb_out),num_B)
        nVi = verticesUnitNormals(verts, facesi, nbrfci, msk_nbrfci)

        if len(la_out)<num_A:
            for qfi, qfs in zip(laf_unq, lfas_unq):
                if qfi[0] not in la_out and qfi[0]==qfs[0]: ############### minor bug! ####################
                    F = np.concatenate((facesi[lfa_[qfi[1]:qfi[2], 1 ]], facess[lfas_[qfs[1]:qfs[2], 2 ]]))
                    Vr = np.unique(F)
                    V = len(Vr)
                    F = len(F)
                    X = V - (3/2)*F + F # E=(3/2)*F for a water-tight mesh of fluid cluster
                    if V>500 and X%2==0 and X<=2:
                        # # in one rare case X=1, which is wrong - might be a bug in orientation corrections
                        # count number of 3-phase lines on cluster
                        # first, create line edges
                        li = facesi[lfa_[qfi[1]:qfi[2], 1 ]]
                        li = np.vstack((li[:,0], li[:,2])).T
                        li = li[ np.logical_and( labc[li[:,0],2]==-4, labc[li[:,1],2]==-4 ) ]
                        li.sort()
                        ls = facess[lfas_[qfs[1]:qfs[2], 2 ]]
                        ls = np.vstack((ls[:,0], ls[:,2])).T
                        ls = ls[ np.logical_and( labc[ls[:,0],2]==-4, labc[ls[:,1],2]==-4 ) ]
                        ls.sort()
                        ls = ls.tolist()
                        # next, search for isolated lines 
                        num = 0
                        tw = time()
                        tw_cond = True
                        lines = [] # will contain list of lines – each sorted as a time series with consecutive vertices –
                        while len(li)>0 and tw_cond==True:
                            line = []
                            for j in range(len(li)):
                                if li[j].tolist() in ls:
                                    v0, v_ = li[j]
                                    li[j] = -1
                                    break
                            line.extend([v0, v_])
                            for i in range(len(li)):
                                ind0, ind1 = np.where(li==v_)
                                ind1 = 1-ind1
                                for i0, i1 in zip(ind0, ind1):
                                    if li[i0].tolist() in ls:
                                        v_ = li[i0, i1]
                                        line.append(v_)
                                        li[i0] = -1
                                        if v_ == v0:
                                            line.pop()  # deletes last element (v_) which is the same as element 0
                                            break
                                    else:
                                        li[i0] = -1
                                if time() - tw > 5:
                                    tw_cond = False
                                    break
                            num += 1
                            lines.append(line) # each line is array of verts circling orderly around the line
                            li = li[li[:,0]!=-1]
                        if tw_cond == True:
                            cluster.append([-3, qfi[0], np.nan, np.nan, X, lines, Vr, lfa_[qfi[1]:qfi[2], 1 ], lfas_[qfs[1]:qfs[2], 2 ]])
                            # each element is [-3 (i.e. phase A), label, np.nan, np.nan, Euler_char_2D_manifold, all 3-phase lines, indices of verts, indices of facesi, indices of facess]
                            # np.nan's are two extra element which will be replaced by contact angle for culster when calculated                        
        else:
            print('All clusters in phase A are in contact with image borders!', flush=True)
        if len(lb_out)<num_B:
            for qfi, qfs in zip(lbf_unq, lfbs_unq):
                if qfi[0] not in lb_out  and qfi[0]==qfs[0]: ############### minor bug! ####################
                    # print('BBB', qfi[0], qfs[0])
                    F = np.concatenate((facesi[lfb_[qfi[1]:qfi[2], 1 ]], facess[lfbs_[qfs[1]:qfs[2], 2 ]]))
                    Vr = np.unique(F)
                    V = len(Vr)
                    F = len(F)
                    X = V - (3/2)*F + F # E=(3/2)*F for water-tight mesh of fluid cluster
                    if V>500 and X%2==0 and X<=2:
                        # # in one rare case X=1, which is wrong - might be a bug in orientation corrections
                        # count number of 3-phase lines on cluster
                        # first, create line edges
                        li = facesi[lfb_[qfi[1]:qfi[2], 1 ]]
                        li = np.vstack((li[:,0], li[:,2])).T
                        li = li[ np.logical_and( labc[li[:,0],2]==-4, labc[li[:,1],2]==-4 ) ]
                        li.sort()
                        ls = facess[lfbs_[qfs[1]:qfs[2], 2 ]]
                        ls = np.vstack((ls[:,0], ls[:,2])).T
                        ls = ls[ np.logical_and( labc[ls[:,0],2]==-4, labc[ls[:,1],2]==-4 ) ]
                        ls.sort()
                        ls = ls.tolist()
                        # next, search for isolated lines
                        num = 0
                        tw = time()
                        tw_cond = True
                        lines = []
                        while len(li)>0 and tw_cond==True:
                            line = []
                            for j in range(len(li)):
                                if li[j].tolist() in ls:
                                    v0, v_ = li[j]
                                    li[j] = -1
                                    break
                            line.extend([v0, v_])
                            for i in range(len(li)):
                                ind0, ind1 = np.where(li==v_)
                                ind1 = 1-ind1
                                for i0, i1 in zip(ind0, ind1):
                                    if li[i0].tolist() in ls:
                                        v_ = li[i0, i1]
                                        line.append(v_)
                                        li[i0] = -1
                                        if v_ == v0:
                                            line.pop()
                                            break
                                    else:
                                        li[i0] = -1
                                if time() - tw > 5:
                                    tw_cond = False
                                    break
                            num += 1
                            lines.append(line)
                            li = li[li[:,0]!=-1]
                        if tw_cond == True:
                            cluster.append([-2, qfi[0], np.nan, X, lines, Vr, lfb_[qfi[1]:qfi[2], 1 ], lfbs_[qfs[1]:qfs[2], 2 ]])    
                            # each element is [-2 (i.e. phase B), label, np.nan, np.nan, Euler_char_2D_manifold, all 3-phase lines, indices of verts, indices of facesi, indices of facess]
                            # np.nan is an extra element which will be replaced by contact angle for culster when calculated
        else:
            print('All clusters in phase B are in contact with image borders!', flush=True)
        del idx, cnt, lfa_, lfb_, laf_unq, lbf_unq, lfas_, lfbs_, lfas_unq, lfbs_unq, ls
        print(f'5) Info on fluid clusters not in contact with image boundaries extracted in {round(time()-t, 4)} sec!', flush=True)

        ##################################################################

    ##################################################################
        result = verts, facesi, facess, labc, nbrfci, nbrfcs, msk_nbrfci, msk_nbrfcs,\
                    nbri, nbrs, ind_nbri, ind_nbrs, msk_nbri, msk_nbrs, interface, cluster
    else:
        result = verts, facesi, facess, labc
    
    ##################################################################
    # # # # visualize
    # nvs = facesUnitNormals(verts, facess)
    # nvi = facesUnitNormals(verts, facesi)
    # ces = verts[facess]
    # ces = (ces[:,0]+ces[:,1]+ces[:,2])/3
    # cei = verts[facesi]
    # cei = (cei[:,0]+cei[:,1]+cei[:,2])/3
    # sp = verts[labc[:,2]==-4]
    # # # sp_, sp__ = verts[lssp_], verts[lssp__]
    # mlab.figure(figure='3-phase', bgcolor=(0.95,0.95,0.95), size=(2400, 1200))
    # msh0 = mlab.triangular_mesh(verts[:,0], verts[:,1], verts[:,2], facess, representation='wireframe', color=(0.9, 0.9, 0.9))
    # msh1 = mlab.triangular_mesh(verts[:,0], verts[:,1], verts[:,2], facess, color=(0.53, 0.84, 0.98))
    # nvs = mlab.quiver3d(ces[:,0], ces[:,1], ces[:,2], nvs[:,0], nvs[:,1], nvs[:,2], line_width=2, scale_factor=0.3, color=(0,0,1))
    # msh0 = mlab.triangular_mesh(verts[:,0], verts[:,1], verts[:,2], facesi, representation='wireframe', color=(0.7, 0.7, 0.7))
    # msh2 = mlab.triangular_mesh(verts[:,0], verts[:,1], verts[:,2], facesi, color=(0.9, 0.33, 0.5))
    # nvi = mlab.quiver3d(cei[:,0], cei[:,1], cei[:,2], nvi[:,0], nvi[:,1], nvi[:,2], line_width=2, scale_factor=0.3, color=(0,0,1))
    # sp = mlab.points3d(sp[:,0], sp[:,1], sp[:,2], np.ones(len(sp)), line_width=0.5, scale_factor=0.5, color=(1,0,0))
    # # sp_ = mlab.points3d(sp_[:,0], sp_[:,1], sp_[:,2], np.ones(len(sp_)), line_width=0.5, scale_factor=0.5, color=(0,1,0))
    # # sp__ = mlab.points3d(sp__[:,0], sp__[:,1], sp__[:,2], np.ones(len(sp__)), line_width=0.5, scale_factor=0.5, color=(0,0,1))
    # mlab.show()
    ##################################################################

    print(f'\nTotal runtime for intersection func. {round(time()-t0,4)} sec!', flush=True)
    print(f'\n{len(facesi)} triangles @ AB (fluid-fluid) & {len(facess)} @ S-AB (solid-fluid) interfaial meshes!', flush=True)
    print(f'{len(verts)} vertices, with {len(labc[labc[:,2]==-4])} @ 3-phase contact line!', flush=True)
    print(f'Returning {len(result)} arrays & lists!\n', flush=True)
    return result


def wrapper(func, *args, **kwargs):
    # to time a func. with arguments(example below)
    # wrapped = wrapper(func, *args)
     def wrapped():
         return func(*args, **kwargs)
     return wrapped


def unitVector(vec):
    # returns array of unit vectors of a np.array with shape=(n,3)
    leng = np.sqrt(vec[:,0]**2 + vec[:,1]**2 + vec[:,2]**2)
    leng[leng==0]= 1 # avoids devision by zero if vector is for (0,0,0)
    vec[:,0] /= leng
    vec[:,1] /= leng
    vec[:,2] /= leng                
    return vec


def facesUnitNormals(verts, faces):
    # returns the unit normals of faces
    tris = verts[faces]      
    nFace = np.cross(tris[:,1] - tris[:,0], tris[:,2] - tris[:,0])
    del tris
    nFace = unitVector(nFace) # normalizing the length (length=1)
    return nFace


def verticesUnitNormals(verts, faces, nbrfc, msk_nbrfc):
    # returns the unit normals of vertices
    tris = verts[faces]
    # normals of faces         
    nFace = np.cross(tris[:,1] - tris[:,0], tris[:,2] - tris[:,0])
    del tris
    nFace = unitVector(nFace) # normalizing (length=1)
    nV_ = nFace[nbrfc[:,1::]]
    nV_[msk_nbrfc==False] = 0
    nV_ = np.sum(nV_, axis=1)
    nV_ = unitVector(nV_) # normalizing (length=1)
    nV = np.zeros(verts.shape, dtype=verts.dtype)
    nV[nbrfc[:,0]] = nV_    
    return nV


def averageAllDotProducts(nbr, nV):
    # returns the sum of dot products of vectors n0 & n1
    # n0 & n1 symbolize unit normals for all pairs of neighbor verts
    # the sum is normalized with the number of all pairs
    n = nV[nbr[:,0]]
    n_ = nV[nbr[:,1]]
    n = n[:,0]*n_[:,0] + n[:,1]*n_[:,1] + n[:,2]*n_[:,2]
    return np.average(n) # np.sum(n)/len(n)


def meanGaussianPrincipalCurvatures(verts, faces, nbrfc, msk_nbrfc, nV, nbr, ind_nbr, msk_nbr, **kwargs):
    ## verts (vertices), nV (unit normal vectors at verts)
    # nbr (neighborhood map)
    # ind_nbr, msk_nbr are indexing and masking structures for nbr
    
    # returns a weight func.(wf) for smoothing triangle mesh data
    # returns voroni area for verts as well
    # returns also max triangle area in mesh
    # defualt wf is the mean curvature as in isotropic diffusion smoothing.
    # wf is calculated by anisotropic diffusion method, only 
    # if 'anis_diff' if given as kwargs. In this method, Gaussian and 
    # principle curvatures are also calculated to find wf.

    # For details of the method, check:
    # "Discrete Differential-Geometry Operators for Triangulated 2-Manifolds"
    # by Meyer, Desbrun, Schroderl, Barr, 2003, Springer
    # @ "Visualization and Mathematics III" book, pp 35-57,  
    method = kwargs.get('method')
    Gauss = kwargs.get('return_Gaussian')
    voroni = kwargs.get('vorArea')
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
    wab = vr0 - vr1
    ub = vr0 - vr3
    vb = vr1 - vr3
    del vr0, vr1, vr2, vr3

    withOutBeta = nbr[:,3]==-1
    ub[withOutBeta] = 0 # setting side ub to zero when beta doesn't exist
    vb[withOutBeta] = 0 # setting side vb to zero when beta doesn't exist

    uava  = ua[:,0]* va[:,0] + ua[:,1]* va[:,1] + ua[:,2]* va[:,2] # dot prods
    uawab = ua[:,0]*wab[:,0] + ua[:,1]*wab[:,1] + ua[:,2]*wab[:,2]
    vawab = va[:,0]*wab[:,0] + va[:,1]*wab[:,1] + va[:,2]*wab[:,2]
    ubvb  = ub[:,0]* vb[:,0] + ub[:,1]* vb[:,1] + ub[:,2]* vb[:,2]
    ubwab = ub[:,0]*wab[:,0] + ub[:,1]*wab[:,1] + ub[:,2]*wab[:,2]
    vbwab = vb[:,0]*wab[:,0] + vb[:,1]*wab[:,1] + vb[:,2]*wab[:,2]

    l2ua = ua[:,0]**2 + ua[:,1]**2 + ua[:,2]**2 # squared of lengths
    l2va = va[:,0]**2 + va[:,1]**2 + va[:,2]**2
    l2wab = wab[:,0]**2 + wab[:,1]**2 + wab[:,2]**2
    l2ub = ub[:,0]**2 + ub[:,1]**2 + ub[:,2]**2
    l2vb = vb[:,0]**2 + vb[:,1]**2 + vb[:,2]**2

    # 2x Triangle area on alpha and beta sides of i,j edge
    areaTa = np.sqrt(l2ua*l2va-(uava**2))
    areaTb = np.sqrt(l2ub*l2vb-(ubvb**2))
    del va, ua, ub, vb, l2va, l2vb #, l2ub
    # smoothing can sometimes squeeze all 3 verts of a face so close
    # that the area becomes nearly zero. This may cause zero-division warning
    # & potentially errors. The two lines below is to prevent this!
    # areaTa[areaTa==np.nan] = 2.2250738585072014e-308
    # areaTb[areaTb==np.nan] = 2.2250738585072014e-308
    # max triangle area in the mesh
    maxa = 0.5*max(np.max(areaTa), np.max(areaTb[areaTb>0]))
    
    cota = uava/areaTa  # cot(alpha)
    cotb = np.zeros(shape=cota.shape, dtype=cota.dtype)
    # cot(beta), when beta exists
    withBeta = nbr[:,3]!=-1
    cotb[withBeta] = ubvb[withBeta]/areaTb[withBeta]
    cotb[withOutBeta] = 0  # when beta doesn't exist (edge in boundary)

    # three dot products to see if a triangle is obtuse
    # axis0 (alpha & beta); axis1 (angle by vert i); axis2 (the other angle)
    aa = np.vstack((uava, uawab, -vawab)).T
    bb = np.vstack((ubvb, ubwab, -vbwab)).T
    del uava, vawab, ubvb, vbwab, # ubwab, withBeta, withOutBeta 

    if voroni == 'DMSB99':
        # Desbrun et al. 1999 used Barycenter area which is a 
        # third of the whole 1-ring area A1-ring used (REF below)
        # Mathieu Desbrun, Mark Meyer, Peter Schroder, and Alan H. Barr. Implicit
        # Fairing of Irregular Meshes using Diffusion and Curvature Flow. In SIG-
        # GRAPH 99 Conference Proceedings, pages 317-·324, 1999.
        Ava = verts[faces]
        Ava = np.cross(Ava[:,1] - Ava[:,0], Ava[:,2] - Ava[:,0])
        Ava = 0.5*np.sqrt(Ava[:,0]**2 + Ava[:,1]**2 + Ava[:,2]**2)
        Ava = Ava[nbrfc[:,1::]]
        Ava[msk_nbrfc==False] = 0
        Ava = np.sum(Ava, axis=1)/3
    else:
        # Meyer et. al. 2003 have slightly complicated definition of voronoi region   
        # "Discrete Differential-Geometry Operators for Triangulated 2-Manifolds"
        # by Meyer, Desbrun, Schroderl, Barr, 2003, Springer
        # @ "Visualization and Mathematics III" book, pp 35-57,
        
        # # Ava & Avb (A_voroni) stores areas of alpha and beta sides
        Ava = np.zeros(len(nbr), dtype=verts.dtype)
        Avb = np.zeros(len(nbr), dtype=verts.dtype)
        
        # True if all three are positive (all angles>=90)
        msk = np.logical_and(aa[:,0]>=0, aa[:,1]>=0, aa[:,2]>=0)
        # voroni area where triangle in alpha-side is not obtuse
        Ava[msk] = cota[msk]*l2wab[msk]/8
        # True if all three are positive (all angles>=90)
        msk = np.logical_and(bb[:,0]>=0, bb[:,1]>=0, bb[:,2]>=0)
        # voroni area where triangle in beta-side is not obtuse
        Avb[msk] = cotb[msk]*l2wab[msk]/8
        
        # voroni area at alpha-side when triangle is obtuse at i-angle
        msk = aa[:,1]<0
        Ava[msk] = areaTa[msk]/4
        # voroni area at alpha-side when triangle is obtuse but not in i-angle
        msk = np.logical_or(aa[:,0]<0, aa[:,2]<0)
        Ava[msk] = areaTa[msk]/8
        
        # voroni area at beta-side when triangle is obtuse at i-angle
        msk = bb[:,1]<0
        Avb[msk] = areaTb[msk]/4    
        # voroni area at beta-side when triangle is obtuse but not in i-angle
        msk = np.logical_or(bb[:,0]<0, bb[:,2]<0)
        Avb[msk] = areaTb[msk]/8
        
        Ava = (Ava + Avb)
        del areaTa, areaTb, aa, bb, msk, Avb
        Ava = Ava[ind_nbr]
        Ava[msk_nbr==False] = 0
        Ava = np.sum(Ava, axis=1) # Ava[i] is now total voroni area (Amxd) for vertex i

    # calc. Area mixed (Amxd) and mean curvature (kH) per vertex
    norm = nV[nbr[:,0]]
    dotprd = wab[:,0]*norm[:,0] + wab[:,1]*norm[:,1] + wab[:,2]*norm[:,2]
    kk = (cota + cotb) * dotprd # per edge (i,j)
    del nV, norm, wab, cota, cotb, dotprd

    kk = kk[ind_nbr]
    kk[msk_nbr==False] = 0
    kk = np.sum(kk, axis=1)
    Amxd = np.zeros(len(verts), dtype=verts.dtype)
    kH = np.zeros(len(verts), dtype=verts.dtype)
    kH[nbr[:,0][ind_nbr[:,0]]] = 0.25*kk/Ava # Ava[Ava==0] = 2.2250738585072014e-308 # to prevent devision-by-zero error
    Amxd[nbr[:,0][ind_nbr[:,0]]] = Ava

    # ######################################################
    # ######################################################
    # # compute mean curvature and normal vectors directly from LB
    # Klb0 = (cota + cotb) * wab[:,0]
    # Klb1 = (cota + cotb) * wab[:,1]
    # Klb2 = (cota + cotb) * wab[:,2]
    
    # Klb0 = Klb0[ind_nbr]
    # Klb0[msk_nbr==False] = 0
    # Klb0 = np.sum(Klb0, axis=1)
    
    # Klb1 = Klb1[ind_nbr]
    # Klb1[msk_nbr==False] = 0
    # Klb1 = np.sum(Klb1, axis=1)
    
    # Klb2 = Klb2[ind_nbr]
    # Klb2[msk_nbr==False] = 0
    # Klb2 = np.sum(Klb2, axis=1)

    # Klb0 = Klb0/(2*Ava)
    # Klb1 = Klb1/(2*Ava)
    # Klb2 = Klb2/(2*Ava)
    
    # # LB operator (K_)
    # K_ = np.vstack((Klb0, Klb1, Klb2)).T
    # # value of mean curvature (sign adjusted later)
    # kHlb = 0.5*(K_[:,0]**2 + K_[:,1]**2 + K_[:,2]**2)
    # # nVlb is unit normal for vertices from LB operator
    # nVlb = unitVector(K_)
    # # replace LB unit normal with unit normal from faces when kH=0
    # nVlb[kHlb==0] = nV[ nbr[:,0][ind_nbr[:,0]] ][kHlb==0]
    # # print('kHlb=0 cases', len(kHlb[kHlb==0]), len(kHlb))
    
    # # adjust indexing
    # nV_ = np.zeros((len(verts),3), dtype=verts.dtype)
    # nV_[nbr[:,0][ind_nbr[:,0]]] = nVlb
    
    # kH_ = np.zeros(len(verts), dtype=verts.dtype)
    # kH_[nbr[:,0][ind_nbr[:,0]]] = kHlb
    
    # # adjust mean curvature sign
    # # nV is unit normal from 1-ring faces, nV_ is from LB operator
    # dp = nV[:,0]*nV_[:,0] + nV[:,1]*nV_[:,1] + nV[:,2]*nV_[:,2]
    # kH_[dp<0] = - kH_[dp<0]
    
    # # nV_ & kH_ in this block were tested in smoothing
    # # smoothing with nV_ & kH_ do not converge!
    # ######################################################
    # ######################################################

    # Gaussian & principal curvatures, wieght func. (wf) for anisotropic diffusion smoothing
    if Gauss == True or method == 'aniso_diff':
        # kH[kH==0] = 2.2250738585072014e-308  # to prevent devision-by-zero error
        # Gaussian curvature (kG)
        # uawab is dotprod of two edges (ua & wab) making theta (ta) angle at vertex i
        # costa = uawab/np.sqrt(l2ua*l2wab) # cos(ta)
        ta = np.arccos(uawab/np.sqrt(l2ua*l2wab)) 
        tb = np.zeros(len(nbr), dtype=verts.dtype) # new
        tb[withOutBeta] = 0                        # new
        tb[withBeta] = np.arccos(ubwab[withBeta]/np.sqrt(l2ub[withBeta]*l2wab[withBeta]))  # new
        ta = (ta+tb)/2                             # new
        ta = ta[ind_nbr]
        ta[msk_nbr==False] = 0
        ta = np.sum(ta, axis=1)
        kG = np.zeros(len(verts), dtype=verts.dtype)
        kG[nbr[:,0][ind_nbr[:,0]]] = (2*np.pi - ta)/Ava

        if Gauss == True and method != 'aniso_diff':
            return kH, kG, Amxd, maxa
            
        elif method == 'aniso_diff':
            # principal curvatures (k1, k2)
            dlta = kH**2 - kG
            dlta[dlta<0] = 0
            dlta = np.sqrt(dlta)
            kHabs = np.absolute(kH)
            k1 = kHabs + dlta
            k2 = kHabs - dlta
            del dlta, uawab, l2ua, l2wab, ta, nbr, kk, Ava, withBeta, withOutBeta, l2ub, ubwab

            # weight function (wf) for smoothing by aniso. diff.
            wf = np.ones(len(verts), dtype=verts.dtype)
            TT = 0.7 # user-defined parameter, feature-noise threshold
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
            for i in range(len(msk2)):
                if kHabs[i] != 0: # to avoid devision by zero @ wf=k1/kH or k2/kH
                    msk2[i] = min(k1abs[i], k2abs[i], kHabs[i])
            # for geometric or feature edges (not mesh edges), 
            # smoothing speed proportional to min curvature
            crit1 = k1abs==msk2
            crit2 = k2abs==msk2
            wf[crit1] = k1[crit1]/kH[crit1]
            wf[crit2] = k2[crit2]/kH[crit2]   
            # 3 lines below commented out; as wf is initialized by np.ones 
            #msk3 = np.logical_and(k1abs<=TT, k2abs<=TT)
            #wf[msk3] = 1    # isotropic smoothing for noisy regions
            #wf[kHabs==msk2] = 1 # isotropic smoothing for noisy regions
            wf[wf<-0.1] = -0.1 # stability thresh. to avoid strong inverse diff.
            wf = wf*kH
            del kH, mx, verts, kHabs, k1abs, k2abs, msk2, crit1, crit2
            # in smoothing by anis. diff. the verts are moved along
            # their unit normal vectors by wf*kH (x_new = x_old -(wf*kH)*normal)
            # in isotropic diffusion wf's simply mean curvature. kH (below)
            if Gauss != True:
                return wf, Amxd, maxa # wf(smoothing wieghts), Amxd (voroni area), maxa (max triangle area)        
            elif Gauss == True:
                return wf, kG, Amxd, maxa # wf(smoothing weights), kG (Gaussian curvature), Amxd (voroni area), maxa (max triangle area)        
    else:
        # return H, kH, kG, k1, k2      # returns all curvatures
        del nbr, verts, uawab, l2ua, l2wab, kk, Ava
        return kH, Amxd, maxa # wf(mean curnvature), Amxd (voroni area), maxa (max triangle area)


def clusterBasedContactCalculations(verts, verts_, facesi, nVi, ang, kGi, kGs, Avori, Avors, labc, cluster, path_, filename):
    # each element is [-3 (i.e. phase A), label, np.nan, np.nan, Euler_char_2D_manifold, 3-phase lines, indices of verts, indices of facesi, indices of facess]
    # np.nan's are two extra element which will be replaced by contact angle for culster when calculated
    t = time()
    kGi_A = kGi*Avori
    kGs_A = kGs*Avors
    
    ls_ = []
    clust = np.zeros(shape=(len(cluster),9), dtype=ang.dtype)

    ang_all = ang[ labc[:,2]==-4 ]
    num_lags = 100
    acf_ = acf(ang_all, unbiased=True, fft=True, nlags=num_lags)

    for i, cl in enumerate(cluster):
        kG_A_ = kGs_A[cl[6]] + kGi_A[cl[6]] # solid and interface array added to find kG_A_ for entire cluster
        msk = labc[cl[6],2]==-4
        kG_A_[ msk ] = 0
        kG_A_ = np.sum(kG_A_) # integral Gaussian curvature for solid-fluid and fluid-fluid mesh
        
        # mean contact angle for cluster by Gauss-Bonnet & given that ang = deficit_curvature/(4*number_of_3phase_lines)
        ang_deficit = (180/np.pi)*(2*np.pi*cl[4] - kG_A_)/(4*len(cl[5]))
        # mean contact angle for cluster by averaging the angles at each vertex on 3phase line
        ang_mean = np.mean(ang[ cl[6][msk] ])

        clust[i, 0:2] = cluster[i][0:2]
        clust[i, 2:4] = ang_mean, ang_deficit
        clust[i, 4] = cl[4]
        clust[i, 5] = len(cl[5])
        clust[i, 6:9] = len(cl[6]), len(cl[7]), len(cl[8])
        ###################################################
        sum_rotS = [] # list for sum of (signed) rotation angles per line
        sum_rot = []  # list for sum of (not-signed) rotation angles per line
        
        ########## autocorrelation function, AFC for contact angles  ##########
        mask = np.logical_or(labc[cl[6],2]==-1, labc[cl[6],2]==-4)
        V_f = len(mask[mask==True])

        # performing some calc. for clusters larger than 5000 vertices
        # ACF of angle (with viz), cluster viz (with/without bounding box),
        if V_f>5000:
            #########################################
            ######### UNCOMMENT BELOW BLOCK #########
            ######### cluster visualization #########
            # xyz = verts_[ cl[6][msk] ]
            # mlab.figure(figure='angles color-coded, cluseter '+str(int(cl[1])), size=(2400, 1200), bgcolor=(1, 1, 1), fgcolor=(0.,0.,0.))
            # mlab.points3d(xyz[:,0], xyz[:,1], xyz[:,2], ang[ cl[6][msk] ], colormap='blue-red', scale_factor=0.01) # line_width=0.5,
            # msh0 = mlab.triangular_mesh(verts_[:,0], verts_[:,1], verts_[:,2], facess[cl[8]], representation='wireframe', color=(0.9, 0.9, 0.9))
            # msh0 = mlab.triangular_mesh(verts_[:,0], verts_[:,1], verts_[:,2], facess[cl[8]], color=(0.7, 0.7, 0.7))
            # mlab.show()
            #########################################

            #########################################
            ######### UNCOMMENT BELOW BLOCK #########
            ### cluster visualization with bounding box on solid ###
            # xyz = verts_[ cl[6][msk] ]
            # vr = verts_[cl[6]]
            # zl, zh = np.min(vr[:,0]), np.max(vr[:,0])
            # yl, yh = np.min(vr[:,1]), np.max(vr[:,1])
            # xl, xh = np.min(vr[:,2]), np.max(vr[:,2])
            # vr = verts_[facess[:,1]]
            # mask = np.logical_and(np.logical_and(vr[:,0]>=zl-2, vr[:,0]<=zh+2), \
            #                       np.logical_and(vr[:,1]>=yl-2, vr[:,1]<=yh+2))
            # mask = np.logical_and(mask, np.logical_and(vr[:,2]>=xl-2, vr[:,2]<=xh+2))
            # fcs = facess[mask]
            # mlab.figure(figure='angles color-coded, cluster '+str(int(cl[1])), size=(2400, 1200), bgcolor=(1, 1, 1), fgcolor=(0.,0.,0.))
            # pts = mlab.points3d(xyz[:,0], xyz[:,1], xyz[:,2], ang[ cl[6][msk] ], colormap='blue-red', scale_factor=0.01) # line_width=0.5,
            # cbar = mlab.colorbar(object=pts, title='contact angle (°)')
            # cbar.label_text_property.font_family = 'arial'
            # cbar.label_text_property.font_size = 16
            # cbar.title_text_property.font_family = 'arial'
            # cbar.title_text_property.font_size = 16
            # cbar.data_range = np.array([25.,100.])
            # cbar.number_of_labels = 5
            # # msh0 = mlab.triangular_mesh(verts_[:,0], verts_[:,1], verts_[:,2], fcs, representation='wireframe', color=(0.9, 0.9, 0.9))
            # msh0 = mlab.triangular_mesh(verts_[:,0], verts_[:,1], verts_[:,2], fcs, color=(0.9, 0.6, 0.6))
            # # msh1 = mlab.triangular_mesh(verts[:,0], verts[:,1], verts[:,2], fcs, representation='wireframe', color=(0.9, 0.9, 0.9))
            # msh1 = mlab.triangular_mesh(verts[:,0], verts[:,1], verts[:,2], fcs, color=(0.7, 0.7, 0.7))
            # msh0 = mlab.triangular_mesh(verts_[:,0], verts_[:,1], verts_[:,2], facesi[cl[7]], color=(0, 0.9, 0.9))
            # mlab.show()
            #########################################

            #########################################
            ######### UNCOMMENT BELOW BLOCK #########
            ### plot angles (time series) for lines – and ACF of angles per line
            # r = 0.7
            # color = [(1,0,0), (0,1,0), (0,0,1), (0,0,0), (1,1,0), (1,0,1), (0.6,0.6,0.6), \
            #         (1*r,0,0), (0,1*r,0), (0,0,1*r), (1*r,1*r,0), (1*r,0,1*r), (0.6*r,0.6*r,0.6*r),\
            #         (1,0,0), (0,1,0), (0,0,1), (0,0,0), (1,1,0), (1,0,1), (0.6,0.6,0.6), \
            #         (1*r,0,0), (0,1*r,0), (0,0,1*r), (1*r,1*r,0), (1*r,0,1*r), (0.6*r,0.6*r,0.6*r)]
            # fig, ax = plt.subplots()
            # for j in range(len(cl[5])): # cl[5] is list of lines
            #     plt.plot(np.arange(len(ang[cl[5][j]])), ang[cl[5][j]], 'o', color=color[j], mfc='none', label='line '+ str(j)) # 'ro'
            # plt.xlabel('consecutive vertices', fontsize=16)
            # plt.ylabel(r'$\theta$ (°)',fontsize=16)
            # # # plt.title(f'cluster {str(cl[1])}', fontsize=16)
            # ax.grid(linestyle=':', linewidth=0.5)
            # plt.xlim(0,)
            # plt.ylim(0,105)
            # ax.legend(fontsize=14)
            # plt.savefig(path_ + 'angle_cluster-'+str(cl[1])+'_'+filename+ '.png', dpi=300, bbox_inches='tight')
            # # plt.show()
            # # plot acf (time series) for lines
            # fig, ax = plt.subplots()
            # for j in range(len(cl[5])): # cl[5] is list of lines
            #     if j <8:
            #         acf_ = acf(ang[cl[5][j]], unbiased=True, fft=True, nlags=100) # nlags=int(0.8*len(ang[cl[5][j]]))
            #         plt.plot(np.arange(len(acf_)), acf_, 'o', color=color[j], mfc='none', label='line '+ str(j)) # 'ro'
            # plt.xlabel(r'$lag$', fontsize=16)
            # plt.ylabel(r'$ACF$',fontsize=16)
            # # plt.title(f'cluster {str(cl[1])}', fontsize=16)
            # ax.grid(linestyle=':', linewidth=0.5)
            # plt.xlim(0,)
            # ax.legend(fontsize=14)
            # plt.savefig(path_ + 'ACF_cluster-'+str(cl[1])+'_'+filename+ '.png', dpi=300, bbox_inches='tight')
            # # plt.show()
            #########################################
            
            #########################################
            ######### UNCOMMENT BELOW BLOCK #########
            ### calc. rotation angles (te) on 3phase lines - see if correlates with contact angle (it doesn't!)
            ### & visualize line curvature vector
            # r = 0.7
            # color = [(1,0,0), (0,1,0), (0,0,1), (0,0,0), (1,1,0), (1,0,1), (0.6,0.6,0.6), \
            #         (1*r,0,0), (0,1*r,0), (0,0,1*r), (1*r,1*r,0), (1*r,0,1*r), (0.6*r,0.6*r,0.6*r),\
            #         (1,0,0), (0,1,0), (0,0,1), (0,0,0), (1,1,0), (1,0,1), (0.6,0.6,0.6), \
            #         (1*r,0,0), (0,1*r,0), (0,0,1*r), (1*r,1*r,0), (1*r,0,1*r), (0.6*r,0.6*r,0.6*r)]
            # fig, ax = plt.subplots()        
            # for j in range(len(cl[5])): # cl[5] is list of lines
            #     ed2 = np.vstack(( cl[5][j], np.append(cl[5][j][1::],cl[5][j][0]) )).T
            #     ed3 = np.vstack(( cl[5][j], np.append(cl[5][j][1::],cl[5][j][0]), np.append(cl[5][j][2::],[cl[5][j][0], cl[5][j][1]]) )).T
                
            #     # below block gives a sign to te (inward line curvature+/outward-)
            #     nEd3 = facesUnitNormals(verts_, ed3)
            #     nv_ = nVi[ed3[:,1]]
            #     dp = nEd3[:,0]*nv_[:,0] + nEd3[:,1]*nv_[:,1] + nEd3[:,2]*nv_[:,2]    
            #     v01 = unitVector(verts_[ed3[:,1]] - verts_[ed3[:,0]])
            #     v12 = unitVector(verts_[ed3[:,2]] - verts_[ed3[:,1]])
            #     te = np.arccos( v01[:,0]*v12[:,0] + v01[:,1]*v12[:,1] + v01[:,2]*v12[:,2] )*180/np.pi
            #     te[dp<0] = -te[dp<0]
            #     sum_rotS.append(np.sum(te))
            #     sum_rot.append(np.sum(np.absolute(te)))
                
            #     # signs not needed in this case (just absolute values of rotation angles)
            #     te = np.absolute(te)
            #     plt.plot(te, ang[cl[5][j]], 'o', color=color[j], mfc='none', label='line '+ str(j)) # 'ro'
            # plt.xlabel(r'$\psi $ (°) – proportional to line curvature', fontsize=16)
            # plt.ylabel(r'$\theta$ (°)',fontsize=16)
            # # plt.title(f'cluster {str(cl[1])}', fontsize=16)
            # ax.grid(linestyle=':', linewidth=0.5)
            # plt.xlim(0,90)
            # plt.ylim(0,110)
            # ax.legend(fontsize=14)
            # plt.savefig(path_ + 'theta-psi_cluster-'+str(cl[1])+'_'+filename+ '.png', dpi=300, bbox_inches='tight')
 
            # ### visualize line curvature vector
            # v1neg = verts_[ed3[:,1][dp<0]]
            # vEd3 = (verts_[ed3[:,0]] + verts_[ed3[:,1]] + verts_[ed3[:,2]] )/3
            # vEd1 = verts_[ed3[:,1]]
            # ve = 0.5*(verts_[ed2[:,0]] + verts_[ed2[:,1]])
            # n_ = unitVector(verts_[ed2[:,0]] - verts_[ed2[:,1]])
            # mlab.figure(figure='3-phase', bgcolor=(0.95,0.95,0.95), size=(2400, 1200))
            # # n_ = mlab.quiver3d(ve[:,0], ve[:,1], ve[:,2], n_[:,0], n_[:,1], n_[:,2], line_width=2.2, scale_factor=0.4, color=(0,1,0))
            # # midedge = mlab.points3d(ve[:,0], ve[:,1], ve[:,2], np.ones(len(ve)), line_width=0.5, scale_factor=0.25, color=(0,1,0))
            # v1neg_ = mlab.points3d(v1neg[:,0], v1neg[:,1], v1neg[:,2], np.ones(len(v1neg)), line_width=0.5, scale_factor=0.25, color=(0,1,0))
            # mshEd3 = mlab.triangular_mesh(verts_[:,0], verts_[:,1], verts_[:,2], ed3, color=(0, 0, 1))
            # nEd3_ = mlab.quiver3d(vEd3[:,0], vEd3[:,1], vEd3[:,2], nEd3[:,0], nEd3[:,1], nEd3[:,2], line_width=2.2, scale_factor=0.6, color=(0,0,1))
            # nv__ = mlab.quiver3d(vEd1[:,0], vEd1[:,1], vEd1[:,2], nv_[:,0], nv_[:,1], nv_[:,2], line_width=2.2, scale_factor=0.6, color=(0,1,0))
            # msh0 = mlab.triangular_mesh(verts_[:,0], verts_[:,1], verts_[:,2], facesi[cluster[i][7]], representation='wireframe', color=(0.7, 0.7, 0.7))
            # msh2 = mlab.triangular_mesh(verts_[:,0], verts_[:,1], verts_[:,2], facesi[cluster[i][7]], color=(0.9, 0.33, 0.5))
            # # msh0 = mlab.triangular_mesh(verts_[:,0], verts_[:,1], verts_[:,2], facess[lfas_[qfs[1]:qfs[2], 2 ]], representation='wireframe', color=(0.9, 0.9, 0.9))
            # # msh1 = mlab.triangular_mesh(verts_[:,0], verts_[:,1], verts_[:,2], facess[lfas_[qfs[1]:qfs[2], 2 ]], color=(0.53, 0.84, 0.98))
            # mlab.show()
            #########################################
            
            #########################################
            for j in range(len(cl[5])): # cl[5] is list of lines
                ed2 = np.vstack(( cl[5][j], np.append(cl[5][j][1::],cl[5][j][0]) )).T
                ed3 = np.vstack(( cl[5][j], np.append(cl[5][j][1::],cl[5][j][0]), np.append(cl[5][j][2::],[cl[5][j][0], cl[5][j][1]]) )).T
                # below block gives a sign to te (inward line curvature+/outward-)
                nEd3 = facesUnitNormals(verts_, ed3)
                nv_ = nVi[ed3[:,1]]
                dp = nEd3[:,0]*nv_[:,0] + nEd3[:,1]*nv_[:,1] + nEd3[:,2]*nv_[:,2]    
                v01 = unitVector(verts_[ed3[:,1]] - verts_[ed3[:,0]])
                v12 = unitVector(verts_[ed3[:,2]] - verts_[ed3[:,1]])
                te = np.arccos( v01[:,0]*v12[:,0] + v01[:,1]*v12[:,1] + v01[:,2]*v12[:,2] )*180/np.pi
                te[dp<0] = -te[dp<0]
                sum_rotS.append(np.sum(te))
                sum_rot.append(np.sum(np.absolute(te)))
            # print('Sum rotation angles per line (with sign)', sum_rotS)
            # print('Sum rotation angles per line (without sign)',sum_rot,'\n')
            ##########################################
        ls_.append([ clust[i,0], clust[i,1], clust[i,2], clust[i,3], clust[i,4], [sum_rotS, sum_rot, cl[5]], cl[6], cl[7], cl[8] ])
    clust = clust[clust[:,0]!=0]
    ind = np.lexsort((clust[:,5], clust[:,4]))
    clust = clust[ ind ]
    ls = []
    for i in ind:
        ls.append(ls_[i])
    ls.insert(0, '(PhaseA -3 or PhaseB -2, cluster label, meanAngle(deg), meanDeficitAngle(deg), 2DManifoldEulerChar, Rotationangles&3PhaseLines, indicesOfVertices, indicesOfABFaces, indicesOfSolidFaces')
    print(f'\nCluster-based calc., mean angle, lines etc (check the function) - in {round(time()-t,4)} sec!', flush=True)
    return ls, clust


def clusterBasedContactCalculations_before(verts, ang, facess, facesi, labc, cluster):
    # each element is [-3 (i.e. phase A), label, np.nan, np.nan, Euler_char_2D_manifold, all 3-phase lines, indices of verts, indices of facesi, indices of facess]
    # np.nan's are two extra element which will be replaced by contact angle for culster when calculated
    t=time()
    ls_ = []
    clust = np.zeros(shape=(len(cluster),9), dtype=verts.dtype)
    for i, cl in enumerate(cluster):
        fc = np.concatenate((facess[cl[8]], facesi[cl[7]]))
        kG_A_ = np.zeros(len(cl[6]), dtype=verts.dtype) # to store Gaussian curvature times Voroni area
        for j, v in enumerate(cl[6]):
            te = 0
            ind_, ind = np.where(fc==v)  ##### faces around v - an n² operation (n number of faces in cluster) !!!! #####
            for i_ in ind_:
                v1, v2 = fc[i_][fc[i_]!=v]
                v_, v1_, v2_ = verts[v], verts[v1], verts[v2]
                v1_, v2_ = v1_-v_, v2_-v_
                dotp = v1_[0]*v2_[0] + v1_[1]*v2_[1] + v1_[2]*v2_[2]
                l2v1v2_ = (v1_[0]**2 + v1_[1]**2 + v1_[2]**2)*(v2_[0]**2 + v2_[1]**2 + v2_[2]**2)
                te += np.arccos(dotp/np.sqrt(l2v1v2_))
            kG_A_[j] = 2*np.pi - te

        msk = labc[cl[6],2]==-4
        kG_A_[ msk ] = 0
        kG_A_ = np.sum(kG_A_) # integral Gaussian curvature for entire cluster
        # mean contact angle for cluster by Gauss-Bonnet & given that ang = deficit_curvature/(4*number_of_3phase_lines)
        # cluster[i][3] = (180/np.pi)*(2*np.pi*cl[4] - kG_A_)/(4*cl[5])
        ang_defi = (180/np.pi)*(2*np.pi*cl[4] - kG_A_)/(4*cl[5])
        # mean contact angle for cluster by averaging the angles at each vertex on 3phase line
        ang_mean = np.mean(ang[ cl[6][msk] ])

        clust[i, 0:2] = cluster[i][0:2]
        clust[i, 2:4] = ang_mean, ang_defi
        clust[i, 4:6] = cluster[i][4:6]
        clust[i, 6:9] = len(cluster[i][6]), len(cluster[i][7]), len(cluster[i][8])
        ls_.append([ clust[i,0], clust[i,1], clust[i,2], clust[i,3], clust[i,4], clust[i,5], cluster[i][6], cluster[i][7], cluster[i][8] ])

    clust = clust[clust[:,0]!=0]
    ind = np.lexsort((clust[:,5], clust[:,4]))
    clust = clust[ ind ]
    ls = []
    for i in ind:
        ls.append(ls_[i])
    ls.insert(0, '(PhaseA -3 or PhaseB -2, cluster label, meanAngle(deg), meanDeficitAngle(deg), 2DManifoldEulerChar, NumberOf3PhaseLines, indicesOfVertices, indicesOfABFaces, indicesOfSolidFaces')
    print(f'\nMean contact angle for individual clusters using Gauss-Bonnet theorem calculated in {round(time()-t,4)} sec!', flush=True)
    return ls, clust


def smoothingThreePhase(verts,facesi, facess, nbrfci, nbrfcs, msk_nbrfci, msk_nbrfcs, nbri, nbrs, ind_nbri, ind_nbrs, msk_nbri, msk_nbrs, interface, **kwargs):
    # smooth simultaneously 2 meshes (solid-fluid and fluid-fluid) which share boundaries
    # receives verts, faces, & other mesh structural array (See ThreePhaseIntersection func)
    # returns smoothed verts
        
    def interfaceAverageMeanCurvature(kH, Avor, interface):
        # receives mean curvature (kH) and voroni area (Avor) for verts ( outputs of meancurvatre func.)
        # interface is list of interfaces between phases A & B - each element is ...
        # ... [ labelA, labelB, np.nan, np.nan, np.nan, indices of verts @ this interface, indices of faces @ this interface ]
        # returns mean curvature (kH_) for individual interfaces - returns for all verts
        kH_Av = kH*Avor
        kH_ = np.zeros(len(kH), dtype=kH.dtype) # average of all kH @ interface (for all points)
        std = np.zeros(len(interface), dtype=kH.dtype) # standard deviation of kH @ each interface
        for i, pr in enumerate(interface):
            kH_[pr[5]] = sum(kH_Av[pr[5]])/sum(Avor[pr[5]])
            std[i]= np.std((kH[pr[5]]))
        return  kH_, std

    t=time()
    method = kwargs.get('method')
    nVi = verticesUnitNormals(verts, facesi, nbrfci, msk_nbrfci) # unit normals of verts
    nVs = verticesUnitNormals(verts, facess, nbrfcs, msk_nbrfcs)

    # below are lists to store new verts, smoothing criteria, verts distance from originals, 
    # & min/max face area @ iterations 
    # averageAllDotProducts returns average for dot products of all neighbors' unit normals
    VV, tau_s, dist, tau_f, maxa = [], [], [], [], []
    tau_s.append(1.-averageAllDotProducts(nbrs, nVs))
    dist.append(0.)
    kHi, Avori, max_ai = meanGaussianPrincipalCurvatures(verts,facesi,nbrfci,msk_nbrfci, nVi, nbri, ind_nbri, msk_nbri)
    kHi_, std = interfaceAverageMeanCurvature(kHi, Avori, interface)
    tau_f.append(np.mean(std))
    maxa.append(0.25) # area of all triangles before smoothing
    VV.append(verts)
    del verts

    # DD max distance of each vertex from its original value
    DD_default = 1.7 # default DD is sqrt(3) - longest diameter in a voxel
    method = kwargs.get('method')
    DD = kwargs.get('verts_constraint', DD_default)
    verts_original = np.copy(VV[0])

    # mm is iter. counter @ while loop (must start @ 1)
    # the convergence is checked every nn iters.
    condition, mm, nn = True, 1, 100
    t_s, t_f = 0.15, 0.3     # tuning parameters for solid-fluid and fluid-fluid meshes
    final_countdown = 0
    
    print('\n##############   Smoothing progress  ###############\n', flush=True)
    print('step #', 3*' ', 'tau_s', 9*' ', '  tau_f   ', '  sum((pnt-pnt0)^2)', 4*' ', 'max_tri_Ar', 1*' ','  t_s  ', 1*' ', '  t_f  ', flush=True)
    print('0', 8*' ', round(tau_s[0],7), 9*' ', '--------', 8*' ', round(dist[0],1), 13*' ', round(maxa[0],4), 5*' ',t_s, 2*' ',t_f, flush=True)
    
    while condition:    # smoothing loop
        # verts tuned by moving along their unit normals
        # movement has a weight function (curvatures)
        # @ isotropic diff. weights are mean curvatures
        # @ aniso. diff. weights have feature/noise detection
        # by a thresholding variable (see TT @ curvature func.)
        # weights are multiplied by coefficients t_s & t_f to ensure not-to-fast
        # changes. This seems to be necessary in complex shapes;
        # especially for kHi which coverages to an unknown mean value
        # t_s & t_f (esp. t_f) act as reward or penalty overall weight in iterations
        # when convergence is faster, t_s & t_f increase & vice versa

        # # find curvatures/weights
        if method == None: # isotropic (diffusion) smoothing
            kHi, Avori, max_ai = meanGaussianPrincipalCurvatures(VV[mm-1],facesi,nbrfci,msk_nbrfci, nVi, nbri, ind_nbri, msk_nbri)
            kHi_, std = interfaceAverageMeanCurvature(kHi, Avori, interface)
            dvi_ = (t_f*(nVi.T*(kHi - kHi_))).T   # kHi --> const (for individual AB interfaces)
            tau_f.append(np.mean(std))
            kHs, Avors, max_as = meanGaussianPrincipalCurvatures(VV[mm-1],facess,nbrfcs,msk_nbrfcs, nVs, nbrs, ind_nbrs, msk_nbrs)
            dvs_ = t_s*(nVs.T*kHs).T               # kHs --> min

        elif method == 'aniso_diff':
            kHi, Avori, max_ai = meanGaussianPrincipalCurvatures(VV[mm-1],facesi,nbrfci,msk_nbrfci, nVi, nbri, ind_nbri, msk_nbri, method='aniso_diff')
            kHs, Avors, max_as = meanGaussianPrincipalCurvatures(VV[mm-1],facess,nbrfcs,msk_nbrfcs, nVs, nbrs, ind_nbrs, msk_nbrs, method='aniso_diff')
            dvi_ = 0.1*(nVi.T*(kHi)).T  # kHi --> min
            dvs_ = 0.1*(nVs.T*(kHs)).T  # kHs --> min
        del kHs, kHi, kHi_,nVs, nVi,

        # # update verts
        # 1st, change in every iteration cannot be larger than le (below); this ensures smoother changes
        # 2nd, compare verts_itr with originals & correct the ones displaced more than (DD = 1.7)
        dvi = sum((dvi_**2).T)
        dvs = sum((dvs_**2).T)
        le = 0.2
        msk = dvi>le
        dvi_[msk] = (le*(dvi_[msk].T)/dvi[msk]).T # length larger than le (0.2), becomes le
        msk = dvs>le
        dvs_[msk] = (le*(dvs_[msk].T)/dvs[msk]).T # length larger than le (0.2), becomes le
        if mm <= int(DD/le):
            verts_itr = VV[mm-1] - dvs_ - dvi_    # kHi --> const (for individual AB interfaces) || kHs --> min
            d__ = sum(((verts_original - verts_itr)**2).T)
            d__ = sum(d__)
        else:
            verts_itr = VV[mm-1] - dvs_ - dvi_    # kHi --> const (for individual AB interfaces) || kHs --> min
            d__ = sum(((verts_original - verts_itr)**2).T)
            msk = d__ > DD**2
            verts_itr[msk] = VV[mm-1][msk] # if a vertex jumped over DD, returns back to previous position
            d__ = sum(d__)
        del dvi_, dvs_, dvi, dvs, msk

        # # update the rest
        dist.append(d__) # sum of squared of distances
        VV.append(verts_itr) # save new verts
        del verts_itr
        maxa.append(max(max_ai, max_as))
        nVi = verticesUnitNormals(VV[-1], facesi, nbrfci, msk_nbrfci) # update unit normal
        nVs = verticesUnitNormals(VV[-1], facess, nbrfcs, msk_nbrfcs) # update unit normal
        tau_s.append(1.-averageAllDotProducts(nbrs, nVs))
        
        # # tune t_s & t_f
        if  t_s>=1e-3:
            if tau_s[-1]>tau_s[-2]:
                t_s *=0.8
            else:
                t_s *=1.05
        if t_f>=1e-3:
            if tau_f[-1]>tau_f[-2]:
                t_f *=0.8
            else:
                t_f *= 1.05
        if t_s <1e-3:
            t_f = t_s
        if t_f <1e-3:
            t_s = t_f
        if t_f<=1e-3 and t_s<=1e-3:
            final_countdown += 1
            if final_countdown>5:
                condition = False
        
        print(mm, 8*' ', round(tau_s[mm],7), 7*' ', round(tau_f[mm],7), 7*' ', \
            round(dist[mm],3), 10*' ', round(maxa[mm],4), 5*' ', round(t_s,4), 2*' ', round(t_f,4), flush=True)

        if  mm >= nn or condition == False:
            # checks to stop iter. @ step 100 & all iter. steps after 
            ks_ = np.argmin(tau_s)
            ki_ = np.argmin(tau_f)
            criti = ki_ < mm    # criterion for smoothness of fluid-fluid mesh (criti)
            if criti or condition == False:
                # update verts with the best ones resulted in best AB interfaces 
                # solid interfaces smooth/converge long before AB & stay smoothed when AB interface is improving
                del nbrs, facess, nVs, ind_nbrs, msk_nbrs, nbri, facesi, nVi, ind_nbri, msk_nbri 
                verts = VV[ki_-1]
                VV[ki_-1] = -1
                print(f'\ntau_s  – solid-fluid smoothing criteria – is min at iteration {ks_}', flush=True)
                print(f'tau_s value is {round(tau_s[ks_],6)}', flush=True)
                print(f'\ntau_f – fluid-fluid smoothing criteria – is min at iteration {ki_-1}', flush=True)
                print(f'tau_f value is {round(tau_f[ki_],6)} at iteration {ki_-1}', flush=True)
                print(f'tau_s is {round(tau_s[ki_-1],6)} when tau_f is min at iteration {ki_-1}', flush=True)

            if ki_ == mm: # or ks_ == mm:
                # smoothing may still improve, so itereation should not stop
                VV[mm-nn:mm-1] = [-1]*(nn-1)
                # replaces unnecessary & large elements of VV with an integer
                # only last two element are needed for further iteration
        mm += 1
    del VV, tau_s, dist, maxa
    print(f'\nsmoothing runtime in {mm-1} iterations is {round(time()-t,4)} sec!\n', flush=True)
    return verts # smoothed verts


def sphericalCap(r, h):
    # r complete ball radius, h is cap height
    # cap angle from vertical axis (also equal to contact angle) 
    rr=r+3   #### (previosuly used r+1), r+3 gives larger boundary margin (no change on curvature, angle etc...)
    z,y,x = np.ogrid[-rr:rr+1, -rr:rr+1, -rr:rr+1] # +1 ensures that center of coordinate is at center of image
    # ball of radius r where the ball doesn't touch boundaries of the array
    img = z**2+y**2+x**2 <= r**2
    img = Aval*img # True/False to 1/0
    img[img!=Aval] = Bval
    # inside is Aval=1, outside is Bval=0
    # cap is phase A (Aval=1), fluid outside cap is phase B (Bval=0)
    # img[h+1::] = Sval # below cap is solid (Sval=2) - SPHERICAL CAP HORINZONTAL
    img[z+y+x<np.sqrt(3)*(r-h)] = Sval              # - SPHERICAL CAP LATERRAL
    return img


def spheresCrossedByTwoPlanes(r, h):
    # r complete ball radius, h is cap height
    # cap angle from vertical axis (also equal to contact angle) 
    rr=r+3
    z,y,x = np.ogrid[-rr:rr+1, -rr:rr+1, -rr:rr+1]
    # ball of radius r where the ball doesn't touch boundaries of the array
    img = z**2+y**2+x**2 < r**2  # # one ball
    # img = np.logical_and(z**2+y**2+x**2 < r**2 , (z-12)**2+(y-7)**2+(x-7)**2 < (r+5)**2)  # # two balls
    img = Aval*img # True/False to 1/0
    img[img!=Aval] = Bval
    # inside is Aval=1, outside is Bval=0
    # fluid inside is phase A (Aval=1), fluid outside is phase B (Bval=0)

    # # TWO PLANES # #
    # # used below link to find distance of plane and sphere center
    # # http://www.ambrsoft.com/TrigoCalc/Sphere/SpherePlaneIntersection_.htm
    a1,b1,c1 = 1,-1,-1
    a2,b2,c2 = 1,-1,-1
    # a1=a2 and b1=b2 and c1=c2 -> two parallel planes
    img[ a1*z + b1*y + c1*x >= (r-h)*np.sqrt(a1**2+b1**2+c1**2) ] = Sval # Sval=2 is solid
    img[ a2*z + b2*y + c2*x <=-(r-h)*np.sqrt(a2**2+b2**2+c2**2) ] = Sval
    
    return img



if __name__ == '__main__':
    main()
