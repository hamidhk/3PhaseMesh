
import numpy as np
import pickle
from mayavi import mlab
from mayavi.core import module_manager
from time import time
from statsmodels.tsa.stattools import acf
import matplotlib.pyplot as plt
from skimage import io, measure
from scipy import ndimage


path = 'PATH_TO_FOLDER_WITH_SMOOTHING_RESULTS'
filename = 'WRTIE_IMAGAE_NAME_WITH_EXTENSION'
path_ = path # path of folder where the new results will be saved (for instance ACF of angles for lines on clusters)

# read cluster list
with open(path + filename + '_cluster_summary', 'rb') as fp:   # unpickle
	cluster = pickle.load(fp)

# read arrays
# init_ contains results before smoothing (in order): verts, labc, facesi, facess
init_ = np.load(path + filename + '_init.npz')
# final_ contains results after smoothing (in order): verts_, nVi, nVs, ang, kHi, kGi, Avori, kHs, kGs, Avors
final_ = np.load(path + filename + '_final.npz')



t0=time()
verts =  init_[init_.files[0]] # original vertices
labc =  init_[init_.files[1]]  # labels of vertices
facesi =  init_[init_.files[2]] # faces in fluid–fluid mesh
facess =  init_[init_.files[3]] # faces in solid–fluid mesh

verts_ = final_[final_.files[0]] # smoothed vertice
nVi = final_[final_.files[1]]    # normals of smoothed vertices in fluid–fluid mesh
nVs = final_[final_.files[2]]    # normals of smoothed vertices in solid–fluid mesh
ang = final_[final_.files[3]]    # contact angles
# kHi = final_[final_.files[4]]  # point-wise mean curvatures in fluid–fluid mesh
kGi = final_[final_.files[5]]    # point-wise Gaussian curvatures in fluid–fluid mesh
Avori = final_[final_.files[6]]  # point-wise voronoi area in fluid–fluid mesh
# kHs = final_[final_.files[7]]  # point-wise mean curvatures in solid–fluid mesh
kGs = final_[final_.files[8]]    # point-wise Gaussian curvatures in solid–fluid mesh
Avors = final_[final_.files[9]]  # point-wise voronoi area in solid–fluid mesh


del init_, final_
print(f'all arrays loaded in {round(time()-t0,2)} sec!')


def facesUnitNormals(verts, faces):
	# returns the unit normals of faces
	tris = verts[faces]      
	nFace = np.cross(tris[:,1] - tris[:,0], tris[:,2] - tris[:,0])
	del tris
	nFace = unitVector(nFace) # normalizing the length (length=1)
	return nFace


def unitVector(vec):
	# returns array of unit vectors of a np.array with shape=(n,3)
	leng = np.sqrt(vec[:,0]**2 + vec[:,1]**2 + vec[:,2]**2)
	leng[leng==0]= 1 # avoids devision by zero if vector is for (0,0,0)
	vec[:,0] /= leng
	vec[:,1] /= leng
	vec[:,2] /= leng                
	return vec


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


clusterF, clust = clusterBasedContactCalculations(verts, verts_, facesi, nVi, ang, kGi, kGs, Avori, Avors, labc, cluster)