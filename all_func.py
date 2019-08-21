  
from sys import version
print('python version is,', version)
import timeit
from skimage import io, measure
import numpy as np
from mayavi import mlab
from scipy import spatial, ndimage




def wrapper(func, *args, **kwargs):
    # to time a func. with argsuments(example below)
    #wrapped = wrapper(norm_faces, verts, faces)
    #print(timeit.timeit(wrapped, number=10))
     def wrapped():
         return func(*args, **kwargs)
     return wrapped


def unit_vec(vctr):
    # returns unit vector of one 3D vector
    z, y, x = vctr
    d_v = np.sqrt(z**2 + y**2 + x**2)
    vctr = np.array([z, y, x])/d_v
    return vctr


def unitVector(vec):
    # returns array of unit vectors of a np.array with shape=(n,3)
    leng = np.sqrt(vec[:,0]**2 + vec[:,1]**2 + vec[:,2]**2)
    vec[:,0] /= leng
    vec[:,1] /= leng
    vec[:,2] /= leng                
    return vec


def norm_faces(verts, faces):
    # returns normals of faces for a np.array of faces
    # also min/max tirangle (face) area at the mesh
    tris = verts[faces]          
    nFace = np.cross(tris[::,1] - tris[::,0], tris[::,2] - tris[::,0])
    # investigating fineness of faces by  min/max face area
    s = 0.5*(np.sqrt(nFace[:,0]**2 + nFace[:,1]**2 + nFace[:,2]**2))
    return nFace, min(s), max(s)


def norm_verts(verts, faces, nF):
    # returns normals of verts for a np.array of faces
    # nF is unit normals of faces
    nVert = np.zeros(verts.shape, dtype=verts.dtype)
    nVert[faces[:,0]] += nF
    nVert[faces[:,1]] += nF
    nVert[faces[:,2]] += nF
    return nVert


def facesUnitNormals(verts, faces):
    # returns the unit normals of faces
    tris = verts[faces]      
    nFace = np.cross(tris[:,1] - tris[:,0], tris[:,2] - tris[:,0])
    nFace = unitVector(nFace) # normalizing (length=1)
    return nFace


def verticesUnitNormals(verts, faces):
    # returns the unit normals of vertices
    tris = verts[faces]
    # normals of faces         
    nFace = np.cross(tris[:,1] - tris[:,0], tris[:,2] - tris[:,0])
    nFace = unitVector(nFace) # normalizing (length=1)
    nVerts = np.zeros(verts.shape, dtype=verts.dtype)
    # norms of a vertex found by adding norms of faces surrounding vertex
    nVerts[faces[:,0]] += nFace
    nVerts[faces[:,1]] += nFace
    nVerts[faces[:,2]] += nFace
    nVerts = unitVector(nVerts) # normalizing (length=1)
    return nVerts


def mean_curv_before(verts, nV, nbrV):
    # this func. is not precise when the valence of verts on mesh
    # are not equal
    # curvature = (n2-n1).(p2-p1)/|p2-p1|**2
    # for vert0 with its neighbors, find min and max to compute mean cruv.
    m_curv = np.zeros(nV[:,0].shape, dtype=nV.dtype)
    for i in range(len(nbrV)):
        gradnV = []
        for j in nbrV[i]:
            dV = verts[j]-verts[i]
            dN = nV[j]-nV[i]
            gradnV.append(np.dot(dN, dV)/(dV[0]**2+dV[1]**2+dV[2]**2))
        m_curv[i] = -0.5*(min(gradnV) + max(gradnV))
    return m_curv


def mesh_blbs_slw(nghbrsMap):
    # receives a verts neighborhood map (nghbrs_verts), returns mesh blobs
    # finding mesh blobs is not needed if the separate volumes are labeled and meshed separately
    A = nghbrsMap.copy()
    for i in range(len(A)):
        for j in range(i+1, len(A)):
            if not set(A[i]).isdisjoint(A[j]):
                A[i] += A[j]
                A[i] = sorted(set(A[i]))
                A[j] = A[i]
    #now elements of A are repetitions of non-colliding mesh blobs
    meshBlobs = []
    for i in range(len(A)-1):
        for j in range(i+1, len(A)):
            if A[i] == A[j]:
                if not A[i] in meshBlobs:
                    meshBlobs.append(A[i])
    '''every element of meshBlobs is a collection of indices of verts
       connected immediately or via other verts.
       elements of meshBlobs have no intersection '''
    return meshBlobs


def mesh_blbs(nghbrsMap):
    # receives a verts neighborhood map (nghbrs_verts), returns mesh blobs
    # more than 1000x faster than the previous one for a 30x30x30 image
    # finding mesh blobs is not needed if the separate volumes are labeled and meshed separately
    A = nghbrsMap.copy()
    meshBlobs = []
    condition = True
    p = range(len(A))
    q = []

    while condition:
        n = 0
        mesh = A[p[0]]
        for i in p:
            if not set(mesh).isdisjoint(A[i]):
                mesh += A[i]
                mesh = sorted(set(mesh))
                A[i] = [-1] # turning off the analyzed item with (-1) 
                            #  which does not exist in the verts
                n +=1
            else:
                q.append(i)
        r = q.copy()
        if q:
            for j in q:
                if not set(mesh).isdisjoint(A[j]):
                    mesh += A[j]
                    mesh = sorted(set(mesh))
                    A[j] = [-1]
                    r.remove(j)
                    n += 1
        if n != 0:
            if mesh != [-1]:
                meshBlobs.append(mesh)
        if len(set(r)) == 0:
            condition = False
        else:
            p = r
    '''every element of meshBlobs is a collection of indices of verts
       connected immediately or via other verts.
       elements of meshBlobs have no intersection '''
    return meshBlobs


def nghbrs_faces(faces):
    # returns faces neigbours of a face, for all faces
    nghbrsMap = [[]]*len(faces)
    edges = np.zeros((len(faces),3,2), dtype=faces.dtype)
    edges[:,0] = np.vstack((faces[:,0], faces[:,1])).T
    edges[:,1] = np.vstack((faces[:,1], faces[:,2])).T
    edges[:,2] = np.vstack((faces[:,2], faces[:,0])).T

    edges_lst = []
    for i in range(len(faces)):
        x = set(edges[i,0]), set(edges[i,1]), set(edges[i,2])
        edges_lst.append(x)
    
    for i in range(len(faces)):
        a,b,c = edges_lst[i]
        for j in range(i+1, len(faces)):
            d,e,f = edges_lst[j]
            if ((a in (d,e,f)) or (b in (d,e,f)) or (c in (d,e,f))):
                if not nghbrsMap[i]:
                    nghbrsMap[i] = [j]
                else: nghbrsMap[i].append(j)
                if not nghbrsMap[j]:
                    nghbrsMap[j] = [i]
                else: nghbrsMap[j].append(i)
            if len(nghbrsMap[i]) == 3:
                break
    # i is a face index
    # nghbrsMap[i] are indices of faces neighbor with i
    return nghbrsMap


def vrtx_nghbrs(faces, i):
    # neighbors of a single vertex
    # i is index of a vertex, func. returns indeces of neighbor verts of i
    nghbrs =[]
    for face in faces:
        if i in face:
            nghbrs += list(face)
            nghbrs.remove(i)
    return sorted(set(nghbrs))


def nghbrs_verts_slwst(faces, numVerts):
    # func. receives faces and nr. of verts
    # returns indeces of neighbor verts for all verts
    nghbrsMap = []
    for i in range(numVerts):
        nghbrs =[]
        for face in faces:
            if i in face:
                nghbrs += list(face)
                nghbrs.remove(i)
        nghbrsMap.append(sorted(set(nghbrs)))
        ''' nghbrsMap[j] is list of indices of verts
            which are negibour with vertex j'''
    return nghbrsMap


def nghbrs_verts_slwr(faces, numVerts):
    # func. receives faces and nr. of verts
    # returns indeces of neighbor verts for all verts
    # about 50x faster than nghbrs_verts_slwst for a sample data

    # converting the faces (np.array) to a dictionary for fast lookup
    # in dct, a key is index of a face & a value is a set of the face
    nghbrsMap = []
    faces_dict = {}
    for i in range(len(faces)):
        faces_dict[i] = set(faces[i])
    for i in range(numVerts):
        nghbrs =[]
        for key in faces_dict:
            if i in faces_dict[key]:
                nghbrs += list(faces_dict[key])
                nghbrs.remove(i)
        nghbrsMap.append(sorted(set(nghbrs)))
        ''' nghbrsMap[j] is list of indices of verts
            which are negibors with vertex j'''
    return nghbrsMap


def nghbrs_verts(faces, numVerts):
    # faster for small/medium blobs compared to nghbrs_verts_kdtree
    ''' converting the faces (np.array) to a list of sets;
        creation of those small sets makes the func.
        two orders of magnitude faster in a sample of 5000 verts '''
    nghbrsMap = []
    faces_lst = []
    for i in range(len(faces)):
        faces_lst.append(set(faces[i]))
    for i in range(numVerts):
        nghbrs =[]
        for j in faces_lst:
            if i in j:
                nghbrs += j
                nghbrs.remove(i)
        nghbrsMap.append(sorted(set(nghbrs)))
        ''' nghbrsMap[j] is list of indices of verts
            which are negibors with vertex j'''
    return nghbrsMap


def verticesNeighborhood(faces, numVerts):
    # for every vertex, returns faces vertex is at
    # returns also vertices neighbor with vertex 
    nghbrs_fc = [] # i-th element is faces vertex i is at
    nghbrs_vrt = [] # i-th element is vertices neighbor with vertex i
    for i in range(numVerts):
        fc = []
        a = faces[faces[:,0]==i]
        b = faces[faces[:,1]==i]
        c = faces[faces[:,2]==i]
        for x in (a,b,c):
            if len(x)>=1:
                fc.extend(x.tolist())
        nghbrs_fc.append(fc)

    for ii in range(len(nghbrs_fc)):
        lst=[]
        for y in nghbrs_fc[ii]:
            lst += y
        lst = sorted(set(lst))
        lst.remove(ii)
        nghbrs_vrt.append(lst)
    return nghbrs_fc, nghbrs_vrt


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
            if len(ls) == 1: # edge i-j at boundary of mesh
                nbrLB.append([i, j, ls[0], -1]) # -1 is just an identifier of boundary
    nbrLB = np.array(nbrLB)
    nbrLB = nbrLB[nbrLB[:,3].argsort()] # sorting to have -1's at top of array
    return nbrLB


def verticesLaplaceBeltramiNeighborhood2(faces, verts):
    

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
        nbrLB = nbrLB[nbrLB[:,3].argsort()] # sorting to have -1's at top of array
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
        # nbrLB = nbrLB[nbrLB[:,3].argsort()] # sorting to have -1's at top of array
        return nbrLB


    # splitting the data into smaller batches to run
    # neighbors' search on individual batches
    nn = max(1, int(log10(len(verts))-1))
    if nn == 1:
        nbrLB = verticesNeighborsA(faces, len(verts))
    else:
        zmin, zmax = np.min(verts[:,0]), np.max(verts[:,0])
        ymin, ymax = np.min(verts[:,1]), np.max(verts[:,1])
        xmin, xmax = np.min(verts[:,2]), np.max(verts[:,2])
        dz, dy, dx = (zmax-zmin)/nn, (ymax-ymin)/nn, (xmax-xmin)/nn
        lst = []
        tri = verts[faces]
        for ii in range(1, nn+1):
            zit, zft = zmin + (ii-1)*dz, zmin + ii*dz + 1 # target box (z)
            for jj in range(1, nn+1):
                yit, yft = ymin + (jj-1)*dy, ymin + jj*dy + 1 # target box (y)
                for kk in range(1, nn+1):
                    xit, xft = xmin + (kk-1)*dx, xmin + kk*dx + 1 # target box (x)
                
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
        nbrLB = lst[0]
        for i in range(1, len(lst)):
            nbrLB = np.concatenate((nbrLB, lst[i]))
        nbrLB = nbrLB[nbrLB[:,3].argsort()] # sorting to have -1's at top of array

        for i in range(len(verts)):
            crit = nbrLB[:,0] == i
            ax1 = nbrLB[:,1][crit]
            for j in range(len(ax1)-1):
                for k in range(j+1, len(ax1)):
                    if ax1[k] == ax1[j]:
                        ax1[k] = -2
            nbrLB[:,1][crit] = ax1
        nbrLB = nbrLB[nbrLB[:,1]!=-2]

    return nbrLB


def verticesLaplaceBeltramiNeighborhood3(faces, verts):

    def verticesNeighborsA(facesS, idxVt):
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
        # nbrLB = nbrLB[nbrLB[:,3].argsort()] # sorting to have -1's at top of array
        return nbrLB

    # splitting the data into smaller batches to run
    # neighbors' search on individual patches by 'neighbors'
    nn = max(1, int(log10(len(faces)-1)))
    lst = []
    LL = len(faces)//nn
    rr = len(faces)%nn
    for ii in range(1, nn + 1):
        if ii < nn:
            fcb = faces[(ii-1)*LL : ii*LL]
        elif ii == nn:
            fcb = faces[(ii-1)*LL : ii*LL + rr]
        idxVrt = np.concatenate((fcb[:,0], fcb[:,1], fcb[:,2]))                
        idxVrt = np.unique(idxVrt)
        nbr = neighbors(fcb, idxVrt)
        lst.append(nbr)
    nbrLB = lst[0]
    for i in range(1, len(lst)):
        nbrLB = np.concatenate((nbrLB, lst[i]))


    nbrLB = nbrLB[nbrLB[:,1].argsort()]
    nbrLB = nbrLB[nbrLB[:,3].argsort()] # sorting to have -1's at top of array
    for i in range(len(verts)):
        crit = nbrLB[:,0] == i
        nb = nbrLB[crit]
        for j in range(len(nb)-1):
            for k in range(j+1, len(nb)):
                if nb[:,1][j] == nb[:,1][k]:
                    nb[:,3][k] = nb[:,2][j]
                    nb[:,2][j] = -2
        nbrLB[crit] = nb
    nbrLB = nbrLB[nbrLB[:,2]!=-2]    

    return nbrLB


def verticesLaplaceBeltramiNeighborhood_parallel(faces, verts):
    

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
        nbrLB = nbrLB[nbrLB[:,3].argsort()] # sorting to have -1's at top of array
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
        # nbrLB = nbrLB[nbrLB[:,3].argsort()] # sorting to have -1's at top of array
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
        # multithreading on subarrays
        # and neighbors' search on individual parts
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
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
        mm = max(1, int((10*1.152/15/6000)*len(nbrLB)/5))   # rough num of sub-arrays
        d = int(len(nbrLB)/mm)
        for k in range(mm):
            low = k*d
            if k < mm-1:
                high = (k+1)*d
            else:
                high = (k+1)*d + len(nbrLB)%mm
            sub.append([low,high])
        # removing repeated results in nbrLB (multithreading)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for outp  in executor.map(sequential_cleaning, sub):
                pass
        nbrLB = nbrLB[nbrLB[:,1]!=-2]    
    return nbrLB


def nghbrs_verts_kdtree(verts,r):
    # faster for large blobs compared to nghbrs_verts
    # using KDTree structure to find neighbors of verts
    # at a radius r, regardless of connectivity via edges/faces
    nghbrsMap = []
    # from scipy import spatial
    tree = spatial.KDTree(verts)
    nghbrsMap = tree.query_ball_point(verts, r, 2)  
    for i in range(len(nghbrsMap)):
        nghbrsMap[i].remove(i)
        nghbrsMap[i] = sorted(nghbrsMap[i])
        '''nghbrsMap[i] is list of indices of verts
            which are negibors with vertex i at a radius r'''
    return nghbrsMap


def averageAllDotProducts_before(nbrV, nV):
    # returns the sum of dot products of vectors n1 and n2
    # n1 & n2 symbolize unit normals for all pairs of neighbor verts
    # the sum is normalized with the number of all pairs
    # so the returned value will approach to 1 when surfaces getting smoother
    euclid = []
    for i in range(len(nbrV)):
        for j in nbrV[i]: 
            euclid.append(np.dot(nV[i], nV[j]))
    return sum(euclid)/len(euclid)


def averageAllDotProducts_kdtree_before(verts, r, nV):
    # first finds verts within a radius (r=1), (p-norm=2 --> Eulidean distance)
    tree = spatial.KDTree(verts)
    nghbrsMap = tree.query_ball_point(verts, r, 2)
    # returns the sum of dot products of vectors n1 and n2
    # n1 & n2 symbolize unit normals for all pairs of neighbor verts
    # the sum is normalized with the number of all pairs
    # so the returned value will approach to 1 when surfaces getting smoother
    euclid = []
    for i in range(len(verts)):
        for j in nghbrsMap[i]: 
            euclid.append(np.dot(nV[i], nV[j]))
    return sum(euclid)/len(euclid)


def averageAllDotProducts(nbrLB, nV):
    # returns the sum of dot products of vectors n1 and n2
    # n1 & n2 symbolize unit normals for all pairs of neighbor verts
    # the sum is normalized with the number of all pairs
    # & returned value approaches to 1 when surfaces becoming smoother
    n1 = nV[nbrLB[:,0]]
    n2 = nV[nbrLB[:,1]]
    dotprd = n1[:,0]*n2[:,0] + n1[:,1]*n2[:,1] + n1[:,2]*n2[:,2]
    return np.sum(dotprd)/len(dotprd)


def labeledVolSetA_labeledVolSetB_neighbor_blobs_previous_ver2(lbld_W, nr_blb_W, lbld_N, nr_blb_N):
    # receives labeled wetting (W) and labeled nonwetting (N) imgs (np.arrays)
    # returns a list of labels of W and N which have common WN interface
    # EXTREMELY MEMORY-DEMANDING!
    nghbr_w_n_blb = []
    lst_blb_W = []
    lst_blb_N = []   
    # outer if-else puts the fluid with least number of
    # labels at the outer for loop for faster runtime
    if nr_blb_W <= nr_blb_N:
        for i in range(1, nr_blb_W + 1):
            blb_W = np.where(lbld_W==i, lbld_W, 0) # keeping only label i in labeled_W
            blb_W_dlt = ndimage.binary_dilation(blb_W) # booltype
            lst_blb_W.append(blb_W_dlt)
        for j in range(1, nr_blb_N + 1):
            blb_N = np.where(lbld_N==j, lbld_N, 0)          
            lst_blb_N.append(blb_N)
    else:
        for j in range(1, nr_blb_N + 1):
            blb_N = np.where(lbld_N==j, lbld_N, 0)
            blb_N_dlt = ndimage.binary_dilation(blb_N)          
            lst_blb_N.append(blb_N_dlt)
        for i in range(1, nr_blb_W + 1):
            blb_W = np.where(lbld_W==i, lbld_W, 0)
            lst_blb_W.append(blb_W)
    print('neighbor pairs: (wetting label, nonwetting label)')
    for i in range(nr_blb_W):
        for j in range(nr_blb_N):
            if np.logical_and(lst_blb_W[i], lst_blb_N[j]).any():
                nghbr_w_n_blb.append((i+1,j+1))
                print(nghbr_w_n_blb[-1])
    del lst_blb_W, lst_blb_N
    # nghbr_w_n_blbs is a list of tuples (i,j)
    # i is label of a blob in labeled image of wetting fluid -- j for nonwetting
    return nghbr_w_n_blb


def labeledVolSetA_labeledVolSetB_neighbor_blobs_previous(lbld_W, nr_blb_W, lbld_N, nr_blb_N):
    # receives labeled wetting (W) and labeled nonwetting (N) imgs (np.arrays)
    # returns a list of labels of W and N which have common WN interface
    nghbr_w_n_blb = []    
    # outer if-else puts the fluid with least number of
    # labels at the outer for loop for faster runtime
    print('neighbor pairs: (wetting label, nonwetting label)')
    if nr_blb_W <= nr_blb_N:
        for i in range(1, nr_blb_W + 1):
            blb_W = np.where(lbld_W==i, lbld_W, 0) # keeping only label i in labeled_W
            blb_W_dlt = ndimage.binary_dilation(blb_W) # booltype            
            for j in range(1, nr_blb_N + 1):
                blb_N = np.where(lbld_N==j, lbld_N, 0)          
                # True if the dilated blobs have intersection
                if np.logical_and(blb_W_dlt, blb_N).any():
                    nghbr_w_n_blb.append((i,j))
                    print(nghbr_w_n_blb[-1])
        del blb_W, blb_W_dlt, blb_N
    else:
        for j in range(1, nr_blb_N + 1):
            blb_N = np.where(lbld_N==j, lbld_N, 0)
            blb_N_dlt = ndimage.binary_dilation(blb_N)           
            for i in range(1, nr_blb_W + 1):
                blb_W = np.where(lbld_W==i, lbld_W, 0)
                if np.logical_and(blb_N_dlt, blb_W).any():
                    nghbr_w_n_blb.append((i,j))
                    print(nghbr_w_n_blb[-1])
        del blb_N, blb_N_dlt, blb_W
    # nghbr_w_n_blbs is a list of tuples (i,j)
    # i is label of a blob in labeled image of wetting fluid -- j for nonwetting
    return nghbr_w_n_blb


def labeledVolSetA_labeledVolSetB_neighbor_blobs(lA, lB):
    # receives two sets of labeled volumes (lA & lB) as imgs (np.arrays) where
    # the islotated blobs are labeled by scipy.ndimage.label for individual sets
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
                # (+1)'s in array slicing below is to ensure neighbour blocks 
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
    # removing repeated items from nghbr_ab
    lst = []
    arr = np.array(nghbr_ab)
    arr = arr[arr[:,0].argsort()]
    ax0 = np.unique(arr[:,0])
    for ss in ax0:
        crit = arr[:,0]==ss
        ax1 = np.unique(arr[:,1][crit])
        for tt in ax1:
            lst.append((ss,tt))
    nghbr_ab = np.array(lst)
    print('\npairs of [volA, volB] neighbor vol. blobs:', nghbr_ab,'\n')      
    # nghbr_ab is a list of tuples (i,j)
    # i is blob label in labeled array (lA) for vol set A; j for vol set B
    return nghbr_ab


def labeledVolSetA_labeledVolSetB_neighbor_blobs_last(lA, lB):
    # receives two sets of labeled volumes (lA & lB) as imgs (np.arrays) where
    # the islotated blobs are labeled by scipy.ndimage.label for individual sets
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
                # (+1)'s in array slicing below is to ensure neighbour blocks 
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
    # returns (i,j1,j2,j3...)'s' where i is a blob in A and j1,j2,j3 are neigbors of i in B 
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


def tidy_up_mesh(verts, faces):
    # inputs are np.arrays
    # when a number of faces & their corresponding verts are removed from 
    # a mesh, faces may contain intermittent verts indices.
    # This happens also when faces are subsampled from another mesh,
    # eg in search for common surface between two mesh.
    # func. receives faces changed in such a manner, together with vertices
    # of the original mesh, & returns faces with tidy indexing, keeping
    # their original connection with verts
    faces = faces[faces[:,2].argsort()] # sorts axes (2 first , 0 last) 
    faces = faces[faces[:,1].argsort()]
    faces = faces[faces[:,0].argsort()]
    # constructing np.array for verts
    idx = np.concatenate((faces[:,0], faces[:,1], faces[:,2]))
    idx = np.unique(idx)
    idx = idx[idx.argsort()]
    verts = verts[idx]
    # tidying faces
    faces -= idx[0]
    idx -= idx[0]
    for i in range(len(idx) - 1):
        d = idx[i + 1] - idx[i] - 1
        faces[faces >= idx[i + 1]] -= d
        idx[idx >= idx[i + 1]] -= d 
    return verts, faces    


def tidy_up_mesh_last(verts, faces):
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


def bracket_mesh(verts, faces, qq, zmin, zmax, ymin, ymax, xmin, xmax):
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
        faces = faces[idx]
        verts, faces = tidy_up_mesh(verts, faces)       
        return verts, faces


def mesh_border_edges(faces):
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


def meshA_meshB_common_surface_previous(vertsW, facesW, vertsN, facesN):
    # receives verts/faces of wetting (W) and nonwetting (N)
    # returns the verts/faces for the wetting-nonwetting (WN) interface 
    facesWN_lst = []
    if len(facesW) <= len(facesN):
        vertsN_lst = []
        for i in range(len(vertsN)):
            vertsN_lst.append(list(vertsN[i]))
        for i in range(len(facesW)):
            v0, v1, v2 = vertsW[facesW[i]]
            v0, v1, v2 = list(v0), list(v1), list(v2)
            if ((v0 in vertsN_lst) and (v1 in vertsN_lst) and (v2 in vertsN_lst)):
                facesWN_lst.append(facesW[i]) # faster than np.append
        del vertsN_lst  
    else:
        vertsW_lst = []
        for i in range(len(vertsW)):
            vertsW_lst.append(list(vertsW[i]))
        for i in range(len(facesN)):
            v0, v1, v2 = vertsN[facesN[i]]
            v0, v1, v2 = list(v0), list(v1), list(v2)
            if ((v0 in vertsW_lst) and (v1 in vertsW_lst) and (v2 in vertsW_lst)):
                facesWN_lst.append(facesN[i]) # faster than np.append
        del vertsW_lst          
    # converting list to np.array & sorting axes (2 first , 0 last) 
    facesWN = np.array(facesWN_lst)
    facesWN = facesWN[facesWN[:,2].argsort()]
    facesWN = facesWN[facesWN[:,1].argsort()]
    facesWN = facesWN[facesWN[:,0].argsort()]
    # constructing np.array for vertsWN
    vertWN_idx = np.concatenate((facesWN[:,0], facesWN[:,1], facesWN[:,2]))
    vertWN_idx = vertWN_idx[vertWN_idx.argsort()]
    vertWN_idx = np.unique(vertWN_idx)
    if len(facesW) <= len(facesN):
        vertsWN = vertsW[vertWN_idx]
    else:
        vertsWN = vertsN[vertWN_idx]
    # facesWN may still contain intermittent verts indices
    # because the indices were subsampled from facesW (or facesN)
    # making neat the verts indices in facesWN using vertWN_idx
    facesWN -= vertWN_idx[0]
    vertWN_idx -= vertWN_idx[0]
    for i in range(len(vertWN_idx)-1):
        d = vertWN_idx[i+1] - vertWN_idx[i] - 1
        facesWN[facesWN >= vertWN_idx[i+1]] -= d
        vertWN_idx[vertWN_idx >= vertWN_idx[i+1]] -= d    
    del facesWN_lst, vertWN_idx # clean memory
    return vertsWN, facesWN


def meshA_meshB_common_surface(vertsA, facesA, vertsB, facesB):
    # receives verts/faces of mesh A & mesh B for two neighbor volumes
    # returns mesh (verts/faces) for  AB common surface
    for i in range(2):
        # runs bracket_mesh 2 times 
        # 1st run: is helpulf if two mesh have large difference in size, as
        # bracket_mesh removes a large part of larger mesh. Later, CPU time to
        # search for common surface, will be proportional to the number 
        # of verts in smaller mesh
        # 2nd run: is helpful if both mesh are extremly large,
        # but have small common surface.
        # if-else is to ensure bracket_mesh acts on the larger mesh
        if len(facesA) <= len(facesB):
            zmin, zmax = np.min(vertsA[:,0]), np.max(vertsA[:,0])
            ymin, ymax = np.min(vertsA[:,1]), np.max(vertsA[:,1])
            xmin, xmax = np.min(vertsA[:,2]), np.max(vertsA[:,2])
            vertsB, facesB = bracket_mesh(vertsB, facesB, 1, \
                                          zmin, zmax, ymin, ymax, xmin, xmax)
        else:
            zmin, zmax = np.min(vertsB[:,0]), np.max(vertsB[:,0])
            ymin, ymax = np.min(vertsB[:,1]), np.max(vertsB[:,1])
            xmin, xmax = np.min(vertsB[:,2]), np.max(vertsB[:,2])
            vertsA, facesA = bracket_mesh(vertsA, facesA, 1, \
                                          zmin, zmax, ymin, ymax, xmin, xmax)   
    
    # search for faces which have all three vertices common in the two mesh
    facesAB = []
    # if-else is for further speedup and memory efficiency
    if len(facesA) <= len(facesB):
        triA = vertsA[facesA]
        triA0= triA[:,0].tolist()
        triA1= triA[:,1].tolist()
        triA2= triA[:,2].tolist()
        vertsB = vertsB.tolist()
        for i in range(len(triA)):
            if (triA0[i] in vertsB) and  \
                (triA1[i] in vertsB) and \
                (triA2[i] in vertsB):
                facesAB.append(facesA[i])
        facesAB = np.array(facesAB) # converts list to np.array
        vertsAB, facesAB = tidy_up_mesh(vertsA, facesAB)
    else:
        triB = vertsB[facesB]
        triB0= triB[:,0].tolist()
        triB1= triB[:,1].tolist()
        triB2= triB[:,2].tolist()
        vertsA = vertsA.tolist()
        for i in range(len(triB)):
            if (triB0[i] in vertsA) and  \
                (triB1[i] in vertsA) and \
                (triB2[i] in vertsA):
                facesAB.append(facesB[i])
        facesAB = np.array(facesAB)
        vertsAB, facesAB = tidy_up_mesh(vertsB, facesAB)
    return vertsAB, facesAB


def meshA_meshB_common_surface2(vertsA, facesA, vertsB, facesB):
    # receives verts/faces of mesh A & mesh B for two neighbor volumes
    # returns mesh (verts/faces) for  AB common surface
    for i in range(2):
        # runs bracket_mesh 2 times 
        # 1st run: is helpulf if two mesh have large difference in size, as
        # bracket_mesh removes a large part of larger mesh. Later, CPU time to
        # search for common surface, will be proportional to the number 
        # of verts in smaller mesh
        # 2nd run: is helpful if both mesh are extremly large,
        # but have small common surface.
        # if-else is to ensure bracket_mesh acts on the larger mesh
        if len(facesA) <= len(facesB):
            zmin, zmax = np.min(vertsA[:,0]), np.max(vertsA[:,0])
            ymin, ymax = np.min(vertsA[:,1]), np.max(vertsA[:,1])
            xmin, xmax = np.min(vertsA[:,2]), np.max(vertsA[:,2])
            vertsB, facesB = bracket_mesh(vertsB, facesB, 1, \
                                          zmin, zmax, ymin, ymax, xmin, xmax)
        else:
            zmin, zmax = np.min(vertsB[:,0]), np.max(vertsB[:,0])
            ymin, ymax = np.min(vertsB[:,1]), np.max(vertsB[:,1])
            xmin, xmax = np.min(vertsB[:,2]), np.max(vertsB[:,2])
            vertsA, facesA = bracket_mesh(vertsA, facesA, 1, \
                                          zmin, zmax, ymin, ymax, xmin, xmax)   
    
    # search for faces which have all three vertices common in the two mesh
    facesAB = []
    # if-else is for further speedup and memory efficiency
    if len(facesA) <= len(facesB):
        triA = vertsA[facesA]
        triA0= triA[:,0].tolist()
        triA1= triA[:,1].tolist()
        triA2= triA[:,2].tolist()
        vertsB = vertsB.tolist()
        for i in range(len(triA)):
            cond0 = triA0[i] in vertsB
            cond1 = triA1[i] in vertsB
            cond2 = triA2[i] in vertsB
            if ((cond0 and cond1) or \
                (cond1 and cond2) or \
                (cond2 and cond0)):
                facesAB.append(facesA[i])
        facesAB = np.array(facesAB) # converts list to np.array
        vertsAB, facesAB = tidy_up_mesh(vertsA, facesAB)
    else:
        triB = vertsB[facesB]
        triB0= triB[:,0].tolist()
        triB1= triB[:,1].tolist()
        triB2= triB[:,2].tolist()
        vertsA = vertsA.tolist()
        for i in range(len(triB)):
            cond0 = triB0[i] in vertsA
            cond1 = triB1[i] in vertsA
            cond2 = triB2[i] in vertsA
            if ((cond0 and cond1) or \
                (cond1 and cond2) or \
                (cond2 and cond0)):
                facesAB.append(facesB[i])
        facesAB = np.array(facesAB)
        vertsAB, facesAB = tidy_up_mesh(vertsB, facesAB)
    return vertsAB, facesAB


def meshA_meshB_common_surface3(vertsA, facesA, vertsB, facesB):
    # receives verts/faces of mesh A & mesh B for two neighbor volumes
    # returns mesh (verts/faces) for  AB common surface
    for i in range(2):
        # runs bracket_mesh 2 times 
        # 1st run: is helpulf if two mesh have large difference in size, as
        # bracket_mesh removes a large part of larger mesh. Later, CPU time to
        # search for common surface, will be proportional to the number 
        # of verts in smaller mesh
        # 2nd run: is helpful if both mesh are extremly large,
        # but have small common surface.
        # if-else is to ensure bracket_mesh acts on the larger mesh
        if len(facesA) <= len(facesB):
            zmin, zmax = np.min(vertsA[:,0]), np.max(vertsA[:,0])
            ymin, ymax = np.min(vertsA[:,1]), np.max(vertsA[:,1])
            xmin, xmax = np.min(vertsA[:,2]), np.max(vertsA[:,2])
            vertsB, facesB = bracket_mesh(vertsB, facesB, 1, \
                                          zmin, zmax, ymin, ymax, xmin, xmax)
        else:
            zmin, zmax = np.min(vertsB[:,0]), np.max(vertsB[:,0])
            ymin, ymax = np.min(vertsB[:,1]), np.max(vertsB[:,1])
            xmin, xmax = np.min(vertsB[:,2]), np.max(vertsB[:,2])
            vertsA, facesA = bracket_mesh(vertsA, facesA, 1, \
                                          zmin, zmax, ymin, ymax, xmin, xmax)   
    
    # search for faces which have all three vertices common in the two mesh
    facesAB = []
    # if-else is for further speedup and memory efficiency
    if len(facesA) <= len(facesB):
        triA = vertsA[facesA]
        triA0= triA[:,0].tolist()
        triA1= triA[:,1].tolist()
        triA2= triA[:,2].tolist()
        vertsB = vertsB.tolist()
        for i in range(len(triA)):
            cond0 = triA0[i] in vertsB
            cond1 = triA1[i] in vertsB
            cond2 = triA2[i] in vertsB
            if (cond0 or cond1 or cond2):
                facesAB.append(facesA[i])
        facesAB = np.array(facesAB) # converts list to np.array
        vertsAB, facesAB = tidy_up_mesh(vertsA, facesAB)
    else:
        triB = vertsB[facesB]
        triB0= triB[:,0].tolist()
        triB1= triB[:,1].tolist()
        triB2= triB[:,2].tolist()
        vertsA = vertsA.tolist()
        for i in range(len(triB)):
            cond0 = triB0[i] in vertsA
            cond1 = triB1[i] in vertsA
            cond2 = triB2[i] in vertsA
            if (cond0 or cond1 or cond2):
                facesAB.append(facesB[i])
        facesAB = np.array(facesAB)
        vertsAB, facesAB = tidy_up_mesh(vertsB, facesAB)
    return vertsAB, facesAB


def meshA_meshB_common_surface_last(vertsA, facesA, vertsB, facesB):
    # receives verts/faces of mesh A & mesh B for two neighbor volumes
    # returns mesh (verts/faces) for  AB common surface

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
        # this is helpulf if two mesh have large difference in size,
        # bracket_mesh removes a large part of larger mesh. Later, CPU time to
        # search for common surface, will be proportional to the number 
        # of verts in smaller mesh
        # if-else is to ensure bracket_mesh acts on the larger mesh
        if len(facesA) <= len(facesB):
            zmin, zmax = np.min(vertsA[:,0]), np.max(vertsA[:,0])
            ymin, ymax = np.min(vertsA[:,1]), np.max(vertsA[:,1])
            xmin, xmax = np.min(vertsA[:,2]), np.max(vertsA[:,2])
            vertsB, facesB = bracket_mesh(vertsB, facesB, 1, \
                                          zmin, zmax, ymin, ymax, xmin, xmax)
        else:
            zmin, zmax = np.min(vertsB[:,0]), np.max(vertsB[:,0])
            ymin, ymax = np.min(vertsB[:,1]), np.max(vertsB[:,1])
            xmin, xmax = np.min(vertsB[:,2]), np.max(vertsB[:,2])
            vertsA, facesA = bracket_mesh(vertsA, facesA, 1, \
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
        vertsAB, facesAB = tidy_up_mesh(vertsA, facesAB)
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
        vertsB, facesB = tidy_up_mesh(vertsB, facesB)
        vertsA, facesA = tidy_up_mesh(vertsA, facesA)
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
            


        print('Advanced/parallel interface extraction for large meshes')
        facesAB = []
        facesA2 = facesA[:,0].copy()         
        # calculating the boundaries of sub-arrays of facesA
        n = int((10*1.152/15/8000)*len(facesA)) + 5  # rough num of sub-sample of A for optimiziation
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
        # multithreading on subarrays
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     for inp, outp  in zip(lst, executor.map(sequential, lst)):
        #         pass
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for outp  in executor.map(sequential, lst):
                pass

        facesA = facesA[facesA2!=-2]
        facesAB = np.array(facesAB)
        vertsAB, facesAB = tidy_up_mesh(vertsA, facesAB)
        vertsA, facesA = tidy_up_mesh(vertsA, facesA)
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
        vertsA, facesA = tidy_up_mesh(vertsA, facesA)
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


        n = int((10*1.152/15/8000)*len(facesA)) + 5  # rough num of sub-sample of A for optimiziation
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
        # mutithreading
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for inp, outp  in zip(lst, executor.map(sequential, lst)):
                pass
        facesA = facesA[facesA2!=-2]
        vertsA, facesA = tidy_up_mesh(vertsA, facesA)
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
        # surrounding border; func returns edges in all borders
        edgeAB = mesh_border_edges(facesAB)
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
        # so the face normals points the correct direction
        normF1 = facesUnitNormals(vertsAB, facesAB_new)
        nV = verticesUnitNormals(vertsAB, facesAB)
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
            nfcs1 = facesUnitNormals(vertsAB, fcs1)
            nV1 = nV[fcs1[:,1]]  # nV (verts normals ) calculated above
            dotp1 = nV1[:,0]*nfcs1[:,0] + nV1[:,1]*nfcs1[:,1] + nV1[:,2]*nfcs1[:,2]
            dotp1 = np.sign(dotp1)
            fcs1 = fcs1[dotp1>0]
            facesAB = np.concatenate((facesAB, fcs1))
        if len(fcs2)>0:
            nfcs2 = facesUnitNormals(vertsAB, fcs2)
            nV2 = nV[fcs2[:,1]]
            dotp2 = nV2[:,0]*nfcs2[:,0] + nV2[:,1]*nfcs2[:,1] + nV2[:,2]*nfcs2[:,2]
            dotp2 = np.sign(dotp2)
            fcs2 = fcs2[dotp2>0]
            facesAB = np.concatenate((facesAB, fcs2))

        del seg, seg1, seg2, idx, fcs1, fcs2, dotp1, dotp2, nfcs1, nfcs2
    return vertsAB, facesAB


def meshA_meshB_meshC_common_line(vertsA, facesA, vertsB, facesB, vertsC, facesC):
   
    
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

    print(len(vAB), len(vAC), len(vBC), len(vABC))
    xx = vAB, vAC, vBC, vABS
    return xx


def meshAB_meshC_common_line(nbrLB_AB, vertsAB, vertsC, facesC):
    
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
    print(len(ABCinAB), len(ABCinC))
    print('Testing...',sum(vertsC[ABCinC] - vertsAB[ABCinAB]))
    return ABCinAB, ABCinC


def smoothingWeightsAndCurvatures(verts, nV, nbrLB, *args):
    # returns a weight func.(wf) for smoothing triangle mesh data
    # returns also min/max triangle area
    # defualt wf is the mean curvature as in isotropic diffusion smoothing.
    # wf is calculated by anisotropic diffusion method, only 
    # if 'anis_diff' if given as args. In this method, Gaussian and 
    # principle curvatures are also calculated to find wf.

    # For details of the method, check:
    # "Discrete Differential-Geometry Operators for Triangulated 2-Manifolds"
    # by Meyer, Desbrun, Schroderl, Barr, 2003, Springer
    # @ "Visualization and Mathematics III" book, pp 35-57,  

    # verts inside mesh have both alpha and beta angles
    # verts on the bondaries have only alpha angle
    # a for alpha-side and b for beta-side of an edge (i-j)
    withOutBeta = nbrLB[:,3]==-1
    withBeta = nbrLB[:,3]!=-1
    vr = verts[nbrLB] # vertices
    # triangle sides (edges)
    ua = vr[:,0] - vr[:,2] # vectors for edges
    va = vr[:,1] - vr[:,2]
    wab = vr[:,0] - vr[:,1]
    ub = vr[:,0] - vr[:,3]
    vb = vr[:,1] - vr[:,3]
    ub[withOutBeta] = 0 # setting side ub to zero when beta doesn't exist
    vb[withOutBeta] = 0 # setting side vb to zero when beta doesn't exist

    uava = ua[:,0]*va[:,0] + ua[:,1]*va[:,1] + ua[:,2]*va[:,2] # dot prods
    uawab = ua[:,0]*wab[:,0] + ua[:,1]*wab[:,1] + ua[:,2]*wab[:,2]
    vawab = va[:,0]*wab[:,0] + va[:,1]*wab[:,1] + va[:,2]*wab[:,2]
    ubvb = ub[:,0]*vb[:,0] + ub[:,1]*vb[:,1] + ub[:,2]*vb[:,2]
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
    cota = uava/areaTa  # cot(alpha)
    cotb = np.zeros(shape=cota.shape, dtype=cota.dtype)
    # cot(beta), when beta exists
    cotb[withBeta] = ubvb[withBeta]/areaTb[withBeta]
    cotb[withOutBeta] = 0  # when beta doesn't exist (edge in boundary)
    
    # three dot products to see if a triangle is obtuse
    # axis0 (alpha & beta); axis1 (angle by vert i); axis2 (the other angle)
    aa = np.vstack((uava, uawab, -vawab)).T
    bb = np.vstack((ubvb, ubwab, -vbwab)).T
    # True if all three are positive (all angles>=90)
    aasgn = np.logical_and(aa[:,0]>=0, aa[:,1]>=0, aa[:,2]>=0)
    bbsgn = np.logical_and(bb[:,0]>=0, bb[:,1]>=0, bb[:,2]>=0)
    
    # Av (A_voroni) stores areas of alpha and beta sides in its axes 0, 1
    Av = np.zeros(shape=(len(nbrLB),2), dtype=verts.dtype)
    aaAllPos = aasgn == True
    bbAllPos = bbsgn == True
    aa1Neg = aa[:,1]<0
    bb1Neg = bb[:,1]<0
    aa0Neg = aa[:,0]<0
    bb0Neg = bb[:,0]<0
    aa2Neg = aa[:,2]<0
    bb2Neg = bb[:,2]<0
    # voroni area where triangle in alpha-side is not obtuse
    Av[:,0][aaAllPos] = cota[aaAllPos]*l2wab[aaAllPos]/8
    # voroni area where triangle in beta-side is not obtuse
    Av[:,1][bbAllPos] = cotb[bbAllPos]*l2wab[bbAllPos]/8
    # voroni area at alpha-side when triangle is obtuse at i-angle
    Av[:,0][aa1Neg] = areaTa[aa1Neg]/4
    # voroni area at beta-side when triangle is obtuse at i-angle
    Av[:,1][bb1Neg] = areaTb[bb1Neg]/4
    # voroni area at alpha-side when triangle is obtuse but not in i-angle
    Av[:,0][aa0Neg] = areaTa[aa0Neg]/8
    Av[:,0][aa2Neg] = areaTa[aa2Neg]/8
    # voroni area at beta-side when triangle is obtuse but not in i-angle
    Av[:,1][bb0Neg] = areaTb[bb0Neg]/8 
    Av[:,1][bb2Neg] = areaTb[bb2Neg]/8

    # calc. Area mixed (Amxd) and mean curvature (kH) per vertex
    Amxd = np.zeros(len(verts), dtype=verts.dtype)
    kH = np.zeros(len(verts), dtype=verts.dtype)
    norm = nV[nbrLB[:,0]]
    dotprd = wab[:,0]*norm[:,0] + wab[:,1]*norm[:,1] + wab[:,2]*norm[:,2] 
    kk = (cota + cotb) * dotprd # per edge (i,j)
    for i in range(len(verts)):
        # Av's are voroni area per edge (i,j)
        # Av[:,0] & Av[:,1] for alpha & beta sides respec.
        # Amxd's are voroni area per vertex (i)
        crit = nbrLB[:,0]==i
        Amxd[i] = np.sum(Av[:,0][crit]) + np.sum(Av[:,1][crit])
        kH[i] = 0.25*np.sum(kk[crit])/Amxd[i] # for vertex i

    # wieght func. (wf) for anisotropic diffusion
    if 'aniso_diff' in args:
        # Gaussian curvature (kG)
        kG = np.zeros(len(verts), dtype=verts.dtype)
        tasum = np.zeros(len(verts), dtype=verts.dtype)
        # uawab is dotprod of two edges making theta (ta) angle at vertex i
        costa = uawab/np.sqrt(l2ua*l2wab) # cos(ta)
        ta = np.arccos(costa)
        for i in range(len(tasum)):
            tasum[i] = np.sum(ta[nbrLB[:,0]==i])
            kG[i] = (2*np.pi - tasum[i])/Amxd[i]

        # priciple curvatures (k1, k2)
        dlta = kH**2 - kG
        dlta[dlta<0] = 0
        dlta = np.sqrt(dlta)
        k1 = kH + dlta
        k2 = kH - dlta

        # weight function (wf) for smoothing by anis. diff.
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
        # initialized niether by np.ones, because k1,k2,kH can be 1
        # initialization by values larger than all curvatures
        mx = 1.1*max(np.max(kHabs), np.max(k1abs), np.max(k2abs))
        msk2 = mx*np.ones(len(verts), dtype=verts.dtype)
        for i in range(len(msk2)):
            if kHabs[i] != 0: # to avoid devision by zero @ wf=k1/kH or k2/kH
                msk2[i] = min(k1abs[i], k2abs[i], kHabs[i])
        # for geometric or feature edges (not mesh edges), 
        # smoothing speed proportional to min cruvature
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
        # in smoothing by anis. diff. the verts are moved along
        # their unit normal vectors by wf*kH (x_new = x_old -(wf*kH)*normal)
        # in isotropic diffusion wf's simply mean curvature. kH (below)
    else:
        wf = kH
    
    # min/max triangle area in the mesh
    mina = 0.5*min(np.min(areaTa), np.min(areaTb[areaTb>0]))
    maxa = 0.5*max(np.max(areaTa), np.max(areaTb[areaTb>0]))

    res = (wf, mina, maxa)
    return res


def MeanGaussianPrincipalCurvatures_last(verts, nV, nbrLB, *args):
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

    # verts inside mesh have both alpha and beta angles
    # verts on the bondaries have only alpha angle
    # a for alpha-side and b for beta-side of an edge (i-j)
    withOutBeta = nbrLB[:,3]==-1
    withBeta = nbrLB[:,3]!=-1
    vr = verts[nbrLB] # vertices
    # triangle sides (edges)
    ua = vr[:,0] - vr[:,2] # vectors for edges
    va = vr[:,1] - vr[:,2]
    wab = vr[:,0] - vr[:,1]
    ub = vr[:,0] - vr[:,3]
    vb = vr[:,1] - vr[:,3]
    ub[withOutBeta] = 0 # setting side ub to zero when beta doesn't exist
    vb[withOutBeta] = 0 # setting side vb to zero when beta doesn't exist

    uava = ua[:,0]*va[:,0] + ua[:,1]*va[:,1] + ua[:,2]*va[:,2] # dot prods
    uawab = ua[:,0]*wab[:,0] + ua[:,1]*wab[:,1] + ua[:,2]*wab[:,2]
    vawab = va[:,0]*wab[:,0] + va[:,1]*wab[:,1] + va[:,2]*wab[:,2]
    ubvb = ub[:,0]*vb[:,0] + ub[:,1]*vb[:,1] + ub[:,2]*vb[:,2]
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
    # smoothing may sometimes squeeze all 3 verts of a face so close
    # that the area becomes zero. This causes zero-division warning
    # & potentially errors. The two lines below is to prevent this!
    areaTa[areaTa==0] = 2.2250738585072014e-308
    areaTb[areaTb==0] = 2.2250738585072014e-308

    cota = uava/areaTa  # cot(alpha)
    cotb = np.zeros(shape=cota.shape, dtype=cota.dtype)
    # cot(beta), when beta exists
    cotb[withBeta] = ubvb[withBeta]/areaTb[withBeta]
    cotb[withOutBeta] = 0  # when beta doesn't exist (edge in boundary)
    
    # three dot products to see if a triangle is obtuse
    # axis0 (alpha & beta); axis1 (angle by vert i); axis2 (the other angle)
    aa = np.vstack((uava, uawab, -vawab)).T
    bb = np.vstack((ubvb, ubwab, -vbwab)).T
    # True if all three are positive (all angles>=90)
    aasgn = np.logical_and(aa[:,0]>=0, aa[:,1]>=0, aa[:,2]>=0)
    bbsgn = np.logical_and(bb[:,0]>=0, bb[:,1]>=0, bb[:,2]>=0)
    
    # Av (A_voroni) stores areas of alpha and beta sides in its axes 0, 1
    Av = np.zeros(shape=(len(nbrLB),2), dtype=verts.dtype)
    aaAllPos = aasgn == True
    bbAllPos = bbsgn == True
    aa1Neg = aa[:,1]<0
    bb1Neg = bb[:,1]<0
    aa0Neg = aa[:,0]<0
    bb0Neg = bb[:,0]<0
    aa2Neg = aa[:,2]<0
    bb2Neg = bb[:,2]<0
    # voroni area where triangle in alpha-side is not obtuse
    Av[:,0][aaAllPos] = cota[aaAllPos]*l2wab[aaAllPos]/8
    # voroni area where triangle in beta-side is not obtuse
    Av[:,1][bbAllPos] = cotb[bbAllPos]*l2wab[bbAllPos]/8
    # voroni area at alpha-side when triangle is obtuse at i-angle
    Av[:,0][aa1Neg] = areaTa[aa1Neg]/4
    # voroni area at beta-side when triangle is obtuse at i-angle
    Av[:,1][bb1Neg] = areaTb[bb1Neg]/4
    # voroni area at alpha-side when triangle is obtuse but not in i-angle
    Av[:,0][aa0Neg] = areaTa[aa0Neg]/8
    Av[:,0][aa2Neg] = areaTa[aa2Neg]/8
    # voroni area at beta-side when triangle is obtuse but not in i-angle
    Av[:,1][bb0Neg] = areaTb[bb0Neg]/8 
    Av[:,1][bb2Neg] = areaTb[bb2Neg]/8

    # calc. Area mixed (Amxd) and mean curvature (kH) per vertex
    Amxd = np.zeros(len(verts), dtype=verts.dtype)
    kH = np.zeros(len(verts), dtype=verts.dtype)
    norm = nV[nbrLB[:,0]]
    dotprd = wab[:,0]*norm[:,0] + wab[:,1]*norm[:,1] + wab[:,2]*norm[:,2] 
    kk = (cota + cotb) * dotprd # per edge (i,j)
    for i in range(len(verts)):
        # Av's are voroni area per edge (i,j)
        # Av[:,0] & Av[:,1] for alpha & beta sides respec.
        # Amxd's are voroni area per vertex (i)
        crit = nbrLB[:,0]==i
        Amxd[i] = np.sum(Av[:,0][crit]) + np.sum(Av[:,1][crit])
        if Amxd[i] == 0:    # to prevent devision-by-zero error
            Amxd[i] = 2.2250738585072014e-308
        kH[i] = 0.25*np.sum(kk[crit])/Amxd[i] # for vertex i


    # wieght func. (wf) for anisotropic diffusion
    if 'aniso_diff' in args:
        kH[kH==0] = 2.2250738585072014e-308  # to prevent devision-by-zero error
        # Gaussian curvature (kG)
        kG = np.zeros(len(verts), dtype=verts.dtype)
        tasum = np.zeros(len(verts), dtype=verts.dtype)
        # uawab is dotprod of two edges making theta (ta) angle at vertex i
        l2 = np.sqrt(l2ua*l2wab)
        l2[l2==0] = 2.2250738585072014e-308 # to prevent devision-by-zero error
        costa = uawab/l2 # cos(ta)
        ta = np.arccos(costa)
        for i in range(len(tasum)):
            tasum[i] = np.sum(ta[nbrLB[:,0]==i])
            kG[i] = (2*np.pi - tasum[i])/Amxd[i]

        # principal curvatures (k1, k2)
        dlta = kH**2 - kG
        dlta[dlta<0] = 0
        dlta = np.sqrt(dlta)
        k1 = kH + dlta
        k2 = kH - dlta

        # weight function (wf) for smoothing by anis. diff.
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
        # initialized niether by np.ones, because k1,k2,kH can be 1
        # initialization by values larger than all curvatures
        mx = 1.1*max(np.max(kHabs), np.max(k1abs), np.max(k2abs))
        msk2 = mx*np.ones(len(verts), dtype=verts.dtype)
        for i in range(len(msk2)):
            if kHabs[i] != 0: # to avoid devision by zero @ wf=k1/kH or k2/kH
                msk2[i] = min(k1abs[i], k2abs[i], kHabs[i])
        # for geometric or feature edges (not mesh edges), 
        # smoothing speed proportional to min cruvature
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
        # in smoothing by anis. diff. the verts are moved along
        # their unit normal vectors by wf*kH (x_new = x_old -(wf*kH)*normal)
        # in isotropic diffusion wf's simply mean curvature. kH (below)
    else:
        wf = kH
    
    # max triangle area in the mesh
    maxa = 0.5*max(np.max(areaTa), np.max(areaTb[areaTb>0]))

    # res = kH, kG, k1, k2      # returns all curvatures
    res = (wf, maxa)
    return res


def MeanCurvatures_last(verts, nV, nbrLB):
    # this func. is copied from MeanGaussianPrincipalCurvatures()
    # returns mean curvature only
    # For details of the method, check:
    # "Discrete Differential-Geometry Operators for Triangulated 2-Manifolds"
    # by Meyer, Desbrun, Schroderl, Barr, 2003, Springer
    # @ "Visualization and Mathematics III" book, pp 35-57,  

    # verts inside mesh have both alpha and beta angles
    # verts on the bondaries have only alpha angle
    # a for alpha-side and b for beta-side of an edge (i-j)
    withOutBeta = nbrLB[:,3]==-1
    withBeta = nbrLB[:,3]!=-1
    vr = verts[nbrLB] # vertices
    # triangle sides (edges)
    ua = vr[:,0] - vr[:,2] # vectors for edges
    va = vr[:,1] - vr[:,2]
    wab = vr[:,0] - vr[:,1]
    ub = vr[:,0] - vr[:,3]
    vb = vr[:,1] - vr[:,3]
    ub[withOutBeta] = 0 # setting side ub to zero when beta doesn't exist
    vb[withOutBeta] = 0 # setting side vb to zero when beta doesn't exist

    uava = ua[:,0]*va[:,0] + ua[:,1]*va[:,1] + ua[:,2]*va[:,2] # dot prods
    uawab = ua[:,0]*wab[:,0] + ua[:,1]*wab[:,1] + ua[:,2]*wab[:,2]
    vawab = va[:,0]*wab[:,0] + va[:,1]*wab[:,1] + va[:,2]*wab[:,2]
    ubvb = ub[:,0]*vb[:,0] + ub[:,1]*vb[:,1] + ub[:,2]*vb[:,2]
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
    # smoothing may sometimes squeeze all 3 verts of a face so close
    # that the area becomes zero. This causes zero-division warning
    # & potentially errors. The two lines below is to prevent this!
    areaTa[areaTa==0] = 2.2250738585072014e-308
    areaTb[areaTb==0] = 2.2250738585072014e-308

    cota = uava/areaTa  # cot(alpha)
    cotb = np.zeros(shape=cota.shape, dtype=cota.dtype)
    # cot(beta), when beta exists
    cotb[withBeta] = ubvb[withBeta]/areaTb[withBeta]
    cotb[withOutBeta] = 0  # when beta doesn't exist (edge in boundary)
    
    # three dot products to see if a triangle is obtuse
    # axis0 (alpha & beta); axis1 (angle by vert i); axis2 (the other angle)
    aa = np.vstack((uava, uawab, -vawab)).T
    bb = np.vstack((ubvb, ubwab, -vbwab)).T
    # True if all three are positive (all angles>=90)
    aasgn = np.logical_and(aa[:,0]>=0, aa[:,1]>=0, aa[:,2]>=0)
    bbsgn = np.logical_and(bb[:,0]>=0, bb[:,1]>=0, bb[:,2]>=0)
    
    # Av (A_voroni) stores areas of alpha and beta sides in its axes 0, 1
    Av = np.zeros(shape=(len(nbrLB),2), dtype=verts.dtype)
    aaAllPos = aasgn == True
    bbAllPos = bbsgn == True
    aa1Neg = aa[:,1]<0
    bb1Neg = bb[:,1]<0
    aa0Neg = aa[:,0]<0
    bb0Neg = bb[:,0]<0
    aa2Neg = aa[:,2]<0
    bb2Neg = bb[:,2]<0
    # voroni area where triangle in alpha-side is not obtuse
    Av[:,0][aaAllPos] = cota[aaAllPos]*l2wab[aaAllPos]/8
    # voroni area where triangle in beta-side is not obtuse
    Av[:,1][bbAllPos] = cotb[bbAllPos]*l2wab[bbAllPos]/8
    # voroni area at alpha-side when triangle is obtuse at i-angle
    Av[:,0][aa1Neg] = areaTa[aa1Neg]/4
    # voroni area at beta-side when triangle is obtuse at i-angle
    Av[:,1][bb1Neg] = areaTb[bb1Neg]/4
    # voroni area at alpha-side when triangle is obtuse but not in i-angle
    Av[:,0][aa0Neg] = areaTa[aa0Neg]/8
    Av[:,0][aa2Neg] = areaTa[aa2Neg]/8
    # voroni area at beta-side when triangle is obtuse but not in i-angle
    Av[:,1][bb0Neg] = areaTb[bb0Neg]/8 
    Av[:,1][bb2Neg] = areaTb[bb2Neg]/8

    # calc. Area mixed (Amxd) and mean curvature (kH) per vertex
    Amxd = np.zeros(len(verts), dtype=verts.dtype)
    kH = np.zeros(len(verts), dtype=verts.dtype)
    norm = nV[nbrLB[:,0]]
    dotprd = wab[:,0]*norm[:,0] + wab[:,1]*norm[:,1] + wab[:,2]*norm[:,2] 
    kk = (cota + cotb) * dotprd # per edge (i,j)
    for i in range(len(verts)):
        # Av's are voroni area per edge (i,j)
        # Av[:,0] & Av[:,1] for alpha & beta sides respec.
        # Amxd's are voroni area per vertex (i)
        crit = nbrLB[:,0]==i
        Amxd[i] = np.sum(Av[:,0][crit]) + np.sum(Av[:,1][crit])
        if Amxd[i] == 0:    # to prevent devision-by-zero error
            Amxd[i] = 2.2250738585072014e-308
        kH[i] = 0.25*np.sum(kk[crit])/Amxd[i] # for vertex i
    
    return kH


def smoothing(verts, faces, nbrLB, *args):
    # smoothes given triangulated mesh e.g.the WN-interface
    # receives verts, faces, & Laplace-Beltrami neighborhood map (nbrLB) of verts
    # returns smoothed verts
    print('\nsmoothing...')
    verts_original = np.copy(verts)
    nV = verticesUnitNormals(verts, faces) # unit normals of verts
    # weight called to calc. min/max triangle area; wt unimportant here
    wt, min_A, max_A = smoothingWeightsAndCurvatures(verts, nV, nbrLB)

    # new verts, smoothing criteria, verts distance from originals, 
    # & min/max face area @ iterations 
    # averageAllDotProducts returns average for dot products of all 
    # neighbors' unit normals
    # smoother surface will have a average approaching unity
    VV, smooth, constr, mina, maxa = [], [], [], [], []
    smooth.append(averageAllDotProducts(nbrLB, nV)) # at original verts
    constr.append(np.float64(0))
    mina.append(min_A)
    maxa.append(max_A)
    VV.append(verts)
    # mm is iter. counter @ while loop (must start @ 1)
    # the convergence is checked every nn iters.
    condition, mm, nn = True, 1, 25
    DD = 0.35   # DD max distance of each vertex from its original value
                # DD larger than 0.5 (half a pixel) NOT recommended
    print('\nInitial average for dot products of all', \
            'neighbour unit normals:', smooth[0].round(5))
    print('Initial min/max triangle areas:', \
            mina[0].round(4), maxa[0].round(4),'\n')
    while condition:    # smoothing loop
        # verts tuned by moving along their unit normals
        # movement has a weight function (func. weight)
        # @ isotropic diff. weights are mean curvatures
        # @ uniso. diff. wieghts have feature/noise detection
        # by a thresholding variable (see TT @ wieght func.)
        # weights are multiplied by 0.1 to ensure not-to-fast
        # changes. This seems to be necessary in complex shapes.
        # new_vert = vert - 0.1*(weight)*(unit normal at vert)
        if 'aniso_diff' in args:
            tune, min_a, max_a = smoothingWeightsAndCurvatures(\
                                    VV[mm-1], nV, nbrLB, 'aniso_diff')
        else:
            tune, min_a, max_a = smoothingWeightsAndCurvatures(\
                                    VV[mm-1], nV, nbrLB)
        tune = (nV.T*tune).T
        verts_itr = VV[mm-1] - 0.1*tune
        # comparing verts_itr with the originals & correcting the jumped ones
        # constraint is not to have dislacement more than DD at every vert
        # if disp. is more than DD, vertex goes back to value @ previous iter.
        dd = sum(((verts_original - verts_itr)**2).T) # squared of distances
        nojump = dd >= DD**2
        verts_itr[:,0][nojump] = VV[mm-1][:,0][nojump]
        verts_itr[:,1][nojump] = VV[mm-1][:,1][nojump]
        verts_itr[:,2][nojump] = VV[mm-1][:,2][nojump]

        nV = verticesUnitNormals(verts_itr, faces) # update norms with new verts
        VV.append(verts_itr) # save new verts
        mina.append(min_a)
        maxa.append(max_a)
        # sum of squared of distance difference of updated and original verts
        constr.append(sum(dd)) #constr.append(sum(np.sqrt(dd)))
        smooth.append(averageAllDotProducts(nbrLB, nV))

        if mm % nn == 0: # true at every nn-th iter.
            # checks if iteration should be ended                 
            kk = np.argmax(smooth)
            if kk < mm:
                # true when max smooth in the last nn iterations
                # happened before the very last iteration.
                # update verts with the ones gave max avg. dot prods. & stop iter.
                verts = VV[kk]
                condition = False
                print('\n ##############   Summary   ###############\n')
                print('iter. nr.', 2*' ','ave. dot prods', 2*' ',\
                        'sum squared dist. from initial',\
                        3*' ', 'min triangle area', 2*' ', 'max triangle area')
                for ii in range(mm + 1):
                    print(ii, 14*' ', smooth[ii].round(5), 11*' ',\
                            constr[ii].round(4), 22*' ', mina[ii].round(4),\
                            12*' ', maxa[ii].round(4))         
                print('\naverage for dot products of all neighbour unit', \
                        'normals is max at itr.', kk)
                print('the max avg. dot product is', smooth[kk].round(5), \
                        'and sum of squared distance of verts from originals is',\
                        constr[kk].round(2),'\n')
            if kk == mm:
                # smooth may still increase, so iter. does not stop
                VV[mm-nn:mm] = [-1]*nn
                # replaces unnecessary & large elements of VV with an integer
                # only last element is needed for further iter.
        mm += 1
    del VV, tune, smooth, constr, mina, maxa       
    return verts # smoothed verts


def smoothing_last(verts, faces, nbrLB, **kwargs):
    # smoothes a triangulated mesh e.g.the WN-interface
    # receives verts, faces, & Laplace-Beltrami neighborhood map (nbrLB) of verts
    # returns smoothed verts
    print('\nsmoothing iteration num.')
    verts_original = np.copy(verts)
    nV = verticesUnitNormals(verts, faces) # unit normals of verts
    # weight called to calc. min/max triangle area; wt unimportant here
    wt, max_A = MeanGaussianPrincipalCurvatures(verts, nV, nbrLB)

    # new verts, smoothing criteria, verts distance from originals, 
    # & min/max face area @ iterations 
    # averageAllDotProducts returns average for dot products of all 
    # neighbors' unit normals
    # smoother surface will have an average approaching unity
    VV, smooth, constr, maxa = [], [], [], []
    smooth.append(averageAllDotProducts(nbrLB, nV)) # at original verts
    constr.append(np.float64(0))
    maxa.append(max_A)
    VV.append(verts)
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
        # @ uniso. diff. wieghts have feature/noise detection
        # by a thresholding variable (see TT @ wieght func.)
        # weights are multiplied by 0.1 to ensure not-to-fast
        # changes. This seems to be necessary in complex shapes.
        # new_vert = vert - 0.1*(weight)*(unit normal at vert)
        if method == 'aniso_diff':
            tune, max_a = MeanGaussianPrincipalCurvatures(\
                                    VV[mm-1], nV, nbrLB, 'aniso_diff')
        else:
            tune, max_a = MeanGaussianPrincipalCurvatures(\
                                    VV[mm-1], nV, nbrLB)
        tune = (nV.T*tune).T
        verts_itr = VV[mm-1] - 0.1*tune
        # comparing verts_itr with the originals & correcting the jumped ones
        # constraint is not to have dislacement more than DD at every vert
        # if disp. is more than DD, vertex goes back to value @ previous iter.
        dd = sum(((verts_original - verts_itr)**2).T) # squared of distances
        nojump = dd >= DD**2
        verts_itr[:,0][nojump] = VV[mm-1][:,0][nojump]
        verts_itr[:,1][nojump] = VV[mm-1][:,1][nojump]
        verts_itr[:,2][nojump] = VV[mm-1][:,2][nojump]

        nV = verticesUnitNormals(verts_itr, faces) # update norms with new verts
        VV.append(verts_itr) # save new verts
        maxa.append(max_a)
        # sum of squared of distance difference of updated and original verts
        constr.append(sum(dd)) #constr.append(sum(np.sqrt(dd)))
        smooth.append(averageAllDotProducts(nbrLB, nV))

        if mm % nn == 0: # true at every nn-th iter.
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


def smoothing_stdev_last(verts, faces, nbrLB, **kwargs):
    # smoothes a triangulated mesh e.g.the WN-interface
    # receives verts, faces, & Laplace-Beltrami neighborhood map (nbrLB) of verts
    # returns smoothed verts
    print('\nsmoothing iteration num.')
    verts_original = np.copy(verts)
    nV = verticesUnitNormals(verts, faces) # unit normals of verts
    # weight called to calc. min/max triangle area; wt unimportant here
    wt, max_A = MeanGaussianPrincipalCurvatures(verts, nV, nbrLB)

    # new verts, smoothing criteria, verts distance from originals, 
    # & min/max face area @ iterations 
    # averageAllDotProducts returns average for dot products of all 
    # neighbors' unit normals
    # smoother surface will have a average approaching unity
    VV, smooth, constr,  maxa = [], [], [], []
    smooth.append(np.std(wt)) # at original verts
    constr.append(np.float64(0))
    maxa.append(max_A)
    VV.append(verts)
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
        # @ uniso. diff. wieghts have feature/noise detection
        # by a thresholding variable (see TT @ wieght func.)
        # weights are multiplied by 0.1 to ensure not-to-fast
        # changes. This seems to be necessary in complex shapes.
        # new_vert = vert - 0.1*(weight)*(unit normal at vert)
        if method == 'aniso_diff':
            tune0, max_a = MeanGaussianPrincipalCurvatures(\
                                    VV[mm-1], nV, nbrLB, 'aniso_diff')
        else:
            tune0, max_a = MeanGaussianPrincipalCurvatures(\
                                    VV[mm-1], nV, nbrLB)
        tune = (nV.T*tune0).T
        verts_itr = VV[mm-1] - 0.1*tune
        # comparing verts_itr with the originals & correcting the jumped ones
        # constraint is not to have dislacement more than DD at every vert
        # if disp. is more than DD, vertex goes back to value @ previous iter.
        dd = sum(((verts_original - verts_itr)**2).T) # squared of distances
        nojump = dd >= DD**2
        verts_itr[:,0][nojump] = VV[mm-1][:,0][nojump]
        verts_itr[:,1][nojump] = VV[mm-1][:,1][nojump]
        verts_itr[:,2][nojump] = VV[mm-1][:,2][nojump]

        nV = verticesUnitNormals(verts_itr, faces) # update norms with new verts
        VV.append(verts_itr) # save new verts
        maxa.append(max_a)
        # sum of squared of distance difference of updated and original verts
        constr.append(sum(dd)) #constr.append(sum(np.sqrt(dd)))
        smooth.append(np.std(tune0))

        if mm % nn == 0: # true at every nn-th iter.
            # checks if iteration should be ended
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
                verts = VV[kk-1]
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
                print('\naverage for dot products of all neighbour unit', \
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
    del VV, tune, smooth, constr, maxa       
    return verts # smoothed verts


def smoothing_ball(r, *args):
    # tests the smoothing method by
    # generating and smoothing a ball of radius (r)
    rr=r+1
    z,y,x = np.ogrid[-rr:rr+1, -rr:rr+1, -rr:rr+1]
    # ball of radius r where the ball doesn't touch boundaries of the array
    ball = z**2+y**2+x**2 <= (rr-1)**2
    verts, faces, norms, vals = measure.marching_cubes_lewiner(ball)
    print('\nBall with radius {} has {} verts and {} faces.'\
            .format(r, len(verts), len(faces)))
    mlab.triangular_mesh(verts[:,0], verts[:,1], verts[:,2], faces)
    mlab.show()
    nbrLB = verticesLaplaceBeltramiNeighborhood(faces, len(verts))
    if 'aniso_diff' in args:
        verts2 = smoothing(verts, faces, nbrLB, method='aniso_diff')
    else:
        verts2 = smoothing(verts, faces, nbrLB)
    mlab.triangular_mesh(verts2[:,0], verts2[:,1], verts2[:,2], faces)
    mlab.show()


def smoothing_double_torus(*args):
    # source for equation: https://www.mathcurve.com/surfaces.gb/tore/tn.shtml
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
    nbrLB = verticesLaplaceBeltramiNeighborhood(faces, len(verts))
    if 'aniso_diff' in args:
        verts2 = smoothing(verts, faces, nbrLB, method='aniso_diff')
    else:
        verts2 = smoothing(verts, faces, nbrLB)
    mlab.triangular_mesh(verts2[:,0], verts2[:,1], verts2[:,2], faces)
    mlab.show()


def smoothing_tripple_torus(*args):
    # source for equation: https://www.mathcurve.com/surfaces.gb/tore/tn.shtml
    n = 100 # increase for finer result
    R = 1.05
    x = np.linspace(-R, R, n)
    y = np.linspace(-R, R, n)
    z = np.linspace(-0.8, 0.8, n)
    x, y, z = np.meshgrid(x,y, z)

    ee = 0.003  # a very small number
    # T is a vol. contained by a tripple torus
    T = z**2 - (ee - ((x**2 + y**2)**2 - x*(x**2 - 3*y**2))**2) <=0 
    ii,jj,kk = T.shape
    Torus = np.zeros(shape=(ii + 2, jj + 2, kk + 2))
    Torus[1 : ii + 1, 1 : jj + 1, 1 : kk + 1] = T
    verts, faces, norms, vals = measure.marching_cubes_lewiner(Torus)
    print(len(verts), len(faces))
    mlab.triangular_mesh(verts[:,0], verts[:,1], verts[:,2], faces)
    mlab.show()
    nbrLB = verticesLaplaceBeltramiNeighborhood(faces, len(verts))
    if 'aniso_diff' in args:
        verts2 = smoothing(verts, faces, nbrLB, method='aniso_diff')
    else:
        verts2 = smoothing(verts, faces, nbrLB)
    mlab.triangular_mesh(verts2[:,0], verts2[:,1], verts2[:,2], faces)
    mlab.show()


def main_inner(vertsA, facesA, vertsB, facesB, pixel_size):
    # pixel_size is in meter
    t0=time()
    print('Interface extraction...')
    # extracts verts/faces common in W & N meshes (WN interface)
    vertsAB, facesAB = meshA_meshB_common_surface(vertsA, facesA, vertsB, facesB)
    if vertsAB is not None:
        print('num. verts & faces @ interface: ', len(vertsAB), len(facesAB), '\n')
        # # constraining solid mesh to qq = 5 pixel(s) vicinity of WN interface 
        # # (used in contact angle measurement)
        # zmin, zmax = np.min(vertsAB[:,0]), np.max(vertsAB[:,0])
        # ymin, ymax = np.min(vertsAB[:,1]), np.max(vertsAB[:,1])
        # xmin, xmax = np.min(vertsAB[:,2]), np.max(vertsAB[:,2])
        # vertsSb, facesSb = bracket_mesh(vertsS, facesS, 5,\
        #                                  zmin, zmax, ymin, ymax, xmin, xmax)
        # # verts neighbourhood map for WN - required for smoothing
        t1=time()
        print('Constructing neighborhood map for interface...') 
        nbrAB = verticesLaplaceBeltramiNeighborhood_parallel(facesAB, vertsAB)
        # # isotropic smoothing, calc. surface area and mean curvature at WN interface
        t2=time()
        print('Isotropic smoothing of interface...')
        vertsAB2 = smoothing(vertsAB, facesAB, nbrAB, verts_constraint=1.7)     
        t3 = time()
        aAB = measure.mesh_surface_area(vertsAB2, facesAB)*(pixel_size**2)  # in squared meter
        nV = verticesUnitNormals(vertsAB2, facesAB)
        kH = MeanCurvatures(vertsAB2, nV, nbrAB)
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
