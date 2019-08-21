# curvatures-smoothing-triangle-mesh

## Curvature computation & smoothing for triangular mesh data by isotropic/anisotropic diffusion 
This program is for smoothing triangulated mesh surfaces by calculation of mean curvatures. Smoothing can be performed by either isotropic or anisotropic diffusion. The latter has heavier computation where Gaussian and principal curvatures are also calculated in addition to the mean curvatures. The anisotropic diffusion smoothing is capable of feature/noise recognition. The isotropic diffusion suits smoothing of for instance fluid-fluid interfaces (in two fluid flow) whereas the anisotropic one respects the natural roughness of a surface; and it suits smoothing of for instance solid surfaces with natural roughness. The meshes are smoothed using contangent discretization of the Laplace-Beltrami operator. For details of the smoothing method, check

["Discrete Differential-Geometry Operators for Triangulated 2-Manifolds"](http://www.geometry.caltech.edu/pubs/DMSB_III.pdf)
by Meyer, Desbrun, Schroderl, Barr, 2003, Springer
"Visualization and Mathematics III" book, pp 35-57.

In case of fluid flow in porous media, after smoothing fluid-fluid interfaces, the mean curvatures at each interface is calculated. This can be used to find the local capillary pressure in two fluid flow in porous material.

Further, the code can be used in general for smoothing any triangular meshes, finding intersection of two arbitrary meshes, creating neighborhood map of mesh vertices, detection of mesh borders, etc. 


## Descriptions

### 1)
The .raw/.mhd images used in this work can be loaded as numpy arrays
either using simpleITK library (See main()), or simply by simpleitk plugin of skimage library after simpleitk library is installed.
The .raw/.mhd images we use have been published by [Schlüter et al. in 2016](https://doi.org/10.1002/2016WR019815) and [2017](https://doi.org/10.1002/2015WR018254).
In the images, pixel value of 0 is nonwetting fluid (an oil-based fluid), 1 is wetting fluid (salt water) and 2 is solid (sintered glass spheres) plus the boundary.

### 2)
Smoothing of fluid-fluid interfaces should be done locally (minimization of Helmholtz free energy) i.e. all the interfaces in an image should be identified and extracted. Therefore, the wetting and nonwetting droplets are labeled separately using ndimage.measurements.label(img_phaseA) and ndimage.measurements.label(img_phaseB) from scipy library. Afterwards, function

```
labeledVolSetA_labeledVolSetB_neighbor_blobs(lbld_W, lbld_N)
```

takes the two images and identifies which wetting droplet (label) is neighbor with which nonwetting droplet (label).

### 3)
The next step is to create mesh on the wetting and nonwetting blobs where we want to extract wetting-nonwetting (WN) interface. Marching cube algorithm for laying triangulated mesh on volumes accessible in measure.marching_cubes_lewiner(wetting_blob) in skimage library used to do this task.

```
vertsW, facesW, normsW, valsW = measure.marching_cubes_lewiner(wetting_blob)
vertsN, facesN, normsN, valsN = measure.marching_cubes_lewiner(nonwetting_blob)
```
verts (vertices) are (z,y,x) coordinates; faces represent triangles. Each face contains three indices of verts [v0,v1,v2]. Norms (normals) are the normal vectors of verts; and vals (values) are taken from the pixel value of the original image.

### 4)
For the neighbor blobs, the WN interface or the common faces of the triangulated surfaces of wetting and nonwetting fluids can be extracted using the function,

```
meshA_meshB_common_surface(vertsA, facesA, vertsB, facesB)
```

The function returns verts and faces of the common interface. The function is optimized for different sizes of the two meshes for which the common interface needs to be extracted. Different cases of large-large, small-large and small-small meshes are designed. The large-large case is parallelized using concurrent.futures.ThreadPoolExecutor() in Python. Detection of which case the function is going to execute is automatically performed via nested functions. 

### 5)
Smoothing of WN interface requires identification of the verts that are neighbor on the mesh, because in smoothing, we eventually need to maximize the dot product of normal vectors of all neighbor verts in order to smooth the surfaces.
Function

```   
verticesLaplaceBeltramiNeighborhood_parallel(faces, verts)
```

receives faces and verts on a mesh and returns the neighborhood map in a structure that suits the Laplace-Beltrami smoothing. Creation of this structure requires extra time when constructing the neighborhood map, but it pays off later when curvatures are calculated in smoothing iterations, because regular loops can be avoided and the vectorization nature of the numpy calculations can be benefited. The construction of neighborhood map is sequential for small meshes and parallel for large meshes.
Sequential function

```
verticesLaplaceBeltramiNeighborhood(faces, numVerts)
```
is nested in the parallel one, too.

### 6)
Smoothing requires the calculation of normal vectors at faces (triangles) and verts and it also requires the normalization of vectors (length = 1). Small and effective functions for these purposes are created using numpy.

``` 
facesUnitNormals(verts, faces) 	  # normals of faces
verticesUnitNormals(verts, faces) # normals of verts
unitVector(vec)					  # normalization of vectors (length = 1)
```

### 7)
The function

```
smoothing(verts, faces, nbrLB, **kwargs)
```

receives verts, faces and the neighborhood map (nbrLB). It changes the positions of verts along their normal vectors to maximize their dot product. The amount of change is determined by the weight function (mean curvature) in isotropic smoothing, and slightly complicated in the anisotropic smoothing.

```
# new_vert = vert - 0.1*(weight)*(unit normal at vert)  
```

Fraction of 0.1 is used to guarantee the quality of smoothing in the natural surface/interfaces which often have a complicated geometry.

The weight function (curvatures) are calculated in every iteration using function 

```
MeanGaussianPrincipalCurvatures(verts, nV, nbrLB, *args)
```

Smoothing and calculation of curvature/weights are done by isotropic diffusion by default. If anisotropic method is required, keyword argument should be passed to the smoothing function. Smoothing function passes the argument to the curvature function.
```
# method='aniso_diff' 
new_verts = smoothing(verts, faces, nbrLB, method='aniso_diff')
```

The amount of change of verts from their originals is by default constrained to 1.7 pixel (longest diagonal of a 1x1x1 voxel, given 1 pixel uncertainty in imaging), unless another constraint is given to the smoothing function using

```
# verts_constraint=1.4
new_verts = smoothing(verts, faces, nbrLB, verts_constraint=1.4 )
```

Smoothing the points in space without a constraint will likely end up as a flat plane for a surface with boundaries. In a water-tight mesh (no boundary), for example in a sphere, the points may jump indefinitely. Smoothing function checks all the updated verts (vertices) with their original positions, if a vert has changed more than the constraint, it will be taken back to its original position. 

The other smoothing function

```
smoothing_stdev(verts, faces, nbrLB, **kwargs)
```

is the same as the regular one, with only one difference. Instead of maximizing the average of all dot products, it aims for the minimization of standard deviation of the mean curvatures of all vertices. The two functions lead to the same or very similar results.

I suggest the reader checks the code block below, and runs smoothing simpler geometric surfaces(sphere, torus...) using both isotropic and anisotropic methods, different constraints, etc. in order to become better familiar with the code.

```
########### test smoothing a ball/torus ############
smoothing_ball(5, 'aniso_diff') # smooths  a ball (r=5) by anisotropic diff.
smoothing_ball(8)               # smooths by isotropic diff.
smoothing_triple_torus()       # smooths triple torus by isotropic diff.
smoothing_double_torus('aniso_diff') # smooths double torus by anisotropic diff.
smoothing_double_torus()        # smooths double torus by isotropic diff.
```
### 8)
Parallel computation (multi-threading) has been used in a number of functions:

```
verticesLaplaceBeltramiNeighborhood_parallel(faces, verts)
# The two func. below are nested @ meshA_meshB_common_surface(vertsA, facesA, vertsB, facesB)
maskA_advanced_parallel(vertsA, facesA, vertsB, facesB)
intersectAB_maskA_advanced_parallel(vertsA, facesA, vertsB, facesB) 
```
These functions are written to handle large mesh data by splitting them into smaller subsamples and running a sequential function simultaneously on several subsamples. ThreadPoolExecutor() module of concurrent.futures library has been utilized for this task. The tasks are created almost immediately and stored in a line on the memory. If the memory is limited or you wish to keep a part of the memory free for other purposes, use keyword "max_workers = n" to limit number of the created tasks to a maximum of "n". This could be helpful for instance when you are running the code for several sets of images simultaneously and need to limit the memory consumption of individual processes. For further reading about concurrent.futures [click here](https://docs.python.org/3/library/concurrent.futures.html). 

```
# up to 60 tasks are created almost immediately
with concurrent.futures.ThreadPoolExecutor() as executor:
# only 12 tasks are created 
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
```
### 9)
If the intention is estimation of contact angle of the fluid-fluid interface with the solid surface, isotropic and anisotropic methods can be implemented to smooth fluid-fluid interfaces and solid surfaces, respectively. Contact angle computation requires extraction of the three-phase contact line by functions

```
meshA_meshB_meshC_common_line(vertsA, facesA, vertsB, facesB, vertsC, facesC)
meshAB_meshC_common_line(nbrLB_AB, vertsAB, vertsC, facesC)
```
The two functions are under development.


### NOTE: 
The main code is in the file curvatures_smoothing.py. All the written functions (old and recent) are archived in file all_func.py.



```
#smoothing #triangular #mesh #Laplace #Beltrami
#differential # geometry #cotangent #discretization
#mean #Gaussian #principal #curvature
#torus #double_torus #triple_torus #sphere #ball
#manifold #visualization #neighborhood #map
#vertex #vertices #edge #face #normal #vector
#Young-Laplace #capillary #pressure 
#X-ray #tomography #image #fluid #flow #porous #material
#Helmholtz #free #energy #state #variable

```