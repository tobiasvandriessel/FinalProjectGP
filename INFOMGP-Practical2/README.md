# Practical 2: Finite-Element Soft-Body Deformation with Constraints

##Handout date: 16/Mar/2018.

##Deadline: 27/Mar/2018 09:00AM.


The second practical is about simulating soft bodies with the finite-element method learned in class. In addition, the system will handle barrier, collision, and user-based distance constraints.


The objective of the practical are:

1. Implement the full finite-element dynamic equation integration, to create soft-body internal forces on a tetrahedral mesh.
</br>
2. Implement sequential velocity and position impulse-based system for constraint resolution.
</br>
3. Extend the framework with some chosen effects.  

This is the repository for the skeleton on which you will build your practical. Using CMake allows you to work and submit your code in all platforms. This is practically the same system as in the previous practical, in terms of compilation and setup.

##Scope

The practical runs in the same time loop (that can also be run step-by-step) as the previous practical. The objects are not limited to convex ones; every triangle mesh would do. The software automatically converts a surface OFF mesh into a tetrahedral mesh with [Tetgen](http://wias-berlin.de/software/index.jsp?id=TetGen&lang=1). There are no ambient forces but gravity in the basic version. However, a tetrahedral mesh can be provided by the scene files with the .MESH format.

A considerable difference from the previous practical is that the platform is no longer a mesh in the usual sense (although it appears as such)---it is treated as a barrier constraint, where objects cannot fall beyond the supporting plane. As a consequence, they cannot also fall down the eternal pit of damnation, and will float on air on that plane, as if the platform is infinite. 

The objects move in time steps by altering each vertex of the tet mesh, by solving the finite-element equation for movement as learnt in class (*Lecture 8*). For this, you will set up the mass, damping, and stiffness matrices in the beginning of time, and whenever the time step changes. Moreover, you will set up the Cholesky solver that factorizes these matrices to be ready for solutions in each time step. 

The user can draw distance constraints between vertices as they move. Those constraints would maintain the distance at the point of drawing, as if you put a rigid rod between them.

Within each time step, you will also iteratively solve for velocity and position correction using the methods learned in class (*Lectures 6 and 9*) to resolve constraints.

Note that every vertex has one velocity; this practical will not explicitly model rigid-body behavior with angular velocity in the basic setting.

##The Time Loop

The algorithm will perform the following within each time loop:

1. Integrate velocities and positions from external and internal forces (the finite-element stage).
</br>
2. Collect collision and other constraints if active.
</br>
3. Resolve velocities and then positions to valid ones according to the constraints.

##Finite-Element Integration

Finite-element integration consists of two main steps that you will implement:

###Matrix constrution

At the beginning of time, or any change to the time step, you have to construct the basic matrices $M$, $K$ and $D$, and consequently the left-hand side matrix $A=M+\Delta t D+\Delta t^2 K$. Those all have to be sparse matrices (of the data type ``Eigen::SparseMatrix<double>``). You have to fill them in using COO triplet format with `Eigen::Triplet<double>` as learnt in class. Then, you can see the Cholesky decomposition code preparing the solver. This is done in functon `createGlobalMatrices()`.

###Integration.

At each time step, you have to create the right-hand side from current positions and velocities, and solve the system to get the new velocities (and consequently positions). This is in the usual function `integrateVelocities()`. As this function only changes right-hand side, it should be quite cheap in time.

The scene file contains the necessary material parameters: Young's Modulus, Poisson's ratio, and mass density. You need to produce the proper masses and stiffness tensors from them. Damping parameters $\alpha$ and $\beta$ are given as inputs to the function, and hardcoded as $0.02$ in `main.cpp`.

##Resolving constraints

The constraints are resolved on a loop until all of them are satisfied. There are two similar loops: one that solves for impulses to correct velocities (*lecture 6*) and a subsequent one to fix positions (*lecture 9*). The loops run until a full streak of checking all constraints makes no change (zero impulses and position difference), or the max iterations limit has passed. In each iteration, a single constraint is satisfied. 


The practical is currently configured to encode three possible constraints:

1. **Barrier**: one coordinate of an object is bounded. The "natural" constraints are being always higher than the platform: $C_B=y - h \geq 0$.
<br />
2. **Collision**: two tetrahedra are penetrating in points $p_1$ and $p_2$, with contact normal $n$, similarly to the first practical. The constraint is then $C(p_1,p_2)=n^T\cdot (p_2-p_1) \geq 0$
<br />
3. **Distance**: Two vertices $v_1$ and $v_2$ have a prescribed distance $d_{12}$ so that the constraint is $C(v_1,v_2) = \left|v_1-v_2\right|-d_{12}=0$

###Constraints and gradients

A single constraint is usually only expressed as the function of the coordinates of two vertices (so 6 variables). You need to analytically compute the gradient of these functions, and code them in the constraint code (below explained how). Example: the distance constraint: $C(v_1,v_2) = \left|v_1-v_2\right|-d_{12}=0$ has the gradient:

$$\nabla C_e = \left( \frac{v_1-v_2}{\left| v_1-v_2 \right|}, -\frac{v_1-v_2}{\left| v_1-v_2 \right|} \right)$$

Where the gradient expressed in the 6 varibles of the three coordinates to $v_1$ and $v_2$ each.

The other constraints have mostly straightforward gradients. Note that the barrier constraint, for instance, only applies to a single coordinate in a vertex. This is made possible due to the indexing scheme we apply in this practical.

The code is arranged so that the constraint-resolution loop(s) are not aware of the type of constraint and how to resolve it; the entire code is within the `Constraint` class, and it gives back the necessary impulse and position difference outputs.

You will note that constraints can be either equality or inequality; while the resolution math is the same, inequality constraints would not receive impulses if they are already satisfied.

Each constraint will have an additional coefficient of restitution to allow for bouncing. That means that you need to multiply each impulse by $(1+CR)$ to get the proper boost. Note that this should only apply to inequality constraints (for instance collision or barrier), and velocity impulses.

###Extensions

The above will earn you $70\%$ of the grade. To get a full $100$, you must choose 2 of these 6 extension options, and augment the practical. Some will require minor adaptations to the GUI or the function structure which are easy to do. Each extension will earn you $15\%$, and the exact grading will commensurate with the difficulty. Note that this means that all extensions are equal in grade; if you take on a hard extension, it's your own challenge to complete it well.


1. Create extension and compression constraints - like distance constraints but with lower and upper bounds to the possible distance between two vertices, instead of a rigid single distance. **Level: easy**.
 </br>
 2. Allow to apply random user-prescribed deformations of objects. For instance, instantaenous "squeezing" of an object to see it deform back after a time. **Level: easy-intermediate**. 
 </br>
 3. Introduce perfectly rigid bodies that integrate like the first practical, just in tetrahedral-mesh mode, so with a single linear velocity and angular velocity. Note that their collisions should still be handled by the same collision detection and constraint loop as the soft bodies (just the integration should be different). **Level: easy-intermediate**.
 </br>
4. Implement the corotational elements method as learnt in class (*lecture 8*). That would require to use method thatdoes not rely on matrix factorization, like conjugate gradients. This is available through Eigen, but you'll have to figure out the details and proper initialization. **Level: intermediate-hard**.
</br>
5. Improve the slow-ish collision detection by introduction some cheap heirarchical bounding-volume method (for instance, group several tets under difference boxes). As an even harder challenge, allow for self-intersections between a body and itself, which are not currently covered. **Level: intermediate-hard**.
 </br>

You may invent your own extension as substitute to **one** in the list above, but it needs approval on the Lecturer's behalf **beforehand**.


##Installation

The installation follows the exact path as the previous practical. It is repeated for completeness.

The skeleton uses the following dependencies: [libigl](http://libigl.github.io/libigl/), and consequently [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page), for the representation and viewing of geometry, and [libccd](https://github.com/danfis/libccd) for collision detection (now between individual terahedra). Again, everything is bundled as either submodules, or just incorporated code within the environment, and you do not have to take care of any installation details. To get the library, use:

```bash
git clone --recursive https://github.com/avaxman/INFOMGP-Practical2.git
```

to compile the environment, go into the `practical2` folder and enter in a terminal (macOS/Linux):

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ../
make
```

In windows, you need to use [cmake-gui](https://cmake.org/runningcmake/). Pressing twice ``configure`` and then ``generate`` will generate a Visual Studio solution in which you can work. The active soution should be ``practical2_bin``. *Note*: it only seems to work in 64-bit mode. 32-bit mode might give alignment errors.

##Working with the repository

All the code you need to update is in the ``practical2`` folder. Please do not attempt to commit any changes to the repository. <span style="color:red">You may ONLY fork the repository for your convenience and work on it if you can somehow make the forked repository PRIVATE afterwards</span>. Solutions to the practical which are published online in any public manner will **disqualify** the students! submission will be done in the "classical" department style of submission servers, published separately.


##The coding environment and GUI

THe GUI is very similar to the one you used in the first practical, with these notable differences:

1. The platform is not considered as a mesh anymore in the scene, but only for visualization purposes; it is represented as geometry through barrier constraints given to the scene. It is also therefore never tetrahedralized.

2. By clicking one vertex and dragging the right mouse button, releasing it on another vertex, you can draw a distance connection between two vertices in the scene on the fly. The distance will snapshot the distance at the time instance of drawing.

Most of the action, and where you will have to write code, happens in `scene.h` and `constraint.h`. The functions in `scene.h` implement the time loop and the algorithmic steps, while `constraint.h` implements the `Constraint` class that evaluates and constraints and resolves them. 

The code you have to complete is always marked as:

```cpp
/***************
TODO
***************/
```

For the most part, the description of the functions will tell you what exactly you need to put in, according to what you need to implement as described above.

###Input

The program is loaded by giving two TXT files as arguments to the executable (three arguments in total; the first being the data folder again): one describing the scene, and another the user-prescribed distance constraints. It is similar to the format in the first practical, but with extensions. That means that there is no backward compatibility; you will have to update files you already made (a bit).

The format of the scene file is:

```
#num_objects
object1.off/mesh     density1   young1  poisson1     is_fixed1    COM1     o1
object2.off/mesh     density2   young2  poisson2     is_fixed2    COM2     o2
...
```

Where:

1. ``objectX.off/mesh`` - either an OFF file in the same manner as the first practical, or a tetahedral MESH file. OFF files will be automatically tetrahedralized upon loading, so you don't have to spend the effort creating tetrahedral meshes offline.
</br>
2. ``density`` - the uniform density of the object. 
</br>
2. ``Young`` - Young's modulus, used to create the stiffness tensor. Use very big values; look up values for general materials, which are usually expressed in Mega or Giga Pascals. The value you give here will be interpreted as Pascal.
</br>
3. ``poisson`` - Poisson's ratio, which describes the incompressibility of the object. Values should be between $[0,0.4999...]$---note that exact 0.5 will produce infinite stiffness that will kill your simulation by dividing by zero, so avoid it.
</br>
4. ``is_fixed`` - if the object should be immobile (fixed in space) or not.
</br>
5. ``COM`` - the initial position in space where the object would be translated to. That means, where the COM is at time $t=0$.
</br>
6. ``o`` - the initial orientation of the object, expressed as a quaternion that rotates the geometry to $o*object*inv(o)$ at time $t=0$.


The constraints file has the following format:

```
#num_constraints

m1 v1 m2 v2

```

Where it creates a distance constraints between vertex ``v1`` on mesh ``m1`` to vertex ``v2`` on mesh ``m2`` in their initial configuration (after COM+orientation transformations). For your convenience, there is an ``emptyconstraints.txt`` file, if you do not want to use constraints.


###Indexing

We use two indexing methods for vertices and coordinates:

1. **Local indexing**: the vertices in each mesh are kept in one big column vector of $3\left| V \right|$ coordinates in $x_1y_1z_1x_2y_2z_2x_3y_3z_3...$ format (dominant order by vertices and subdominant order by coordinate). In general, corresponding quantities like velocities, forces, and impulses will be encoded the same way. Quantities that are the same for all $xyz$ coordinates of a single vertex---like mass, Voronoi volumes, etc.---are encoded in a vector of $\left| V \right|$ with one quantity per vertex. The comments in the code will tell you which is which.
</br>
2. **Global indexing**: this is the indexing used by the scene class, which aggregates all vertex-coordinates in the scene. The index is a flat single column vector of **all** coordinates in the scene in the same format as the local indexing. As such, the individual coordinates of every mesh are a continuous segment within this global vector; the ``globalOffset`` variable in each mesh indicates where this segment begins. The same goes again for impulses and velocities---and inverse masses in this case! (unlike the mesh class which keeps one inverse mass quantity per vertex). That is because the scene class handles the constraints resolution, so this encoding is much more comfortable; the mass fitting for each coordinate. At any rate, check the comments attached to the variable definitions.

Two functions: ``global2mesh()`` and ``mesh2global()`` update the values back and forth between the scene and the individual meshes.


###Data structure

There are three main classes, `Scene`, `Mesh`, and `Constraint`. They are commented throughout, so their individual properties are understood. 


A single constraint is encoded as follows in the `Constraint` class:

```cpp
VectorXi globalIndices;         //|V_c| list of participating indices (out of global indices of the scene)
double currValue;               //The current value of the constraint
RowVectorXd currGradient;       //Current gradient of the constraint
MatrixXd invMassMatrix;         //M^{-1} matrix size |V_c| X |V_c| only over the participating  vertices
double refValue;                //Reference values to use in the constraint, when needed
VectorXd refVector;             //Reference vector when needed
ConstraintType constraintType;  //The type of the constraint, and will affect the value and the gradient. This SHOULD NOT change after initialization!
ConstraintEqualityType constraintEqualityType;  //whether the constraint is an equality or an inequality
```

where:

1. ``globalIndices`` is a list of the vertex-coordinate indices in the global scene index that participate in this constraints. This is typically very small (1--6 indices in total for the default constraints).

2. ``currValue`` is the scalar value of the constraint, and ``currGradient`` is the vector of partial derivative values, where the gradient size is the size of ``globalIndices``.

3. ``invMassMatrix`` is a diagonal matrix of all the inverse masses corresponding to coordinates. 

4. ``refValue`` and ``refVector`` are auxiliary quantities needed for the use of the constraints; for instance, the normal to a collision, or the distance to a distance constraint.


The constraint should compute resolution impulses and position differences in ``resolveVelocityConstraint`` and ``resolvePositionConstraint``. Note that all the elements here are dense; a single constraint solves a very small system (*lectures 6 and 9*). The impulses should be then fed back to the appropriate global indices in scene, per ``globalIndices``.


##Submission

The entire code of the practical has to be submitted in a zip file to the designated submission server that will be anounced. The deadline is **27/Mar/2018 09:00AM**. 

The practical must be done **in pairs**. Doing it alone requires a-priori permission. Any other combination (more than 2 people, or some fractional number) is not allowed. 

The practical will be checked during the lecture time of the deadline date. Every pair will have 10 minutes to shortly present their practical, and be tested by the lecturer with some fresh scene files. In addition, the lecturer will ask every person a short question that should be easy to answer if this person was fully involved in the exercise. We might need more than the allocated two hours for the lecture time; you will very soon receive a notification. The registration for time slots will appear soon.

<span style="color:red">**Very important:**</span> come to the submission with a compiled and working exercise on your own computers. If you cannot do so, tell the lecturer **in advance**. Do not put the command-line arguments in code, so that we waste time on compilation with every scene (is there any logic in doing so? it was common for some reason in the submissions).

##Frequently Asked Questions

Here are detailed answers to common questions. Please read through whenever you have a problem, since in most cases someone else would have had it as well.


<span style="color:blue">Q: </span> How do we "translate" the constraint formulas we have for each constraint to currGradient and currValue
<span style="color:blue">A:</span> currValue is a scalar with the value of the constraint. currGradient is an $n$ vector, if $n$ are the participating variables, and each entry $i$ of the vector is $\frac{\partial C}{\partial p_i}$, where $p_i$ is the participating variable. See lengthy discussion above. Remember that a position is 3 different $x,y,z$ variables.





#Good work!











