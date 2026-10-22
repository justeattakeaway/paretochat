# Source 

Instances are taken from [vOptLib GitHub repository](https://github.com/vOptSolver/vOptLib), which is licensed under the MIT License (see [LICENSE](LICENSE) in this folder; Copyright (c) 2017: Xavier Gandibleux, and contributors). This folder is not covered by the Apache 2.0 license of the rest of this repository.

The input files are in the `instances` folder and the generated nondominated sets of those instaces are placed under `nondominated_sets` folder.


## Format
The format is originally from the [OR library](http://people.brunel.ac.uk/~mastjjb/jeb/orlib/sppinfo.html) extended for handling multiple objectives.

The format of all of these data files is:

    number of rows (m), number of columns (n)   
    for each column j (j=1,...,n) in turn:    
        column cost for objective 1 c1(j)
        column cost for objective 2 c2(j)
        number of rows covered by j
        list of the rows covered by j


## Instances

The data files are:
+    biodidactic
+    biosppaa01, biosppaa02, biosppaa03, biosppaa05
+    biosppnw01, biosppnw02, ... , biosppnw43
+    biorail4284

For these instances, the coefficients of the second objective are a random permutation of the coefficients of the first objective.   


The set of non dominated points have been computed with [vOptGeneric](https://github.com/vOptSolver/vOptGeneric.jl) for several instances.

The format of all of these files is:

    CPU time   
    number of non-dominated points (K)
    for each non-dominated point (k=1,...,K) in turn:
        value on the 1st objective, z1(k)
        value on the 2nd objective, z2(k)

The nondominated set of an instance named as `instance1` is available in the file named as `Y_N_instance1.txt`. Both instance and nondominated set files are `txt` files.


## Parser

A parser is available in the [folder "examples"](https://github.com/vOptSolver/vOptGeneric.jl/tree/master/examples) of vOptGeneric (function `loadInstance2SPA` in the file `setPartitionning.jl`).


## Reference

Instances referenced in:

> Xavier Gandibleux, Guillaume Gasnier and Saïd Hanafi.
  A primal heuristic to compute an upper bound set for multi-objective 0-1 linear optimisation problems.
  MODeM '21: 1st Multi-Objective Decision Making Workshop, July 14-16, 2021.
  [http://modem2021.cs.nuigalway.ie/](http://modem2021.cs.nuigalway.ie/)
