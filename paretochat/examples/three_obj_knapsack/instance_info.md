## Instance Info

- This instance is taken from the GitHub repository of [Gonçalo Lopes](https://scholar.google.com/citations?user=BxylAt4AAAAJ&hl=en): [Link to the instance](https://github.com/gaplopes/mobkp-instances/blob/main/instances/random/3D/100_1.in)
- I separated the input from the generated nondominated points.
- In instance.txt:
    - First value: The number of items
    - Second value: The number of objectives
    - Third value: The knapsack capacity
    - The following rows:
        - (weight, obj_1_coeff, obj_2_coeff, obj_3_coeff)
- In nd_points.csv:
    - Each row is a nondominated point vector
        - (obj_1_value, obj_2_value, obj_3_value)