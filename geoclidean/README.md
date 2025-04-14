Geoclidean-Elements includes 17 concepts and 34 tasks of Close and Far, while Geoclidean-Constraints includes 20 concepts and 40 tasks. The data is structured as following:

```
geoclidean
│
└───elements -> the Geoclidean-Elements dataset
│   │
│   └───concept_name
│       │   concept.txt
│       │   close_concept.txt
│       │   far_concept.txt
│       │   train
│       │   │   1_fin.png -> rendered from concept.txt
│       │   │   2_fin.png
│       │   │   3_fin.png
│       │   │   4_fin.png
│       │   │   5_fin.png
│       │   test
│       │   │   in_1_fin.png -> rendered from concept.txt
│       │   │   in_2_fin.png
│       │   │   in_3_fin.png
│       │   │   in_4_fin.png
│       │   │   in_5_fin.png
│       │   │   out_close_1_fin.png -> rendered from close_concept.txt
│       │   │   out_close_2_fin.png
│       │   │   out_close_3_fin.png
│       │   │   out_close_4_fin.png
│       │   │   out_close_5_fin.png
│       │   │   out_far_1_fin.png -> rendered from far_concept.txt
│       │   │   out_far_2_fin.png
│       │   │   out_far_3_fin.png
│       │   │   out_far_4_fin.png
│       │   │   out_far_5_fin.png
│       ...
│
└───constraints -> the Geoclidean-Constraints dataset
    │   ...

```

For the full framework and additional details, see: https://github.com/joyhsu0504/geoclidean_framework.
