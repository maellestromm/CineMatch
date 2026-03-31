# Results from evaluations 
models = ["SVD-50", "Item-KNN", "Content-KNN", "Deep-AutoRec", "User-KNN", "Meta"]

weight_results = [
    {# 1.
        "weights": {
            "SVD-50": 0.4532,
            "User-KNN": 0.0012,
            "Item-KNN": 0.3178,
            "Deep-AutoRec": 0.1321,
            "Content-KNN": 0.0957
        },
        "hit": 0.7817,
        "prec": 0.2683
    },
    { # 2.
        "weights": {
            "SVD-50": 0.3016,
            "User-KNN": 0.1381,
            "Item-KNN": 0.2511,
            "Deep-AutoRec": 0.1411,
            "Content-KNN": 0.1682
        },
        "hit": 0.7664,
        "prec": 0.2557
    },
    { # 3.
        "weights": {
            "SVD-50": 0.3065,
            "User-KNN": 0.0389,
            "Item-KNN": 0.1877,
            "Deep-AutoRec": 0.0837,
            "Content-KNN": 0.3832
        },
        "hit": 0.7689,
        "prec": 0.2548
    },
    { # 4.
        "weights": {
            "SVD-50": 0.3145,
            "User-KNN": 0.0232,
            "Item-KNN": 0.2729,
            "Deep-AutoRec": 0.0671,
            "Content-KNN": 0.3222
        },
        "hit": 0.7642,
        "prec": 0.2515
    },
    { # 5.
        "weights": {
            "SVD-50": 0.1771,
            "User-KNN": 0.3816,
            "Item-KNN": 0.3716,
            "Deep-AutoRec": 0.0115,
            "Content-KNN": 0.0582
        },
        "hit": 0.7882,
        "prec": 0.2491
    },
    { # 6.
        "weights": {
            "SVD-50": 0.3694,
            "User-KNN": 0.1891,
            "Item-KNN": 0.0555,
            "Deep-AutoRec": 0.2280,
            "Content-KNN": 0.1581
        },
        "hit": 0.7402,
        "prec": 0.2450
    },
    { # 7.
        "weights": {
            "SVD-50": 0.1974,
            "User-KNN": 0.1553,
            "Item-KNN": 0.2144,
            "Deep-AutoRec": 0.2571,
            "Content-KNN": 0.1758
        },
        "hit": 0.7511,
        "prec": 0.2437
    },
    { # 8.
        "weights": {
            "SVD-50": 0.3467,
            "User-KNN": 0.0419,
            "Item-KNN": 0.1384,
            "Deep-AutoRec": 0.2475,
            "Content-KNN": 0.2254
        },
        "hit": 0.7424,
        "prec": 0.2426
    },
    { # 9.
        "weights": {
            "SVD-50": 0.1772,
            "User-KNN": 0.1910,
            "Item-KNN": 0.2369,
            "Deep-AutoRec": 0.1357,
            "Content-KNN": 0.2594
        },
        "hit": 0.7533,
        "prec": 0.2421
    },
    { # 10.
        "weights": {
            "SVD-50": 0.1682,
            "User-KNN": 0.3882,
            "Item-KNN": 0.2979,
            "Deep-AutoRec": 0.1163,
            "Content-KNN": 0.0295
        },
        "hit": 0.7445,
        "prec": 0.2389
    }
]

confusion_results = [
    {
        "User-KNN":{
            "TP":117,
            "FP":4963,
            "FN":24343,
            "Hit-Rate": 0.7336
        },
        "Deep-AutoRec":{
            "TP":614,
            "FP":5466,
            "FN":24846,
            "Hit-Rate": 0.5033
        },
        "Item-KNN":{
            "TP":1186,
            "FP":4894,
            "FN":24274,
            "Hit-Rate": 0.7336
        },
        "SVD-50":{
            "TP":1607,
            "FP":4473,
            "FN":23853,
            "Hit-Rate": 0.7582
        },
        "Content-KNN":{
            "TP":196,
            "FP":5884,
            "FN":25264,
            "Hit-Rate": 0.2089
        },
        "Meta":{
            "TP":1649,
            "FP":4431,
            "FN":23811,
            "Hit-Rate": 0.7829
        }
    }
]