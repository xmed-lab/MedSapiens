# dataset.py

# Hand Dataset Info
dataset_info = dict(
    dataset_name='x-ray',
    paper_info=dict(
        author='Marawan Elbatel',
        title='Combined X-ray Landmark Detection',
        container='arXiv',
        year='2025',
        homepage='marwankefah.github.io',
    ),
    keypoint_info={
        i: dict(name=name, id=i, color=color, type='', swap='')
        for i, (name, color) in enumerate(zip(
            [
                # Keypoints names
                "Head_Point1", "Head_Point2", "Head_Point3", "Head_Point4", "Head_Point5",
                "Head_Point6", "Head_Point7", "Head_Point8", "Head_Point9", "Head_Point10",
                "Head_Point11", "Head_Point12", "Head_Point13", "Head_Point14", "Head_Point15",
                "Head_Point16", "Head_Point17", "Head_Point18", "Head_Point19",
                "Chest_Point1", "Chest_Point2", "Chest_Point3", "Chest_Point4", "Chest_Point5", "Chest_Point6",
                "Hand_Point1", "Hand_Point2", "Hand_Point3", "Hand_Point4", "Hand_Point5", "Hand_Point6",
                "Hand_Point7", "Hand_Point8", "Hand_Point9", "Hand_Point10", "Hand_Point11", "Hand_Point12",
                "Hand_Point13", "Hand_Point14", "Hand_Point15", "Hand_Point16", "Hand_Point17", "Hand_Point18",
                "Hand_Point19", "Hand_Point20", "Hand_Point21", "Hand_Point22", "Hand_Point23", "Hand_Point24",
                "Hand_Point25", "Hand_Point26", "Hand_Point27", "Hand_Point28", "Hand_Point29", "Hand_Point30",
                "Hand_Point31", "Hand_Point32", "Hand_Point33", "Hand_Point34", "Hand_Point35", "Hand_Point36",
                "Hand_Point37", "Leg_Point1", "Leg_Point2", "Leg_Point3", "Leg_Point4", "Leg_Point5",
                "Leg_Point6", "Leg_Point7", "Leg_Point8", "Leg_Point9", "Leg_Point10", "Leg_Point11",
                "Leg_Point12", "Leg_Point13", "Leg_Point14", "Leg_Point15", "Leg_Point16", "Leg_Point17",
                "Leg_Point18", "Leg_Point19", "Leg_Point20", "Leg_Point21", "Leg_Point22", "Leg_Point23",
                "Leg_Point24", "Leg_Point25", "Leg_Point26"
            ],
            [
                # Distinct colors (RGB)
                [51, 153, 255], [255, 128, 0], [0, 255, 0], [255, 255, 0], [255, 51, 51],
                [102, 178, 255], [153, 51, 255], [255, 102, 178], [128, 0, 255], [0, 255, 128],
                [255, 204, 0], [0, 153, 255], [255, 153, 204], [255, 102, 102], [204, 255, 51],
                [0, 102, 204], [153, 255, 153], [102, 51, 255], [255, 255, 153],
                [255, 204, 102], [102, 255, 255], [178, 102, 255], [255, 255, 204], [204, 102, 0],
                [51, 255, 204], [0, 204, 51], [153, 0, 255], [255, 153, 51], [102, 153, 255],
                [153, 255, 102], [204, 51, 51], [255, 102, 255], [255, 153, 255], [102, 255, 178],
                [51, 153, 102], [153, 255, 204], [255, 204, 153], [102, 102, 255], [204, 255, 153],
                [255, 102, 0], [255, 255, 51], [102, 204, 255], [153, 51, 102], [178, 255, 102],
                [51, 255, 102], [255, 51, 204], [153, 102, 255], [0, 255, 204], [255, 153, 0],
                [51, 204, 102], [153, 204, 255], [102, 51, 153], [204, 153, 255], [255, 255, 102],
                [255, 102, 102], [0, 255, 153], [102, 153, 204], [178, 255, 204], [51, 102, 204],
                [153, 102, 204], [255, 255, 255], [255, 204, 51], [102, 204, 102], [51, 102, 255],
                [102, 255, 51], [255, 102, 153], [51, 153, 51], [153, 204, 102], [204, 51, 153],
                [51, 51, 255], [255, 204, 204], [255, 255, 204], [255, 102, 51], [153, 153, 102],
                [51, 255, 255], [102, 153, 255], [255, 204, 255], [255, 153, 153], [153, 51, 153],
                [102, 102, 102], [255, 153, 51], [255, 102, 255], [0, 255, 51], [255, 204, 0],
                [255, 20, 0],[20, 204, 0],[255, 204, 20],[50, 20, 0]
            ]
        ))
    },
    skeleton_info={},  # Define relationships if needed
    joint_weights=[1.] * 88,
    sigmas=[0.00463] * 19 +  # Head points
           [0.00356] * 6 +  # Chest points
           [0.00541] * 37 +  # Hand points
           [0.00237] * 26  # Leg points

)
