# dataset.py

# Utility function to generate unique colors

# Head Dataset Info
dataset_info = dict(
    dataset_name='head_xray',
    paper_info=dict(
        author='Marawan Elbatel',
        title='Head X-ray Landmark Detection',
        container='arXiv',
        year='2025',
        homepage='marwankefah.github.io',
    ),
    keypoint_info={
        0: dict(name='L0', id=0, color=[255, 0, 0], type='', swap=''),
        1: dict(name='L1', id=1, color=[255, 128, 0], type='', swap=''),
        2: dict(name='L2', id=2, color=[255, 255, 0], type='', swap=''),
        3: dict(name='L3', id=3, color=[128, 255, 0], type='', swap=''),
        4: dict(name='L4', id=4, color=[0, 255, 0], type='', swap=''),
        5: dict(name='L5', id=5, color=[0, 255, 128], type='', swap=''),
        6: dict(name='L6', id=6, color=[0, 255, 255], type='', swap=''),
        7: dict(name='L7', id=7, color=[0, 128, 255], type='', swap=''),
        8: dict(name='L8', id=8, color=[0, 0, 255], type='', swap=''),
        9: dict(name='L9', id=9, color=[128, 0, 255], type='', swap=''),
        10: dict(name='L10', id=10, color=[255, 0, 255], type='', swap=''),
        11: dict(name='L11', id=11, color=[255, 0, 128], type='', swap=''),
        12: dict(name='L12', id=12, color=[255, 0, 64], type='', swap=''),
        13: dict(name='L13', id=13, color=[255, 64, 0], type='', swap=''),
        14: dict(name='L14', id=14, color=[255, 128, 64], type='', swap=''),
        15: dict(name='L15', id=15, color=[128, 128, 128], type='', swap=''),
        16: dict(name='L16', id=16, color=[64, 64, 128], type='', swap=''),
        17: dict(name='L17', id=17, color=[128, 64, 128], type='', swap=''),
        18: dict(name='L18', id=18, color=[64, 128, 128], type='', swap=''),
    },
    skeleton_info={},  # Optional: Define relationships if needed
    joint_weights=[1.] * 19,
    sigmas=[0.00463] * 19
)


