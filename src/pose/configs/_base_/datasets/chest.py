# dataset.py

# Utility function to generate unique colors

# Chest Dataset Info
dataset_info = dict(
    dataset_name='chest_xray',
    paper_info=dict(
        author='Marawan Elbatel',
        title='Chest X-ray Landmark Detection',
        container='arXiv',
        year='2025',
        homepage='marwankefah.github.io',
    ),
    keypoint_info={
        0: dict(name='K0', id=0, color=[255, 0, 0], type='', swap=''),
        1: dict(name='K1', id=1, color=[255, 128, 0], type='', swap=''),
        2: dict(name='K2', id=2, color=[255, 255, 0], type='', swap=''),
        3: dict(name='K3', id=3, color=[128, 255, 0], type='', swap=''),
        4: dict(name='K4', id=4, color=[0, 255, 0], type='', swap=''),
        5: dict(name='K5', id=5, color=[0, 255, 128], type='', swap=''),
    },
    skeleton_info={},  # Optional: Define relationships if needed
    joint_weights=[1.] * 6,
    sigmas=[0.00356] * 6
)
