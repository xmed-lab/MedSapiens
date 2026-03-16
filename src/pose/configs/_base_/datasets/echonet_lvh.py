dataset_info = dict(
    dataset_name='echo_lvh_measurements',
    paper_info=dict(
        author='Marawan Elbatel',
        title='Left Ventricular Measurement Keypoint Detection in Echocardiography',
        container='(to appear)',
        year='2025',
        homepage='https://marwankefah.github.io/',
    ),
    keypoint_info={
        0: dict(
            name='ivs_endocardial_top',          # top IVS point
            id=0,
            color=[255, 0, 0],                   # red
            type='structure',
            swap=''),
        1: dict(
            name='ivs_lvid_junction',           # IVS–LVID junction
            id=1,
            color=[0, 255, 0],                   # green
            type='structure',
            swap=''),
        2: dict(
            name='lvid_lvpw_junction',          # LVID–LVPW junction
            id=2,
            color=[255, 255, 0],                 # yellow
            type='structure',
            swap=''),
        3: dict(
            name='lvpw_endocardial_bottom',     # bottom LVPW point
            id=3,
            color=[0, 255, 255],                 # cyan
            type='structure',
            swap=''),
    },
    skeleton_info={
        0: dict(
            link=('ivs_endocardial_top', 'ivs_lvid_junction'),
            id=0,
            color=[255, 0, 0]),                  # IVSd segment
        1: dict(
            link=('ivs_lvid_junction', 'lvid_lvpw_junction'),
            id=1,
            color=[0, 255, 0]),                  # LVIDd segment
        2: dict(
            link=('lvid_lvpw_junction', 'lvpw_endocardial_bottom'),
            id=2,
            color=[255, 255, 0]),                # LVPWd segment
    },
    joint_weights=[1.] * 4,
    sigmas=[0.01] * 4,
)
