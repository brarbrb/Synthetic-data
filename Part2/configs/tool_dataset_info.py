dataset_info = dict(
    dataset_name='tools_coco_like',
    paper_info=dict(),  # optional
    keypoint_info={
        0: dict(name='bb_tl', id=0, color=[255,0,0], type='bb', swap=''),
        1: dict(name='bb_tr', id=1, color=[0,255,0], type='bb', swap=''),
        2: dict(name='bb_br', id=2, color=[0,0,255], type='bb', swap=''),
        3: dict(name='bb_bl', id=3, color=[255,255,0], type='bb', swap=''),
        4: dict(name='bb_center', id=4, color=[255,0,255], type='bb', swap=''),
    },
    skeleton_info={
        0: dict(link=('bb_tl','bb_tr'), id=0, color=[255,128,0]),
        1: dict(link=('bb_tr','bb_br'), id=1, color=[255,128,0]),
        2: dict(link=('bb_br','bb_bl'), id=2, color=[255,128,0]),
        3: dict(link=('bb_bl','bb_tl'), id=3, color=[255,128,0]),
        4: dict(link=('bb_tl','bb_center'), id=4, color=[128,255,0]),
        5: dict(link=('bb_tr','bb_center'), id=5, color=[128,255,0]),
        6: dict(link=('bb_br','bb_center'), id=6, color=[128,255,0]),
        7: dict(link=('bb_bl','bb_center'), id=7, color=[128,255,0]),
    },
    # OKS sigmas per keypoint (tune if needed; same for all is OK to start)
    joint_weights=[1., 1., 1., 1., 1.],
    sigmas=[0.05, 0.05, 0.05, 0.05, 0.05],
)
