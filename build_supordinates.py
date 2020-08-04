"""
Construct csv of superorinate groups
that like what I have for the imageNet categories.
"""
import pandas as pd
from robustness.tools.imagenet_helpers import ImageNetHierarchy

from TRAIN.utils.data_utils import load_classes

in_path = '/fast-data21/datasets/ILSVRC/2012/clsloc/'
in_info_path = 'data_local'
in_hier = ImageNetHierarchy(in_path, in_info_path)

# ct = 0
# for cnt, (wnid, ndesc_in, ndesc_total) in enumerate(in_hier.wnid_sorted):
#     print(f"WordNet ID: {wnid}, Name: {in_hier.wnid_to_name[wnid]}, #ImageNet descendants: {ndesc_in}")
#     if ndesc_in == 0:
#         ct += 1
# print(ct)

ancestor_wnids = ['n01661091', 'n01503061', 'n02512053', 'n01627424', 'n02469914']

for ancestor_wnid in ancestor_wnids:
    ancestor_name = in_hier.wnid_to_name[ancestor_wnid]

    want_wnids = []
    want_indices = []
    want_names = []

    # full ImagetNet
    wnids, indices, names = load_classes(1000, 'ranked')
    wnid2index = dict(zip(wnids, indices))
    wnid2name = dict(zip(wnids, names))

    for cnt, wnid in enumerate(in_hier.tree[ancestor_wnid].descendants_all):
        if wnid in wnids:
            want_wnids.append(wnid)
            want_names.append(wnid2name[wnid])
            want_indices.append(wnid2index[wnid])

    df = pd.DataFrame(
        {"idx": want_indices,
        "wnid": want_wnids,
        "description": want_names})
    df.to_csv(f'groupings-csv/{ancestor_name}.csv', index=False)

