"""P2 split protocol (80/10/10), shared by train_v2.py and eval_ckpt.py.

The outer 80/20 boundary is identical to the backbone's public protocol
(same random_state = dataseed), so the training set is byte-identical to a
P1 run of the same seed. The held-out 20% is halved with a derived seed
into a validation half (used for epoch selection and training-time logs)
and a test half (never touched during training; reported post hoc).
"""
from sklearn.model_selection import train_test_split

_TEST_SEED_OFFSET = 31337


def split_p2(img_ids, dataseed):
    train_ids, hold_ids = train_test_split(img_ids, test_size=0.2, random_state=dataseed)
    val_ids, test_ids = train_test_split(sorted(hold_ids), test_size=0.5,
                                         random_state=dataseed + _TEST_SEED_OFFSET)
    return sorted(train_ids), sorted(val_ids), sorted(test_ids)
