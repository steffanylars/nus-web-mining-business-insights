"""Step 4 inference must score every split without refitting or leaking labels."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import step4_baselines_lgbm as step4


class CompleteStep4(unittest.TestCase):
    def test_svd_scores_known_items_and_zeroes_unknowns_across_blocks(self):
        self.assertTrue(hasattr(step4, 'score_svd'), 'Missing reusable SVD inference for train and validation')
        frame = pd.DataFrame({'session_id': [0, 1, 0], 'cand_wid': ['a', 'b', 'new']})
        users = np.array([[2., 3.], [4., 5.]], dtype=np.float32)
        factors = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
        before = factors.copy()
        scores = step4.score_svd(frame, {'a': 0, 'b': 1}, users, factors, block_size=2)
        np.testing.assert_array_equal(scores, [8., 32., 0.])
        np.testing.assert_array_equal(factors, before)
        shuffled = frame.iloc[[2, 0, 1]]
        np.testing.assert_array_equal(step4.score_svd(shuffled, {'a': 0, 'b': 1}, users, factors), [0., 8., 32.])

    def test_reference_scores_ignore_labels_and_preserve_candidate_identity(self):
        self.assertTrue(hasattr(step4, 'reference_scores'), 'Missing shared reference scoring for all splits')
        frame = pd.DataFrame({'session_id': [0, 0], 'cand_wid': ['a', 'b'],
            'label': [0, 3], 'ov_name': [1., 0.], 'ov_cat': [2., 0.],
            'ov_brand': [4., 0.], 'jacc_name': [.25, 0.], 'pop_wid': [5., 2.]})
        scored = step4.reference_scores(frame)
        np.testing.assert_array_equal(scored.content_score, [5.25, 0.])
        np.testing.assert_array_equal(scored.pop_wid, [5., 2.])
        without_labels = step4.reference_scores(frame.drop(columns='label').iloc[::-1])
        np.testing.assert_array_equal(scored.random_score, without_labels.random_score.iloc[::-1])
        self.assertNotIn('random_score', frame)


if __name__ == '__main__':
    unittest.main()
