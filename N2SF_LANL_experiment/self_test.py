"""Self-test for metric math and schema checks; does not need LANL files or AI models."""
import math
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

# Expected Llama-3 confusion matrix from the old manuscript; this verifies only
# arithmetic, not the provenance of those four counts.
y_true = np.array([0]*20_000 + [1]*1_000)
y_pred = np.array([0]*20_000 + [0]*119 + [1]*881)
cm = confusion_matrix(y_true, y_pred, labels=[0,1])
assert tuple(cm.ravel()) == (20_000, 0, 119, 881)
assert math.isclose(precision_score(y_true, y_pred), 1.0, abs_tol=1e-12)
assert math.isclose(recall_score(y_true, y_pred), 0.881, abs_tol=1e-12)
assert math.isclose(f1_score(y_true, y_pred), 0.936735778841042, abs_tol=5e-6)
print("PASS: TN/FP/FN/TP -> precision/recall/F1 arithmetic")
print(cm)
print("F1=", f1_score(y_true, y_pred))
