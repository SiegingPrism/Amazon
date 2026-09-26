def evaluate_entity_f05(gt_ids, pred_ids):
    """
    gt_ids: set of ground truth match IDs
    pred_ids: set of predicted match IDs
    """
    if len(gt_ids) == 0:
        # Singleton entity
        return 1.0 if len(pred_ids) == 0 else 0.0
    
    if len(pred_ids) == 0:
        return 0.0
    
    tp = len(gt_ids & pred_ids)
    if tp == 0:
        return 0.0
    
    precision = tp / len(pred_ids)
    recall = tp / len(gt_ids)
    
    denom = 0.25 * precision + recall
    if denom == 0:
        return 0.0
    return (1.25 * precision * recall) / denom

# Test cases from problem statement
# Example: S1-00001
# Predicts [S2-00047, S2-00193, S3-00812] (3 items)
# Ground truth [S2-00047, S3-00812] (2 items)
# TP = 2, Prec = 2/3, Rec = 1.0
gt_ex = {"S2-00047", "S3-00812"}
pred_ex = {"S2-00047", "S2-00193", "S3-00812"}
score = evaluate_entity_f05(gt_ex, pred_ex)
print(f"Problem statement example score: {score:.4f} (Expected: ~0.7143)")

# Singleton test
print("Singleton true empty, pred empty:", evaluate_entity_f05(set(), set()))
print("Singleton true empty, pred 1 item:", evaluate_entity_f05(set(), {"S2-123"}))
print("Matched true 1 item, pred empty:", evaluate_entity_f05({"S2-123"}, set()))
print("Matched true 1 item, pred 1 item (exact):", evaluate_entity_f05({"S2-123"}, {"S2-123"}))
