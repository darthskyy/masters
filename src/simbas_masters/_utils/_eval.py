def eval_segments(predicts: list[list[str]], targets: list[list[str]], level: str = "boundary") -> dict:
    """
    Evaluates segmentation predictions against targets at either the boundary or morphemic level.

    Arguments:
        predicts (list[list[str]]): Lists of predicted segments, where each inner list contains the segments for a single prediction.
        targets (list[list[str]]): Lists of target segments, where each inner list contains the segments for a single target.
        level (str): The evaluation level, either "boundary" or "morphemic". Defaults to "boundary".
    If "boundary", evaluates based on segment boundaries; if "morphemic", evaluates based on morpheme matches.

    Returns:
        dict: Dictionary with keys "precision", "recall", and "f1" containing the corresponding scores.

    Raises:
        ZeroDivisionError: If there are no true positives and/or no predicted positives, leading to division by zero.
    
    Example:
        >>> predicts = [["Ġhello", "Ġworld"], ["Ġfoo", "Ġbar"]]
        >>> targets = [["Ġhello", "Ġworld"], ["Ġfoo", "Ġbaz"]]
        >>> eval_segments(predicts, targets, level="morphemic")
        {'precision': 0.75, 'recall': 0.75, 'f1': 0.75}
        >>> eval_segments(predicts, targets, level="boundary")
        {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
    """

    if level == "boundary":
        # Initialize counters for true positives, false positives, true negatives, false negatives
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        for pred, targ in zip(predicts, targets):
            # Compute boundary positions for predictions
            pred_bounds = []
            cur_index = -1
            for p in pred[0: -1]:
                pred_bounds.append(cur_index + len(p))
                cur_index += len(p)
            # Compute non-boundary positions for predictions
            pred_non_bounds = [index for index in range(len("".join(pred)) - 1) if index not in pred_bounds]

            # Compute boundary positions for targets
            targ_bounds = []
            cur_index = -1
            for t in targ[0: -1]:
                targ_bounds.append(cur_index + len(t))
                cur_index += len(t)
            # Compute non-boundary positions for targets
            targ_non_bounds = [index for index in range(len("".join(targ)) - 1) if index not in targ_bounds]

            # Calculate metrics by comparing sets of boundaries
            tp += len(set(pred_bounds) & set(targ_bounds))  # Correctly predicted boundaries
            fp += len(set(pred_bounds) & set(targ_non_bounds))  # Incorrectly predicted boundaries
            tn += len(set(pred_non_bounds) & set(targ_non_bounds))  # Correctly predicted non-boundaries
            fn += len(set(pred_non_bounds) & set(targ_bounds))  # Missed boundaries

        # Calculate precision, recall, and F1 score
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f_score = 2 / (1 / precision + 1 / recall)

    elif level == "morphemic":
        # Count correctly predicted morphemes
        correct = 0.0
        for pred, targ in zip(predicts, targets):
            for p in pred:
                if p in targ:
                    correct += 1

        # Calculate precision, recall, and F1 score based on morpheme counts
        predicted_length = sum([len(pred) for pred in predicts])
        target_length = sum([len(targ) for targ in targets])
        precision, recall = correct / predicted_length, correct / target_length
        f_score = 2 / (1 / precision + 1 / recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f_score
    }

def fix_tokens(tokens):
    word_start = "Ġ"
    g_strip = lambda x: x.lstrip(word_start)
    tokens = [g_strip(token) for token in tokens]
    return [token for token in tokens if token]