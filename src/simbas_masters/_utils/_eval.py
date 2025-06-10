import json
from sklearn.metrics import f1_score, precision_score, recall_score

def get_boundaries(tokens: list[str]) -> list[bool]:
    """
    Get the boundaries of the tokens in a list. True means a character pair
    is within a token, False means it's a boundary between tokens.
    This function assumes that tokens are non-empty strings and that the
    last token does not have a boundary after it.
    f

    Arguments:
        tokens (list[str]): List of tokens to analyze.

    Returns:
        list[bool]: A list of booleans meeting the conditions:
            - The index will be a particular pair of characters in the word.
            - A boolean indicating whether the pair is part of the same token or not.
            - tokens = ['ab', 'cd', 'efg', 'h']
            - char_pairs = ["ab", "bc", "cd", "de", "ef", "fg", "gh"]
            - output = [True, False, True, False, True, True, False]

    Example:
        >>> get_boundaries(["hello", "world"])
        [True, True, False, True, True]
    """
    if not tokens:
        return []

    boundaries = []
    for i in range(len(tokens) - 1):
        token = tokens[i]
        if len(token) > 1:
            boundaries.extend([True] * (len(token) - 1))
        boundaries.append(False)

    last_token = tokens[-1]
    if len(last_token) > 1:
        boundaries.extend([True] * (len(last_token) - 1))

    return boundaries

def calculate_tokenization_scores_simba(reference_tokens: list[str], predicted_tokens: list[str], 
                                        mode: str = "boundary") -> dict:
    """
    Calculates F1, precision, and recall for tokenization using boundary detection.

    This is an implementation of evaluation I devised for what i thought boundary identification
    would be a good way to evaluate tokenization. It is based on the idea that we can treat
    token boundaries as a binary classification problem, where we identify whether a character
    pair is a boundary between tokens or not.

    NOTE: It has some downsides:
        - It does not take into account the morphological soundness tokenisation.
            - e.g. ["ru", "nning"] and ["runn", "ing"] would have the same when the reference is ["run", "ning"]

    Arguments:
        reference_tokens (list[str]): List of tokens from the reference text.
        predicted_tokens (list[str]): List of tokens from the predicted text.
        mode (str): The mode of evaluation, either "boundary" or "morphemic". Default is "boundary".

    Returns:
        dict: A dictionary containing the F1 score, precision, and recall.
            {
                "f1_score": float,
                "precision": float,
                "recall": float
            }
    """
    # [ ] morpheme mode needs to be implemented
    # --- Verification Step ---
    ref_str = "".join(reference_tokens)
    pred_str = "".join(predicted_tokens)
    if ref_str.lower() != pred_str.lower():
        raise ValueError("Reference and predicted tokens do not form the same string.")

    # --- Step 1: Generate Boundary Lists ---
    y_true_boundaries = get_boundaries(reference_tokens)
    y_pred_boundaries = get_boundaries(predicted_tokens)

    # --- Step 2: Define Positive Class ---
    # We are looking for boundaries, which are represented by `False`.
    # To use standard metrics libraries, we'll map False -> 1 and True -> 0.
    # So, 1 is the positive class (a boundary).
    y_true = [1 if not b else 0 for b in y_true_boundaries]
    y_pred = [1 if not b else 0 for b in y_pred_boundaries]

    # --- Step 3 & 4: Calculate Scores ---
    # Using scikit-learn for a robust and simple calculation
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
    }

def calculate_tokenisation_scores_francois(reference_tokens, predicted_tokens, mode="boundary"):
    """
    Calculate tokenization scores based on the mode specified.
    This function computes precision, recall, and F1 score for tokenization
    using either boundary detection or morphemic evaluation.

    This implementation is by Francois Meyer.

    Arguments:
        reference_tokens (list[str]): List of tokens from the reference text.
        predicted_tokens (list[str]): List of tokens from the predicted text.
        mode (str): The mode of evaluation, either "boundary" or "morphemic".
    Returns:
        tuple: A tuple containing precision, recall, and F1 score.
            (precision, recall, f_score)
    """

    if mode == "boundary":
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        for pred, targ in zip(predicted_tokens, reference_tokens):
            pred_bounds = []
            cur_index = -1
            for p in pred[0: -1]:
                pred_bounds.append(cur_index + len(p))
                cur_index += len(p)
            pred_non_bounds = [index for index in range(len("".join(pred)) - 1) if index not in pred_bounds]

            targ_bounds = []
            cur_index = -1
            for t in targ[0: -1]:
                targ_bounds.append(cur_index + len(t))
                cur_index += len(t)
            targ_non_bounds = [index for index in range(len("".join(targ)) - 1) if index not in targ_bounds]

            tp += len(set(pred_bounds) & set(targ_bounds))
            fp += len(set(pred_bounds) & set(targ_non_bounds))
            tn += len(set(pred_non_bounds) & set(targ_non_bounds))
            fn += len(set(pred_non_bounds) & set(targ_bounds))

        if tp + fp == 0 or tp + fn == 0:
            precision, recall, f1 = 0.0, 0.0, 0.0
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 / (1 / precision + 1 / recall)

    elif mode == "morphemic":
        correct = 0.0
        for pred, targ in zip(predicted_tokens, reference_tokens):
            for p in pred:
                if p in targ:
                    correct += 1

        predicted_length = sum([len(pred) for pred in predicted_tokens])
        target_length = sum([len(targ) for targ in reference_tokens])
        if predicted_length == 0 or target_length == 0 or correct == 0:
            precision, recall, f1 = 0.0, 0.0, 0.0
        else:
            precision, recall = correct / predicted_length, correct / target_length
            f1 = 2 / (1 / precision + 1 / recall)


    return {
        "f1_score": f1,
        "precision": precision,
        "recall": recall
    }

def fix_tokens(tokens):
    word_start = "Ġ"
    g_strip = lambda x: x.lstrip(word_start)
    tokens = [g_strip(token) for token in tokens]
    return [token for token in tokens if token]