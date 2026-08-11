def predict_bulk(reviews):
    """
    Temporary bulk prediction function.

    The actual BERT model will be connected here later.
    """
    results = []

    for review in reviews:
        if review and str(review).strip():
            results.append("Model not connected yet")
        else:
            results.append("Invalid review")

    return results
