from services.prediction_service import predict_sentiment


def predict_bulk(reviews):
    """
    Predict sentiment for multiple reviews.
    """

    results = []

    for review in reviews:

        if review and str(review).strip():
            results.append(
                predict_sentiment(review)
            )
        else:
            results.append("Invalid review")

    return results
