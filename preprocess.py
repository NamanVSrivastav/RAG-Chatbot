import re
from textblob import TextBlob

def preprocess_query(query):
    # Correct spelling
    corrected_query = str(TextBlob(query).correct())

    # Determine query type
    if "summarize" in corrected_query.lower() or "summary" in corrected_query.lower():
        query_type = "summary"
    else:
        query_type = "general"

    return corrected_query, query_type
