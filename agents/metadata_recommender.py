import re
from collections import Counter
from nltk.corpus import stopwords
import nltk

# Download stopwords if not already present
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords', quiet=True)

class MetadataRecommenderAgent:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def _clean_text(self, text: str) -> str:
        """Cleans text by removing special characters and converting to lowercase."""
        text = text.lower()
        # Remove non-alphanumeric characters (keep spaces)
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text

    def extract_keywords(self, content: str, top_n: int = 8) -> list:
        """Extracts the most frequent non-stop words as keywords."""
        cleaned_text = self._clean_text(content)
        words = cleaned_text.split()
        
        # Filter out stopwords and short words
        keywords = [word for word in words if word not in self.stop_words and len(word) > 2]
        
        word_counts = Counter(keywords)
        most_common = word_counts.most_common(top_n)
        
        return [word for word, count in most_common]

    def suggest_metadata(self, content: str):
        """Suggests keywords, tags, categories, and badge ideas."""
        keywords = self.extract_keywords(content, top_n=12)
        
        # Simple heuristic for categories based on common repo phrases
        categories = []
        lower_content = content.lower()
        if "rag" in lower_content or "agent" in lower_content or "llm" in lower_content:
            categories.append("LLM/Generative AI")
        if "python" in lower_content or "code" in lower_content:
            categories.append("Python Development")
        if "data" in lower_content or "analysis" in lower_content or "model" in lower_content:
            categories.append("Data Science")
        
        # Placeholder badge suggestions (can be improved by LLM later)
        badges = [
            "[Maintenance] Needs Review", 
            "[License] Recommended MIT",
            "[Status] Work in Progress"
        ]
        
        print(f"Generated Keywords: {keywords}")
        
        return {
            "keywords": keywords,
            "tags": list(set(keywords[:5] + categories)), # Use top 5 keywords + categories as tags
            "categories": categories,
            "badges_to_add": badges 
        }