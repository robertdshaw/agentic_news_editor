# headline_metrics.py

import re
import logging

class HeadlineMetrics:
    def __init__(self, client=None):
        self.client = client
        
        # Define scoring weights
        self.power_words = [
            'secret', 'proven', 'shocking', 'amazing', 'incredible', 
            'breakthrough', 'revolutionary', 'essential', 'critical',
            'stunning', 'revealed', 'discover', 'transform', 'powerful',
            'exclusive', 'urgent', 'limited', 'guarantee', 'scientific'
        ]
        
        self.emotional_words = [
            'love', 'hate', 'fear', 'joy', 'surprise', 'anger', 
            'happy', 'sad', 'excited', 'worried', 'confident'
        ]
        
        self.weak_words = [
            'might', 'maybe', 'could', 'possibly', 'perhaps',
            'somewhat', 'fairly', 'quite', 'rather', 'seems'
        ]
    
    def calculate_ctr_score(self, headline):
        """Calculate predicted CTR based on headline characteristics"""
        try:
            score = 50.0  # Start with baseline score of 50%
            
            # Length optimization (40-60 characters is optimal)
            length = len(headline)
            if 40 <= length <= 60:
                score += 10.0
            elif 30 <= length < 40 or 60 < length <= 70:
                score += 5.0
            elif length > 80:
                score -= 10.0
            elif length < 25:
                score -= 15.0
            
            # Numbers in headline (specific data increases CTR)
            numbers = re.findall(r'\d+', headline)
            if numbers:
                score += 15.0
                if any(int(num) > 1000 for num in numbers if num.isdigit()):
                    score += 5.0  # Big numbers are more compelling
            
            # Question format
            if headline.endswith('?'):
                score += 12.0
            
            # Power words
            headline_lower = headline.lower()
            power_word_count = sum(1 for word in self.power_words if word in headline_lower)
            score += power_word_count * 8.0  # Each power word adds 8 points
            
            # Emotional words
            emotional_word_count = sum(1 for word in self.emotional_words if word in headline_lower)
            score += emotional_word_count * 6.0  # Each emotional word adds 6 points
            
            # Weak words (reduce score)
            weak_word_count = sum(1 for word in self.weak_words if word in headline_lower)
            score -= weak_word_count * 10.0  # Each weak word reduces 10 points
            
            # Capitalization issues
            if headline.isupper():  # ALL CAPS is bad
                score -= 20.0
            elif headline.islower():  # all lowercase is bad
                score -= 10.0
            
            # Check for "How to" format
            if headline_lower.startswith('how to') or 'how you can' in headline_lower:
                score += 10.0
            
            # Check for listicle format
            if re.match(r'^\d+\s+\w+', headline):  # Starts with number
                score += 8.0
            
            # Punctuation
            if headline.count('!') > 1:  # Too many exclamation marks
                score -= 15.0
            elif headline.count('!') == 1:  # One is okay
                score += 3.0
            
            # Specificity bonus
            if any(word in headline_lower for word in ['study', 'research', 'report', 'data']):
                score += 5.0
            
            # Urgency words
            urgency_words = ['now', 'today', 'breaking', 'urgent', 'immediately']
            if any(word in headline_lower for word in urgency_words):
                score += 8.0
            
            # Convert to percentage (ensure it's between 10% and 90%)
            ctr_percentage = min(max(score, 10), 90)
            
            logging.debug(f"Headline: {headline}")
            logging.debug(f"Base score: {score}, CTR: {ctr_percentage}%")
            
            return score, ctr_percentage
            
        except Exception as e:
            logging.error(f"Error calculating CTR score: {e}")
            return 50.0, 50.0  # Return baseline on error
    
    def compare_headlines(self, original, rewritten):
        """Compare original and rewritten headlines"""
        try:
            original_score, original_ctr = self.calculate_ctr_score(original)
            rewritten_score, rewritten_ctr = self.calculate_ctr_score(rewritten)
            
            improvement = rewritten_ctr - original_ctr
            score_percent_change = (improvement / original_ctr) * 100 if original_ctr > 0 else 0
            
            # Identify key improvements
            key_improvements = []
            
            # Check for question format
            if rewritten.endswith('?') and not original.endswith('?'):
                key_improvements.append("Added question format")
            
            # Check for numbers
            original_numbers = re.findall(r'\d+', original)
            rewritten_numbers = re.findall(r'\d+', rewritten)
            if rewritten_numbers and not original_numbers:
                key_improvements.append("Added specific numbers")
            elif len(rewritten_numbers) > len(original_numbers):
                key_improvements.append("Added more specific data")
            
            # Check for power words
            original_power = sum(1 for word in self.power_words if word in original.lower())
            rewritten_power = sum(1 for word in self.power_words if word in rewritten.lower())
            if rewritten_power > original_power:
                key_improvements.append("Added power words")
            
            # Check for length optimization
            original_len = len(original)
            rewritten_len = len(rewritten)
            if (original_len > 70 or original_len < 40) and (40 <= rewritten_len <= 60):
                key_improvements.append("Optimized length")
            
            # Check for weak word removal
            original_weak = sum(1 for word in self.weak_words if word in original.lower())
            rewritten_weak = sum(1 for word in self.weak_words if word in rewritten.lower())
            if original_weak > rewritten_weak:
                key_improvements.append("Removed weak language")
            
            # If no improvements found but score increased
            if not key_improvements and improvement > 0:
                key_improvements.append("General optimization")
            
            return {
                'original_score': original_score,
                'rewritten_score': rewritten_score,
                'original_ctr': original_ctr / 100.0,  # Convert to decimal
                'rewritten_ctr': rewritten_ctr / 100.0,
                'score_percent_change': score_percent_change,
                'key_improvements': key_improvements,
                'headline_improvement': improvement  # Add this for consistency
            }
            
        except Exception as e:
            logging.error(f"Error comparing headlines: {e}")
            return {
                'original_score': 50.0,
                'rewritten_score': 50.0,
                'original_ctr': 0.5,
                'rewritten_ctr': 0.5,
                'score_percent_change': 0,
                'key_improvements': ["Error in comparison"],
                'headline_improvement': 0
            }
    
    def get_headline_feedback(self, original, rewritten):
        """Generate detailed feedback on the headline rewrite"""
        comparison = self.compare_headlines(original, rewritten)
        
        feedback = []
        
        if comparison['score_percent_change'] > 0:
            feedback.append(f"✅ Improved CTR by {comparison['score_percent_change']:.1f}%")
        else:
            feedback.append(f"⚠️ Decreased CTR by {abs(comparison['score_percent_change']):.1f}%")
        
        for improvement in comparison['key_improvements']:
            feedback.append(f"• {improvement}")
        
        # Specific recommendations
        if len(rewritten) > 70:
            feedback.append("💡 Consider shortening to 40-60 characters")
        
        if not any(char.isdigit() for char in rewritten):
            feedback.append("💡 Consider adding specific numbers")
        
        if not rewritten.endswith('?') and 'how' not in rewritten.lower():
            feedback.append("💡 Questions or 'How to' formats often perform well")
        
        return "\n".join(feedback)

# Optional: Test the metrics
if __name__ == "__main__":
    metrics = HeadlineMetrics()
    
    # Test examples
    original = "Scientists Make Discovery About Climate Change"
    rewritten = "7 Shocking Ways Climate Change Will Transform Your City by 2030"
    
    comparison = metrics.compare_headlines(original, rewritten)
    print(f"Original CTR: {comparison['original_ctr']*100:.1f}%")
    print(f"Rewritten CTR: {comparison['rewritten_ctr']*100:.1f}%")
    print(f"Improvement: {comparison['score_percent_change']:.1f}%")
    print(f"Key improvements: {', '.join(comparison['key_improvements'])}")