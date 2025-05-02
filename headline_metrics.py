class HeadlineMetrics:
    """
    A comprehensive class for analyzing and scoring headline effectiveness
    """
    
    def __init__(self, client=None):
        """Initialize the headline metrics analyzer"""
        self.client = client  # Optional OpenAI client for AI-powered analysis
        
        # Power words that typically drive higher engagement
        self.power_words = {
            'exclusive', 'secret', 'free', 'revealed', 'stunning', 'surprising',
            'remarkable', 'unauthorized', 'shocking', 'amazing', 'incredible',
            'essential', 'crucial', 'critical', 'proven', 'guaranteed',
            'powerful', 'ultimate', 'instant', 'easy', 'simple',
            'groundbreaking', 'revolutionary', 'breakthrough', 'game-changing',
            'extraordinary', 'jaw-dropping', 'mind-blowing', 'unbelievable',
            'unprecedented', 'urgent', 'vital', 'limited', 'special',
            'official', 'real', 'authentic', 'genuine', 'legitimate',
            'insider', 'hidden', 'untold', 'unique', 'rare',
            'unfiltered', 'raw', 'exposed', 'debunked', 'verified'
        }
        
        # Clickbait words that might harm credibility
        self.clickbait_words = {
            'actually', 'absolutely', 'literally', 'seriously', 'honestly',
            'insane', 'crazy', 'unreal', 'won\'t believe', 'mind blown',
            'speechless', 'jaw-dropping', 'life-changing', 'melt', 'perfect',
            'epic', 'legendary', 'going viral', 'broke the internet', 'this is why',
            'here\'s why', 'the reason is', 'you need to', 'what happened next',
            'changed forever', 'just happened', 'right now', 'this one thing'
        }
    
    def analyze_headline(self, headline):
        """
        Analyze a headline and return comprehensive metrics
        
        Args:
            headline (str): The headline text to analyze
            
        Returns:
            dict: Dictionary of headline metrics
        """
        if not headline or not isinstance(headline, str):
            return {
                'score': 0,
                'error': 'Invalid headline'
            }
        
        # Basic metrics
        length = len(headline)
        word_count = len(headline.split())
        chars_per_word = length / word_count if word_count > 0 else 0
        
        # Advanced metrics
        has_number = any(c.isdigit() for c in headline)
        is_question = '?' in headline
        words_lower = set(word.lower() for word in headline.split())
        power_word_count = len(words_lower.intersection(self.power_words))
        clickbait_word_count = sum(1 for cw in self.clickbait_words if cw.lower() in headline.lower())
        
        # Calculate readability
        try:
            import textstat
            fre = textstat.flesch_reading_ease(headline)
            fkg = textstat.flesch_kincaid_grade(headline)
            complex_words = textstat.difficult_words(headline)
        except (ImportError, Exception) as e:
            fre = 50  # Default value
            fkg = 8   # Default value
            complex_words = 0
        
        # Emotional impact (simple heuristic)
        emotional_impact = 0
        emotional_words = ['new', 'best', 'free', 'top', 'first', 'last', 'big', 'great', 'key', 'major']
        emotional_impact = sum(1 for word in words_lower if word in emotional_words)
        
        # Calculate overall score
        # Optimal length: 6-9 words, 60-100 characters
        length_score = 1.0
        if word_count < 5:
            length_score = 0.6  # Too short
        elif word_count > 15:
            length_score = 0.4  # Too long
        elif 6 <= word_count <= 9:
            length_score = 1.0  # Optimal
        else:
            length_score = 0.8  # Acceptable
            
        # Numbers typically increase CTR by 20-30%
        number_bonus = 0.2 if has_number else 0
            
        # Questions can increase CTR but overused
        question_bonus = 0.15 if is_question else 0
            
        # Power words bonus
        power_word_bonus = min(power_word_count * 0.1, 0.3)  # Cap at 0.3
            
        # Clickbait penalty
        clickbait_penalty = min(clickbait_word_count * 0.15, 0.5)  # Cap at 0.5
            
        # Readability sweet spot (FRE around 60-70 is optimal for news)
        readability_score = 0.0
        if 55 <= fre <= 75:
            readability_score = 1.0  # Optimal
        elif 40 <= fre < 55 or 75 < fre <= 85:
            readability_score = 0.8  # Good
        elif 30 <= fre < 40 or 85 < fre <= 95:
            readability_score = 0.6  # Acceptable
        else:
            readability_score = 0.4  # Poor
            
        # Emotional impact
        emotional_score = min(emotional_impact * 0.1, 0.3)  # Cap at 0.3
        
        # Calculate final score (base 100)
        base_score = 50  # Start at neutral
        components = {
            'length': length_score * 15,  # Max 15 points
            'numbers': number_bonus * 100,  # Max 20 points
            'question': question_bonus * 100,  # Max 15 points
            'power_words': power_word_bonus * 100,  # Max 30 points
            'clickbait_penalty': -clickbait_penalty * 100,  # Max -50 points
            'readability': readability_score * 20,  # Max 20 points
            'emotional': emotional_score * 100  # Max 30 points
        }
        
        # Calculate final score
        final_score = base_score
        for component, value in components.items():
            final_score += value
            
        # Cap between 0-100
        final_score = max(0, min(100, final_score))
        
        # Create result object
        result = {
            'score': final_score,
            'length': length,
            'word_count': word_count,
            'has_number': has_number,
            'is_question': is_question,
            'reading_ease': fre,
            'grade_level': fkg,
            'complex_words': complex_words,
            'power_words': power_word_count,
            'clickbait_words': clickbait_word_count,
            'emotional_words': emotional_impact,
            'score_components': components,
            'prediction': {
                'ctr_estimate': self._convert_score_to_ctr(final_score),
                'ctr_range': self._get_ctr_range(final_score)
            }
        }
        
        # Add AI analysis if client is available
        if self.client:
            try:
                ai_analysis = self._get_ai_analysis(headline)
                result['ai_analysis'] = ai_analysis
            except Exception as e:
                result['ai_analysis_error'] = str(e)
        
        return result
    
    def _convert_score_to_ctr(self, score):
        """Convert score to estimated CTR percentage"""
        # Empirical mapping from score to CTR
        # Based on typical news headline CTR ranges
        if score < 40:
            return 0.5 + (score * 0.025)  # 0.5% to 1.5%
        elif score < 60:
            return 1.5 + ((score - 40) * 0.05)  # 1.5% to 2.5%
        elif score < 80:
            return 2.5 + ((score - 60) * 0.075)  # 2.5% to 4.0%
        else:
            return 4.0 + ((score - 80) * 0.1)  # 4.0% to 6.0%
    
    def _get_ctr_range(self, score):
        """Get the CTR range as a tuple (min, max)"""
        ctr = self._convert_score_to_ctr(score)
        return (max(0.1, ctr * 0.8), ctr * 1.2)
    
    def _get_ai_analysis(self, headline):
        """Use AI to analyze headline appeal"""
        if not self.client:
            return None
            
        prompt = f"""Analyze this headline objectively for its potential engagement level:

Headline: "{headline}"

Please evaluate on:
1. Clarity: Is it clear what the article is about?
2. Curiosity: Does it create interest without being clickbait?
3. Value: Does it signal value to the reader?
4. Specificity: Is it specific rather than vague?
5. Emotional appeal: Does it connect emotionally?

Provide a brief analysis and a score from 1-10.
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert headline analyst who objectively evaluates headline effectiveness."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=150,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"AI analysis failed: {str(e)}"
    
    def compare_headlines(self, original, rewritten):
        """
        Compare original and rewritten headlines
        
        Args:
            original (str): Original headline
            rewritten (str): Rewritten headline
            
        Returns:
            dict: Comparison metrics
        """
        original_metrics = self.analyze_headline(original)
        rewritten_metrics = self.analyze_headline(rewritten)
        
        # Calculate improvements
        score_diff = rewritten_metrics['score'] - original_metrics['score']
        score_percent = (score_diff / original_metrics['score']) * 100 if original_metrics['score'] > 0 else 0
        
        ctr_original = original_metrics['prediction']['ctr_estimate']
        ctr_rewritten = rewritten_metrics['prediction']['ctr_estimate']
        ctr_diff = ctr_rewritten - ctr_original
        ctr_percent = (ctr_diff / ctr_original) * 100 if ctr_original > 0 else 0
        
        # Determine key improvements
        improvements = []
        if rewritten_metrics['has_number'] and not original_metrics['has_number']:
            improvements.append("Added specific numbers")
        
        if rewritten_metrics['power_words'] > original_metrics['power_words']:
            improvements.append(f"Added {rewritten_metrics['power_words'] - original_metrics['power_words']} power words")
        
        if original_metrics['clickbait_words'] > rewritten_metrics['clickbait_words']:
            improvements.append("Reduced clickbait language")
        
        if abs(rewritten_metrics['reading_ease'] - 65) < abs(original_metrics['reading_ease'] - 65):
            improvements.append("Improved readability")
        
        if rewritten_metrics['word_count'] < original_metrics['word_count'] and original_metrics['word_count'] > 12:
            improvements.append("More concise wording")
        
        # Return comparison
        return {
            'original_score': original_metrics['score'],
            'rewritten_score': rewritten_metrics['score'],
            'score_difference': score_diff,
            'score_percent_change': score_percent,
            'original_ctr': ctr_original,
            'rewritten_ctr': ctr_rewritten,
            'ctr_difference': ctr_diff,
            'ctr_percent_change': ctr_percent,
            'key_improvements': improvements,
            'original_metrics': original_metrics,
            'rewritten_metrics': rewritten_metrics,
            'verdict': "Improved" if score_diff > 0 else "Worsened" if score_diff < 0 else "Unchanged"
        }