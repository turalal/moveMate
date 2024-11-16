# pages/services.py
class EmailSecurityService:
    """Handles email security checks and rate limiting."""
    
    @staticmethod
    def check_submission_history(email, ip_address):
        """
        Check submission history for an email/IP combination.
        Returns (is_allowed, message)
        """
        cache_key = f'email_submission_{email}_{ip_address}'
        submission_count = cache.get(cache_key, 0)
        
        # Check submission frequency
        if submission_count >= 3:
            return False, "Too many submissions. Please try again later."
        
        # Increment submission count
        cache.set(cache_key, submission_count + 1, 3600)  # 1 hour expiry
        
        return True, None