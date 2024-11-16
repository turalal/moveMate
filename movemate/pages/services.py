# pages/services.py
from django.core.cache import cache
from datetime import timedelta

class EmailSecurityService:
    """Service for managing email submission security."""
    
    @staticmethod
    def check_submission_history(email, ip_address):
        """
        Check if email/IP combination should be allowed to submit.
        Returns tuple of (is_allowed, message).
        """
        # Check IP-based submissions
        ip_cache_key = f'email_ip_{ip_address}'
        ip_count = cache.get(ip_cache_key, 0)
        
        if ip_count >= 10:  # Maximum 10 submissions per IP per day
            return False, "Too many submissions from this IP address. Please try again tomorrow."
            
        # Check email-based submissions
        email_cache_key = f'email_address_{email}'
        email_count = cache.get(email_cache_key, 0)
        
        if email_count >= 3:  # Maximum 3 submissions per email per day
            return False, "Too many submissions from this email address. Please try again tomorrow."
            
        # Update counters
        cache.set(ip_cache_key, ip_count + 1, timeout=86400)  # 24 hours
        cache.set(email_cache_key, email_count + 1, timeout=86400)  # 24 hours
        
        return True, None
