# pages/validators.py
import re
from django.core.exceptions import ValidationError
from django.core.validators import validate_email
from django.core.cache import cache
from rest_framework.throttling import AnonRateThrottle
from datetime import datetime, timedelta

class EmailDomainValidator:
    """Validates email domains against allowed/blocked lists."""
    
    BLOCKED_DOMAINS = {
        'tempmail.com', 'throwawaymail.com', 'guerrillamail.com',
        # Add more disposable email domains
    }
    
    @classmethod
    def validate_domain(cls, email):
        try:
            # Basic email format validation
            validate_email(email)
            
            # Extract domain
            domain = email.split('@')[1].lower()
            
            # Check against blocked domains
            if domain in cls.BLOCKED_DOMAINS:
                raise ValidationError('This email domain is not allowed.')
            
            # Additional domain validation rules can be added here
            
        except IndexError:
            raise ValidationError('Invalid email format.')

class EmailThrottler(AnonRateThrottle):
    """Custom throttle for email submissions."""
    
    rate = '3/hour'  # Limit to 3 emails per hour per IP
    cache_format = 'email_throttle_{}'
    
    def get_cache_key(self, request, view):
        if not request.META.get('REMOTE_ADDR'):
            return None
        
        return self.cache_format.format(request.META.get('REMOTE_ADDR'))




