# pages/validators.py
from rest_framework.throttling import AnonRateThrottle
from django.core.exceptions import ValidationError
from django.core.validators import validate_email

class EmailDomainValidator:
    """Validates email domains against blocked lists."""
    
    BLOCKED_DOMAINS = {
        'tempmail.com', 'temp-mail.org', 'guerrillamail.com', 
        'sharklasers.com', 'grr.la', 'guerrillamail.info',
        'yopmail.com', 'disposablemail.com', 'mailinator.com',
        # Add more as needed
    }
    
    @staticmethod
    def validate(email):
        try:
            # Basic email validation
            validate_email(email)
            
            # Extract domain
            domain = email.split('@')[1].lower()
            
            if domain in EmailDomainValidator.BLOCKED_DOMAINS:
                raise ValidationError('Temporary or disposable email addresses are not allowed.')
                
        except IndexError:
            raise ValidationError('Invalid email format.')

class EmailThrottler(AnonRateThrottle):
    """Throttle for limiting email submissions."""
    
    rate = '3/hour'  # Adjust as needed
    
    def get_cache_key(self, request, view):
        if not request.META.get('REMOTE_ADDR'):
            return None
            
        return f'email_throttle_{request.META.get("REMOTE_ADDR")}'
