# pages/throttling.py
from rest_framework.throttling import AnonRateThrottle
from django.core.cache import cache

class EmailThrottle(AnonRateThrottle):
    rate = '1/15m'  # One submission per 15 minutes
    
    def get_cache_key(self, request, view):
        if request.data.get('email'):
            return f'email_submission_{request.data["email"]}'
        return self.get_ident(request)

class IPThrottle(AnonRateThrottle):
    rate = '3/hour'  # Three submissions per hour per IP
    
    def get_cache_key(self, request, view):
        return f'ip_submission_{self.get_ident(request)}'