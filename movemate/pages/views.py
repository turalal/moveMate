# pages/views.py
from rest_framework import generics, status, viewsets
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.decorators import action
from rest_framework.filters import SearchFilter, OrderingFilter
from rest_framework.throttling import AnonRateThrottle
from django.shortcuts import get_object_or_404
from django.template.loader import render_to_string
from django.conf import settings
from django.core.cache import cache
from django.db.models import Count
from django.core.mail import EmailMessage, BadHeaderError
from django_filters import rest_framework as django_filters
from .models import Contact, Service, BlogPost, BlogCategory, Comment
from .serializers import (
    ContactSerializer, 
    ServiceSerializer, 
    BlogPostSerializer, 
    BlogCategorySerializer,
    CommentSerializer
)
import logging

logger = logging.getLogger(__name__)


class ServiceFilter(django_filters.FilterSet):
    class Meta:
        model = Service
        fields = {
            'title': ['icontains'],
            'created_at': ['gte', 'lte'],
            'is_active': ['exact'],
        }

class BlogPostFilter(django_filters.FilterSet):
    class Meta:
        model = BlogPost
        fields = {
            'title': ['icontains'],
            'category': ['exact'],
            'created_at': ['gte', 'lte'],
            'author': ['exact'],
            'status': ['exact'],
        }

class ContactRateThrottle(AnonRateThrottle):
    rate = '3/h'  # 3 requests per hour
    scope = 'contact'

# Email templates as string constants
CUSTOMER_EMAIL_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
</head>
<body>
    <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
        <h2>Thank you for contacting MoveMate!</h2>
        <p>Dear {name},</p>
        <p>We have received your message and appreciate you taking the time to write to us.</p>
        <p>Our team will review your message and get back to you as soon as possible.</p>
        <br>
        <p>Best regards,</p>
        <p>The MoveMate Team</p>
    </div>
</body>
</html>
"""

ADMIN_EMAIL_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
</head>
<body>
    <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
        <h2>New Contact Form Submission</h2>
        <p><strong>From:</strong> {name}</p>
        <p><strong>Email:</strong> {email}</p>
        <p><strong>Date:</strong> {date}</p>
        <p><strong>Message:</strong></p>
        <div style="background-color: #f5f5f5; padding: 15px; border-radius: 5px;">
            {message}
        </div>
        <br>
        <p>Please respond to this inquiry as soon as possible.</p>
    </div>
</body>
</html>
"""

class ContactView(generics.CreateAPIView):
    queryset = Contact.objects.all()
    serializer_class = ContactSerializer
    permission_classes = [AllowAny]
    throttle_classes = [ContactRateThrottle]

    def create(self, request, *args, **kwargs):
        try:
            # Validate and save the contact form data
            serializer = self.get_serializer(data=request.data)
            serializer.is_valid(raise_exception=True)
            contact = serializer.save()
            
            # Send auto-reply to the customer
            try:
                self.send_customer_confirmation(contact)
                logger.info(f"Auto-reply sent to customer: {contact.email}")
            except Exception as e:
                logger.error(f"Failed to send customer confirmation: {str(e)}")

            # Send notification to admin/sales
            try:
                self.send_admin_notification(contact)
                logger.info(f"Notification sent to admin: {settings.SALES_EMAIL}")
            except Exception as e:
                logger.error(f"Failed to send admin notification: {str(e)}")

            return Response(
                {"message": "Thank you for your message. We will contact you soon."},
                status=status.HTTP_201_CREATED
            )

        except Exception as e:
            logger.error(f"Contact form submission error: {str(e)}")
            return Response(
                {"error": "Sorry, we couldn't process your request. Please try again later."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def send_customer_confirmation(self, contact):
        """Send an immediate auto-reply to the customer"""
        try:
            email_body = CUSTOMER_EMAIL_TEMPLATE.format(
                name=contact.name
            )

            message = EmailMessage(
                subject="Thank you for contacting MoveMate",
                body=email_body,
                from_email=settings.DEFAULT_FROM_EMAIL,
                to=[contact.email],
                reply_to=[settings.SALES_EMAIL],
                headers={
                    'X-Auto-Response-Suppress': 'OOF, DR, RN, NRN, AutoReply',
                    'Auto-Submitted': 'auto-generated',
                }
            )
            message.content_subtype = "html"
            message.send(fail_silently=False)
            
        except Exception as e:
            logger.error(f"Error sending customer confirmation email: {str(e)}")
            raise

    def send_admin_notification(self, contact):
        """Send notification to admin/sales about the new inquiry"""
        try:
            email_body = ADMIN_EMAIL_TEMPLATE.format(
                name=contact.name,
                email=contact.email,
                message=contact.message,
                date=contact.created_at.strftime('%Y-%m-%d %H:%M:%S')
            )

            message = EmailMessage(
                subject=f"New Contact Form Submission - {contact.name}",
                body=email_body,
                from_email=settings.DEFAULT_FROM_EMAIL,
                to=[settings.SALES_EMAIL],
                reply_to=[contact.email]
            )
            message.content_subtype = "html"
            message.send(fail_silently=False)
            
        except Exception as e:
            logger.error(f"Error sending admin notification email: {str(e)}")
            raise

class ServiceViewSet(viewsets.ModelViewSet):
    queryset = Service.objects.filter(is_active=True)
    serializer_class = ServiceSerializer
    permission_classes = [AllowAny]
    lookup_field = 'slug'
    filter_backends = [django_filters.DjangoFilterBackend, SearchFilter, OrderingFilter]
    filterset_class = ServiceFilter
    search_fields = ['title', 'description']
    ordering_fields = ['created_at', 'title']
    ordering = ['-created_at']

class BlogCategoryViewSet(viewsets.ModelViewSet):
    queryset = BlogCategory.objects.annotate(posts_count=Count('posts'))
    serializer_class = BlogCategorySerializer
    permission_classes = [AllowAny]
    lookup_field = 'slug'
    search_fields = ['name']
    ordering = ['name']

class BlogPostViewSet(viewsets.ModelViewSet):
    serializer_class = BlogPostSerializer
    permission_classes = [AllowAny]
    lookup_field = 'slug'
    filter_backends = [django_filters.DjangoFilterBackend, SearchFilter, OrderingFilter]
    filterset_class = BlogPostFilter
    search_fields = ['title', 'content']
    ordering_fields = ['created_at', 'views', 'title']
    ordering = ['-created_at']

    def get_queryset(self):
        if self.request.user.is_staff:
            return BlogPost.objects.all()
        return BlogPost.objects.filter(status='published')

    @action(detail=True, methods=['post'])
    def increment_view(self, request, slug=None):
        try:
            post = self.get_object()
            post.views += 1
            post.save(update_fields=['views'])
            return Response({'status': 'view count updated'})
        except Exception as e:
            logger.error(f"Error incrementing view count: {str(e)}")
            return Response(
                {'error': 'Failed to update view count'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class CommentViewSet(viewsets.ModelViewSet):
    serializer_class = CommentSerializer
    permission_classes = [IsAuthenticated]
    filter_backends = [django_filters.DjangoFilterBackend, OrderingFilter]
    ordering_fields = ['created_at']
    ordering = ['-created_at']

    def get_queryset(self):
        return Comment.objects.filter(
            post__slug=self.kwargs['post_slug'],
            is_approved=True
        )

    def perform_create(self, serializer):
        post = get_object_or_404(BlogPost, slug=self.kwargs['post_slug'])
        serializer.save(
            author=self.request.user,
            post=post,
            is_approved=True
        )