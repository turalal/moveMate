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
from django.core.mail import EmailMessage
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

class ContactView(generics.CreateAPIView):
    queryset = Contact.objects.all()
    serializer_class = ContactSerializer
    permission_classes = [AllowAny]
    throttle_classes = [ContactRateThrottle]

    def create(self, request, *args, **kwargs):
        try:
            serializer = self.get_serializer(data=request.data)
            serializer.is_valid(raise_exception=True)
            contact = self.perform_create(serializer)
            
            try:
                logger.info(f"Sending notification email to {settings.SALES_EMAIL}")
                self.send_notification_email(contact.name, contact.email, contact.message)
                logger.info("Notification email sent successfully")
                
                logger.info(f"Sending confirmation email to {contact.email}")
                self.send_confirmation_email(contact.name, contact.email)
                logger.info("Confirmation email sent successfully")
            except Exception as email_error:
                logger.error(f"Error sending emails: {str(email_error)}")
            
            return Response(
                {"message": "Message sent successfully"},
                status=status.HTTP_201_CREATED
            )
            
        except Exception as e:
            logger.error(f"Error in contact form submission: {str(e)}")
            return Response(
                {"error": "An error occurred while processing your request"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def send_notification_email(self, name, email, message):
        """Send notification email to admin"""
        email = EmailMessage(
            subject='New Contact Form Submission',
            body=render_to_string('emails/contact_notification.html', {
                'name': name,
                'email': email,
                'message': message,
            }),
            from_email=settings.DEFAULT_FROM_EMAIL,
            to=[settings.SALES_EMAIL],
            reply_to=[email],
            headers={
                'X-Auto-Response-Suppress': 'OOF, DR, RN, NRN, AutoReply',
                'Auto-Submitted': 'auto-generated',
                'X-Priority': '3',
            },
        )
        email.content_subtype = "html"
        email.send(fail_silently=False)

    def send_confirmation_email(self, name, email):
        """Send confirmation email to user"""
        email = EmailMessage(
            subject='Thank you for contacting MoveMate',
            body=render_to_string('emails/contact_confirmation.html', {
                'name': name,
            }),
            from_email=settings.DEFAULT_FROM_EMAIL,
            to=[email],
            reply_to=[settings.SALES_EMAIL],
            headers={
                'X-Auto-Response-Suppress': 'OOF, DR, RN, NRN, AutoReply',
                'Auto-Submitted': 'auto-generated',
                'X-Priority': '3',
                'Precedence': 'bulk',
            },
        )
        email.content_subtype = "html"
        email.send(fail_silently=False)

class ServiceViewSet(viewsets.ModelViewSet):
    queryset = Service.objects.filter(is_active=True)
    serializer_class = ServiceSerializer
    permission_classes = [AllowAny]
    lookup_field = 'slug'
    filter_backends = [django_filters.DjangoFilterBackend, SearchFilter, OrderingFilter]  # Updated
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
    filter_backends = [django_filters.DjangoFilterBackend, SearchFilter, OrderingFilter]  # Updated
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
        post = self.get_object()
        post.views += 1
        post.save(update_fields=['views'])
        return Response({'status': 'view count updated'})

class CommentViewSet(viewsets.ModelViewSet):
    serializer_class = CommentSerializer
    permission_classes = [IsAuthenticated]
    filter_backends = [django_filters.DjangoFilterBackend, OrderingFilter]  # Updated
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