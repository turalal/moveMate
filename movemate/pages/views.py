# pages/views.py
from rest_framework import generics, status, viewsets
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.decorators import action
from rest_framework.filters import SearchFilter, OrderingFilter
from rest_framework.throttling import AnonRateThrottle

from django.shortcuts import get_object_or_404
from django.core.mail import send_mail, BadHeaderError
from django.template.loader import render_to_string
from django.conf import settings
from django.utils.html import strip_tags
from django.db.models import Count
from django.utils import timezone

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

class BlogPostFilter(django_filters.FilterSet):  # Changed from filters.FilterSet
    class Meta:
        model = BlogPost
        fields = {
            'title': ['icontains'],
            'category': ['exact'],
            'created_at': ['gte', 'lte'],
            'author': ['exact'],
            'status': ['exact'],
        }


class ContactView(generics.CreateAPIView):
    throttle_classes = [AnonRateThrottle]
    throttle_scope = 'contact'
    queryset = Contact.objects.all()
    serializer_class = ContactSerializer
    permission_classes = [AllowAny]

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        try:
            contact = self.perform_create(serializer)
            self.send_notification_email(contact)
            self.send_confirmation_email(contact)
            
            return Response(
                {"message": "Message sent successfully"},
                status=status.HTTP_201_CREATED
            )
        except BadHeaderError:
            logger.error("Invalid header found in email submission")
            return Response(
                {"error": "Invalid header found in email"},
                status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            logger.error(f"Error in contact form submission: {str(e)}")
            return Response(
                {"error": "An error occurred while processing your request"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def perform_create(self, serializer):
        return serializer.save()

    def send_notification_email(self, contact):
        try:
            subject = f'New Contact Form Submission - {contact.subject}'
            html_message = render_to_string('emails/contact_notification.html', {
                'name': contact.name,
                'email': contact.email,
                'subject': contact.subject,
                'message': contact.message,
            })
            plain_message = strip_tags(html_message)

            send_mail(
                subject=subject,
                message=plain_message,
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=[settings.SALES_EMAIL],
                html_message=html_message,
                fail_silently=True,
            )
        except Exception as e:
            logger.error(f"Error sending notification email: {str(e)}")
            raise

    def send_confirmation_email(self, contact):
        try:
            subject = 'Thank you for contacting MoveMate'
            html_message = render_to_string('emails/contact_confirmation.html', {
                'name': contact.name,
            })
            plain_message = strip_tags(html_message)

            send_mail(
                subject=subject,
                message=plain_message,
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=[contact.email],
                html_message=html_message,
                fail_silently=True,
            )
        except Exception as e:
            logger.error(f"Error sending confirmation email: {str(e)}")
            raise

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