# pages/tests.py
from django.test import TestCase
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase
from django.contrib.auth import get_user_model
from .models import Service, BlogPost, BlogCategory, Comment

User = get_user_model()

class ServiceTests(APITestCase):
    def setUp(self):
        self.service = Service.objects.create(
            title='Test Service',
            description='Test Description'
        )

    def test_list_services(self):
        url = reverse('service-list')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data['results']), 1)

class BlogPostTests(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )
        self.category = BlogCategory.objects.create(
            name='Test Category',
            slug='test-category'
        )
        self.post = BlogPost.objects.create(
            title='Test Post',
            slug='test-post',
            content='Test Content',
            category=self.category,
            author=self.user
        )

    def test_list_posts(self):
        url = reverse('post-list')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data['results']), 1)

    def test_post_detail(self):
        url = reverse('post-detail', kwargs={'slug': self.post.slug})
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['title'], 'Test Post')

class CommentTests(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )
        self.category = BlogCategory.objects.create(
            name='Test Category',
            slug='test-category'
        )
        self.post = BlogPost.objects.create(
            title='Test Post',
            slug='test-post',
            content='Test Content',
            category=self.category,
            author=self.user
        )

    def test_create_comment(self):
        self.client.force_authenticate(user=self.user)
        url = reverse('comment-create', kwargs={'post_slug': self.post.slug})
        data = {'content': 'Test Comment'}
        response = self.client.post(url, data)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)