# pages/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('services/', views.ServiceListView.as_view(), name='service-list'),
    path('services/<int:pk>/', views.ServiceDetailView.as_view(), name='service-detail'),
    path('blog/categories/', views.BlogCategoryListView.as_view(), name='category-list'),
    path('blog/posts/', views.BlogPostListView.as_view(), name='post-list'),
    path('blog/posts/<slug:slug>/', views.BlogPostDetailView.as_view(), name='post-detail'),
    path('blog/posts/<slug:post_slug>/comments/', views.CommentCreateView.as_view(), name='comment-create'),
    path('blog/comments/<int:pk>/', views.CommentDeleteView.as_view(), name='comment-delete'),
    path('contact/', views.ContactView.as_view(), name='contact'),
]