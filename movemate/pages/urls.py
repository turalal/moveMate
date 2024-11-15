# pages/urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

# Create a router and register our viewsets with it
router = DefaultRouter()
router.register(r'services', views.ServiceViewSet, basename='service')
router.register(r'blog/categories', views.BlogCategoryViewSet, basename='category')
router.register(r'blog/posts', views.BlogPostViewSet, basename='post')
router.register(
    r'blog/posts/(?P<post_slug>[\w-]+)/comments',
    views.CommentViewSet,
    basename='comment'
)

# The API URLs are now determined automatically by the router
urlpatterns = [
    path('', include(router.urls)),
    path('contact/', views.ContactView.as_view(), name='contact'),
]

# Optional: Add additional custom URLs if needed
# urlpatterns += [
#     path('custom-endpoint/', views.CustomView.as_view(), name='custom'),
# ]