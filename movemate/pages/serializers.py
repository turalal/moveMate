# pages/serializers.py
from rest_framework import serializers
from .models import Contact, Service, BlogPost, BlogCategory, Comment
from .validators import EmailDomainValidator


class ContactSerializer(serializers.ModelSerializer):
    class Meta:
        model = Contact
        fields = ('name', 'email', 'subject', 'message')

    def validate_email(self, value):
        if not value:
            raise serializers.ValidationError("Email is required")
            
        # Validate email domain
        EmailDomainValidator.validate_domain(value)
        
        return value.lower()  # Normalize email to lowercase
    
    def validate_name(self, value):
        if not value:
            raise serializers.ValidationError("Name is required")
        return value
    
class ServiceSerializer(serializers.ModelSerializer):
    class Meta:
        model = Service
        fields = '__all__'

class CommentSerializer(serializers.ModelSerializer):
    author_name = serializers.CharField(source='author.username', read_only=True)

    class Meta:
        model = Comment
        fields = ['id', 'content', 'author', 'author_name', 'created_at']
        read_only_fields = ['author']

class BlogPostSerializer(serializers.ModelSerializer):
    category_name = serializers.CharField(source='category.name', read_only=True)
    author_name = serializers.CharField(source='author.username', read_only=True)
    comments = CommentSerializer(many=True, read_only=True)

    class Meta:
        model = BlogPost
        fields = ['id', 'title', 'slug', 'content', 'image', 'category', 
                 'category_name', 'author', 'author_name', 'comments', 
                 'created_at', 'updated_at']

class BlogCategorySerializer(serializers.ModelSerializer):
    class Meta:
        model = BlogCategory
        fields = '__all__'