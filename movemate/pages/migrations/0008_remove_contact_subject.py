# Generated by Django 5.1.3 on 2024-11-16 12:59

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('pages', '0007_alter_contact_subject'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='contact',
            name='subject',
        ),
    ]
