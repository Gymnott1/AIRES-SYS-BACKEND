from django.db import models
from django.contrib.auth.models import User

class Resume(models.Model):
    user = models.ForeignKey(User, null=True, blank=True, on_delete=models.SET_NULL)
    file = models.FileField(upload_to='resumes/')
    text = models.TextField(blank=True)
    analysis = models.TextField(blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    # Add these new fields for resume rewriting functionality
    rewritten_content = models.TextField(blank=True)
    last_revision_date = models.DateTimeField(null=True, blank=True)
    revision_count = models.IntegerField(default=0)

    def __str__(self):
        return f"Resume {self.id} - {self.user or 'Anonymous'}"

class ChatMessage(models.Model):
    SENDER_CHOICES = (
        ('user', 'User'),
        ('ai', 'AI'),
    )
    resume = models.ForeignKey(Resume, null=True, blank=True, on_delete=models.SET_NULL)
    user = models.ForeignKey(User, null=True, blank=True, on_delete=models.SET_NULL)
    sender = models.CharField(max_length=10, choices=SENDER_CHOICES)
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.sender}: {self.message[:30]}"


