from rest_framework import serializers
from .models import Resume, ChatMessage

#we are creating a serializer class for the Resume model so that we can convert the pdf input into something that can be stored in the database and can be snalyzed by the ai

class ResumeSerializer(serializers.ModelSerializer):
    class Meta:
        model = Resume
        fields = '__all__'


class ChatMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatMessage
        fields = '__all__'