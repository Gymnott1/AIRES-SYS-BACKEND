from django.contrib import admin
from .models import (
    Resume,
    Analysis,
    ChatMessage,
    Conversation,
    Message,
    UpgradedResume,
    JobRequirement,
    RecruiterAnalysis,
    CandidateResume
)

@admin.register(Resume)
class ResumeAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'title', 'upload_date', 'is_active', 'file')
    list_filter = ('is_active', 'upload_date', 'user')
    search_fields = ('title', 'user__username', 'file')
    readonly_fields = ('upload_date',)

@admin.register(Analysis)
class AnalysisAdmin(admin.ModelAdmin):
    list_display = ('id', 'resume', 'analysis_date', 'user_role', 'overall_score')
    list_filter = ('user_role', 'analysis_date')
    search_fields = ('resume__id', 'resume__user__username')
    readonly_fields = ('analysis_date',)

@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ('id', 'resume', 'user', 'sender', 'created_at')
    list_filter = ('sender', 'created_at', 'user')
    search_fields = ('message', 'user__username', 'resume__id')
    readonly_fields = ('created_at',)

@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'resume', 'analysis', 'start_date', 'status')
    list_filter = ('status', 'start_date', 'user')
    search_fields = ('title', 'user__username', 'resume__id', 'analysis__id')
    readonly_fields = ('start_date', 'end_date')

@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ('id', 'conversation', 'sender', 'user_sender', 'timestamp', 'is_read')
    list_filter = ('sender', 'is_read', 'timestamp', 'conversation')
    search_fields = ('content', 'user_sender__username', 'conversation__id')
    readonly_fields = ('timestamp',)

@admin.register(UpgradedResume)
class UpgradedResumeAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'original_resume', 'title', 'creation_date', 'status', 'revision_count')
    list_filter = ('status', 'creation_date', 'user')
    search_fields = ('title', 'user__username', 'original_resume__id')
    readonly_fields = ('creation_date', 'last_updated_date')

@admin.register(JobRequirement)
class JobRequirementAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'title', 'upload_date')
    list_filter = ('upload_date', 'user')
    search_fields = ('title', 'content', 'user__username')
    readonly_fields = ('upload_date',)

@admin.register(RecruiterAnalysis)
class RecruiterAnalysisAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'job_requirement', 'analysis_date', 'status')
    list_filter = ('status', 'analysis_date', 'user')
    search_fields = ('user__username', 'job_requirement__title', 'job_requirement__id')
    readonly_fields = ('analysis_date',)

@admin.register(CandidateResume)
class CandidateResumeAdmin(admin.ModelAdmin):
    list_display = ('id', 'recruiter_analysis', 'resume_identifier', 'upload_date', 'status', 'file')
    list_filter = ('status', 'upload_date', 'recruiter_analysis')
    search_fields = ('resume_identifier', 'recruiter_analysis__id', 'recruiter_analysis__user__username')
    readonly_fields = ('upload_date',)
