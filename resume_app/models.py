import os
from django.db import models
from django.contrib.auth.models import User, Group 
from django.utils import timezone

USER_ROLE_CHOICES = [
    ('jobseeker', 'Job Seeker'),
    ('recruiter', 'Recruiter'),
]

CONVERSATION_STATUS_CHOICES = [
    ('active', 'Active'),
    ('archived', 'Archived'),
    ('closed', 'Closed'),
]

MESSAGE_SENDER_CHOICES = [
    ('user', 'User'),
    ('ai', 'AI'),
]

UPGRADED_RESUME_STATUS_CHOICES = [
    ('pending', 'Pending'),
    ('in_progress', 'In Progress'),
    ('completed', 'Completed'),
    ('failed', 'Failed'),
]

RECRUITER_ANALYSIS_STATUS_CHOICES = [
    ('pending', 'Pending'),
    ('processing', 'Processing'),
    ('completed', 'Completed'),
    ('failed', 'Failed'),
]

CANDIDATE_RESUME_STATUS_CHOICES = [
    ('uploaded', 'Uploaded'),
    ('processing', 'Processing'),
    ('analyzed', 'Analyzed'),
    ('error', 'Error'),
]

class Resume(models.Model):
    
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='resumes',
        help_text="The user who uploaded this resume."
    )
    
    uploaded_as_role = models.CharField(
        max_length=20,
        choices=USER_ROLE_CHOICES,
        default='jobseeker',
        help_text="The role the user had when this resume was uploaded."
    )
    title = models.CharField(
        max_length=255,
        blank=True,
        help_text="Optional title for the resume (e.g., 'Software Engineer Resume')."
    )
    file = models.FileField(
        upload_to='resumes/', 
        help_text="The original uploaded resume file."
    )
    extracted_text = models.TextField(
        blank=True,
        help_text="Text extracted from the resume file for analysis."
    )
    upload_date = models.DateTimeField(
        auto_now_add=True,
        help_text="Timestamp when the resume was uploaded."
    )
    file_type = models.CharField(
        max_length=50,
        blank=True,
        help_text="Detected file type (e.g., application/pdf)."
    )
    file_size = models.BigIntegerField(
        null=True, blank=True,
        help_text="Size of the uploaded file in bytes."
    )
    is_active = models.BooleanField(
        default=True,
        help_text="Indicates if the resume is currently active/usable."
    )
    
    rewritten_content = models.TextField(
        blank=True, null=True,
        help_text="AI-generated rewritten version of the resume (Markdown)."
    )

    class Meta:
        ordering = ['-upload_date']
        verbose_name = "Resume"
        verbose_name_plural = "Resumes"

    def __str__(self):
        filename = os.path.basename(self.file.name) if self.file else "No file"
        
        role_display = self.get_uploaded_as_role_display()
        return f"Resume {self.id} ({role_display}): '{self.title or filename}' by {self.user.username}"

    def save(self, *args, **kwargs):
        
        super().save(*args, **kwargs)


class Analysis(models.Model):
    
    resume = models.OneToOneField( 
        Resume,
        on_delete=models.CASCADE,
        related_name='jobseeker_analysis',
        help_text="The resume that was analyzed."
    )
    analysis_date = models.DateTimeField(
        auto_now_add=True,
        help_text="Timestamp when the analysis was performed."
    )
    user_role = models.CharField( 
        max_length=20,
        choices=USER_ROLE_CHOICES,
        default='jobseeker',
        help_text="Role of the user when analysis was done (mostly jobseeker here)."
    )
    
    skills_score = models.IntegerField(null=True, blank=True, help_text="Score (0-100) for skills match.")
    experience_score = models.IntegerField(null=True, blank=True, help_text="Score (0-100) for experience relevance.")
    education_score = models.IntegerField(null=True, blank=True, help_text="Score (0-100) for education alignment.")
    overall_score = models.IntegerField(null=True, blank=True, help_text="Overall score (0-100) for the resume.")
    key_insights = models.JSONField(null=True, blank=True, help_text="AI-generated key insights (list or dict).")
    improvement_suggestions = models.JSONField(null=True, blank=True, help_text="AI-generated improvement suggestions (list or dict).")
    class Meta:
        ordering = ['-analysis_date']
        verbose_name = "Resume Analysis (Job Seeker)"
        verbose_name_plural = "Resume Analyses (Job Seeker)"

    def __str__(self):
        return f"Analysis for Resume {self.resume.id} ({self.analysis_date.strftime('%Y-%m-%d')})"


class ChatMessage(models.Model):
    
    resume = models.ForeignKey(
        Resume,
        null=True, blank=True,
        on_delete=models.SET_NULL,
        related_name='chat_messages', 
        help_text="The resume this chat message relates to (optional)."
        )
    user = models.ForeignKey(
        User,
        null=True, blank=True, 
        on_delete=models.SET_NULL,
        related_name='chat_messages', 
        help_text="The authenticated user who sent/received this message (optional)."
        )
    sender = models.CharField(
        max_length=10,
        choices=MESSAGE_SENDER_CHOICES 
        )
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']
        verbose_name = "Chat Message"
        verbose_name_plural = "Chat Messages"

    def __str__(self):
        sender_id = self.user.username if self.user else self.sender
        resume_id = f"Resume {self.resume.id}" if self.resume else "No Resume"
        return f"Chat Msg ({sender_id} on {resume_id}): {self.message[:50]}..."


class Conversation(models.Model):
    
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='conversations',
        help_text="The user participating in the conversation."
    )
    resume = models.ForeignKey(
        Resume,
        null=True, blank=True,
        on_delete=models.SET_NULL,
        related_name='conversations',
        help_text="The resume being discussed (optional)."
    )
    analysis = models.ForeignKey(
        Analysis, 
        null=True, blank=True,
        on_delete=models.SET_NULL,
        related_name='conversations',
        help_text="The analysis being discussed (optional)."
    )
    start_date = models.DateTimeField(
        auto_now_add=True,
        help_text="Timestamp when the conversation started."
    )
    end_date = models.DateTimeField(
        null=True, blank=True,
        help_text="Timestamp when the conversation ended (if applicable)."
    )
    title = models.CharField(
        max_length=255,
        blank=True,
        help_text="Optional title for the conversation."
    )
    status = models.CharField(
        max_length=20,
        choices=CONVERSATION_STATUS_CHOICES,
        default='active',
        help_text="Current status of the conversation."
    )

    class Meta:
        ordering = ['-start_date']
        verbose_name = "Conversation Session"
        verbose_name_plural = "Conversation Sessions"

    def __str__(self):
        topic = f"Resume {self.resume.id}" if self.resume else f"Analysis {self.analysis.id}" if self.analysis else "General"
        return f"Conversation {self.id}: {self.title or topic} with {self.user.username}"


class Message(models.Model):
    
    conversation = models.ForeignKey(
        Conversation,
        on_delete=models.CASCADE,
        related_name='messages', 
        help_text="The conversation this message belongs to."
    )
    sender = models.CharField(
        max_length=10,
        choices=MESSAGE_SENDER_CHOICES, 
        help_text="Who sent the message ('user' or 'ai')."
    )
    user_sender = models.ForeignKey( 
        User,
        null=True, blank=True,
        on_delete=models.SET_NULL,
        related_name='sent_messages', 
        help_text="The user who sent this message (if sender='user')."
    )
    content = models.TextField(
        help_text="The text content of the message."
    )
    timestamp = models.DateTimeField(
        auto_now_add=True,
        help_text="Timestamp when the message was created."
    )
    is_read = models.BooleanField(
        default=False,
        help_text="Indicates if the message has been read."
    )

    class Meta:
        ordering = ['timestamp']
        verbose_name = "Conversation Message"
        verbose_name_plural = "Conversation Messages"

    def __str__(self):
        sender_name = self.user_sender.username if self.user_sender else self.sender.upper()
        return f"Msg {self.id} in Conv {self.conversation.id} by {sender_name}: {self.content[:50]}..."


class UpgradedResume(models.Model):
    
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='upgraded_resumes',
        help_text="The user this upgraded resume belongs to."
    )
    original_resume = models.ForeignKey(
        Resume,
        on_delete=models.SET_NULL, null=True, blank=True,
        related_name='upgraded_versions',
        help_text="The original resume this was based on (optional)."
    )
    ai_raw_response = models.JSONField(
        null=True,
        blank=True,
        help_text="Raw JSON response object received from the AI during rewrite/revision."
    )
    analysis = models.ForeignKey(
        Analysis, 
        on_delete=models.SET_NULL, null=True, blank=True,
        related_name='upgraded_resumes',
        help_text="The analysis that might have triggered this upgrade (optional)."
    )
    conversation = models.ForeignKey(
        Conversation, 
        on_delete=models.SET_NULL, null=True, blank=True,
        related_name='upgraded_resumes',
        help_text="The conversation context for this upgrade (optional)."
    )
    content_markdown = models.TextField(
        blank=True,
        help_text="The upgraded resume content, typically in Markdown format."
    )
    creation_date = models.DateTimeField(
        auto_now_add=True,
        help_text="Timestamp when this upgraded version was created."
    )
    last_updated_date = models.DateTimeField(
        auto_now=True,
        help_text="Timestamp when this upgraded version was last modified."
    )
    title = models.CharField(
        max_length=255,
        blank=True,
        help_text="Optional title for this upgraded version (e.g., 'v2 - Tech Lead Focus')."
    )
    status = models.CharField(
        max_length=20,
        choices=UPGRADED_RESUME_STATUS_CHOICES,
        default='completed',
        help_text="Status of the generation/revision process."
    )
    revision_count = models.IntegerField(
        default=0,
        help_text="Number of revisions applied to this specific upgraded version."
    )

    class Meta:
        ordering = ['-creation_date']
        verbose_name = "Upgraded Resume"
        verbose_name_plural = "Upgraded Resumes"

    def __str__(self):
        orig_id = f"Orig: {self.original_resume.id}" if self.original_resume else "No Original Linked"
        return f"Upgraded Resume {self.id}: '{self.title or 'Untitled'}' ({orig_id}) by {self.user.username}"


class JobRequirement(models.Model):
    
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        limit_choices_to={'groups__name': 'Recruiter'}, 
        related_name='job_requirements',
        help_text="The recruiter who owns this job requirement."
    )
    title = models.CharField(
        max_length=255,
        blank=True, 
        help_text="Title of the job (e.g., 'Senior Backend Engineer')."
    )
    content = models.TextField(
        help_text="The full text content of the job description/requirements."
    )
    upload_date = models.DateTimeField(
        auto_now_add=True,
        help_text="Timestamp when the requirement was created/uploaded."
    )

    class Meta:
        ordering = ['-upload_date']
        verbose_name = "Job Requirement"
        verbose_name_plural = "Job Requirements"

    def __str__(self):
        return f"Job Requirement {self.id}: '{self.title or 'Untitled'}' by {self.user.username}"


class RecruiterAnalysis(models.Model):
   
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        limit_choices_to={'groups__name': 'Recruiter'},
        related_name='recruiter_analyses',
        help_text="The recruiter who initiated this analysis."
    )
    job_requirement = models.ForeignKey(
        JobRequirement,
        on_delete=models.SET_NULL, null=True, 
        related_name='analyses',
        help_text="The job requirement used for this analysis."
    )
    analysis_date = models.DateTimeField(
        auto_now_add=True,
        help_text="Timestamp when the analysis was performed."
    )
    job_description_snapshot = models.TextField(
        blank=True,
        help_text="Snapshot of the job description used for this specific analysis."
    )
    
    comparative_analysis = models.JSONField(
        null=True, blank=True,
        help_text="AI-generated comparative analysis output (JSON structure)."
    )
    ranking = models.JSONField(
        null=True, blank=True,
        help_text="Ordered list of candidate identifiers (e.g., filenames or IDs)."
    )
    candidate_analysis_details = models.JSONField(
        null=True, blank=True,
        help_text="Detailed analysis results for each candidate in JSON format."
    )
    status = models.CharField(
        max_length=20,
        choices=RECRUITER_ANALYSIS_STATUS_CHOICES,
        default='pending',
        help_text="Status of the recruiter analysis process."
    )

    class Meta:
        ordering = ['-analysis_date']
        verbose_name = "Recruiter Analysis"
        verbose_name_plural = "Recruiter Analyses"

    def __str__(self):
        req_title = self.job_requirement.title if self.job_requirement else "No Job Req Linked"
        return f"Recruiter Analysis {self.id} for '{req_title}' by {self.user.username}"


class CandidateResume(models.Model):
    
    recruiter_analysis = models.ForeignKey(
        RecruiterAnalysis,
        on_delete=models.CASCADE,
        related_name='candidate_resumes',
        help_text="The recruiter analysis this resume belongs to."
    )
    resume_identifier = models.CharField(
        max_length=255,
        help_text="Identifier for this resume (e.g., original filename)."
    )
    file = models.FileField(
        upload_to='candidate_resumes/', 
        help_text="The candidate resume file uploaded by the recruiter."
    )
    extracted_text = models.TextField(
        blank=True,
        help_text="Text extracted from the candidate's resume file."
    )
    upload_date = models.DateTimeField(
        auto_now_add=True,
        help_text="Timestamp when this candidate resume was added."
    )
    status = models.CharField(
        max_length=20,
        choices=CANDIDATE_RESUME_STATUS_CHOICES,
        default='uploaded',
        help_text="Processing status of this specific candidate resume."
    )

    class Meta:
        ordering = ['upload_date']
        verbose_name = "Candidate Resume (for Recruiter)"
        verbose_name_plural = "Candidate Resumes (for Recruiter)"

    def __str__(self):
        return f"Candidate Resume '{self.resume_identifier}' for Analysis {self.recruiter_analysis.id}"

    def save(self, *args, **kwargs):
        
        super().save(*args, **kwargs)