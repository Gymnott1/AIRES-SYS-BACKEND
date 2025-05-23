# Generated by Django 5.1.6 on 2025-04-18 11:22

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('resume_app', '0005_alter_resume_options_remove_resume_analysis_and_more'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='analysis',
            options={'ordering': ['-analysis_date'], 'verbose_name': 'Resume Analysis (Job Seeker)', 'verbose_name_plural': 'Resume Analyses (Job Seeker)'},
        ),
        migrations.AlterModelOptions(
            name='chatmessage',
            options={'ordering': ['created_at'], 'verbose_name': 'Chat Message', 'verbose_name_plural': 'Chat Messages'},
        ),
        migrations.AlterModelOptions(
            name='conversation',
            options={'ordering': ['-start_date'], 'verbose_name': 'Conversation Session', 'verbose_name_plural': 'Conversation Sessions'},
        ),
        migrations.AlterModelOptions(
            name='message',
            options={'ordering': ['timestamp'], 'verbose_name': 'Conversation Message', 'verbose_name_plural': 'Conversation Messages'},
        ),
        migrations.AddField(
            model_name='resume',
            name='rewritten_content',
            field=models.TextField(blank=True, help_text='AI-generated rewritten version of the resume (Markdown).', null=True),
        ),
        migrations.AddField(
            model_name='resume',
            name='uploaded_as_role',
            field=models.CharField(choices=[('jobseeker', 'Job Seeker'), ('recruiter', 'Recruiter')], default='jobseeker', help_text='The role the user had when this resume was uploaded.', max_length=20),
        ),
        migrations.AlterField(
            model_name='analysis',
            name='resume',
            field=models.OneToOneField(help_text='The resume that was analyzed.', on_delete=django.db.models.deletion.CASCADE, related_name='jobseeker_analysis', to='resume_app.resume'),
        ),
        migrations.AlterField(
            model_name='candidateresume',
            name='upload_date',
            field=models.DateTimeField(auto_now_add=True, help_text='Timestamp when this candidate resume was added.'),
        ),
        migrations.AlterField(
            model_name='chatmessage',
            name='resume',
            field=models.ForeignKey(blank=True, help_text='The resume this chat message relates to (optional).', null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='chat_messages', to='resume_app.resume'),
        ),
        migrations.AlterField(
            model_name='chatmessage',
            name='user',
            field=models.ForeignKey(blank=True, help_text='The authenticated user who sent/received this message (optional).', null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='chat_messages', to=settings.AUTH_USER_MODEL),
        ),
        migrations.AlterField(
            model_name='jobrequirement',
            name='title',
            field=models.CharField(blank=True, help_text="Title of the job (e.g., 'Senior Backend Engineer').", max_length=255),
        ),
        migrations.AlterField(
            model_name='jobrequirement',
            name='user',
            field=models.ForeignKey(help_text='The recruiter who owns this job requirement.', limit_choices_to={'groups__name': 'Recruiter'}, on_delete=django.db.models.deletion.CASCADE, related_name='job_requirements', to=settings.AUTH_USER_MODEL),
        ),
        migrations.AlterField(
            model_name='message',
            name='is_read',
            field=models.BooleanField(default=False, help_text='Indicates if the message has been read.'),
        ),
        migrations.AlterField(
            model_name='recruiteranalysis',
            name='comparative_analysis',
            field=models.JSONField(blank=True, help_text='AI-generated comparative analysis output (JSON structure).', null=True),
        ),
        migrations.AlterField(
            model_name='recruiteranalysis',
            name='user',
            field=models.ForeignKey(help_text='The recruiter who initiated this analysis.', limit_choices_to={'groups__name': 'Recruiter'}, on_delete=django.db.models.deletion.CASCADE, related_name='recruiter_analyses', to=settings.AUTH_USER_MODEL),
        ),
    ]
