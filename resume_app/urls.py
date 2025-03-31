from django.urls import path
from .views import ValidateResumeView, UploadResumeView, AnalyzeResumeView, ChatView, SignupView, LoginView, LogoutView, account_detail, update_profile, update_password, delete_account, ChatMessagesView, user_conversations 
from . import views

urlpatterns = [
    path('api/validate_resume/', ValidateResumeView.as_view(), name='validate-resume'),
    path('upload_resume/', UploadResumeView.as_view(), name='upload_resume'),
    path('analyze_resume/', AnalyzeResumeView.as_view(), name='analyze_resume'),
    path('chat/', ChatView.as_view(), name='chat'),
    path('signup/', SignupView.as_view(), name='signup'),
    path('login/', LoginView.as_view(), name='login'),
    path('logout/', LogoutView.as_view(), name='logout'),
    path('account/', account_detail, name='account_detail'),
    path('account/update/', update_profile, name='update_profile'),
    path('account/password/', update_password, name='update_password'),
    path('account/delete/', delete_account, name='delete_account'),
    path('chat-messages/', ChatMessagesView.as_view(), name='chat_messages'),
    path('user-conversations/', user_conversations, name='user_conversations'),
    
    path('rewrite_resume/', views.rewrite_resume, name='rewrite_resume'),
    path('revise_resume/', views.revise_resume, name='revise_resume'),
    path('generate_pdf/', views.generate_pdf, name='generate_pdf'),


    
]

